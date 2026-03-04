#!/usr/bin/env python
import os
import re
import json
import shutil
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# =============================
# CONFIG
# =============================
# Modelo de TEXTO (HF estándar, guardado con save_pretrained)
TEXT_MODEL_DIR = "model_avvillas_beto"

# Priors de TIEMPO (JSON) guardado por training/train_text_plus_timeprior.py
TIME_PRIORS_JSON = "reentreno/Colombia/Avvillas/model/time_priors_subtag.json"

TEXT_COLUMN = "text"      # expected column in transcript CSV
WINDOW_SIZE = 14          # words per window
STRIDE      = 6           # step in words
MAX_LENGTH  = 64          # tokenizer max length
SMOOTH_K    = 0          # majority window on each side (0 to disable)

USE_TEXT_TAIL = False


# =============================
# LOAD MODEL + TIME PRIORS
# =============================
def _resolve_id2label(model):
    id2label = model.config.id2label
    # HF a veces guarda keys como strings
    if isinstance(id2label, dict):
        out = {}
        for k, v in id2label.items():
            try:
                out[int(k)] = v
            except Exception:
                # si ya es int o algo raro
                out[k] = v
        return out
    return {i: lab for i, lab in enumerate(id2label)}

def load_text_model(model_dir: str, device: torch.device):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"No existe el directorio del modelo texto: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    # prefer safetensors si existe
    safe_path = os.path.join(model_dir, "model.safetensors")
    use_safetensors = os.path.exists(safe_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        use_safetensors=use_safetensors,
    ).to(device)

    model.eval()
    id2label = _resolve_id2label(model)
    return tokenizer, model, id2label

def load_time_priors(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No existe el JSON de time priors: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # sanity mínimo
    required = ["labels", "rel_bins", "prob_time_bin_given_y", "prob_relbin_given_y"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"TIME_PRIORS_JSON inválido. Faltan keys: {missing}")

    # parámetros de fusión (si no hay, defaults)
    fp = meta.get("fusion_params", {})
    meta["_alpha"] = float(fp.get("alpha", 0.8))
    meta["_beta"]  = float(fp.get("beta", 0.8))
    meta["_eps"]   = 1e-12
    return meta

def normalize_priors_labels(meta):
    # si labels vienen como ints (0..N-1), convertirlos a strings usando meta["id2label"]
    labels = meta.get("labels", [])
    if labels and isinstance(labels[0], int):
        id2label = {int(k): v for k, v in meta["id2label"].items()}
        meta["labels"] = [id2label[i] for i in labels]

        def remap_prob_dict(d):
            out = {}
            for k, v in d.items():
                kk = int(k) if str(k).isdigit() else k
                out[id2label[kk]] = v
            return out

        if "prob_time_bin_given_y" in meta:
            meta["prob_time_bin_given_y"] = remap_prob_dict(meta["prob_time_bin_given_y"])
        if "prob_relbin_given_y" in meta:
            meta["prob_relbin_given_y"] = remap_prob_dict(meta["prob_relbin_given_y"])

    return meta


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer, text_model, id2label = load_text_model(TEXT_MODEL_DIR, device)
TIME_META = load_time_priors(TIME_PRIORS_JSON)
TIME_META = normalize_priors_labels(TIME_META)

# labels (orden final) se toma del JSON de priors
TIME_LABELS = TIME_META["labels"]
LABEL2IDX_TIME = {lab: i for i, lab in enumerate(TIME_LABELS)}

# Mapeo entre labels del modelo texto (id2label) y labels del time model (strings)
# Queremos que probs del texto queden en el mismo orden TIME_LABELS
TEXT_ID2LABEL = id2label
TEXT_LABEL2ID = {v: k for k, v in TEXT_ID2LABEL.items()}

# Validación: todas las TIME_LABELS deben existir en el modelo texto
missing_in_text = [lab for lab in TIME_LABELS if lab not in TEXT_LABEL2ID]
if missing_in_text:
    raise ValueError(
        "Tus time priors tienen labels que no existen en el modelo texto. "
        f"Faltan en texto: {missing_in_text}\n"
        f"Labels en texto: {sorted(list(TEXT_LABEL2ID.keys()))[:30]}..."
    )

ALPHA = TIME_META["_alpha"]
BETA  = TIME_META["_beta"]


# =============================
# HELPERS: tiempo + texto
# =============================
def parse_spanish_decimal(x):
    """'303,75256' -> 303.75256; returns np.nan if not parseable."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def preprocess_text_identity(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()

def _mmss(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "??:??"
    x = max(0.0, float(x))
    mm = int(x // 60)
    ss = int(round(x % 60))
    return f"{mm:02d}:{ss:02d}"

def rel_time_bin(rel):
    # mantén esto consistente con tu enrichment
    if rel is None or (isinstance(rel, float) and (np.isnan(rel) or np.isinf(rel))):
        return "unknown"
    if rel <= 0.15:
        return "early"
    if rel <= 0.65:
        return "mid"
    return "late"

def build_tail(t_mid, conv_dur):
    rel = (t_mid / conv_dur) if (conv_dur and not np.isnan(conv_dur) and conv_dur > 0) else np.nan
    rel_str = f"{rel:.2f}" if isinstance(rel, float) and not np.isnan(rel) else "UNK"
    bin_tag = rel_time_bin(rel)
    t_tag   = _mmss(t_mid)
    return f"[rt={rel_str}][bin={bin_tag}][t={t_tag}]"

def smooth_labels(labels, k=2):
    if k <= 0:
        return labels
    from collections import Counter
    n = len(labels)
    out = []
    for i in range(n):
        L = max(0, i - k)
        R = min(n, i + k + 1)
        win = labels[L:R]
        out.append(Counter(win).most_common(1)[0][0])
    return out


# =============================
# TIME FUSION (multimodal)
# =============================
def rel_bin_index(rel_time: float, rel_bins: list):
    """Retorna bin index en [0..len(rel_bins)-2] o -1 si unknown."""
    if rel_time is None or (isinstance(rel_time, float) and (np.isnan(rel_time) or np.isinf(rel_time))):
        return -1
    r = float(rel_time)
    for i in range(len(rel_bins) - 1):
        if rel_bins[i] <= r < rel_bins[i + 1]:
            return i
    return len(rel_bins) - 2

def apply_time_fusion(text_probs_time_order: np.ndarray, time_bin_str: str, rel_time_val: float) -> np.ndarray:
    """
    Fusiona probabilidades del modelo de texto con priors temporales en log-space:

    log P(y | x, t) ∝ log P_text(y | x)
                     + ALPHA * log P(time_bin | y)
                     + BETA  * log P(rel_bin  | y)

    Donde:
    - time_bin_str: {"early","mid","late","unknown"}
    - rel_bin: índice obtenido al discretizar rel_time_val sobre TIME_META["rel_bins"]

    Implementación:
    - Convierte text_probs a log.
    - Para cada label y:
        suma un término por time_bin y uno por rel_bin (si existe).
    - Estabiliza restando max(logp).
    - Exponencia y renormaliza para obtener probs finales.

    Args:
        text_probs_time_order (np.ndarray): probs del texto ordenadas según TIME_LABELS.
        time_bin_str (str): bin categórico.
        rel_time_val (float): tiempo relativo en [0..1] o NaN.

    Returns:
        np.ndarray shape [C], probs fusionadas.
    """

    eps = TIME_META["_eps"]
    prob_tb = TIME_META["prob_time_bin_given_y"]  # dict label -> dict time_bin -> prob
    prob_rb = TIME_META["prob_relbin_given_y"]    # dict label -> list prob per rel-bin
    rel_bins = TIME_META["rel_bins"]

    tb = "unknown" if (time_bin_str is None or (isinstance(time_bin_str, float) and np.isnan(time_bin_str))) else str(time_bin_str)
    rb = rel_bin_index(rel_time_val, rel_bins)

    logp = np.log(np.clip(text_probs_time_order, eps, 1.0))

    for i, y in enumerate(TIME_LABELS):
        # time_bin term
        p_tb = prob_tb[y].get(tb, prob_tb[y].get("unknown", 1e-6))
        logp[i] += ALPHA * np.log(max(float(p_tb), eps))

        # rel_bin term
        if rb >= 0:
            p_rb = prob_rb[y][rb]
            logp[i] += BETA * np.log(max(float(p_rb), eps))

    logp = logp - np.max(logp)
    p = np.exp(logp)
    p = p / np.sum(p)
    return p



# =============================
# CORE BUILDERS
# =============================
def build_word_timestamps(df_conversation: pd.DataFrame):
    """
    Expande conversación en palabras con (start,end) aproximando duración uniforme por palabra
    dentro de cada segmento.
    """
    word_timeline = []
    for _, row in df_conversation.iterrows():
        seg_text = preprocess_text_identity(row.get("text", ""))
        if not seg_text:
            continue

        seg_start = parse_spanish_decimal(row.get("start", np.nan))
        seg_end   = parse_spanish_decimal(row.get("end",   np.nan))

        words = seg_text.split()
        if not words:
            continue

        if np.isnan(seg_start) or np.isnan(seg_end) or seg_end <= seg_start:
            for w in words:
                word_timeline.append((w, np.nan, np.nan))
            continue

        total_dur = seg_end - seg_start
        dur_per   = total_dur / len(words)
        cur = seg_start
        for w in words:
            nxt = cur + dur_per
            word_timeline.append((w, cur, nxt))
            cur = nxt

    return word_timeline


def predict_text_probs(fragment_text: str, max_length: int = MAX_LENGTH) -> np.ndarray:
    """
    Ejecuta el modelo de texto y retorna un vector de probabilidades en el orden TIME_LABELS.

    Pasos:
    1) Preprocesa texto con `preprocess_text_identity` (normaliza espacios).
    2) Tokeniza con truncation + padding fijo a `max_length`.
    3) Ejecuta `text_model` en modo inference (torch.no_grad).
    4) Aplica softmax a logits → probs en el orden interno del modelo.
    5) Reordena esas probs al orden TIME_LABELS usando el mapping TEXT_LABEL2ID.
    6) Renormaliza por seguridad.

    Returns:
        np.ndarray shape [C] donde C = len(TIME_LABELS)
    """
    processed = preprocess_text_identity(fragment_text)
    inputs = tokenizer(
        processed,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = text_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # en orden de label ids del modelo

    # Reordenar a TIME_LABELS
    probs_time_order = np.zeros((len(TIME_LABELS),), dtype=np.float64)
    for i, lab in enumerate(TIME_LABELS):
        text_id = TEXT_LABEL2ID[lab]
        probs_time_order[i] = probs[text_id]

    # renormalizar por seguridad
    s = probs_time_order.sum()
    if s > 0:
        probs_time_order /= s
    return probs_time_order


def classify_fragment(fragment_text: str, rel_time_val: float, time_bin_str: str, max_length: int = MAX_LENGTH) -> dict:
    """
    Clasifica un fragmento (ventana) usando un enfoque multimodal sencillo:
      1) Modelo de texto produce probabilidades por clase (en orden TIME_LABELS).
      2) Se ajustan (fusionan) esas probabilidades con priors temporales:
         - prior por time_bin (early/mid/late/unknown)
         - prior por rel_bin (binning de rel_time)
      3) Se renormaliza y se toma argmax como etiqueta final.

    Args:
        fragment_text (str): Texto del fragmento (posiblemente con tail temporal).
        rel_time_val (float): Tiempo relativo [0..1] o NaN.
        time_bin_str (str): Bin categórico "early"/"mid"/"late"/"unknown".
        max_length (int): max tokens del tokenizer.

    Returns:
        dict:
          - predicted_subtag: etiqueta elegida
          - predicted_cluster: alias de compatibilidad
          - {label: prob} para cada label en TIME_LABELS
    """
    probs_text = predict_text_probs(fragment_text, max_length=max_length)
    probs_fused = apply_time_fusion(probs_text, time_bin_str=time_bin_str, rel_time_val=rel_time_val)

    pred_id = int(np.argmax(probs_fused))
    pred_label = TIME_LABELS[pred_id]

    result = {
        "predicted_subtag": pred_label,
        "predicted_cluster": pred_label,  # compat
    }

    # probas por label (en orden TIME_LABELS)
    for i, lab in enumerate(TIME_LABELS):
        result[lab] = float(probs_fused[i])

    return result


def classify_entire_conversation(
    df_conversation: pd.DataFrame,
    window_size: int,
    stride: int,
    max_length: int
) -> pd.DataFrame:
    """
    Aplica sliding window sobre una conversación para generar fragmentos de texto, enriquecerlos con
    información temporal y clasificarlos con un modelo de texto + fusión temporal (priors).

    Concepto:
    - Convierte el CSV (segmentos con start/end) en una secuencia de palabras con timestamps aproximados.
    - Recorre esa secuencia con ventanas (window_size palabras) moviéndose cada `stride` palabras.
    - Para cada ventana:
        * Calcula start/end (del primer y último token de la ventana).
        * Calcula tiempo medio t_mid.
        * Calcula rel_time = t_mid / duración_total_conversación.
        * time_bin = bin categórico (early/mid/late/unknown).
        * Construye fragment_text y (opcional) añade un “tail” con metadatos.
        * Llama `classify_fragment` para obtener probs + label final (texto + priors temporales).
    - (Opcional) Suaviza etiquetas con majority voting local (smooth_labels).

    Retorna:
        DataFrame con una fila por ventana. Incluye probabilidades por clase y predicciones.
    """
    word_timeline = build_word_timestamps(df_conversation)
    if not word_timeline:
        return pd.DataFrame(columns=[
            "text", "start", "end", "turn_idx",
            "rel_time", "time_bin",
            "predicted_subtag", "predicted_cluster"
        ])

    words = [w for (w, _, _) in word_timeline]
    total_words = len(words)

    ends = [e for (_, _, e) in word_timeline if e is not None and not np.isnan(e)]
    conv_dur = max(ends) if ends else np.nan

    results = []
    for i in range(0, max(1, total_words - window_size + 1), stride):
        w_slice = words[i: i + window_size] if total_words >= window_size else words
        if not w_slice:
            continue

        if total_words >= window_size:
            start_window = word_timeline[i][1]
            end_window   = word_timeline[i + window_size - 1][2]
        else:
            start_window = word_timeline[0][1]
            end_window   = word_timeline[-1][2]

        if not np.isnan(start_window) and not np.isnan(end_window):
            t_mid = (start_window + end_window) / 2.0
        else:
            t_mid = np.nan

        rel = (t_mid / conv_dur) if (
            conv_dur and not np.isnan(conv_dur) and conv_dur > 0 and not np.isnan(t_mid)
        ) else np.nan

        tb = rel_time_bin(rel)

        fragment_text = " ".join(w_slice)
        if USE_TEXT_TAIL:
            tail = build_tail(t_mid, conv_dur)
            fragment_text = f"{fragment_text} {tail}".strip()

        out = classify_fragment(fragment_text, rel_time_val=rel, time_bin_str=tb, max_length=max_length)
        out.update({
            "text": fragment_text,
            "start": start_window,
            "end": end_window,
            "turn_idx": i,
            "rel_time": rel if isinstance(rel, float) and not np.isnan(rel) else np.nan,
            "time_bin": tb,
        })
        results.append(out)

        if total_words < window_size:
            break

    df_result = pd.DataFrame(results)

    if SMOOTH_K and SMOOTH_K > 0 and not df_result.empty:
        df_result = df_result.sort_values("turn_idx").reset_index(drop=True)
        smooth = smooth_labels(df_result["predicted_subtag"].tolist(), k=SMOOTH_K)
        df_result["predicted_subtag_smooth"] = smooth
        df_result["predicted_cluster_smooth"] = smooth

    # columnas de probas
    prob_cols = [c for c in df_result.columns
                 if c not in [
                     "text", "start", "end", "turn_idx",
                     "rel_time", "time_bin",
                     "predicted_subtag", "predicted_cluster",
                     "predicted_subtag_smooth", "predicted_cluster_smooth"
                 ]]
    df_result.attrs["cluster_columns"] = prob_cols
    return df_result


# =============================
# CSV ENTRY POINTS
# =============================
def fit_csv_sliding_transformer(
    input_csv: str,
    output_csv: str,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    max_length: int = MAX_LENGTH
):
    """
    Clasifica una conversación completa contenida en un CSV, usando sliding window sobre palabras.

    Flujo:
    1) Lee `input_csv` a DataFrame.
    2) Valida que existan columnas mínimas: {"text","start","end"}.
    3) Llama `classify_entire_conversation` que devuelve un DataFrame por ventana con:
       - texto del fragmento
       - start/end de la ventana
       - features temporales (rel_time, time_bin)
       - predicciones (label + probabilidades por clase)
    4) Añade `conversation_id` (siempre 0) y guarda en `output_csv`.

    Args:
        input_csv (str): Ruta del CSV de transcripción.
        output_csv (str): Ruta donde se guardará el CSV clasificado.
        window_size (int): Tamaño de ventana en palabras.
        stride (int): Paso entre ventanas en palabras.
        max_length (int): max tokens para tokenización del fragmento.

    Returns:
        None (guarda archivo).
    """
    df = pd.read_csv(input_csv, encoding="utf-8")
    required = {"text", "start", "end"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"El CSV debe contener columnas {required} "
            "(se ignorarán columnas extra como confidence/speaker)."
        )

    classified_df = classify_entire_conversation(df, window_size, stride, max_length)
    classified_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Predicciones guardadas en: {output_csv}")


def fitCSVConversations(input_folder, output_folder, window_size, stride, max_length):
    """
    Orquestador batch:
    - Recorre todos los .csv en `input_folder`.
    - Para cada CSV, ejecuta `fit_csv_sliding_transformer` que genera un nuevo CSV
      con predicciones por ventanas (sliding window) y lo guarda en `output_folder`.
    - Si ocurre un error procesando un archivo, intenta aislar el audio emparejado
      y elimina la transcripción “rota” para evitar que el pipeline se quede atascado.

    Suposiciones:
    - `input_folder` contiene CSVs de transcripción (ej. "*_transcript.csv") con columnas mínimas:
      ["text", "start", "end"].
    - Existe un “base_folder” un nivel arriba de `input_folder`, donde viven audios .mp3
      con un nombre correlacionado (ej. base_name + ".mp3").
    - Si el CSV se llama "..._transcript.csv", el audio se busca como "... .mp3"
      (sin el sufijo "_transcript").

    Args:
        input_folder (str): Carpeta de entrada con CSVs de transcripción.
        output_folder (str): Carpeta de salida donde se guardan CSVs clasificados.
        window_size (int): Tamaño de ventana (en palabras) usado por sliding window.
        stride (int): Paso (en palabras) para mover la ventana.
        max_length (int): Longitud máxima (tokens) para el tokenizer del modelo de texto.

    Side effects:
        - Crea `output_folder` si no existe.
        - Crea carpeta `isolated/` en el folder padre de `input_folder`.
        - En caso de error: mueve audio a `isolated/` y borra transcripción original.
    """
    print(f"CALIFICANDO {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

    base_folder = os.path.abspath(os.path.join(input_folder, os.pardir))
    isolated_folder = os.path.join(base_folder, "isolated")
    os.makedirs(isolated_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_csv  = os.path.join(input_folder, filename)
            output_csv = os.path.join(output_folder, os.path.splitext(filename)[0] + ".csv")

            try:
                print(f"Processing {input_csv} -> {output_csv}")
                fit_csv_sliding_transformer(
                    input_csv=input_csv,
                    output_csv=output_csv,
                    window_size=window_size,
                    stride=stride,
                    max_length=max_length
                )
            except Exception as e:
                print(f"Error processing {filename}: {e}")

                # Move paired audio (if any) & remove the broken transcript
                base_name = filename.replace("_transcript.csv", "")
                audio_file = os.path.join(base_folder, base_name + ".mp3")
                transcript_file = os.path.join(base_folder, filename)

                if os.path.exists(audio_file):
                    shutil.move(audio_file, os.path.join(isolated_folder, os.path.basename(audio_file)))
                    print(f"Audio {audio_file} movido a {isolated_folder}")

                if os.path.exists(transcript_file):
                    os.remove(transcript_file)
                    print(f"Transcripción {transcript_file} eliminada")
