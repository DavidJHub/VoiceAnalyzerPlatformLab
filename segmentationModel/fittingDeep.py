#!/usr/bin/env python
import os
import re
import json
import shutil
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Tuple

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logger = logging.getLogger(__name__)

# =============================
# CONFIG
# =============================
load_dotenv()

TEXT_MODEL_DIR  = os.getenv("TEXT_MODEL_DIR")
TIME_PRIORS_JSON = os.getenv("TIME_PRIORS_JSON")

TEXT_COLUMN = "text"      # expected column in transcript CSV
WINDOW_SIZE = 14          # words per window
STRIDE      = 6           # step in words

# Issue #1 (critical): MAX_LENGTH must match training (trainingDeep.py MAX_LENGTH=128).
# Previously 64 — caused silent truncation and train/inference distribution mismatch.
MAX_LENGTH  = 128

SMOOTH_K    = 0           # majority window on each side (0 = disabled; CRF handles sequencing)

# Issue #2 (critical): model was fine-tuned on enriched text with temporal tail tags.
# Setting this to False creates a train/inference distribution shift on every prediction.
USE_TEXT_TAIL = True


# =============================
# HELPERS: time-bin thresholds (Issue #3)
# =============================
def _make_rel_time_bin_fn(thresholds: dict):
    """
    Returns a rel_time_bin function built from the thresholds stored in
    time_priors.json.  This guarantees that inference uses exactly the same
    categorical boundaries that were used when building the priors.

    Args:
        thresholds: dict with keys "early" and "mid" (upper inclusive bounds).
                    Everything above "mid" is classified as "late".
    """
    early_max = float(thresholds.get("early", 0.15))
    mid_max   = float(thresholds.get("mid",   0.65))

    def rel_time_bin(rel):
        if rel is None or (isinstance(rel, float) and (np.isnan(rel) or np.isinf(rel))):
            return "unknown"
        r = float(rel)
        if r <= early_max:
            return "early"
        if r <= mid_max:
            return "mid"
        return "late"

    return rel_time_bin


# Placeholder — overridden after TIME_META is loaded below.
def rel_time_bin(rel):
    if rel is None or (isinstance(rel, float) and (np.isnan(rel) or np.isinf(rel))):
        return "unknown"
    if rel <= 0.15:
        return "early"
    if rel <= 0.65:
        return "mid"
    return "late"


# =============================
# LOAD MODEL + TIME PRIORS
# =============================
def _resolve_id2label(model):
    id2label = model.config.id2label
    if isinstance(id2label, dict):
        out = {}
        for k, v in id2label.items():
            try:
                out[int(k)] = v
            except Exception:
                out[k] = v
        return out
    return {i: lab for i, lab in enumerate(id2label)}


def load_text_model(model_dir: str, device: torch.device):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"No existe el directorio del modelo texto: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

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

    required = ["labels", "rel_bins", "prob_time_bin_given_y", "prob_relbin_given_y"]
    missing  = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"TIME_PRIORS_JSON inválido. Faltan keys: {missing}")

    fp = meta.get("fusion_params", {})
    meta["_alpha"] = float(fp.get("alpha", 0.8))
    meta["_beta"]  = float(fp.get("beta",  0.3))
    # Issue #4: load gamma (defaults to 0 for backward compat with old JSON)
    meta["_gamma"] = float(fp.get("gamma", 0.0))
    meta["_eps"]   = 1e-12

    # Issue #3: load bin thresholds (defaults preserve old behaviour if missing)
    tbt = meta.get("time_bin_thresholds", {"early": 0.15, "mid": 0.65})
    meta["_time_bin_thresholds"] = tbt

    # Issue #4: label frequency prior (uniform fallback for old JSON)
    if "label_prior" in meta:
        meta["_prior_y"] = np.array(meta["label_prior"], dtype=np.float64)
    else:
        n = len(meta.get("labels", []))
        meta["_prior_y"] = np.ones(n, dtype=np.float64) / max(n, 1)

    return meta


def normalize_priors_labels(meta):
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


# =============================
# MODEL CONTEXT (multi-model support)
# =============================

@dataclass
class ModelContext:
    """
    Encapsula todo el estado de un modelo de segmentación cargado.
    Permite usar múltiples modelos (uno por sponsor) en el mismo proceso.
    """
    tokenizer:      object
    text_model:     object
    id2label:       Dict[int, str]
    TIME_META:      dict
    TIME_LABELS:    list
    LABEL2IDX_TIME: Dict[str, int]
    TEXT_ID2LABEL:  Dict[int, str]
    TEXT_LABEL2ID:  Dict[str, int]
    ALPHA:          float
    BETA:           float
    GAMMA:          float
    PRIOR_Y:        np.ndarray
    CRF_DECODER:    object          # CRFSequenceDecoder o None
    rel_time_bin:   Callable        # función (rel: float) -> str
    device:         torch.device
    model_dir:      str             # ruta local del modelo (para logging)


# Cache global: clave = (model_dir, time_priors_json)
_CONTEXT_CACHE: Dict[Tuple[str, str], ModelContext] = {}


def _build_context(model_dir: str, time_priors_json: str) -> ModelContext:
    """Carga un modelo completo desde disco y devuelve un ModelContext."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, mdl, i2l = load_text_model(model_dir, dev)

    meta = load_time_priors(time_priors_json)
    meta = normalize_priors_labels(meta)

    rtb = _make_rel_time_bin_fn(meta["_time_bin_thresholds"])
    t_labels = meta["labels"]
    txt_i2l  = i2l
    txt_l2i  = {v: k for k, v in txt_i2l.items()}

    missing = [lab for lab in t_labels if lab not in txt_l2i]
    if missing:
        raise ValueError(
            "time_priors tiene labels que no existen en el modelo texto. "
            f"Faltan: {missing}\n"
            f"Labels en texto: {sorted(list(txt_l2i.keys()))[:30]}..."
        )

    crf = None
    if meta.get("crf_params"):
        try:
            from segmentationModel.crfDecoder import CRFSequenceDecoder
            crf = CRFSequenceDecoder.from_dict(meta["crf_params"])
            logger.info("[CRF] Decoder cargado (%d tags, fitted=%s)", len(t_labels), crf._fitted)
        except Exception as exc:
            logger.warning("[CRF] No se pudo cargar el decoder: %s. Usando argmax.", exc)

    return ModelContext(
        tokenizer      = tok,
        text_model     = mdl,
        id2label       = i2l,
        TIME_META      = meta,
        TIME_LABELS    = t_labels,
        LABEL2IDX_TIME = {lab: i for i, lab in enumerate(t_labels)},
        TEXT_ID2LABEL  = txt_i2l,
        TEXT_LABEL2ID  = txt_l2i,
        ALPHA          = meta["_alpha"],
        BETA           = meta["_beta"],
        GAMMA          = meta["_gamma"],
        PRIOR_Y        = meta["_prior_y"],
        CRF_DECODER    = crf,
        rel_time_bin   = rtb,
        device         = dev,
        model_dir      = model_dir,
    )


def get_or_load_context(
    model_dir: Optional[str] = None,
    time_priors_json: Optional[str] = None,
) -> ModelContext:
    """
    Devuelve el ModelContext para la combinación (model_dir, time_priors_json).

    Si los parámetros son None se usan las variables de entorno TEXT_MODEL_DIR /
    TIME_PRIORS_JSON (comportamiento original del pipeline antes de modelos por sponsor).
    Los env vars se re-leen en tiempo de ejecución para no depender del momento
    de importación del módulo.  El contexto se cachea en memoria para evitar
    recargas en el mismo proceso.
    """
    # Re-leer env vars en tiempo de ejecución como fallback.
    # Esto cubre el caso en que el módulo fue importado antes de que load_dotenv()
    # cargara los valores (ej. módulo-nivel TEXT_MODEL_DIR podría ser None).
    mdir = model_dir      or os.getenv("TEXT_MODEL_DIR") or TEXT_MODEL_DIR
    tpj  = time_priors_json or os.getenv("TIME_PRIORS_JSON") or TIME_PRIORS_JSON

    if not mdir:
        raise ValueError(
            "No se encontró modelo de segmentación. "
            "Registra el modelo en vap_models o define la variable de entorno TEXT_MODEL_DIR."
        )
    if not tpj:
        raise ValueError(
            "No se encontró time_priors.json. "
            "Coloca el archivo dentro del directorio del modelo o define TIME_PRIORS_JSON."
        )

    cache_key = (mdir, tpj)
    if cache_key not in _CONTEXT_CACHE:
        logger.info("[fittingDeep] Cargando modelo desde: %s", mdir)
        _CONTEXT_CACHE[cache_key] = _build_context(mdir, tpj)
        logger.info("[fittingDeep] Modelo cargado y cacheado: %s", mdir)

    return _CONTEXT_CACHE[cache_key]


# ---- module-level load (modelo por defecto desde env vars) ----
# Se carga de forma lazy al primer uso para evitar errores en entornos donde
# TEXT_MODEL_DIR no está configurado pero se importa el módulo.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_default_context: Optional[ModelContext] = None

def _get_default_context() -> ModelContext:
    """Carga (si no está cacheado) y devuelve el contexto del modelo por defecto."""
    global _default_context
    if _default_context is None:
        _default_context = get_or_load_context(TEXT_MODEL_DIR, TIME_PRIORS_JSON)
    return _default_context


# ---------------------------------------------------------------------------
# Helper: devuelve un atributo del contexto por defecto (lazy).
# Usado únicamente por código legacy que acceda a estos nombres en el módulo.
# ---------------------------------------------------------------------------
def _ctx() -> ModelContext:
    """Devuelve el contexto del modelo por defecto, cargándolo si es necesario."""
    return _get_default_context()


# =============================
# HELPERS: time + text
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


def build_tail(t_mid, conv_dur, ctx: Optional["ModelContext"] = None):
    rel = (t_mid / conv_dur) if (conv_dur and not np.isnan(conv_dur) and conv_dur > 0) else np.nan
    rel_str = f"{rel:.2f}" if isinstance(rel, float) and not np.isnan(rel) else "UNK"
    _rtb = (ctx.rel_time_bin if ctx is not None else _ctx().rel_time_bin)
    bin_tag = _rtb(rel)
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
# TIME FUSION (multimodal)  — Issue #4: includes gamma / prior_y
# =============================
def rel_bin_index(rel_time: float, rel_bins: list):
    """Returns bin index in [0..len(rel_bins)-2] or -1 if unknown."""
    if rel_time is None or (isinstance(rel_time, float) and (np.isnan(rel_time) or np.isinf(rel_time))):
        return -1
    r = float(rel_time)
    for i in range(len(rel_bins) - 1):
        if rel_bins[i] <= r < rel_bins[i + 1]:
            return i
    return len(rel_bins) - 2


def apply_time_fusion(text_probs_time_order: np.ndarray,
                      time_bin_str: str,
                      rel_time_val: float,
                      gamma: float = 0.0,
                      prior_y: np.ndarray = None,
                      ctx: Optional["ModelContext"] = None) -> np.ndarray:
    """
    Fuses text-model probabilities with temporal priors in log-space:

      log P(y | x, t) ∝  log P_text(y | x)
                       + ALPHA * log P(time_bin | y)
                       + BETA  * log P(rel_bin  | y)
                       + gamma * log P(y)           [label frequency prior]

    Args:
        text_probs_time_order : shape [C] in TIME_LABELS order, sums to 1.
        time_bin_str          : "early" / "mid" / "late" / "unknown".
        rel_time_val          : relative position in [0..1] or NaN.
        gamma                 : weight for the label-frequency prior term.
        prior_y               : shape [C] empirical label probabilities from training.
        ctx                   : ModelContext a usar. Si None usa el modelo por defecto.

    Returns:
        np.ndarray shape [C], fused and renormalised probabilities.
    """
    _meta     = ctx.TIME_META    if ctx is not None else _ctx().TIME_META
    _t_labels = ctx.TIME_LABELS  if ctx is not None else _ctx().TIME_LABELS
    _alpha    = ctx.ALPHA        if ctx is not None else _ctx().ALPHA
    _beta     = ctx.BETA         if ctx is not None else _ctx().BETA

    eps      = _meta["_eps"]
    prob_tb  = _meta["prob_time_bin_given_y"]
    prob_rb  = _meta["prob_relbin_given_y"]
    rel_bins = _meta["rel_bins"]

    tb = ("unknown"
          if (time_bin_str is None or (isinstance(time_bin_str, float) and np.isnan(time_bin_str)))
          else str(time_bin_str))
    rb = rel_bin_index(rel_time_val, rel_bins)

    logp = np.log(np.clip(text_probs_time_order, eps, 1.0))

    for i, y in enumerate(_t_labels):
        p_tb = prob_tb[y].get(tb, prob_tb[y].get("unknown", 1e-6))
        logp[i] += _alpha * np.log(max(float(p_tb), eps))

        if rb >= 0:
            p_rb = prob_rb[y][rb]
            logp[i] += _beta * np.log(max(float(p_rb), eps))

        if gamma != 0.0 and prior_y is not None:
            logp[i] += gamma * np.log(max(float(prior_y[i]), eps))

    logp -= np.max(logp)
    p = np.exp(logp)
    p /= p.sum()
    return p


# =============================
# CORE BUILDERS
# =============================
def build_word_timestamps(df_conversation: pd.DataFrame):
    """
    Expands each segment into word-level (word, start, end) triples.

    Issue #5: word durations are now weighted by character count (a proxy
    for spoken duration) rather than uniformly distributed.  This produces
    more accurate rel_time estimates for sliding windows, especially when
    a segment contains words of very different lengths.
    """
    word_timeline = []
    for _, row in df_conversation.iterrows():
        seg_text  = preprocess_text_identity(row.get("text", ""))
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

        total_dur   = seg_end - seg_start
        # Character-weighted duration: longer words take proportionally more time
        char_lens   = [max(1, len(w)) for w in words]
        total_chars = sum(char_lens)

        cur = seg_start
        for w, cl in zip(words, char_lens):
            word_dur = total_dur * (cl / total_chars)
            nxt = cur + word_dur
            word_timeline.append((w, cur, nxt))
            cur = nxt

    return word_timeline


def predict_text_probs(
    fragment_text: str,
    max_length: int = MAX_LENGTH,
    ctx: Optional["ModelContext"] = None,
) -> np.ndarray:
    """
    Runs the text model and returns a probability vector in TIME_LABELS order.

    Args:
        fragment_text : text fragment to classify.
        max_length    : tokeniser max length.
        ctx           : ModelContext a usar. Si None usa el modelo por defecto.
    """
    _ctx_use    = ctx if ctx is not None else _ctx()
    _tok        = _ctx_use.tokenizer
    _mdl        = _ctx_use.text_model
    _dev        = _ctx_use.device
    _t_labels   = _ctx_use.TIME_LABELS
    _txt_l2i    = _ctx_use.TEXT_LABEL2ID

    processed = preprocess_text_identity(fragment_text)
    inputs = _tok(
        processed,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(_dev) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _mdl(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    # Reorder to TIME_LABELS
    probs_time_order = np.zeros(len(_t_labels), dtype=np.float64)
    for i, lab in enumerate(_t_labels):
        probs_time_order[i] = probs[_txt_l2i[lab]]

    s = probs_time_order.sum()
    if s > 0:
        probs_time_order /= s
    return probs_time_order


def classify_fragment(
    fragment_text: str,
    rel_time_val: float,
    time_bin_str: str,
    max_length: int = MAX_LENGTH,
    ctx: Optional["ModelContext"] = None,
) -> dict:
    """
    Classifies a single sliding-window fragment using BERT + time fusion.
    The result dict contains per-label fused probabilities used downstream
    by both argmax (fallback) and the CRF Viterbi decoder.

    Args:
        ctx : ModelContext a usar. Si None usa el modelo por defecto.
    """
    _ctx_use  = ctx if ctx is not None else _ctx()
    _t_labels = _ctx_use.TIME_LABELS

    probs_text  = predict_text_probs(fragment_text, max_length=max_length, ctx=_ctx_use)
    probs_fused = apply_time_fusion(
        probs_text,
        time_bin_str=time_bin_str,
        rel_time_val=rel_time_val,
        gamma=_ctx_use.GAMMA,
        prior_y=_ctx_use.PRIOR_Y,
        ctx=_ctx_use,
    )

    pred_id    = int(np.argmax(probs_fused))
    pred_label = _t_labels[pred_id]

    result = {
        "predicted_subtag":   pred_label,
        "predicted_cluster":  pred_label,
    }
    for i, lab in enumerate(_t_labels):
        result[lab] = float(probs_fused[i])

    return result


def classify_entire_conversation(
    df_conversation: pd.DataFrame,
    window_size: int,
    stride: int,
    max_length: int,
    ctx: Optional["ModelContext"] = None,
) -> pd.DataFrame:
    """
    Applies a sliding window over a conversation, classifies each window
    with BERT + time fusion, then (if available) refines the full label
    sequence using CRF Viterbi decoding (Issue #6).

    Args:
        ctx : ModelContext a usar. Si None usa el modelo por defecto.
    """
    _ctx_use  = ctx if ctx is not None else _ctx()
    _rtb      = _ctx_use.rel_time_bin
    _t_labels = _ctx_use.TIME_LABELS
    _meta     = _ctx_use.TIME_META
    _crf      = _ctx_use.CRF_DECODER

    word_timeline = build_word_timestamps(df_conversation)
    if not word_timeline:
        return pd.DataFrame(columns=[
            "text", "start", "end", "turn_idx",
            "rel_time", "time_bin",
            "predicted_subtag", "predicted_cluster",
        ])

    words       = [w for (w, _, _) in word_timeline]
    total_words = len(words)

    ends     = [e for (_, _, e) in word_timeline if e is not None and not np.isnan(e)]
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

        rel = (
            t_mid / conv_dur
            if (conv_dur and not np.isnan(conv_dur) and conv_dur > 0 and not np.isnan(t_mid))
            else np.nan
        )
        tb = _rtb(rel)

        fragment_text = " ".join(w_slice)
        if USE_TEXT_TAIL:
            tail = build_tail(t_mid, conv_dur, ctx=_ctx_use)
            fragment_text = f"{fragment_text} {tail}".strip()

        out = classify_fragment(
            fragment_text, rel_time_val=rel, time_bin_str=tb,
            max_length=max_length, ctx=_ctx_use,
        )
        out.update({
            "text":     fragment_text,
            "start":    start_window,
            "end":      end_window,
            "turn_idx": i,
            "rel_time": rel if isinstance(rel, float) and not np.isnan(rel) else np.nan,
            "time_bin": tb,
        })
        results.append(out)

        if total_words < window_size:
            break

    if not results:
        return pd.DataFrame()

    # Issue #6: CRF Viterbi decoding over the full window sequence
    if _crf is not None and _crf._fitted and len(results) > 1:
        eps = _meta["_eps"]
        log_emissions = np.array([
            np.log(np.clip([r[lab] for lab in _t_labels], eps, 1.0))
            for r in results
        ])  # shape [n_windows, n_labels]

        viterbi_path = _crf.decode(log_emissions)

        for j, r in enumerate(results):
            best_label = _t_labels[viterbi_path[j]]
            r["predicted_subtag"]  = best_label
            r["predicted_cluster"] = best_label

    df_result = pd.DataFrame(results)

    if SMOOTH_K and SMOOTH_K > 0 and not df_result.empty:
        df_result = df_result.sort_values("turn_idx").reset_index(drop=True)
        smooth = smooth_labels(df_result["predicted_subtag"].tolist(), k=SMOOTH_K)
        df_result["predicted_subtag_smooth"]  = smooth
        df_result["predicted_cluster_smooth"] = smooth

    prob_cols = [c for c in df_result.columns
                 if c not in {
                     "text", "start", "end", "turn_idx",
                     "rel_time", "time_bin",
                     "predicted_subtag", "predicted_cluster",
                     "predicted_subtag_smooth", "predicted_cluster_smooth",
                 }]
    df_result.attrs["cluster_columns"] = prob_cols
    return df_result


# =============================
# CSV ENTRY POINTS
# =============================
def fit_csv_sliding_transformer(
    input_csv: str,
    output_csv: str,
    window_size: int = WINDOW_SIZE,
    stride: int      = STRIDE,
    max_length: int  = MAX_LENGTH,
    ctx: Optional["ModelContext"] = None,
):
    """
    Classifies a full conversation CSV via sliding window and writes results.

    Args:
        ctx : ModelContext a usar. Si None usa el modelo por defecto.
    """
    df = pd.read_csv(input_csv, encoding="utf-8")
    required = {"text", "start", "end"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"El CSV debe contener columnas {required} "
            "(se ignorarán columnas extra como confidence/speaker)."
        )

    classified_df = classify_entire_conversation(df, window_size, stride, max_length, ctx=ctx)
    classified_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Predicciones guardadas en: {output_csv}")


def fitCSVConversations(
    input_folder,
    output_folder,
    window_size,
    stride,
    max_length,
    model_dir: Optional[str] = None,
    time_priors_json: Optional[str] = None,
):
    """
    Batch orchestrator: processes all CSVs in input_folder and writes
    classified CSVs to output_folder.  Isolates broken transcripts on error.

    Args:
        model_dir        : ruta local al directorio del modelo de segmentación.
                           Si None, se usa TEXT_MODEL_DIR del entorno.
        time_priors_json : ruta local al JSON de time priors.
                           Si None, se usa TIME_PRIORS_JSON del entorno.
    """
    print(f"CALIFICANDO {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

    # Resolver contexto del modelo (cacheado por (model_dir, time_priors_json))
    model_ctx = get_or_load_context(model_dir, time_priors_json)
    logger.info(
        "[fitCSVConversations] Usando modelo: %s", model_ctx.model_dir
    )

    base_folder     = os.path.abspath(os.path.join(input_folder, os.pardir))
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
                    max_length=max_length,
                    ctx=model_ctx,
                )
            except Exception as e:
                print(f"Error processing {filename}: {e}")

                base_name      = filename.replace("_transcript.csv", "")
                audio_file     = os.path.join(base_folder, base_name + ".mp3")
                transcript_file = os.path.join(base_folder, filename)

                if os.path.exists(audio_file):
                    shutil.move(audio_file, os.path.join(isolated_folder, os.path.basename(audio_file)))
                    print(f"Audio {audio_file} movido a {isolated_folder}")

                if os.path.exists(transcript_file):
                    os.remove(transcript_file)
                    print(f"Transcripción {transcript_file} eliminada")
