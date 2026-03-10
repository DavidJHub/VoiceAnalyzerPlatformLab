#!/usr/bin/env python
import os
import re
import json
import shutil
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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


# ---- module-level load ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer, text_model, id2label = load_text_model(TEXT_MODEL_DIR, device)
TIME_META = load_time_priors(TIME_PRIORS_JSON)
TIME_META = normalize_priors_labels(TIME_META)

# Issue #3: override rel_time_bin with version derived from stored thresholds
rel_time_bin = _make_rel_time_bin_fn(TIME_META["_time_bin_thresholds"])

TIME_LABELS      = TIME_META["labels"]
LABEL2IDX_TIME   = {lab: i for i, lab in enumerate(TIME_LABELS)}
TEXT_ID2LABEL    = id2label
TEXT_LABEL2ID    = {v: k for k, v in TEXT_ID2LABEL.items()}

missing_in_text = [lab for lab in TIME_LABELS if lab not in TEXT_LABEL2ID]
if missing_in_text:
    raise ValueError(
        "Tus time priors tienen labels que no existen en el modelo texto. "
        f"Faltan en texto: {missing_in_text}\n"
        f"Labels en texto: {sorted(list(TEXT_LABEL2ID.keys()))[:30]}..."
    )

ALPHA   = TIME_META["_alpha"]
BETA    = TIME_META["_beta"]
GAMMA   = TIME_META["_gamma"]
PRIOR_Y = TIME_META["_prior_y"]       # shape [n_labels], in TIME_LABELS order

# Issue #6: load CRF decoder from time priors (None if not present / not fitted)
CRF_DECODER = None
if TIME_META.get("crf_params"):
    try:
        from segmentationModel.crfDecoder import CRFSequenceDecoder
        CRF_DECODER = CRFSequenceDecoder.from_dict(TIME_META["crf_params"])
        print(f"[CRF] Decoder loaded ({len(TIME_LABELS)} tags, fitted={CRF_DECODER._fitted})")
    except Exception as e:
        print(f"[WARNING] CRF decoder could not be loaded: {e}. Falling back to argmax.")


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
                      prior_y: np.ndarray = None) -> np.ndarray:
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

    Returns:
        np.ndarray shape [C], fused and renormalised probabilities.
    """
    eps     = TIME_META["_eps"]
    prob_tb = TIME_META["prob_time_bin_given_y"]
    prob_rb = TIME_META["prob_relbin_given_y"]
    rel_bins = TIME_META["rel_bins"]

    tb = ("unknown"
          if (time_bin_str is None or (isinstance(time_bin_str, float) and np.isnan(time_bin_str)))
          else str(time_bin_str))
    rb = rel_bin_index(rel_time_val, rel_bins)

    logp = np.log(np.clip(text_probs_time_order, eps, 1.0))

    for i, y in enumerate(TIME_LABELS):
        p_tb = prob_tb[y].get(tb, prob_tb[y].get("unknown", 1e-6))
        logp[i] += ALPHA * np.log(max(float(p_tb), eps))

        if rb >= 0:
            p_rb = prob_rb[y][rb]
            logp[i] += BETA * np.log(max(float(p_rb), eps))

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


def predict_text_probs(fragment_text: str, max_length: int = MAX_LENGTH) -> np.ndarray:
    """
    Runs the text model and returns a probability vector in TIME_LABELS order.
    """
    processed = preprocess_text_identity(fragment_text)
    inputs = tokenizer(
        processed,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = text_model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    # Reorder to TIME_LABELS
    probs_time_order = np.zeros(len(TIME_LABELS), dtype=np.float64)
    for i, lab in enumerate(TIME_LABELS):
        probs_time_order[i] = probs[TEXT_LABEL2ID[lab]]

    s = probs_time_order.sum()
    if s > 0:
        probs_time_order /= s
    return probs_time_order


def classify_fragment(fragment_text: str, rel_time_val: float,
                      time_bin_str: str, max_length: int = MAX_LENGTH) -> dict:
    """
    Classifies a single sliding-window fragment using BERT + time fusion.
    The result dict contains per-label fused probabilities used downstream
    by both argmax (fallback) and the CRF Viterbi decoder.
    """
    probs_text  = predict_text_probs(fragment_text, max_length=max_length)
    probs_fused = apply_time_fusion(
        probs_text,
        time_bin_str=time_bin_str,
        rel_time_val=rel_time_val,
        gamma=GAMMA,
        prior_y=PRIOR_Y,
    )

    pred_id    = int(np.argmax(probs_fused))
    pred_label = TIME_LABELS[pred_id]

    result = {
        "predicted_subtag":   pred_label,
        "predicted_cluster":  pred_label,
    }
    for i, lab in enumerate(TIME_LABELS):
        result[lab] = float(probs_fused[i])

    return result


def classify_entire_conversation(
    df_conversation: pd.DataFrame,
    window_size: int,
    stride: int,
    max_length: int,
) -> pd.DataFrame:
    """
    Applies a sliding window over a conversation, classifies each window
    with BERT + time fusion, then (if available) refines the full label
    sequence using CRF Viterbi decoding (Issue #6).
    """
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
        tb = rel_time_bin(rel)

        fragment_text = " ".join(w_slice)
        if USE_TEXT_TAIL:
            tail = build_tail(t_mid, conv_dur)
            fragment_text = f"{fragment_text} {tail}".strip()

        out = classify_fragment(fragment_text, rel_time_val=rel, time_bin_str=tb, max_length=max_length)
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
    if CRF_DECODER is not None and CRF_DECODER._fitted and len(results) > 1:
        eps = TIME_META["_eps"]
        # Emissions: log of fused probabilities in TIME_LABELS order
        log_emissions = np.array([
            np.log(np.clip([r[lab] for lab in TIME_LABELS], eps, 1.0))
            for r in results
        ])  # shape [n_windows, n_labels]

        viterbi_path = CRF_DECODER.decode(log_emissions)

        for j, r in enumerate(results):
            best_label = TIME_LABELS[viterbi_path[j]]
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
):
    """
    Classifies a full conversation CSV via sliding window and writes results.
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
    Batch orchestrator: processes all CSVs in input_folder and writes
    classified CSVs to output_folder.  Isolates broken transcripts on error.
    """
    print(f"CALIFICANDO {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

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
