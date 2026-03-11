#!/usr/bin/env python
import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import database.dbConfig as dbcfg

from segmentationModel.crfDecoder import CRFSequenceDecoder


def sanitize_dirname(name: str) -> str:
    name = str(name).strip()
    name = name.replace("\\", "_").replace("/", "_")
    return name

def get_db_connection():
    return dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP,
    )

def get_country_sponsor_by_sponsor_id(conn, sponsor_id: int):
    q = """
    SELECT country, sponsor
    FROM marketing_campaigns
    WHERE sponsor_id = %s
    LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(q, (int(sponsor_id),))
        row = cur.fetchone()

    if not row:
        raise RuntimeError(f"No existe sponsor_id={sponsor_id} en marketing_campaigns")

    country, sponsor = row[0], row[1]
    if not country or not sponsor:
        raise RuntimeError(f"Country/Sponsor inválidos para sponsor_id={sponsor_id}: {row}")
    return str(country), str(sponsor)

def build_default_paths_from_campaign(sponsor_id: int):
    """
    Retorna:
      train_file, output_text_dir, output_time_json
    según convención:
      reentreno/{country}/{sponsor}/rawtraining/master_{sponsor}_enriched.tsv
    """
    conn = get_db_connection()
    try:
        country, sponsor = get_country_sponsor_by_sponsor_id(conn, sponsor_id)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    country_dir = sanitize_dirname(country)
    sponsor_dir = sanitize_dirname(sponsor)

    base = os.path.join("reentreno", country_dir, sponsor_dir)
    train_file = os.path.join(base, "rawtraining", f"master_{sponsor_dir}_enriched.tsv")
    output_text_dir = os.path.join(base, "model", "model_text")
    output_time_json = os.path.join(base, "model", "time_priors_subtag.json")
    return train_file, output_text_dir, output_time_json


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset


def load_tokenizer(model_name: str):
    try:
        tok = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            add_prefix_space=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"No pude cargar tokenizer para '{model_name}'. "
        ) from e

    if not getattr(tok, "is_fast", False):
        raise RuntimeError(
            f"Tokenizer lento detectado para {model_name}. "
            "Instala/actualiza: pip install -U tokenizers transformers"
        )
    return tok


# ----------------- CONFIG ----------------- #
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
SEED       = 42
MAX_LENGTH = 128
VAL_SIZE   = 0.10
MIN_SAMPLES_PER_CLASS = 2

# Relative-time bins for prior estimation (0..1). Must match preprocessing.
REL_BINS = [0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.01]

# Issue #3: categorical time-bin thresholds stored in time_priors JSON so
# fittingDeep can load and use the exact same boundaries used here.
TIME_BIN_THRESHOLDS = {"early": 0.15, "mid": 0.65}   # late = everything else

LAPLACE = 1.0  # Laplace smoothing for priors


# ----------------- REPRO ----------------- #
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------- WEIGHTED TRAINER (Issue #8) ----------------- #
class WeightedTrainer(Trainer):
    """
    Trainer subclass that applies inverse-frequency class weights to
    CrossEntropyLoss so rare labels are not systematically under-penalised.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ----------------- TIME PRIORS ----------------- #
def rel_bin_index(rel):
    if rel is None or (isinstance(rel, float) and (np.isnan(rel) or np.isinf(rel))):
        return -1
    r = float(rel)
    for i in range(len(REL_BINS) - 1):
        if REL_BINS[i] <= r < REL_BINS[i + 1]:
            return i
    return len(REL_BINS) - 2


def build_time_priors(train_df, label_col="label"):
    """
    Learns P(time_bin | y) and P(rel_bin | y) with Laplace smoothing.
    Returns a metadata dict ready to be extended and serialised to JSON.
    """
    labels = sorted(train_df[label_col].unique().tolist())

    bins = sorted(train_df["time_bin"].fillna("unknown").astype(str).unique().tolist())
    if "unknown" not in bins:
        bins.append("unknown")

    count_tb = {lab: {b: 0 for b in bins} for lab in labels}
    count_rb = {lab: [0] * (len(REL_BINS) - 1) for lab in labels}

    for _, r in train_df.iterrows():
        y  = r[label_col]
        tb = str(r.get("time_bin", "unknown")) if pd.notna(r.get("time_bin", np.nan)) else "unknown"
        if tb not in count_tb[y]:
            count_tb[y][tb] = 0
        count_tb[y][tb] += 1

        rb = rel_bin_index(r.get("rel_time", np.nan))
        if rb >= 0:
            count_rb[y][rb] += 1

    prob_tb = {}
    for y in labels:
        total = sum(count_tb[y].values())
        denom = total + LAPLACE * len(count_tb[y])
        prob_tb[y] = {b: (count_tb[y].get(b, 0) + LAPLACE) / denom for b in count_tb[y]}

    prob_rb = {}
    for y in labels:
        total = sum(count_rb[y])
        denom = total + LAPLACE * len(count_rb[y])
        prob_rb[y] = [(c + LAPLACE) / denom for c in count_rb[y]]

    meta = {
        "labels": labels,
        "time_bin_categories": sorted({b for y in prob_tb for b in prob_tb[y]}),
        "rel_bins": REL_BINS,
        "laplace": LAPLACE,
        "prob_time_bin_given_y": prob_tb,
        "prob_relbin_given_y":   prob_rb,
        # Issue #3: store thresholds so fittingDeep uses identical boundaries
        "time_bin_thresholds": TIME_BIN_THRESHOLDS,
    }
    return meta


def apply_time_fusion(text_probs, time_meta, time_bin, rel_time,
                      alpha=0.8, beta=0.8, gamma=0.0, prior_y=None):
    """
    Fuse text-model probabilities with temporal priors in log-space:

      log P(y | x, t) ∝  log P_text(y | x)
                       + alpha * log P(time_bin | y)
                       + beta  * log P(rel_bin  | y)
                       + gamma * log P(y)           [label frequency prior]

    gamma / prior_y implement Issue #4's third optimisation dimension.
    """
    labels  = time_meta["labels"]
    prob_tb = time_meta["prob_time_bin_given_y"]
    prob_rb = time_meta["prob_relbin_given_y"]

    tb = ("unknown"
          if (time_bin is None or (isinstance(time_bin, float) and np.isnan(time_bin)))
          else str(time_bin))
    rb = rel_bin_index(rel_time)

    eps  = 1e-12
    logp = np.log(np.clip(text_probs, eps, 1.0))

    for i, y in enumerate(labels):
        p_tb = prob_tb[y].get(tb, prob_tb[y].get("unknown", 1e-6))
        logp[i] += alpha * np.log(max(float(p_tb), eps))

        if rb >= 0:
            p_rb = prob_rb[y][rb]
            logp[i] += beta * np.log(max(float(p_rb), eps))

        # Issue #4: label frequency prior term
        if gamma != 0.0 and prior_y is not None:
            logp[i] += gamma * np.log(max(float(prior_y[i]), eps))

    logp -= np.max(logp)
    p = np.exp(logp)
    p /= p.sum()
    return p


# ----------------- IO ----------------- #
def load_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t", dtype=str)
    if ext == ".csv":
        return pd.read_csv(path, dtype=str)
    return pd.read_excel(path, dtype=str)


# ----------------- MAIN ----------------- #
def main():
    set_seed(SEED)
    parser = argparse.ArgumentParser(
        description="Train text model + time priors from *_enriched.tsv"
    )
    parser.add_argument(
        "--campaign_id", "--sponsor_id", dest="sponsor_id", type=int, default=None,
        help="ID (sponsor_id/campaign_id) para resolver country/sponsor y construir rutas default."
    )
    parser.add_argument(
        "--train_file", type=str, default=None,
        help="Ruta al TSV/XLSX enriquecido. Si no se pasa, se construye desde campaign_id."
    )
    parser.add_argument(
        "--text_col", type=str, default="input_text",
        help="Columna a usar como texto. Default: input_text (fallback a name si no existe)."
    )
    parser.add_argument(
        "--output_text_dir", type=str, default="model_output",
        help="Directorio de salida del modelo. Si no se pasa, se deriva desde campaign_id."
    )
    parser.add_argument(
        "--output_time_json", type=str, default=None,
        help="JSON de priors. Si no se pasa, se deriva desde campaign_id."
    )
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    max_length = int(args.max_length)

    # -------- PATH RESOLUTION --------
    if args.train_file:
        train_file       = args.train_file
        output_text_dir  = args.output_text_dir  or "model_out_text"
        output_time_json = args.output_time_json or "time_priors_subtag.json"
    else:
        if args.sponsor_id is None:
            raise ValueError(
                "Debes pasar --train_file o --campaign_id/--id_sponsor "
                "para construir rutas automáticamente."
            )
        train_file, default_out_text, default_out_time = build_default_paths_from_campaign(
            args.sponsor_id
        )
        output_text_dir  = args.output_text_dir  or default_out_text
        output_time_json = args.output_time_json or default_out_time

    print("[PATH] train_file:", train_file)
    print("[PATH] output_text_dir:", output_text_dir)
    print("[PATH] output_time_json:", output_time_json)

    # -------- DATA LOADING --------
    df = load_table(train_file)
    print("Columnas:", list(df.columns))

    required = {"name", "subtag", "rel_time", "time_bin"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}. Se esperaban: {required}")

    text_col = args.text_col
    if text_col not in df.columns:
        text_col = "input_text" if "input_text" in df.columns else "name"
    print(f"[INFO] Usando columna de texto: {text_col}")

    df["subtag"]   = df["subtag"].astype(str).str.strip()
    df[text_col]   = df[text_col].astype(str).fillna("").str.strip()
    df["name"]     = df["name"].astype(str).fillna("").str.strip()
    df["rel_time"] = pd.to_numeric(df["rel_time"], errors="coerce")
    df["time_bin"] = (
        df["time_bin"].astype(str)
        .fillna("unknown")
        .replace({"nan": "unknown", "None": "unknown"})
    )
    df = df[df[text_col] != ""].reset_index(drop=True)

    freq = df["subtag"].value_counts()
    rare = freq[freq < MIN_SAMPLES_PER_CLASS].index.tolist()
    if rare:
        print("Eliminando clases raras:", {r: int(freq[r]) for r in rare})
        df = df[~df["subtag"].isin(rare)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No quedaron ejemplos después de filtrar clases raras.")

    labels   = sorted(df["subtag"].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    df["label"] = df["subtag"].map(label2id).astype(int)

    # -------- TRAIN / VAL SPLIT --------
    train_df, val_df = train_test_split(
        df, test_size=VAL_SIZE, random_state=SEED, stratify=df["label"]
    )

    print(f"[DATA] Train: {len(train_df)} | Val: {len(val_df)}")
    print("[DATA] Class distribution (train):")
    for lab in labels:
        n = (train_df["subtag"] == lab).sum()
        print(f"  {lab:30s}: {n}")

    # -------- ISSUE #8: CLASS WEIGHTS --------
    cw_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(labels)),
        y=train_df["label"].values,
    )
    class_weights = torch.tensor(cw_array, dtype=torch.float32)
    print("\n[CLASS WEIGHTS]")
    for lab, w in zip(labels, cw_array):
        print(f"  {lab:30s}: {w:.4f}")

    # -------- TOKENISATION --------
    tokenizer = load_tokenizer(MODEL_NAME)

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_ds = Dataset.from_pandas(
        pd.DataFrame({"text": train_df[text_col].tolist(), "label": train_df["label"].tolist()})
    )
    val_ds = Dataset.from_pandas(
        pd.DataFrame({"text": val_df[text_col].tolist(), "label": val_df["label"].tolist()})
    )

    train_ds = train_ds.map(tok, batched=True).remove_columns(["text"])
    val_ds   = val_ds.map(tok,   batched=True).remove_columns(["text"])
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format(  type="torch", columns=["input_ids", "attention_mask", "label"])

    # -------- MODEL --------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_pred):
        logits, y = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy":    accuracy_score(y, preds),
            "macro_f1":    f1_score(y, preds, average="macro",    zero_division=0),
            "weighted_f1": f1_score(y, preds, average="weighted", zero_division=0),
        }

    training_args = TrainingArguments(
        output_dir=output_text_dir,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        seed=SEED,
        save_safetensors=True,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print("Eval texto:", eval_results)

    # -------- SAVE TEXT MODEL --------
    os.makedirs(output_text_dir, exist_ok=True)
    trainer.model.save_pretrained(output_text_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_text_dir)
    print("Modelo texto guardado en:", output_text_dir)

    # -------- TIME PRIORS --------
    time_meta = build_time_priors(train_df, label_col="label")

    # Issue #4: label frequency prior P(y) from training distribution
    label_counts = np.bincount(train_df["label"].values, minlength=len(labels))
    prior_y      = label_counts / label_counts.sum()
    time_meta["label_prior"] = prior_y.tolist()

    # -------- ISSUE #4: SCIPY OPTIMISATION (alpha, beta, gamma) --------
    val_logits = trainer.predict(val_ds).predictions
    # Convert raw logits to probabilities via stable softmax
    val_logits = val_logits - val_logits.max(axis=1, keepdims=True)
    val_probs  = np.exp(val_logits)
    val_probs  = val_probs / val_probs.sum(axis=1, keepdims=True)
    y_true     = val_df["label"].values

    # Baseline: text-only predictions (no temporal fusion)
    text_only_preds = np.argmax(val_probs, axis=1)
    text_only_f1    = f1_score(y_true, text_only_preds, average="macro", zero_division=0)

    def _neg_macro_f1(params):
        a = float(params[0])
        b = float(params[1])
        g = float(params[2])
        preds = []
        for i in range(len(val_df)):
            row = val_df.iloc[i]
            p = apply_time_fusion(
                val_probs[i], time_meta,
                time_bin=row["time_bin"],
                rel_time=row["rel_time"],
                alpha=a, beta=b, gamma=g,
                prior_y=prior_y,
            )
            preds.append(int(np.argmax(p)))
        return -f1_score(y_true, preds, average="macro", zero_division=0)

    # Multi-start Nelder-Mead to avoid local minima
    starts = [
        [0.8, 0.3, 0.0],
        [0.5, 0.5, 0.1],
        [1.0, 0.5, 0.2],
        [0.3, 0.1, 0.0],
        [0.0, 0.0, 0.5],
    ]
    best_result = None
    for x0 in starts:
        res = minimize(
            _neg_macro_f1,
            x0=x0,
            method="Nelder-Mead",
            options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-4},
        )
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    alpha_opt = float(best_result.x[0])
    beta_opt  = float(best_result.x[1])
    gamma_opt = float(best_result.x[2])
    best_f1   = float(-best_result.fun)

    # Build fused predictions at optimal params for the per-class report
    fused_preds = []
    for i in range(len(val_df)):
        row = val_df.iloc[i]
        p = apply_time_fusion(
            val_probs[i], time_meta,
            time_bin=row["time_bin"],
            rel_time=row["rel_time"],
            alpha=alpha_opt, beta=beta_opt, gamma=gamma_opt,
            prior_y=prior_y,
        )
        fused_preds.append(int(np.argmax(p)))

    time_meta["fusion_params"] = {
        "alpha":    alpha_opt,
        "beta":     beta_opt,
        "gamma":    gamma_opt,
        "macro_f1": best_f1,
    }

    # -------- ISSUE #6: CRF TRANSITION FITTING --------
    call_id_col = next(
        (c for c in ["call_id", "call_id_num"] if c in train_df.columns), None
    )
    sort_col = "turn_idx" if "turn_idx" in train_df.columns else None

    crf_decoder = CRFSequenceDecoder(num_tags=len(labels), labels=labels)
    if call_id_col:
        call_sequences = []
        for _, grp in train_df.groupby(call_id_col):
            if sort_col:
                grp = grp.sort_values(sort_col)
            seq = grp["label"].tolist()
            if seq:
                call_sequences.append(seq)
        crf_decoder.fit_transitions(call_sequences, laplace=LAPLACE)
        time_meta["crf_params"] = crf_decoder.to_dict()
    else:
        print("[WARNING] No se encontró columna call_id/call_id_num. CRF no entrenado.")
        time_meta["crf_params"] = None

    # -------- ISSUE #4: COMPREHENSIVE DEBUG OUTPUT --------
    per_class_report = classification_report(
        y_true, fused_preds, target_names=labels, zero_division=0
    )

    print("\n" + "=" * 70)
    print("  HYPERPARAMETERS & MONITORING DASHBOARD")
    print("=" * 70)
    print(f"\n  [DATASET]")
    print(f"    Training samples   : {len(train_df)}")
    print(f"    Validation samples : {len(val_df)}")
    print(f"    Classes            : {len(labels)}")
    print(f"    MAX_LENGTH (tokens): {max_length}")
    print(f"\n  [BERT EVAL METRICS — val]")
    print(f"    accuracy    : {eval_results.get('eval_accuracy',    0):.4f}")
    print(f"    macro_f1    : {eval_results.get('eval_macro_f1',    0):.4f}  <-- PRIMARY METRIC")
    print(f"    weighted_f1 : {eval_results.get('eval_weighted_f1', 0):.4f}")
    print(f"\n  [FUSION HYPERPARAMETERS — scipy Nelder-Mead, multi-start]")
    print(f"    alpha (time_bin prior weight)  : {alpha_opt:.4f}  [range 0..2, typ 0.5-1.0]")
    print(f"    beta  (rel_bin prior weight)   : {beta_opt:.4f}  [range 0..2, typ 0.1-0.5]")
    print(f"    gamma (label frequency prior)  : {gamma_opt:.4f}  [range -1..1, 0=disabled]")
    print(f"\n  [FUSION QUALITY — val]")
    print(f"    text-only macro_f1             : {text_only_f1:.4f}")
    print(f"    fused    macro_f1              : {best_f1:.4f}  <-- WATCH THIS IMPROVE")
    print(f"    improvement from fusion        : {best_f1 - text_only_f1:+.4f}")
    print(f"    scipy converged                : {best_result.success}")
    print(f"    scipy iterations used          : {best_result.nit}")
    print(f"\n  [CLASS WEIGHTS — inverse frequency]")
    for lab, w in zip(labels, cw_array):
        print(f"    {lab:30s}: {w:.4f}")
    if call_id_col and "crf_params" in time_meta and time_meta["crf_params"]:
        n_seqs = len(time_meta["crf_params"].get("transitions", []))
        print(f"\n  [CRF]")
        print(f"    call_id column used            : {call_id_col}")
        print(f"    num_tags                       : {len(labels)}")
        print(f"    Transitions fitted             : yes")
    print(f"\n  [PER-CLASS REPORT — fused predictions on val]\n")
    print(per_class_report)
    print("=" * 70)
    print("  KEY THINGS TO MONITOR ACROSS RETRAINING RUNS:")
    print("    1. eval_macro_f1 (BERT)      — should increase or hold steady")
    print("    2. fused macro_f1            — must be >= text-only macro_f1")
    print("    3. alpha / beta / gamma      — large shifts signal distribution change")
    print("    4. per-class F1 for rare labels (e.g. TRATAMIENTO DATOS, LEY RETRACTO)")
    print("    5. scipy convergence flag    — if False, increase maxiter or relax tolerances")
    print("=" * 70 + "\n")

    # -------- SAVE TIME PRIORS + CRF --------
    time_meta["label2id"]       = {k: int(v)  for k, v in label2id.items()}
    time_meta["id2label"]       = {str(k): v  for k, v in id2label.items()}
    time_meta["text_col_used"]  = text_col
    time_meta["model_name"]     = MODEL_NAME
    time_meta["max_length"]     = max_length
    time_meta["train_file"]     = os.path.abspath(train_file)

    os.makedirs(os.path.dirname(output_time_json) or ".", exist_ok=True)
    with open(output_time_json, "w", encoding="utf-8") as f:
        json.dump(time_meta, f, ensure_ascii=False, indent=2)

    print("Time priors + CRF guardados en:", output_time_json)


if __name__ == "__main__":
    main()
