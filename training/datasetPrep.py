# datasetprep.py
import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np


# -----------------------------
# Config knobs (fallback only)
# -----------------------------
RESET_TOLERANCE_SEC = 0.5
FORWARD_JUMP_NEW_CALL_SEC = None  # e.g., 7200 for >2h jumps


# -----------------------------
# Helpers
# -----------------------------
def time_bin(rel):
    if rel is None or (isinstance(rel, float) and (np.isnan(rel) or np.isinf(rel))):
        return "unknown"
    if rel <= 0.15:
        return "early"
    elif rel <= 0.65:
        return "mid"
    else:
        return "late"


def parse_time_to_seconds(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def _mmss_or_unknown(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "??:??"
    x = max(0.0, float(x))
    mm = int(x // 60)
    ss = int(round(x % 60))
    return f"{mm:02d}:{ss:02d}"


def format_time_tag(t):
    if t is None or (isinstance(t, float) and (np.isnan(t) or np.isinf(t))):
        return "[T=??:??]"
    t = max(0.0, float(t))
    mm = int(t // 60)
    ss = int(round(t % 60))
    return f"[T={mm:02d}:{ss:02d}]"


def build_input_tail(time_sec, call_duration_sec):
    rel = (float(time_sec) / float(call_duration_sec)) if (
        isinstance(time_sec, (int, float)) and
        isinstance(call_duration_sec, (int, float)) and
        call_duration_sec > 0 and
        not np.isnan(time_sec) and
        not np.isnan(call_duration_sec)
    ) else np.nan

    rel_str = f"{rel:.2f}" if isinstance(rel, float) and not np.isnan(rel) else "UNK"
    bin_str = time_bin(rel)
    t_str   = _mmss_or_unknown(time_sec)
    return f"[rt={rel_str}][bin={bin_str}][t={t_str}]"


def assign_call_ids_from_time(
    series_time_sec,
    reset_tol_sec=RESET_TOLERANCE_SEC,
    forward_jump_new_call_sec=FORWARD_JUMP_NEW_CALL_SEC
):
    """
    Fallback: si NO existe call_id en el dataset.
    Nuevo call cuando el tiempo disminuye > reset_tol_sec.
    Opcionalmente también cuando hay un salto grande hacia adelante.
    """
    times = series_time_sec.values
    call_ids = np.zeros_like(times, dtype=np.int64)
    call_id = 0
    prev = None

    for i, t in enumerate(times):
        if i == 0:
            call_ids[i] = call_id
            prev = t
            continue

        new_call = False
        if not np.isnan(t) and not np.isnan(prev):
            if (prev - t) > reset_tol_sec:
                new_call = True
            if (forward_jump_new_call_sec is not None) and ((t - prev) > forward_jump_new_call_sec):
                new_call = True

        if new_call:
            call_id += 1

        call_ids[i] = call_id
        prev = t

    return pd.Series(call_ids, index=series_time_sec.index, name="call_id_num")


# -----------------------------
# Core API (importable)
# -----------------------------
def enrich_conversations(
    df: pd.DataFrame,
    text_col: str = "name",
    time_col: str = "time",
    call_id_col: str = "call_id",   # preferido si existe
):
    """
    Enrich con features temporales + input_text.

    Requisitos mínimos:
    - text_col (default: name)
    - time_col (default: time)
    - y preferiblemente call_id_col (default: call_id).
      Si no existe call_id_col -> fallback a resets de tiempo.

    Agrega:
    - time_sec
    - call_id_num (int factorized)
    - turn_idx
    - call_duration_sec
    - rel_time
    - time_tag
    - time_bin
    - input_text = "<text> [rt=..][bin=..][t=mm:ss]"
    """
    df = df.copy()

    if text_col not in df.columns:
        raise ValueError(f"No existe columna texto '{text_col}' en df. Columnas: {list(df.columns)}")
    if time_col not in df.columns:
        raise ValueError(f"No existe columna tiempo '{time_col}' en df. Columnas: {list(df.columns)}")

    # 1) time_sec
    df["time_sec"] = df[time_col].apply(parse_time_to_seconds)

    # 2) call_id (string) preferido; si no, fallback num por resets
    if call_id_col in df.columns:
        df["call_id"] = df[call_id_col].astype(str)
    else:
        # fallback compatible (no pierdes la funcionalidad vieja)
        df["call_id_num"] = assign_call_ids_from_time(df["time_sec"])
        df["call_id"] = df["call_id_num"].astype(str)

    # 3) call_id_num siempre presente (útil para training)
    if "call_id_num" not in df.columns:
        df["call_id_num"] = pd.factorize(df["call_id"])[0].astype(np.int64)

    # 4) turn_idx
    df["turn_idx"] = df.groupby("call_id").cumcount()

    # 5) call_duration / rel_time
    call_max = df.groupby("call_id")["time_sec"].transform("max").replace(0.0, np.nan)
    df["call_duration_sec"] = call_max
    df["rel_time"] = (df["time_sec"] / df["call_duration_sec"]).clip(0, 1)

    # 6) tags/bins
    df["time_tag"] = df["time_sec"].apply(format_time_tag)
    df["time_bin"] = df["rel_time"].apply(time_bin)

    # 7) input_text
    def _row_input_text(row):
        txt = str(row.get(text_col, "")).strip()
        tail = build_input_tail(row.get("time_sec", np.nan), row.get("call_duration_sec", np.nan))
        return f"{txt} {tail}".strip()

    df["input_text"] = df.apply(_row_input_text, axis=1)

    return df


def enrich_master_tsv(
    master_tsv_path: str,
    out_dir: str = None,
    out_stem: str = None,
    text_col: str = "name",
    time_col: str = "time",
    call_id_col: str = "call_id",
    write_xlsx: bool = True,
    write_tsv: bool = True,
):
    """
    API pensada para tu pipeline:
    - Lee master TSV (del autotrain)
    - Enrich usando call_id existente (si está)
    - Escribe master_*_enriched.tsv/.xlsx en out_dir

    Returns: (out_tsv_path_or_None, out_xlsx_path_or_None)
    """
    if not os.path.exists(master_tsv_path):
        raise FileNotFoundError(f"No existe: {master_tsv_path}")

    df = pd.read_csv(master_tsv_path, sep="\t", dtype=str)

    # salida
    master_path = Path(master_tsv_path)
    if out_dir is None:
        out_dir = str(master_path.parent)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if out_stem is None:
        out_stem = master_path.stem + "_enriched"

    df_enriched = enrich_conversations(
        df,
        text_col=text_col,
        time_col=time_col,
        call_id_col=call_id_col
    )

    out_tsv = os.path.join(out_dir, out_stem + ".tsv") if write_tsv else None
    out_xlsx = os.path.join(out_dir, out_stem + ".xlsx") if write_xlsx else None

    if write_tsv:
        df_enriched.to_csv(out_tsv, sep="\t", index=False)
    if write_xlsx:
        df_enriched.to_excel(out_xlsx, index=False)

    return out_tsv, out_xlsx


# -----------------------------
# CLI (sigue existiendo)
# -----------------------------
def _cli():
    parser = argparse.ArgumentParser(description="Enrich master TSV/XLSX with time features.")
    parser.add_argument("--master_tsv", type=str, required=True, help="Ruta al master TSV (name, subtag, time, call_id, ...).")
    parser.add_argument("--out_dir", type=str, default=None, help="Directorio de salida (default: mismo del master).")
    parser.add_argument("--out_stem", type=str, default=None, help="Nombre base sin extensión (default: master_stem_enriched).")
    parser.add_argument("--no_xlsx", action="store_true", help="No escribir XLSX.")
    parser.add_argument("--no_tsv", action="store_true", help="No escribir TSV.")
    args = parser.parse_args()

    out_tsv, out_xlsx = enrich_master_tsv(
        master_tsv_path=args.master_tsv,
        out_dir=args.out_dir,
        out_stem=args.out_stem,
        write_xlsx=not args.no_xlsx,
        write_tsv=not args.no_tsv,
    )
    print("OK:")
    if out_tsv: print("-", out_tsv)
    if out_xlsx: print("-", out_xlsx)

if __name__ == "__main__":
    _cli()
