#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import math
import argparse
import traceback
import pandas as pd
from datetime import datetime

# -------------------------
# Fast regex helpers (baratos)
# -------------------------

FILLER_REGEX = re.compile(
    r"^(sí|si|ok|okay|dale|listo|perfecto|bueno|ajá|aja|mmm|eh|uh|perdón|perdon|gracias|bueno\?|¿bueno\?)$",
    re.IGNORECASE
)

# Pistas rápidas de PII: si no hay nada de esto, NO llamamos Presidio (ahorra mucho)
PII_HINT_RE = re.compile(
    r"(\b\d{2,}\b|https?://|www\.|@|\b(señor|señora|sr\.?|sra\.?|don|doña)\b)",
    re.IGNORECASE
)

# "www punto ..." y números largos: pre-masking barato
URL_SPELLED_RE = re.compile(r"\bwww\b(?:\s+punto\s+\w+)+", re.IGNORECASE)
URL_DIRECT_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

# secuencias numéricas tipo cédula/teléfono/montos con separadores
NUM_LONG_RE = re.compile(r"(?:(?<=\D)|^)\d[\d\s,.-]{2,}\d(?:(?=\D)|$)")
NUM_4PLUS_RE = re.compile(r"(?:(?<=\D)|^)\d{4,}(?:(?=\D)|$)")

# -------------------------
# Presidio (lazy init + contadores)
# -------------------------

_PRESIDIO_READY = False
_analyzer = None
_anonymizer = None

# stats globales del masking
MASK_STATS = {
    "mask_calls_total": 0,
    "mask_calls_with_hint": 0,
    "presidio_inits_ok": 0,
    "presidio_calls": 0,
    "presidio_calls_ok": 0,
    "presidio_calls_failed": 0,
}

def _spacy_model_available(model_name: str = "es_core_news_md") -> bool:
    """
    Verifica si el modelo spaCy está instalado sin cargarlo (evita segfault).
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec(model_name.replace("-", "_"))
        return spec is not None
    except Exception:
        return False


def _init_presidio():
    """
    Inicializa Presidio + spaCy solo cuando se necesita.
    Si falla, deja fallback activo (regex).
    Incluye pre-check del modelo spaCy para evitar segfaults.
    """
    global _PRESIDIO_READY, _analyzer, _anonymizer

    if _PRESIDIO_READY:
        return True

    try:
        # Pre-check: verificar que el modelo spaCy existe antes de intentar cargarlo
        if not _spacy_model_available("es_core_news_md"):
            print("[PRESIDIO][WARN] Modelo spaCy 'es_core_news_md' no instalado. Usando fallback regex.", flush=True)
            _PRESIDIO_READY = False
            return False

        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "es", "model_name": "es_core_news_md"}],
        })
        nlp_engine = provider.create_engine()
        _analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["es"])
        _anonymizer = AnonymizerEngine()
        _PRESIDIO_READY = True
        MASK_STATS["presidio_inits_ok"] += 1
        print("[PRESIDIO] Inicializado correctamente.", flush=True)
        return True
    except Exception as e:
        print(f"[PRESIDIO][WARN] Fallo al inicializar Presidio: {e}. Usando fallback regex.", flush=True)
        _PRESIDIO_READY = False
        return False



def _merge_rows(a, b, text_col, start_col, end_col, speaker_col=None):
    """
    Devuelve una nueva fila = a + b (concatenación), preservando timestamps:
    start=min, end=max.
    """
    aa = a.copy()
    ta = normalize_text(aa[text_col])
    tb = normalize_text(b[text_col])

    aa[text_col] = normalize_text(ta + (" " if ta and tb else "") + tb)

    # timestamps
    a_start = safe_float(aa.get(start_col))
    b_start = safe_float(b.get(start_col))
    a_end   = safe_float(aa.get(end_col))
    b_end   = safe_float(b.get(end_col))

    if a_start is None and b_start is not None:
        aa[start_col] = b.get(start_col)
    elif a_start is not None and b_start is not None:
        aa[start_col] = min(a_start, b_start)

    if a_end is None and b_end is not None:
        aa[end_col] = b.get(end_col)
    elif a_end is not None and b_end is not None:
        aa[end_col] = max(a_end, b_end)

    # speaker: si existe y a está vacío, hereda b
    if speaker_col and speaker_col in aa.index:
        asp = "" if pd.isna(aa.get(speaker_col)) else str(aa.get(speaker_col)).strip()
        bsp = "" if pd.isna(b.get(speaker_col)) else str(b.get(speaker_col)).strip()
        if not asp and bsp:
            aa[speaker_col] = b.get(speaker_col)

    return aa

def normalize_text(x):
    if pd.isna(x):
        return ""
    s = str(x).replace("\t", " ").replace("\n", " ").strip()
    return " ".join(s.split())


def _premask_cheap(text: str) -> str:
    """
    Pre-masking barato para reducir tokens y ayudar a Presidio:
    - URLs directas o "www punto ..."
    - secuencias numéricas largas
    """
    t = normalize_text(text)
    if not t:
        return ""
    t = URL_DIRECT_RE.sub("<URL>", t)
    t = URL_SPELLED_RE.sub("<URL>", t)
    t = NUM_LONG_RE.sub("<NUM>", t)
    t = NUM_4PLUS_RE.sub("<NUM>", t)
    return t


def presidio_mask(text: str) -> str:
    """
    Enmascara PERSON/URL/PHONE/EMAIL/etc con Presidio.
    Asume que _init_presidio() ya fue llamado.
    """
    results = _analyzer.analyze(
        text=text,
        language="es",
        entities=["PERSON", "URL", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "IBAN_CODE"],
    )
    anonymized = _anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={
            "PERSON": {"type": "replace", "new_value": "<NAME>"},
            "URL": {"type": "replace", "new_value": "<URL>"},
            "PHONE_NUMBER": {"type": "replace", "new_value": "<PHONE>"},
            "EMAIL_ADDRESS": {"type": "replace", "new_value": "<EMAIL>"},
            "CREDIT_CARD": {"type": "replace", "new_value": "<CARD>"},
            "IBAN_CODE": {"type": "replace", "new_value": "<IBAN>"},
        }
    )
    return anonymized.text


def mask_pii(text: str, use_presidio: bool = True) -> str:
    """
    - Normaliza
    - Pre-masking barato (URLs/números)
    - Si hay hints de PII, intenta Presidio (nombres/PII avanzada)
    - Fallback seguro si Presidio no está disponible
    """
    MASK_STATS["mask_calls_total"] += 1

    t = normalize_text(text)
    if not t:
        return ""

    t = _premask_cheap(t)

    if not use_presidio:
        return t

    if not PII_HINT_RE.search(t):
        return t

    MASK_STATS["mask_calls_with_hint"] += 1

    if not _init_presidio():
        return t

    MASK_STATS["presidio_calls"] += 1
    try:
        t2 = presidio_mask(t)
        MASK_STATS["presidio_calls_ok"] += 1
        return " ".join(t2.split())
    except Exception:
        MASK_STATS["presidio_calls_failed"] += 1
        return t

# -------------------------
# Parsing / IO
# -------------------------

def parse_spanish_float(x):
    if pd.isna(x):
        return float("nan")
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return float("nan")


def read_transcript(path, sep=None, encoding="utf-8"):
    """
    Lee XLSX/CSV/TSV robusto.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, dtype=str, engine="openpyxl")

    if sep is not None:
        return pd.read_csv(path, sep=sep, encoding=encoding, dtype=str)

    try:
        df = pd.read_csv(path, sep="\t", encoding=encoding, dtype=str)
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass

    return pd.read_csv(path, sep=",", encoding=encoding, dtype=str)


def safe_float(x):
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None


def word_count(s: str) -> int:
    s = normalize_text(s)
    return 0 if not s else len(s.split())


def is_filler_only(s: str) -> bool:
    s = normalize_text(s)
    if not s:
        return True
    return bool(FILLER_REGEX.match(s))


def chunk_indices(n_items, max_items):
    if max_items <= 0:
        return [(0, n_items)]
    return [(start, min(n_items, start + max_items)) for start in range(0, n_items, max_items)]


def derive_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_preproc{ext}"


def write_output(df_out: pd.DataFrame, output_path: str, input_path: str, sep: str = None, encoding: str = "utf-8"):
    """
    Escribe con la misma extensión que el input:
    - xlsx/xls -> to_excel
    - csv/tsv  -> to_csv (manteniendo sep si se proporcionó; si no, inferir por extensión)
    """
    ext = os.path.splitext(input_path)[1].lower()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if ext in [".xlsx", ".xls"]:
        df_out.to_excel(output_path, index=False, engine="openpyxl")
        return

    if ext == ".tsv":
        df_out.to_csv(output_path, sep="\t", index=False, encoding=encoding)
        return

    # default csv
    if sep is None:
        sep = ","
    df_out.to_csv(output_path, sep=sep, index=False, encoding=encoding)

# -------------------------
# Compactación
# -------------------------


def compact_units_bounded(
    dfg: pd.DataFrame,
    text_col: str,
    start_col: str,
    end_col: str,
    speaker_col: str = None,
    min_words: int = 6,
    max_words: int = 20,
    merge_fillers: bool = True,
):
    """
    Compacta con límites estrictos:
    - Si >max_words: se descarta
    - Si <min_words: intenta sí o sí merge con prev/next MÁS CORTO que quepa <=max_words
    - Preserva start/end usando end_col
    """
    # stats de esta conversación
    st = {
        "dropped_gt_max_original": 0,
        "dropped_gt_max_after_merge": 0,
        "forced_merge_prev": 0,
        "forced_merge_next": 0,
        "unmerged_lt_min": 0,   # quedó <min porque NO había forma de pegar sin pasarse
    }

    # filas limpias
    items = []
    for _, r in dfg.iterrows():
        rr = r.copy()
        rr[text_col] = normalize_text(rr[text_col])
        if rr[text_col] == "":
            continue
        wc = word_count(rr[text_col])
        if wc > max_words:
            st["dropped_gt_max_original"] += 1
            continue
        items.append(rr)

    out = []
    i = 0
    n = len(items)

    def is_merge_candidate(row):
        t = normalize_text(row[text_col])
        if merge_fillers and is_filler_only(t):
            return True
        return word_count(t) < min_words

    while i < n:
        cur = items[i]
        cur_wc = word_count(cur[text_col])

        # ya es "normal"
        if cur_wc >= min_words:
            out.append(cur)
            i += 1
            continue

        # cur < min_words (o filler) => pegar sí o sí si se puede
        prev_exists = len(out) > 0
        next_exists = (i + 1) < n

        # candidatos (solo si resultan <= max_words)
        options = []

        if prev_exists:
            prev = out[-1]
            prev_wc = word_count(prev[text_col])
            if prev_wc + cur_wc <= max_words:
                # score = tamaño final => preferir vecino más corto / resultante menor
                options.append(("prev", prev_wc + cur_wc))

        if next_exists:
            nxt = items[i + 1]
            nxt_wc = word_count(nxt[text_col])
            if cur_wc + nxt_wc <= max_words:
                options.append(("next", cur_wc + nxt_wc))

        if options:
            # elige la opción que deje el bloque más pequeño (vecino “más corto”)
            options.sort(key=lambda x: x[1])
            choice = options[0][0]

            if choice == "prev":
                out[-1] = _merge_rows(out[-1], cur, text_col, start_col, end_col, speaker_col=speaker_col)
                st["forced_merge_prev"] += 1
                i += 1
                continue

            else:  # next
                nxt = items[i + 1]
                merged = _merge_rows(cur, nxt, text_col, start_col, end_col, speaker_col=speaker_col)
                # merged no puede pasar de max porque lo validamos arriba
                out.append(merged)
                st["forced_merge_next"] += 1
                i += 2
                continue

        # Si no se puede pegar sin superar max_words, lo dejamos (pero marcado)
        # (mejor que perder timestamps; si prefieres drop, cambia esto por continue)
        out.append(cur)
        st["unmerged_lt_min"] += 1
        i += 1

    # Segunda pasada: intentar arreglar los que quedaron <min (si es posible) sin pasarse de max
    cleaned = []
    j = 0
    while j < len(out):
        cur = out[j]
        cur_wc = word_count(cur[text_col])
        if cur_wc >= min_words:
            cleaned.append(cur)
            j += 1
            continue

        # intentar merge con vecino más corto (prev o next en cleaned/out)
        prev_exists = len(cleaned) > 0
        next_exists = (j + 1) < len(out)
        options = []

        if prev_exists:
            prev = cleaned[-1]
            pw = word_count(prev[text_col])
            if pw + cur_wc <= max_words:
                options.append(("prev", pw + cur_wc))

        if next_exists:
            nxt = out[j + 1]
            nw = word_count(nxt[text_col])
            if cur_wc + nw <= max_words:
                options.append(("next", cur_wc + nw))

        if options:
            options.sort(key=lambda x: x[1])
            if options[0][0] == "prev":
                cleaned[-1] = _merge_rows(cleaned[-1], cur, text_col, start_col, end_col, speaker_col=speaker_col)
                j += 1
            else:
                nxt = out[j + 1]
                merged = _merge_rows(cur, nxt, text_col, start_col, end_col, speaker_col=speaker_col)
                cleaned.append(merged)
                j += 2
        else:
            cleaned.append(cur)
            j += 1

    # Sanidad final: si algo quedó >max (por algún edge-case), se descarta
    final = []
    for r in cleaned:
        wc = word_count(r[text_col])
        if wc > max_words:
            st["dropped_gt_max_after_merge"] += 1
            continue
        final.append(r)

    return pd.DataFrame(final).reset_index(drop=True), st


def merge_short_utterances(
    dfg: pd.DataFrame,
    text_col: str,
    start_col: str,
    end_col: str,
    speaker_col: str = None,
    min_words: int = 6,
    max_gap_sec: float = 2.0,
    merge_fillers: bool = True,
):
    rows = []
    n = len(dfg)

    def sp(row):
        if speaker_col and speaker_col in dfg.columns:
            v = row.get(speaker_col, "")
            return "" if pd.isna(v) else str(v).strip()
        return ""

    i = 0
    while i < n:
        cur = dfg.iloc[i].copy()
        cur_text = normalize_text(cur[text_col])

        wc = word_count(cur_text)
        filler = is_filler_only(cur_text)
        short = wc < min_words
        mergeable = short or (merge_fillers and filler)

        if not mergeable:
            rows.append(cur)
            i += 1
            continue

        prev_exists = len(rows) > 0
        next_exists = (i + 1) < n
        cur_sp = sp(cur)

        prev_ok = False
        if prev_exists:
            prev = rows[-1]
            prev_sp = sp(prev)

            gap = None
            prev_end = safe_float(prev.get(end_col))
            cur_start = safe_float(cur.get(start_col))
            if prev_end is not None and cur_start is not None:
                gap = cur_start - prev_end

            if (prev_sp and cur_sp and prev_sp == cur_sp) or (gap is not None and gap <= max_gap_sec):
                prev_ok = True

        if prev_ok:
            prev = rows[-1]
            prev_text = normalize_text(prev[text_col])
            prev[text_col] = normalize_text(prev_text + (" " if prev_text and cur_text else "") + cur_text)

            if safe_float(prev.get(start_col)) is None:
                prev[start_col] = cur.get(start_col)

            cur_end = safe_float(cur.get(end_col))
            prev_end = safe_float(prev.get(end_col))
            if cur_end is not None and (prev_end is None or cur_end > prev_end):
                prev[end_col] = cur.get(end_col)

            rows[-1] = prev
            i += 1
            continue

        if next_exists:
            nxt = dfg.iloc[i + 1].copy()
            nxt_text = normalize_text(nxt[text_col])

            merged = cur.copy()
            merged[text_col] = normalize_text(cur_text + (" " if cur_text and nxt_text else "") + nxt_text)

            if end_col in nxt.index:
                merged[end_col] = nxt.get(end_col)

            if speaker_col and speaker_col in dfg.columns:
                if not cur_sp:
                    merged[speaker_col] = nxt.get(speaker_col)

            rows.append(merged)
            i += 2
            continue

        rows.append(cur)
        i += 1

    return pd.DataFrame(rows).reset_index(drop=True)


def merge_consecutive_same_speaker(
    dfg: pd.DataFrame,
    text_col: str,
    start_col: str,
    end_col: str,
    speaker_col: str = None,
    max_gap_sec: float = 1.5,
):
    if speaker_col is None or speaker_col not in dfg.columns or len(dfg) <= 1:
        return dfg.reset_index(drop=True)

    out = []
    cur = dfg.iloc[0].copy()

    for i in range(1, len(dfg)):
        nxt = dfg.iloc[i].copy()

        cur_sp = "" if pd.isna(cur[speaker_col]) else str(cur[speaker_col]).strip()
        nxt_sp = "" if pd.isna(nxt[speaker_col]) else str(nxt[speaker_col]).strip()

        # gap robusto: si no hay end, usa start-start
        cur_end = safe_float(cur.get(end_col))
        nxt_start = safe_float(nxt.get(start_col))
        cur_start = safe_float(cur.get(start_col))

        gap = None
        if cur_end is not None and nxt_start is not None:
            gap = nxt_start - cur_end
        elif cur_start is not None and nxt_start is not None:
            gap = nxt_start - cur_start

        if cur_sp and nxt_sp and cur_sp == nxt_sp and (gap is not None and gap <= max_gap_sec):
            cur_text = normalize_text(cur[text_col])
            nxt_text = normalize_text(nxt[text_col])
            cur[text_col] = normalize_text(cur_text + (" " if cur_text and nxt_text else "") + nxt_text)
            if safe_float(nxt.get(end_col)) is not None:
                cur[end_col] = nxt.get(end_col)
        else:
            out.append(cur)
            cur = nxt

    out.append(cur)
    return pd.DataFrame(out).reset_index(drop=True)


def split_long_units(dfg: pd.DataFrame, text_col: str, max_unit_words: int = 60):
    rows = []
    for _, r in dfg.iterrows():
        t = normalize_text(r[text_col])
        if word_count(t) <= max_unit_words:
            rows.append(r)
            continue
        words = t.split()
        for k in range(0, len(words), max_unit_words):
            rr = r.copy()
            rr[text_col] = " ".join(words[k:k + max_unit_words])
            rows.append(rr)
    return pd.DataFrame(rows).reset_index(drop=True)

# -------------------------
# Stats helpers
# -------------------------

def _wc_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna("").astype(str).apply(lambda s: len(normalize_text(s).split()))


def save_stats_excel(stats_path: str, summary_dict: dict, per_call_rows: list):
    os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
    df_summary = pd.DataFrame([summary_dict])
    df_per_call = pd.DataFrame(per_call_rows)

    with pd.ExcelWriter(stats_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="summary")
        df_per_call.to_excel(writer, index=False, sheet_name="per_call")


# -------------------------
# Main
# -------------------------

def build_compact(
    input_path: str,
    text_col: str = "text",
    time_col: str = "start",
    end_col: str = "end",
    speaker_col: str = "speaker",
    group_col: str = "file_name",
    min_words: int = 4,
    max_gap_short_merge: float = 2.0,
    max_gap_same_speaker: float = 1.5,
    max_unit_words: int = 60,
    max_units_per_call: int = 70,
    do_mask: bool = True,
    use_presidio: bool = True,
    merge_fillers: bool = True,
    sep: str = None,
    encoding: str = "utf-8",
    do_stats: bool = False,
):
    df = read_transcript(input_path, sep=sep, encoding=encoding)
    cols_lower = {c.lower(): c for c in df.columns}
    def fix(col):
        return col if col in df.columns else cols_lower.get(col.lower(), col)

    text_col = fix(text_col)
    time_col = fix(time_col)
    end_col = fix(end_col)
    speaker_col = fix(speaker_col)
    group_col = fix(group_col)

    if text_col not in df.columns:
        raise ValueError(f"No encontré '{text_col}' en el archivo. Columnas: {list(df.columns)}")
    if time_col not in df.columns:
        raise ValueError(f"No encontré '{time_col}' en el archivo. Columnas: {list(df.columns)}")

    # tiempos
    df["_start_sec"] = df[time_col].apply(parse_spanish_float)
    if end_col not in df.columns:
        raise ValueError(f"Este preprocesamiento requiere columna '{end_col}' (timestamp final). Columnas: {list(df.columns)}")
    df["_end_sec"] = df[end_col].apply(parse_spanish_float)

    # texto
    df["_text_norm"] = df[text_col].apply(normalize_text)

    # group
    if group_col not in df.columns:
        df[group_col] = os.path.basename(input_path)

    # ordenar
    df = df.sort_values([group_col, "_start_sec"], ascending=[True, True], na_position="last").reset_index(drop=True)

    out_rows = []
    per_call_stats = []

    # stats globales pre
    total_in_rows = len(df)

    for gname, dfg in df.groupby(group_col, sort=False):
        dfg = dfg.reset_index(drop=True)
        in_rows = len(dfg)
        in_wc = _wc_series(dfg, "_text_norm")
        in_filler_count = int(dfg["_text_norm"].apply(is_filler_only).sum())

        if do_mask:
            dfg["_text_norm"] = dfg["_text_norm"].apply(lambda x: mask_pii(x, use_presidio=use_presidio))

        # 1) merge de cortos / fillers
        dfg3, st_local = compact_units_bounded(
            dfg,
            text_col="_text_norm",
            start_col="_start_sec",
            end_col="_end_sec",
            speaker_col=speaker_col if speaker_col in dfg.columns else None,
            min_words=min_words,
            max_words=20,               
            merge_fillers=merge_fillers
        )
        print(f"[DEBUG] {gname}: compact_units_bounded done. Rows in={in_rows} | out={len(dfg3)}", flush=True)
        # limpiar vacíos
        try:
            dfg3["_text_norm"] = dfg3["_text_norm"].apply(normalize_text)
            dfg3 = dfg3[dfg3["_text_norm"] != ""].reset_index(drop=True)

            # chunk por subcall
            chunks = chunk_indices(len(dfg3), max_units_per_call)

            for subcall_id, (a, b) in enumerate(chunks, start=1):
                sub = dfg3.iloc[a:b].reset_index(drop=True)

                for local_i in range(len(sub)):
                    row = sub.iloc[local_i]
                    t = row["_start_sec"]
                    t_str = "" if (isinstance(t, float) and math.isnan(t)) else str(t)

                    indice = 0 if local_i == 0 else local_i

                    txt = row["_text_norm"]
                    if speaker_col in sub.columns:
                        sp = row.get(speaker_col, "")
                        sp = "" if pd.isna(sp) else str(sp).strip()
                        if sp:
                            txt = f"{sp}: {txt}"

                    out_rows.append({
                        "indice": indice,
                        "texto": txt,
                        "tiempo": t_str,
                        "file_name": str(gname),
                        "subcall_id": subcall_id,
                        "unit_row": int(a + local_i),
                        "unit_start": row["_start_sec"] if not (isinstance(row["_start_sec"], float) and math.isnan(row["_start_sec"])) else "",
                        "unit_end": row["_end_sec"] if not (isinstance(row["_end_sec"], float) and math.isnan(row["_end_sec"])) else "",
                    })
        except Exception as e:
            print(f"[ERROR] Procesando grupo '{gname}': {e}", flush=True)
            continue

        # per-call stats
        if do_stats:
            per_call_stats.append({
                "file_name": str(gname),
                "rows_in": in_rows,
                "rows_out": len(dfg3),
                "chunks_subcalls": len(chunks),
                "avg_words_out": float(_wc_series(dfg3, "_text_norm").mean()) if len(dfg3) else 0.0,
                "p95_words_out": float(_wc_series(dfg3, "_text_norm").quantile(0.95)) if len(dfg3) else 0.0,
                **st_local
            })

    df_out = pd.DataFrame(out_rows)

    # output path auto
    output_path = derive_output_path(input_path)
    write_output(df_out, output_path, input_path=input_path, sep=sep, encoding=encoding)

    print(f"[OK] Output generado: {output_path}", flush=True)
    print(f"[INFO] Filas salida: {len(df_out)} | mask={do_mask} | presidio={use_presidio} | merge_fillers={merge_fillers}", flush=True)

    # stats a excel "stats.xlsx" en mismo dir
    if do_stats:
        stats_path = os.path.join(os.path.dirname(input_path) or ".", "stats.xlsx")

        summary = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_path": input_path,
            "output_path": output_path,
            "rows_input_total": total_in_rows,
            "rows_output_total": len(df_out),
            "min_words": min_words,
            "max_gap_short_merge": max_gap_short_merge,
            "max_gap_same_speaker": max_gap_same_speaker,
            "max_unit_words": max_unit_words,
            "max_units_per_call": max_units_per_call,
            "mask_enabled": do_mask,
            "use_presidio": use_presidio,
            "merge_fillers": merge_fillers,
            # masking stats
            **MASK_STATS
        }

        save_stats_excel(stats_path, summary, per_call_stats)
        print(f"[OK] Stats guardados en: {stats_path}", flush=True)

    return output_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Preprocesa transcripciones para etiquetado GPT con MENOS tokens (sin ventanizaje): compacta + masking PII. Output auto: _preproc."
    )
    p.add_argument("--input", required=True, help="Ruta CSV/TSV/XLSX con columnas text y start.")
    p.add_argument("--text_col", default="text")
    p.add_argument("--time_col", default="start")
    p.add_argument("--end_col", default="end")
    p.add_argument("--speaker_col", default="speaker")
    p.add_argument("--group_col", default="file_name")

    p.add_argument("--min_words", type=int, default=6)
    p.add_argument("--max_gap_short_merge", type=float, default=2.0)
    p.add_argument("--max_gap_same_speaker", type=float, default=1.5)
    p.add_argument("--max_unit_words", type=int, default=60)
    p.add_argument("--max_units_per_call", type=int, default=70)

    # Mask ON por defecto
    p.add_argument("--no_mask", action="store_true", help="Desactiva enmascarado PII.")
    p.add_argument("--no_presidio", action="store_true", help="No usar Presidio (solo pre-masking barato).")
    p.add_argument("--no_merge_fillers", action="store_true", help="No forzar merge de fillers (sí/ok/etc).")

    p.add_argument("--stats", action="store_true", help="Guarda stats.xlsx en la misma carpeta del input.")
    p.add_argument("--sep", default=None, help="Separador forzado para CSV/TSV (opcional).")
    p.add_argument("--encoding", default="utf-8")
    args = p.parse_args()

    try:
        print(f"[PREPROC] Iniciando preprocesamiento: {args.input}", flush=True)
        build_compact(
            input_path=args.input,
            text_col=args.text_col,
            time_col=args.time_col,
            end_col=args.end_col,
            speaker_col=args.speaker_col,
            group_col=args.group_col,
            min_words=args.min_words,
            max_gap_short_merge=args.max_gap_short_merge,
            max_gap_same_speaker=args.max_gap_same_speaker,
            max_unit_words=args.max_unit_words,
            max_units_per_call=args.max_units_per_call,
            do_mask=(not args.no_mask),
            use_presidio=(not args.no_presidio),
            merge_fillers=(not args.no_merge_fillers),
            sep=args.sep,
            encoding=args.encoding,
            do_stats=args.stats,
        )
    except Exception as e:
        print(f"[PREPROC][FATAL] Error en preprocesamiento: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
