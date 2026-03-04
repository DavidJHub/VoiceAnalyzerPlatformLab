#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
convSegmentation.py

Flujo completo (con REANUDAR automático):
- Si YA existen archivos en  workdir/csv_segmented/*.csv  -> salta descarga + json->csv + split + fit
  y reanuda DIRECTO desde Presidio (post-segmentación).
- Si NO existen -> corre todo el pipeline completo y además guarda mapping_base_to_id.json
  para que futuras reanudaciones conserven id_agent_audio_data.

Salida final:
- Guarda CSV por archivo en:  workdir/final_with_pii/<mismo_nombre>.csv
- (Opcional) guarda un CSV concatenado: workdir/final_with_pii/segments_with_pii_ALL.csv

NOTA importante:
- splitConversations() y splitLongText() requieren columna 'words_list' en csv_raw.
  Por eso esta versión de jsonTranscriptionToCsv genera:
    text, start, end, speaker, num_words, words_list, words_str, avg_confidence, avg_speaker_confidence
"""

import os
import re
import json
from typing import List, Dict, Optional, Any

import pandas as pd
from tqdm import tqdm

import database.dbConfig as dbcfg
from utils.VapUtils import jsonTranscriptionToCsv as _jsonTranscriptionToCsv_external  # por si ya existe, no usamos
from lang.VapLangUtils import splitConversations
from segmentationModel.fittingDeep import fitCSVConversations
from clientData.transcriptScrapper import fetch_agent_audio_rows_last_days
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

S3_BUCKET_NAME = "documentos.aihub"


# =========================================================
# DB / S3
# =========================================================
def get_db_connection():
    return dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP,
    )


def get_s3_client():
    return dbcfg.generate_s3_client()


# =========================================================
# Helpers generales
# =========================================================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def segmented_exists(segmented_dir: str) -> bool:
    return os.path.isdir(segmented_dir) and any(
        f.lower().endswith(".csv") for f in os.listdir(segmented_dir)
    )


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    # Ajusta si tu modelo usa un nombre fijo
    candidates = [
        "text",
        "window_text",
        "utterance",
        "sentence",
        "chunk",
        "transcript",
        "content",
        "texto",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in obj_cols:
        sample = df[c].dropna().astype(str).head(6).tolist()
        if any(len(x) >= 12 for x in sample):
            return c
    return None


def _dedupe_keep_order(items: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for x in items:
        k = str(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _normalize_digits(s: str) -> str:
    return re.sub(r"\D+", "", s or "")


def _compact_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# =========================================================
# JSON -> CSV (versión compatible con splitConversations)
# =========================================================
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize_words(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return _WORD_RE.findall(text.lower())


def jsonTranscriptionToCsv(directory: str, directoryOutput: str):
    """
    Convierte JSON con formato:
      results.channels[0].alternatives[0].paragraphs[].sentences[]
    a CSV con columnas que tu pipeline espera (incluye words_list).

    words_list se guarda como JSON-string (ej: ["hola","mundo"])
    para que sea parseable luego (json.loads o literal_eval).
    """
    ensure_dir(directoryOutput)

    json_files = [f for f in os.listdir(directory) if f.lower().endswith(".json")]
    print(f"[*] jsonTranscriptionToCsv: JSON encontrados en {directory}: {len(json_files)}")

    written = 0
    for filename in json_files:
        file_path = os.path.join(directory, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[X] JSON inválido: {file_path} err={e}")
            continue

        paragraphs = []
        try:
            channels = obj.get("results", {}).get("channels", [])
            if channels and isinstance(channels, list):
                alternatives = channels[0].get("alternatives", [])
                if alternatives and isinstance(alternatives, list):
                    paragraphs = alternatives[0].get("paragraphs", []) or []
        except Exception:
            paragraphs = []

        if not paragraphs:
            # igual que tu lógica vieja: crea CSV vacío con columnas esperadas
            top_keys = list(obj.keys())[:20] if isinstance(obj, dict) else []
            print(f"[!] Sin paragraphs en {filename}. top_keys={top_keys}")

            empty_data = {
                "text": [" ", " "],
                "start": [0, 0],
                "end": [0, 0],
                "speaker": [0, 0],
                "num_words": [0, 0],
                "words_list": [json.dumps([], ensure_ascii=False), json.dumps([], ensure_ascii=False)],
                "words_str": ["", ""],
                "avg_confidence": [None, None],
                "avg_speaker_confidence": [None, None],
            }
            emp_df = pd.DataFrame(empty_data)
            out_path = os.path.join(directoryOutput, f"{filename[:-5]}.csv")
            emp_df.to_csv(out_path, index=False, encoding="utf-8")
            written += 1
            continue

        rows = []
        for p in paragraphs:
            p_speaker = p.get("speaker", 0)
            p_start = p.get("start", 0)
            p_end = p.get("end", 0)

            sentences = p.get("sentences", []) or []
            if not sentences:
                p_text = p.get("text", "")
                wl = _tokenize_words(p_text)
                rows.append({
                    "text": p_text,
                    "start": p_start,
                    "end": p_end,
                    "speaker": p_speaker,
                    "num_words": len(wl),
                    "words_list": json.dumps(wl, ensure_ascii=False),
                    "words_str": " ".join(wl),
                    "avg_confidence": None,
                    "avg_speaker_confidence": None,
                })
                continue

            for s in sentences:
                s_text = s.get("text", "")
                s_start = s.get("start", p_start)
                s_end = s.get("end", p_end)

                wl = _tokenize_words(s_text)
                rows.append({
                    "text": s_text,
                    "start": s_start,
                    "end": s_end,
                    "speaker": p_speaker,  # en tu ejemplo speaker está en párrafo
                    "num_words": len(wl),
                    "words_list": json.dumps(wl, ensure_ascii=False),
                    "words_str": " ".join(wl),
                    "avg_confidence": None,
                    "avg_speaker_confidence": None,
                })

        df = pd.DataFrame(rows)
        out_path = os.path.join(directoryOutput, f"{filename[:-5]}.csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
        written += 1

    print(f"[*] jsonTranscriptionToCsv: CSV escritos: {written}/{len(json_files)}")


# =========================================================
# Descargar JSONs desde S3
# =========================================================
def _sanitize_s3_key(key: str) -> str:
    key = (key or "").strip()

    # Si viene como s3://bucket/...
    if key.startswith("s3://"):
        key_wo = key[5:]
        parts = key_wo.split("/", 1)
        key = parts[1] if len(parts) == 2 else ""

    # quita slash inicial
    return key.lstrip("/")


def download_transcripts_from_db_rows(df_aad: pd.DataFrame, json_dir: str) -> Dict[str, int]:
    ensure_dir(json_dir)
    s3 = get_s3_client()

    mapping: Dict[str, int] = {}
    ok, bad = 0, 0

    for _, row in df_aad.iterrows():
        id_aad = row["id"]
        name = row.get("name")
        key_raw = row.get("link_transcription_audio") or ""
        key = _sanitize_s3_key(key_raw)

        if not key:
            print(f"[!] key inválido id={id_aad}, name={name}, key_raw={key_raw}")
            bad += 1
            continue

        base_name = os.path.basename(key)
        local_json = os.path.join(json_dir, base_name)

        if not os.path.exists(local_json):
            try:
                s3.download_file(S3_BUCKET_NAME, key, local_json)
                # print(f"[↓] Descargado id={id_aad}: {base_name}")
            except Exception as e:
                print(f"[!] Error descargando id={id_aad}, key={key}: {e}")
                bad += 1
                continue

        base_no_ext = os.path.splitext(base_name)[0]
        mapping[base_no_ext] = int(id_aad)
        ok += 1

    print(f"[*] Descargas OK: {ok} | Fallidas: {bad}")
    return mapping


# =========================================================
# Presidio: setup + extracción (MONTO, FECHA, TELEFONO, ID, EDAD)
# =========================================================
_PRESIDIO_ANALYZER_ES: Optional[AnalyzerEngine] = None

MONTHS_ES = {
    "enero": "01",
    "febrero": "02",
    "marzo": "03",
    "abril": "04",
    "mayo": "05",
    "junio": "06",
    "julio": "07",
    "agosto": "08",
    "septiembre": "09",
    "setiembre": "09",
    "octubre": "10",
    "noviembre": "11",
    "diciembre": "12",
}

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer import PatternRecognizer, Pattern

def build_presidio_analyzer_es(spacy_model: str = "es_core_news_lg") -> AnalyzerEngine:
    nlp_conf = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "es", "model_name": spacy_model}],
    }
    provider = NlpEngineProvider(nlp_configuration=nlp_conf)
    nlp_engine = provider.create_engine()

    # Registry SOLO en español (evita que intente cargar recognizers en otros idiomas)
    registry = RecognizerRegistry(supported_languages=["es"])

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=["es"],
    )

    # PHONE_NUMBER (LatAm)
    phone_patterns = [
        Pattern(
            name="PHONE_WITH_COUNTRY_CODE",
            regex=r"\b(?:\+?\s*(?:57|52|54|56|51|503))[\s\-]?\(?\d{1,3}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}(?:[\s\-]?\d{2,4})?\b",
            score=0.80,
        ),
        Pattern(
            name="PHONE_WITH_LABEL",
            regex=r"\b(?:tel(?:éfono)?|cel(?:ular)?|móvil|whats(?:app)?|wp)\s*[:#\-]?\s*(\+?\s*(?:57|52|54|56|51|503)?[\s\-]?\(?\d{1,3}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}(?:[\s\-]?\d{2,4})?)\b",
            score=0.85,
        ),
    ]
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=phone_patterns,
            supported_language="es",
        )
    )

    # ID_NUMBER (LATAM)
    id_patterns = [
        Pattern(
            name="CO_ID_LABELED",
            regex=r"\b(?:cc|c\.c\.|cédula|cedula|ti|t\.i\.|tarjeta\s+de\s+identidad|ce|c\.e\.|c[eé]dula\s+de\s+extranjer[ií]a|documento|identificaci[oó]n)\s*[:#\-]?\s*(\d{1,3}(?:\.\d{3}){1,3}|\d{6,10})\b",
            score=0.90,
        ),
        Pattern(
            name="MX_CURP",
            regex=r"\b[A-Z][AEIOUX][A-Z]{2}\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[HM][A-Z]{5}\d{2}\b",
            score=0.92,
        ),
        Pattern(
            name="MX_RFC",
            regex=r"\b(?:[A-Z&Ñ]{3,4})\d{6}[A-Z0-9]{3}\b",
            score=0.88,
        ),
        Pattern(
            name="MX_NSS_LABELED",
            regex=r"\b(?:nss|seguro\s+social)\s*[:#\-]?\s*(\d{11})\b",
            score=0.88,
        ),
        Pattern(
            name="AR_DNI_LABELED",
            regex=r"\b(?:dni|documento)\s*[:#\-]?\s*(\d{1,2}\.?\d{3}\.?\d{3}|\d{7,8})\b",
            score=0.90,
        ),
        Pattern(
            name="AR_CUIT_CUIL_LABELED",
            regex=r"\b(?:cuit|cuil)\s*[:#\-]?\s*(\d{2}\-?\d{8}\-?\d)\b",
            score=0.92,
        ),
        Pattern(
            name="CL_RUT_LABELED",
            regex=r"\b(?:rut)\s*[:#\-]?\s*(\d{1,2}\.\d{3}\.\d{3}\-[0-9kK]|\d{7,8}\-[0-9kK])\b",
            score=0.93,
        ),
        Pattern(
            name="CL_RUT_BARE",
            regex=r"\b\d{7,8}\-[0-9kK]\b",
            score=0.70,
        ),
        Pattern(
            name="PE_DNI_LABELED",
            regex=r"\b(?:dni|documento)\s*[:#\-]?\s*(\d{8})\b",
            score=0.92,
        ),
        Pattern(
            name="PE_RUC_LABELED",
            regex=r"\b(?:ruc)\s*[:#\-]?\s*(\d{11})\b",
            score=0.93,
        ),
        Pattern(
            name="PE_CE_LABELED",
            regex=r"\b(?:ce|carn[eé]\s+de\s+extranjer[ií]a)\s*[:#\-]?\s*(\d{8,12})\b",
            score=0.80,
        ),
        Pattern(
            name="SV_DUI_LABELED",
            regex=r"\b(?:dui)\s*[:#\-]?\s*(\d{8}\-?\d)\b",
            score=0.93,
        ),
        Pattern(
            name="SV_NIT_LABELED",
            regex=r"\b(?:nit)\s*[:#\-]?\s*(\d{4}\-?\d{6}\-?\d{3}\-?\d)\b",
            score=0.90,
        ),
    ]
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="ID_NUMBER",
            patterns=id_patterns,
            supported_language="es",
        )
    )

    # MONEY_AMOUNT
    money_patterns = [
        Pattern(name="MONEY_DOLLAR_SYMBOL", regex=r"\$\s*\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{1,2})?", score=0.80),
        Pattern(name="MONEY_CURRENCY_CODE", regex=r"\b(?:COP|MXN|ARS|CLP|PEN|USD|EUR)\s*\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{1,2})?\b", score=0.85),
        Pattern(name="MONEY_WORDS", regex=r"\b\d+(?:[.,]\d+)?\s*(?:millones?|mill[oó]n|mil|m)\s*(?:de\s+)?(?:pesos|cop|mxn|ars|clp|pen|soles|d[oó]lares|usd)?\b", score=0.70),
        Pattern(name="MONEY_WITH_WORD_PESOS", regex=r"\b\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{1,2})?\s*(?:pesos|cop|mxn|ars|clp|pen|soles|d[oó]lares|usd)\b", score=0.78),
    ]
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="MONEY_AMOUNT",
            patterns=money_patterns,
            supported_language="es",
        )
    )

    # DATE_ES
    date_patterns = [
        Pattern(name="DATE_NUMERIC", regex=r"\b(?:0?[1-9]|[12]\d|3[01])[\/\-.](?:0?[1-9]|1[0-2])[\/\-.](?:\d{2}|\d{4})\b", score=0.80),
        Pattern(name="DATE_ISO", regex=r"\b\d{4}\-(?:0[1-9]|1[0-2])\-(?:0[1-9]|[12]\d|3[01])\b", score=0.85),
        Pattern(name="DATE_TEXT_ES", regex=r"\b(?:0?[1-9]|[12]\d|3[01])\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s+de\s+\d{4}\b", score=0.90),
    ]
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="DATE_ES",
            patterns=date_patterns,
            supported_language="es",
        )
    )

    # AGE
    age_patterns = [
        Pattern(name="AGE_WITH_ANIOS", regex=r"\b(?:tengo|edad|de)\s*[:#\-]?\s*(\d{1,3})\s*(?:a[nñ]os|años)\b", score=0.85),
        Pattern(name="AGE_RANGE", regex=r"\b(\d{1,3})\s*a\s*(\d{1,3})\s*(?:a[nñ]os|años)\b", score=0.70),
        Pattern(name="AGE_X_ANIOS", regex=r"\b(\d{1,3})\s*(?:a[nñ]os|años)\s*(?:de\s+edad)?\b", score=0.55),
    ]
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="AGE",
            patterns=age_patterns,
            supported_language="es",
        )
    )

    return analyzer


def get_presidio_analyzer_es(spacy_model: str = "es_core_news_lg") -> AnalyzerEngine:
    global _PRESIDIO_ANALYZER_ES
    if _PRESIDIO_ANALYZER_ES is None:
        _PRESIDIO_ANALYZER_ES = build_presidio_analyzer_es(spacy_model=spacy_model)
    return _PRESIDIO_ANALYZER_ES


def parse_money(span: str):
    s = _compact_spaces(span).lower()

    currency = None
    for code in ["cop", "mxn", "ars", "clp", "pen", "usd", "eur"]:
        if re.search(rf"\b{code}\b", s):
            currency = code.upper()
            break
    if currency is None:
        if "$" in s:
            currency = "$"
        elif "pesos" in s:
            currency = "PESOS"
        elif "soles" in s or "s/" in s:
            currency = "PEN"
        elif "dolares" in s or "dólares" in s:
            currency = "USD"

    mult = 1.0
    if re.search(r"\bmillones?\b|\bmill[oó]n\b", s):
        mult = 1_000_000.0
    elif re.search(r"\bmil\b", s):
        mult = 1_000.0
    elif re.search(r"\bm\b", s) and re.search(r"\d\s*m\b", s):
        mult = 1_000_000.0

    m = re.search(r"\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?", s)
    if not m:
        return None, currency

    num = m.group(0).replace(" ", "")
    if "." in num and "," in num:
        num = num.replace(".", "").replace(",", ".")
    elif "," in num and "." not in num:
        if re.search(r",\d{1,2}$", num):
            num = num.replace(",", ".")
        else:
            num = num.replace(",", "")
    elif "." in num and "," not in num:
        if not re.search(r"\.\d{1,2}$", num):
            num = num.replace(".", "")

    try:
        return float(num) * mult, currency
    except Exception:
        return None, currency


def normalize_date_to_iso(span: str) -> Optional[str]:
    s = _compact_spaces(span).lower()

    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    m = re.match(r"^(0?[1-9]|[12]\d|3[01])[\/\-.](0?[1-9]|1[0-2])[\/\-.](\d{2}|\d{4})$", s)
    if m:
        dd = int(m.group(1))
        mm = int(m.group(2))
        yy = m.group(3)
        if len(yy) == 2:
            yyi = int(yy)
            yyyy = 2000 + yyi if yyi <= 69 else 1900 + yyi
        else:
            yyyy = int(yy)
        return f"{yyyy:04d}-{mm:02d}-{dd:02d}"

    m = re.match(
        r"^(0?[1-9]|[12]\d|3[01])\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s+de\s+(\d{4})$",
        s,
    )
    if m:
        dd = int(m.group(1))
        mm = MONTHS_ES.get(m.group(2))
        yyyy = int(m.group(3))
        if mm:
            return f"{yyyy:04d}-{int(mm):02d}-{dd:02d}"

    return None


def infer_id_country(span: str) -> str:
    s = _compact_spaces(span).lower()
    if re.search(r"\b(rut)\b", s) or re.search(r"\d{7,8}\-[0-9kK]\b", span):
        return "Chile"
    if re.search(r"\b(dui|nit)\b", s):
        return "El Salvador"
    if re.search(r"\b(ruc)\b", s):
        return "Perú"
    if re.search(r"\b(cuit|cuil)\b", s):
        return "Argentina"
    if re.search(r"\b(curp|rfc|nss)\b", s) or re.search(r"\b[A-Z&Ñ]{3,4}\d{6}[A-Z0-9]{3}\b", span):
        return "México"
    if re.search(r"\b(cc|c\.c\.|cédula|cedula|ti|t\.i\.|ce|c\.e\.)\b", s):
        return "Colombia"
    return ""


def extract_structured_entities(text: str, analyzer: AnalyzerEngine, score_threshold: float = 0.35) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {
            "persons": [],
            "telefonos_raw": [],
            "telefonos_digits": [],
            "ids_raw": [],
            "ids_digits": [],
            "ids_country": [],
            "montos_raw": [],
            "montos_value": [],
            "montos_currency": [],
            "fechas_raw": [],
            "fechas_iso": [],
            "edades_raw": [],
            "edades_value": [],
        }

    results = analyzer.analyze(
        text=text,
        language="es",
        entities=["PERSON", "PHONE_NUMBER", "ID_NUMBER", "MONEY_AMOUNT", "DATE_ES", "AGE"],
        score_threshold=score_threshold,
    )

    persons, tel_raw, tel_digits = [], [], []
    ids_raw, ids_digits, ids_country = [], [], []
    montos_raw, montos_value, montos_currency = [], [], []
    fechas_raw, fechas_iso = [], []
    edades_raw, edades_value = [], []

    for r in results:
        span = _compact_spaces(text[r.start:r.end])

        if r.entity_type == "PERSON":
            persons.append(span)

        elif r.entity_type == "PHONE_NUMBER":
            d = _normalize_digits(span)
            if len(d) >= 7:
                tel_raw.append(span)
                tel_digits.append(d)

        elif r.entity_type == "ID_NUMBER":
            d = _normalize_digits(span)
            if len(d) >= 6 or re.search(r"[A-Z&Ñ]", span, re.I):
                ids_raw.append(span)
                ids_digits.append(d if d else "")
                ids_country.append(infer_id_country(span))

        elif r.entity_type == "MONEY_AMOUNT":
            val, cur = parse_money(span)
            montos_raw.append(span)
            montos_value.append(val)
            montos_currency.append(cur or "")

        elif r.entity_type == "DATE_ES":
            fechas_raw.append(span)
            fechas_iso.append(normalize_date_to_iso(span) or "")

        elif r.entity_type == "AGE":
            edades_raw.append(span)
            m = re.search(r"\b(\d{1,3})\b", span)
            edades_value.append(int(m.group(1)) if m else None)

    # dedupe
    persons = _dedupe_keep_order(persons)
    tel_raw = _dedupe_keep_order(tel_raw)
    tel_digits = _dedupe_keep_order(tel_digits)

    # ids dedupe alineado
    ids_raw_d, ids_digits_d, ids_country_d = [], [], []
    seen = set()
    for a, b, c in zip(ids_raw, ids_digits, ids_country):
        if a in seen:
            continue
        seen.add(a)
        ids_raw_d.append(a); ids_digits_d.append(b); ids_country_d.append(c)
    ids_raw, ids_digits, ids_country = ids_raw_d, ids_digits_d, ids_country_d

    # montos dedupe por raw
    mraw_d, mval_d, mcur_d = [], [], []
    seen = set()
    for a, b, c in zip(montos_raw, montos_value, montos_currency):
        if a in seen:
            continue
        seen.add(a)
        mraw_d.append(a); mval_d.append(b); mcur_d.append(c)
    montos_raw, montos_value, montos_currency = mraw_d, mval_d, mcur_d

    # fechas dedupe por raw
    fraw_d, fiso_d = [], []
    seen = set()
    for a, b in zip(fechas_raw, fechas_iso):
        if a in seen:
            continue
        seen.add(a)
        fraw_d.append(a); fiso_d.append(b)
    fechas_raw, fechas_iso = fraw_d, fiso_d

    # edades dedupe por raw
    eraw_d, eval_d = [], []
    seen = set()
    for a, b in zip(edades_raw, edades_value):
        if a in seen:
            continue
        seen.add(a)
        eraw_d.append(a); eval_d.append(b)
    edades_raw, edades_value = eraw_d, eval_d

    return {
        "persons": persons,
        "telefonos_raw": tel_raw,
        "telefonos_digits": tel_digits,
        "ids_raw": ids_raw,
        "ids_digits": ids_digits,
        "ids_country": ids_country,
        "montos_raw": montos_raw,
        "montos_value": montos_value,
        "montos_currency": montos_currency,
        "fechas_raw": fechas_raw,
        "fechas_iso": fechas_iso,
        "edades_raw": edades_raw,
        "edades_value": edades_value,
    }


# =========================================================
# Guardado final
# =========================================================
def save_final_per_file(df: pd.DataFrame, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    df.to_csv(out_path, index=False, encoding="utf-8")


# =========================================================
# PIPELINE principal (con REANUDAR automático)
# =========================================================
def segment_conversations_from_s3_jsons(
    workdir: str,
    days: int = 2,
    max_words: int = 18,
    window_size: int = 14,
    stride: int = 6,
    max_length: int = 32,
    presidio_spacy_model: str = "es_core_news_lg",
    presidio_score_threshold: float = 0.35,
    save_final: bool = True,
    save_concat_all: bool = False,
) -> List[pd.DataFrame]:

    ensure_dir(workdir)
    json_dir = ensure_dir(os.path.join(workdir, "json_raw"))
    raw_csv_dir = ensure_dir(os.path.join(workdir, "csv_raw"))
    segmented_dir = ensure_dir(os.path.join(workdir, "csv_segmented"))
    final_dir = ensure_dir(os.path.join(workdir, "final_with_pii"))

    mapping_path = os.path.join(workdir, "mapping_base_to_id.json")

    # ==========================
    # REANUDAR automático
    # ==========================
    if segmented_exists(segmented_dir):
        print(f"[*] REANUDAR: ya existen CSV segmentados en {segmented_dir}. Saltando hasta Presidio.")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping_base_to_id = json.load(f)
            print(f"[*] Mapping cargado: {mapping_path} (n={len(mapping_base_to_id)})")
        else:
            mapping_base_to_id = {}
            print("[!] No existe mapping_base_to_id.json, id_agent_audio_data quedará en None.")
    else:
        # --- 1) DB rows ---
        df_aad = fetch_agent_audio_rows_last_days(days=days)
        if df_aad is None or df_aad.empty:
            print("[!] No se encontraron filas en agent_audio_data para el rango de fechas.")
            return []

        # --- 2) Descargar JSON ---
        mapping_base_to_id = download_transcripts_from_db_rows(df_aad, json_dir)
        if not mapping_base_to_id:
            print("[!] No se descargó ningún JSON válido desde S3.")
            return []

        # Guardar mapping para reanudar
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping_base_to_id, f, ensure_ascii=False, indent=2)
        print(f"[*] Mapping guardado: {mapping_path}")

        # --- 3) json -> csv (con words_list) ---
        print("[*] Convirtiendo JSON a CSV de oraciones (con words_list)...")
        jsonTranscriptionToCsv(json_dir, raw_csv_dir)

        # --- 4) split ---
        print("[*] Dividiendo conversaciones largas en trozos...")
        splitConversations(raw_csv_dir, raw_csv_dir, max_words=max_words)

        # --- 5) segmentación ---
        print("[*] Ejecutando fitCSVConversations (modelo de segmentación)...")
        fitCSVConversations(
            input_folder=raw_csv_dir,
            output_folder=segmented_dir,
            window_size=window_size,
            stride=stride,
            max_length=max_length,
        )

    # ==========================
    # PRESIDIO (siempre corre aquí)
    # ==========================
    segmented_files = [f for f in os.listdir(segmented_dir) if f.lower().endswith(".csv")]
    print(f"[*] CSV segmentados encontrados: {len(segmented_files)}")

    analyzer = get_presidio_analyzer_es(spacy_model=presidio_spacy_model)

    dfs_out: List[pd.DataFrame] = []
    print("[*] Ejecutando Presidio sobre CSV segmentados...")

    for filename in tqdm(segmented_files, desc=f"Procesando archivos en {segmented_dir}"):
        in_path = os.path.join(segmented_dir, filename)
        try:
            df_seg = pd.read_csv(in_path)

            # id_agent_audio_data (si mapping existe)
            base_name = os.path.splitext(filename)[0]
            id_original = mapping_base_to_id.get(base_name) if isinstance(mapping_base_to_id, dict) else None
            df_seg["id_agent_audio_data"] = id_original if id_original is not None else None

            text_col = detect_text_column(df_seg)
            if text_col is None:
                # crea columnas vacías para no romper
                df_seg["persons"] = [[] for _ in range(len(df_seg))]
                df_seg["telefonos_raw"] = [[] for _ in range(len(df_seg))]
                df_seg["telefonos_digits"] = [[] for _ in range(len(df_seg))]
                df_seg["ids_raw"] = [[] for _ in range(len(df_seg))]
                df_seg["ids_digits"] = [[] for _ in range(len(df_seg))]
                df_seg["ids_country"] = [[] for _ in range(len(df_seg))]
                df_seg["montos_raw"] = [[] for _ in range(len(df_seg))]
                df_seg["montos_value"] = [[] for _ in range(len(df_seg))]
                df_seg["montos_currency"] = [[] for _ in range(len(df_seg))]
                df_seg["fechas_raw"] = [[] for _ in range(len(df_seg))]
                df_seg["fechas_iso"] = [[] for _ in range(len(df_seg))]
                df_seg["edades_raw"] = [[] for _ in range(len(df_seg))]
                df_seg["edades_value"] = [[] for _ in range(len(df_seg))]
            else:
                extracted = df_seg[text_col].fillna("").astype(str).apply(
                    lambda t: extract_structured_entities(
                        t, analyzer=analyzer, score_threshold=presidio_score_threshold
                    )
                )

                # PERSON
                df_seg["persons"] = extracted.apply(lambda d: d["persons"])
                df_seg["persons_str"] = df_seg["persons"].apply(lambda xs: " | ".join(xs))

                # TELEFONO
                df_seg["telefonos_raw"] = extracted.apply(lambda d: d["telefonos_raw"])
                df_seg["telefonos_digits"] = extracted.apply(lambda d: d["telefonos_digits"])
                df_seg["telefonos_raw_str"] = df_seg["telefonos_raw"].apply(lambda xs: " | ".join(xs))
                df_seg["telefonos_digits_str"] = df_seg["telefonos_digits"].apply(lambda xs: " | ".join(xs))

                # ID
                df_seg["ids_raw"] = extracted.apply(lambda d: d["ids_raw"])
                df_seg["ids_digits"] = extracted.apply(lambda d: d["ids_digits"])
                df_seg["ids_country"] = extracted.apply(lambda d: d["ids_country"])
                df_seg["ids_raw_str"] = df_seg["ids_raw"].apply(lambda xs: " | ".join(xs))
                df_seg["ids_digits_str"] = df_seg["ids_digits"].apply(lambda xs: " | ".join(xs))
                df_seg["ids_country_str"] = df_seg["ids_country"].apply(lambda xs: " | ".join([x for x in xs if x]))

                # MONTO
                df_seg["montos_raw"] = extracted.apply(lambda d: d["montos_raw"])
                df_seg["montos_value"] = extracted.apply(lambda d: d["montos_value"])
                df_seg["montos_currency"] = extracted.apply(lambda d: d["montos_currency"])
                df_seg["montos_raw_str"] = df_seg["montos_raw"].apply(lambda xs: " | ".join(xs))
                df_seg["montos_value_str"] = df_seg["montos_value"].apply(lambda xs: " | ".join([str(x) for x in xs]))
                df_seg["montos_currency_str"] = df_seg["montos_currency"].apply(lambda xs: " | ".join([x for x in xs if x]))

                # FECHA
                df_seg["fechas_raw"] = extracted.apply(lambda d: d["fechas_raw"])
                df_seg["fechas_iso"] = extracted.apply(lambda d: d["fechas_iso"])
                df_seg["fechas_raw_str"] = df_seg["fechas_raw"].apply(lambda xs: " | ".join(xs))
                df_seg["fechas_iso_str"] = df_seg["fechas_iso"].apply(lambda xs: " | ".join([x for x in xs if x]))

                # EDAD
                df_seg["edades_raw"] = extracted.apply(lambda d: d["edades_raw"])
                df_seg["edades_value"] = extracted.apply(lambda d: d["edades_value"])
                df_seg["edades_raw_str"] = df_seg["edades_raw"].apply(lambda xs: " | ".join(xs))
                df_seg["edades_value_str"] = df_seg["edades_value"].apply(lambda xs: " | ".join([str(x) for x in xs]))

            # Guardar por archivo (FINAL)
            if save_final:
                out_path = os.path.join(final_dir, filename)
                save_final_per_file(df_seg, out_path)

            dfs_out.append(df_seg)

        except Exception as e:
            print(f"[!] Error en {filename}: {e}")

    # Guardar concat (opcional)
    if save_final and save_concat_all and dfs_out:
        all_path = os.path.join(final_dir, "segments_with_pii_ALL.csv")
        pd.concat(dfs_out, ignore_index=True).to_csv(all_path, index=False, encoding="utf-8")
        print(f"[*] Guardado concatenado: {all_path}")

    return dfs_out


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    dfs = segment_conversations_from_s3_jsons(
        workdir="./work_segmented",
        days=1,
        max_words=18,
        window_size=14,
        stride=6,
        max_length=32,
        presidio_spacy_model="es_core_news_lg",
        presidio_score_threshold=0.35,
        save_final=True,          # <- guarda SIEMPRE en work_segmented/final_with_pii/
        save_concat_all=False,    # <- pon True si quieres un CSV gigante concatenado
    )

    print(f"DataFrames segmentados generados: {len(dfs)}")
    if dfs:
        print(dfs[0].head())

    print("\n[*] Resultado final guardado en:")
    print("    ./work_segmented/final_with_pii/")