import os
import io
import re
import sys
import argparse
import subprocess

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

import database.dbConfig as dbcfg
import training.datasetPrep as dsprep

# ============================
# CONFIG
# ============================

BUCKET_DEFAULT = "aihubmodelos"
LOCAL_ROOT = "reentreno"  # raíz local requerida

SUBTAGS = [
    "SALUDO",
    "PERFILAMIENTO",
    "PRODUCTO",
    "CONFIRMACION MONITOREO",
    "LEY RETRACTO",
    "TERMINOS LEGALES",
    "TRATAMIENTO DATOS",
    "MAC",
    "MAC REFUERZO",
    "PRECIO",
    "CONFIRMACION DATOS",
    "CONFORMIDAD",
    "ATENCION",
    "DESPEDIDA",
]
SUBTAGS_SET = set(SUBTAGS)


# ============================
# Helpers Path / S3
# ============================

def sanitize_dirname(name: str) -> str:
    name = str(name).strip()
    name = name.replace("\\", "_").replace("/", "_")
    return name

def s3_join(*parts: str) -> str:
    return "/".join([str(p).strip("/").replace("\\", "/") for p in parts if p is not None and str(p) != ""])

def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

import unicodedata
import re

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))

def _collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _normalize_subtag(raw: str, text: str = "") -> str:
    """
    Normaliza variantes típicas y corrige etiquetas NO permitidas a una permitida,
    para que el pipeline NO se caiga por issues de formato del LLM.

    Regla especial: "CONFIRMACION" -> MONITOREO vs DATOS por heurística simple.
    """
    s = "" if raw is None else str(raw)
    s = _strip_accents(s).upper()
    s = s.replace("\t", " ").replace("\n", " ")
    s = _collapse_spaces(s)

    # Canonicalizaciones conocidas
    s = s.replace("TRATAMIENTO DE DATOS", "TRATAMIENTO DATOS")

    # Heurística para "CONFIRMACION" (la causa de tu crash)
    if s in {"CONFIRMACION", "CONFIRMACION.", "CONFIRMACION:"}:
        t = _strip_accents(str(text or "")).lower()

        # señales de monitoreo/calidad/grabación
        if any(k in t for k in [
            "monitore", "monitor", "calidad", "grabada", "grabacion", "grabaremos",
            "esta llamada sera grabada", "la llamada sera grabada"
        ]):
            return "CONFIRMACION MONITOREO"

        # señales de datos
        if any(k in t for k in [
            "dato", "datos", "cedula", "documento", "correo", "email", "direccion",
            "telefono", "celular", "nit", "numero de"
        ]):
            return "CONFIRMACION DATOS"

        # default seguro (elige uno; yo recomiendo DATOS)
        return "CONFIRMACION DATOS"

    return s

def s3_list_keys(s3_client, bucket: str, prefix: str):
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if not k or k.endswith("/"):
                continue
            yield k

def s3_download_file(s3_client, bucket: str, key: str, local_path: str):
    ensure_dir(os.path.dirname(local_path))
    s3_client.download_file(bucket, key, local_path)
    return local_path

def s3_upload_file(s3_client, local_path: str, bucket: str, key: str):
    s3_client.upload_file(local_path, bucket, key)
    return s3_uri(bucket, key)

def s3_key_exists(s3_client, bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


# ============================
# DB
# ============================

def get_db_connection():
    return dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )

def get_sponsor_info(conn, id_sponsor: int):
    q = """
    SELECT sponsor, country
    FROM marketing_campaigns
    WHERE sponsor_id = %s
    LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(q, (id_sponsor,))
        row = cur.fetchone()

    if not row:
        raise RuntimeError(f"No existe sponsor_id={id_sponsor} en marketing_campaigns")

    sponsor, country = row[0], row[1]
    if not sponsor or not country:
        raise RuntimeError(f"Sponsor/Country inválidos para sponsor_id={id_sponsor}: {row}")

    return sponsor, country


# ============================
# CARGA EJEMPLO / MATRIZ
# ============================

def _norm_cols(df: pd.DataFrame):
    return [str(c).strip().lower() for c in df.columns]

def load_example_from_xlsx(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de ejemplo: {path}")

    df = pd.read_excel(path)
    cols = _norm_cols(df)

    if "name" in cols:
        text_col = df.columns[cols.index("name")]
    elif "texto" in cols:
        text_col = df.columns[cols.index("texto")]
    else:
        text_col = df.columns[0]

    if "subtag" in cols:
        lab_col = df.columns[cols.index("subtag")]
    elif "cluster" in cols:
        lab_col = df.columns[cols.index("cluster")]
    elif "etiqueta" in cols:
        lab_col = df.columns[cols.index("etiqueta")]
    else:
        lab_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    if "time" in cols:
        time_col = df.columns[cols.index("time")]
    elif "tiempo" in cols:
        time_col = df.columns[cols.index("tiempo")]
    else:
        time_col = df.columns[2] if len(df.columns) > 2 else df.columns[-1]

    parts = []
    for _, row in df.iterrows():
        texto  = str(row.get(text_col, "")).replace("\t", " ").replace("\n", " ").strip()
        subtag = str(row.get(lab_col, "")).strip()
        tiempo = str(row.get(time_col, "")).strip()

        if (not texto or texto.lower() == "nan") and (not subtag or subtag.lower() == "nan") and (not tiempo or tiempo.lower() == "nan"):
            continue

        parts.append(f"{texto} {subtag} {tiempo}")

    return " ".join(parts)

def load_matrix_from_xlsx(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró la matriz de guion: {path}")

    df = pd.read_excel(path)
    cols = _norm_cols(df)

    if "name" in cols:
        name_col = df.columns[cols.index("name")]
    elif "texto" in cols:
        name_col = df.columns[cols.index("texto")]
    else:
        name_col = df.columns[0]

    if "subtag" in cols:
        lab_col = df.columns[cols.index("subtag")]
    elif "cluster" in cols:
        lab_col = df.columns[cols.index("cluster")]
    elif "etiqueta" in cols:
        lab_col = df.columns[cols.index("etiqueta")]
    else:
        lab_col = df.columns[1] if len(df.columns) > 1 else df.columns[-1]

    df_out = df[[name_col, lab_col]].copy()
    df_out.columns = ["name", "subtag"]

    df_out["name"] = df_out["name"].astype(str).str.replace("\t", " ").str.replace("\n", " ").str.strip()
    df_out["subtag"] = df_out["subtag"].astype(str).str.strip()
    df_out = df_out[
        (df_out["name"].str.lower() != "nan") &
        (df_out["subtag"].str.lower() != "nan") &
        (df_out["name"] != "") &
        (df_out["subtag"] != "")
    ].reset_index(drop=True)

    return df_out


# ============================
# PROMPT BASE
# ============================

def build_base_prompt(tagged_example: str, matrix_df: pd.DataFrame) -> str:
    matrix_lines = []
    for _, row in matrix_df.iterrows():
        text = str(row["name"]).replace("\t", " ").replace("\n", " ").strip()
        lab  = str(row["subtag"]).strip()
        if not text or not lab:
            continue
        matrix_lines.append(f"{text}\t{lab}")
    matrix_block = "\n".join(matrix_lines)

    allowed = ", ".join(SUBTAGS)

    prompt = f"""
Eres un asistente experto en etiquetar llamadas de call center (Multiasistencias IGS en alianza con Banco de Bogotá).

SOLO vas a usar SUBTAGS (un único nivel de etiquetas).

SUBTAGS PERMITIDOS (en MAYÚSCULAS):
{allowed}

IMPORTANTE: Para este proyecto, usa estas definiciones operacionales (no inventes otras):

========================
MAC vs CONFORMIDAD vs MAC REFUERZO
========================
REGLAS CRÍTICAS PARA MAC, MAC REFUERZO Y CONFORMIDAD
(ESTA SECCIÓN ES OBLIGATORIA Y NO ADMITE INTERPRETACIÓN)

1) MAC (MANDATO DE AUTORIZACIÓN COMERCIAL)
- Un MAC ocurre ÚNICAMENTE cuando el asesor solicita de forma EXPLÍCITA
  la autorización del cliente para:
  - activar la multiasistencia
  - realizar la afiliación
  - efectuar la contratación
- Ejemplos claros de MAC:
  - "¿Autoriza la activación de la multiasistencia?"
  - "¿Me autoriza a realizar la afiliación?"
  - "¿Autoriza el cobro del servicio?"

- Si NO se está solicitando autorización explícita,
  NO es MAC bajo ninguna circunstancia.

2) MAC REFUERZO
- MAC REFUERZO ocurre ÚNICAMENTE cuando:
  a) Ya ocurrió un MAC previo, Y
  b) El asesor REPITE o REFORMULA la solicitud de autorización
     porque el cliente no respondió claramente (silencio, duda, evasiva).

- Ejemplos de MAC REFUERZO:
  - "Entonces, para confirmar, ¿me autoriza la activación?"
  - "¿Me confirma si autoriza o no el servicio?"

- IMPORTANTE:
  - Preguntas cortas como "¿Correcto?", "¿Vale?", "¿De acuerdo?"
    NUNCA son MAC REFUERZO por sí solas.
  - Si la pregunta NO repite ni refuerza explícitamente la autorización,
    NO es MAC REFUERZO.

3) CONFORMIDAD
- CONFORMIDAD corresponde a preguntas o expresiones breves de asentimiento
  que NO solicitan autorización legal.
- Ejemplos típicos de CONFORMIDAD:
  - "¿Correcto?"
  - "¿Vale?"
  - "¿Listo?"
  - "¿De acuerdo?"
  - "¿Está bien?"

- Estas expresiones SON CONFORMIDAD,
  SALVO en el caso especial descrito en el punto 4.

4) REGLA ESPECIAL (CRÍTICA Y PRIORITARIA)
- Si y SOLO SI:
  a) La línea inmediatamente anterior fue un MAC, Y
  b) La línea actual es una pregunta de CONFORMIDAD,

  ENTONCES:
  - La línea de CONFORMIDAD SE ETIQUETA TAMBIÉN COMO MAC.
  - NO se etiqueta como CONFORMIDAD.
  - NO se etiqueta como MAC REFUERZO.

- Ejemplo correcto:
  "¿Autoriza la activación de la multiasistencia?"   → MAC
  "¿Correcto?"                                      → MAC

- Ejemplo incorrecto (NO hacer esto):
  "¿Correcto?" → MAC REFUERZO   ❌
  "¿Correcto?" → MAC            ❌ (si no hay MAC justo antes)

5) CONTEXTO NO MAC
- Si "¿Correcto?", "¿Vale?", etc. aparecen:
  - en explicaciones de precio
  - en validaciones informativas
  - en cualquier contexto distinto a una solicitud explícita de autorización

  DEBEN etiquetarse como CONFORMIDAD.

========================

Tienes dos referencias:

1) EJEMPLO ya etiquetado (formato: TEXTO SUBTAG TIEMPO_EN_SEGUNDOS):
{tagged_example}

2) MATRIZ de referencia del guion (cada fila: texto<TAB>subtag):
{matrix_block}

REGLAS GENERALES:

- Te voy a dar una llamada como lista de líneas, cada una con TEXTO y TIEMPO.
- Para CADA línea devuelve EXACTAMENTE una fila.
- "texto" debe ser EXACTO (sin resumir, sin unir).
- "subtag" debe ser SOLO uno de los permitidos.
- Si coincide casi exactamente con la matriz → usa el subtag de la matriz.
- Si no coincide → elige el subtag más cercano por función.
- No inventes etiquetas.

FORMATO DE SALIDA OBLIGATORIO:
TSV SIN encabezados con columnas:
1) texto   2) subtag   3) tiempo

A continuación va la llamada a etiquetar:
""".strip()

    return prompt


# ============================
# Aux etiquetado / parsing
# ============================

def split_calls_by_index_reset(df):
    df = df.copy()
    df.columns = ["indice", "texto", "tiempo"] + list(df.columns[3:])
    df = df[["indice", "texto", "tiempo"]]

    indices_0 = df.index[df["indice"] == 0].tolist()
    if len(indices_0) == 0:
        return [df.reset_index(drop=True)]

    call_dfs = []
    for i, start_idx in enumerate(indices_0):
        if i < len(indices_0) - 1:
            end_idx = indices_0[i + 1]
            call_dfs.append(df.loc[start_idx:end_idx - 1])
        else:
            call_dfs.append(df.loc[start_idx:])

    return [c.reset_index(drop=True) for c in call_dfs]

def build_call_text_for_prompt(call_df):
    lines = []
    for _, row in call_df.iterrows():
        text = str(row["texto"]).replace("\t", " ").replace("\n", " ")
        time = row["tiempo"]
        lines.append(f"TEXTO: {text}\tTIEMPO: {time}")
    return "\n".join(lines)



# ============================
# Master TSV + Cache
# ============================

def get_processed_call_ids_from_master(master_path: str):
    if not os.path.exists(master_path):
        return set()
    try:
        df = pd.read_csv(master_path, sep="\t", dtype=str)
    except Exception:
        return set()
    if "call_id" not in df.columns:
        return set()
    return set(df["call_id"].astype(str).unique())

def append_to_master(df_new, master_path, add_info=None):
    df_new = df_new.copy()
    if add_info:
        for k, v in add_info.items():
            df_new[k] = v

    if os.path.exists(master_path):
        df_master = pd.read_csv(master_path, sep="\t", dtype=str)
        df_out = pd.concat([df_master, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(master_path, sep="\t", index=False)
    print(f"Archivo maestro actualizado: {master_path} (filas totales: {len(df_out)})")

def cache_path_for(local_xlsx_path: str) -> str:
    return local_xlsx_path + ".cache"

def load_cache(local_xlsx_path: str):
    p = cache_path_for(local_xlsx_path)
    if not os.path.exists(p):
        return set()
    out = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            cid = line.strip()
            if cid:
                out.add(cid)
    return out

def update_cache(local_xlsx_path: str, call_id: str):
    p = cache_path_for(local_xlsx_path)
    with open(p, "a", encoding="utf-8") as f:
        f.write(str(call_id) + "\n")


# ============================
# Checkpoint uploader
# ============================

def upload_checkpoint(
    s3_client,
    conn,
    bucket: str,
    raw_prefix: str,
    sponsor_dir: str,
    master_local: str,
    id_sponsor: int,
    llamadas_totales: int,
    reason: str,
    create_empty_master_if_missing: bool = True,
):
    import pandas as pd

    master_key_out = f"{raw_prefix.rstrip('/')}/master_{sponsor_dir}.tsv"
    path_tsv = f"s3://{bucket}/{master_key_out}"

    if not os.path.exists(master_local) and create_empty_master_if_missing:
        empty_cols = ["name", "subtag", "time", "call_id", "source_file", "id_sponsor", "sponsor", "country"]
        df_empty = pd.DataFrame(columns=empty_cols)
        os.makedirs(os.path.dirname(master_local), exist_ok=True)
        df_empty.to_csv(master_local, sep="\t", index=False)
        print(f"[CHECKPOINT:{reason}] Master local no existía -> creado vacío: {master_local}")

    if os.path.exists(master_local):
        s3_client.upload_file(master_local, bucket, master_key_out)
        print(f"[CHECKPOINT:{reason}] Subido master a: {path_tsv}")
    else:
        print(f"[CHECKPOINT:{reason}] No se subió archivo (no existe master_local), pero se hará upsert igual: {path_tsv}")

    q = """
    INSERT INTO vap_reentreno (id_sponsor, llamadas_totales, path_tsv)
    VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE
        llamadas_totales = VALUES(llamadas_totales),
        path_tsv = VALUES(path_tsv)
    """
    with conn.cursor() as cur:
        cur.execute(q, (id_sponsor, int(llamadas_totales), path_tsv))
    conn.commit()

    print(f"[CHECKPOINT:{reason}] Upsert vap_reentreno OK | id_sponsor={id_sponsor} | llamadas_totales={llamadas_totales} | path_tsv={path_tsv}")
    return path_tsv


# ============================
# Preprocesamiento local (NEW)
# ============================

def _derive_preproc_path(input_xlsx: str) -> str:
    base, ext = os.path.splitext(input_xlsx)
    return f"{base}_preproc{ext}"

def _looks_preprocessed_xlsx(path: str) -> bool:
    """
    Heurística: si las primeras 3 cols incluyen indice/texto/tiempo.
    """
    try:
        df = pd.read_excel(path, nrows=5)
        cols = [str(c).strip().lower() for c in df.columns[:3]]
        return cols == ["indice", "texto", "tiempo"]
    except Exception:
        return False

def preprocess_xlsx_local(
    local_raw_xlsx: str,
    preproc_script: str,
    preproc_extra_args: list = None,
) -> str:
    """
    Aplica el preprocesamiento local:
    raw.xlsx -> raw_preproc.xlsx
    Devuelve la ruta del preproc.
    """
    preproc_path = _derive_preproc_path(local_raw_xlsx)

    # Si ya existe y parece preproc, úsalo
    if os.path.exists(preproc_path) and _looks_preprocessed_xlsx(preproc_path):
        print(f"  -> [PREPROC] ya existe: {preproc_path}")
        return preproc_path

    if _looks_preprocessed_xlsx(local_raw_xlsx):
        # Si por error ya te bajaron preproc, lo aceptamos
        print(f"  -> [PREPROC] input ya está preprocesado: {local_raw_xlsx}")
        return local_raw_xlsx

    if not os.path.exists(preproc_script):
        raise FileNotFoundError(f"No encontré preproc_script={preproc_script}. Pásalo con --preproc_script")

    cmd = [sys.executable, preproc_script, "--input", local_raw_xlsx]
    if preproc_extra_args:
        cmd.extend(preproc_extra_args)

    print(f"  -> [PREPROC] ejecutando: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0:
        print("[PREPROC][STDOUT]\n", r.stdout)
        print("[PREPROC][STDERR]\n", r.stderr)
        raise RuntimeError(f"Preprocesamiento falló (code={r.returncode}) para {local_raw_xlsx}")

    if not os.path.exists(preproc_path):
        # si tu script escribe a otra ruta, esto te lo detecta de inmediato
        print("[PREPROC][STDOUT]\n", r.stdout)
        print("[PREPROC][STDERR]\n", r.stderr)
        raise RuntimeError(f"Preproc terminó pero NO encontré output esperado: {preproc_path}")

    print(f"  -> [PREPROC] OK: {preproc_path}")
    return preproc_path


# ============================
# Resolver TSV existente en rawtraining/
# ============================

def resolve_existing_rawtraining_tsv(s3_client, bucket: str, raw_prefix: str, sponsor_dir: str):
    preferred_key = s3_join(raw_prefix, f"master_{sponsor_dir}.tsv")
    if s3_key_exists(s3_client, bucket, preferred_key):
        return preferred_key

    keys = list(s3_list_keys(s3_client, bucket, raw_prefix))
    tsvs = sorted([k for k in keys if k.lower().endswith(".tsv")])
    return tsvs[0] if tsvs else None


# ============================
# Main runner
# ============================

def run_for_sponsor(
    id_sponsor: int,
    gpt_model: str = "gpt-4o-mini",
    bucket: str = "aihubmodelos",
    preproc_script: str = None,
    preproc_extra_args: list = None,
):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY en .env")
    client = OpenAI(api_key=api_key)

    s3_client = dbcfg.generate_s3_client()
    conn = get_db_connection()

    sponsor, country = get_sponsor_info(conn, id_sponsor)

    country_dir = sanitize_dirname(country)
    sponsor_dir = sanitize_dirname(sponsor)

    local_base = os.path.join(LOCAL_ROOT, country_dir, sponsor_dir)
    local_misc = os.path.join(local_base, "misc")
    local_conv = os.path.join(local_base, "conversations")
    local_raw  = os.path.join(local_base, "rawtraining")

    ensure_dir(local_misc)
    ensure_dir(local_conv)
    ensure_dir(local_raw)

    print("\n==============================")
    print("[START] Reentreno automático")
    print("id_sponsor:", id_sponsor)
    print("pais:", country, "| sponsor:", sponsor)
    print("local_root:", local_base)
    print("==============================\n")

    misc_prefix = s3_join(country, sponsor, "misc") + "/"
    conv_prefix = s3_join(country, sponsor, "conversations") + "/"
    raw_prefix  = s3_join(country, sponsor, "rawtraining") + "/"

    # default preproc_script: preproc.py en el mismo dir que este archivo
    if preproc_script is None:
        here = os.path.dirname(os.path.abspath(__file__))
        preproc_script = os.path.join(here, "buildTrainingWindows.py")

    # 1) Descargar ejemplo/matriz
    example_key = s3_join(misc_prefix, "ejemplo.xlsx")
    matrix_key  = s3_join(misc_prefix, "matriz.xlsx")

    example_local = os.path.join(local_misc, "ejemplo.xlsx")
    matrix_local  = os.path.join(local_misc, "matriz.xlsx")

    print("[INFO] Descargando misc files...")
    s3_download_file(s3_client, bucket, example_key, example_local)
    s3_download_file(s3_client, bucket, matrix_key,  matrix_local)
    print("[OK] misc files descargados:", example_local, "|", matrix_local)

    tagged_example   = load_example_from_xlsx(example_local)
    class_matrix_df  = load_matrix_from_xlsx(matrix_local)
    BASE_PROMPT      = build_base_prompt(tagged_example, class_matrix_df)

    # 2) Resolver TSV existente
    existing_tsv_key = resolve_existing_rawtraining_tsv(s3_client, bucket, raw_prefix, sponsor_dir)
    master_local = os.path.join(local_raw, f"master_{sponsor_dir}.tsv")

    if existing_tsv_key:
        s3_download_file(s3_client, bucket, existing_tsv_key, master_local)
        print(f"[INFO] TSV base descargado: {s3_uri(bucket, existing_tsv_key)}")
        print(f"[INFO] TSV local: {master_local}")
    else:
        print("[INFO] No existe TSV en rawtraining/. Se creará uno nuevo localmente cuando llegue la primera llamada.")
        print(f"[INFO] TSV local (a crear): {master_local}")

    # 3) Listar XLSX en conversations/
    conv_keys = list(s3_list_keys(s3_client, bucket, conv_prefix))
    xlsx_keys = sorted([k for k in conv_keys if k.lower().endswith(".xlsx")])

    if not xlsx_keys:
        raise RuntimeError(f"No encontré .xlsx en s3://{bucket}/{conv_prefix}")

    print(f"[INFO] Archivos XLSX encontrados en conversations/: {len(xlsx_keys)}")

    llamadas_totales = 0

    def label_call_with_gpt(call_df, max_retries=2):
        nonlocal llamadas_totales

        call_text_block = build_call_text_for_prompt(call_df)

        base_messages = [
            {"role": "system", "content": "Eres un asistente experto en etiquetar llamadas de call center con SUBTAGS."},
            {"role": "user", "content": BASE_PROMPT + "\n\n" + call_text_block},
        ]

        last_err = None
        raw_output = None

        for attempt in range(max_retries + 1):
            resp = client.chat.completions.create(
                model=gpt_model,
                messages=base_messages if attempt == 0 else repair_messages,
                temperature=0,
                top_p=1,
            )

            raw_output = (resp.choices[0].message.content or "").strip()

            # Parse TSV
            tsv_buffer = io.StringIO(raw_output)
            df_labels = pd.read_csv(
                tsv_buffer,
                sep="\t",
                header=None,
                names=["texto", "subtag", "tiempo"],
                dtype=str
            )

            # Normaliza/corrige
            df_labels["texto"]  = df_labels["texto"].astype(str)
            df_labels["tiempo"] = df_labels["tiempo"].astype(str).str.strip()
            df_labels["subtag"] = df_labels.apply(lambda r: _normalize_subtag(r["subtag"], r["texto"]), axis=1)

            # Validación
            bad = df_labels[~df_labels["subtag"].isin(SUBTAGS_SET)]
            if bad.empty:
                llamadas_totales += 1
                return df_labels.rename(columns={"texto": "name", "tiempo": "time"})

            last_err = f"Subtags inválidos: {sorted(bad['subtag'].unique().tolist())}"

            # Construye repair prompt para el siguiente intento
            allowed = ", ".join(SUBTAGS)
            repair_messages = [
                {"role": "system", "content": "Corrige salidas. No inventes etiquetas."},
                {"role": "user", "content": (
                    "Tu salida TSV contiene SUBTAGS inválidos.\n"
                    f"SUBTAGS PERMITIDOS (exactos): {allowed}\n\n"
                    "Reescribe TODA la salida en TSV SIN encabezados con columnas:\n"
                    "texto<TAB>subtag<TAB>tiempo\n"
                    "- Mantén el texto EXACTO por línea.\n"
                    "- Mantén el tiempo EXACTO.\n"
                    "- Usa SOLO subtags permitidos (exact match).\n\n"
                    "Esta fue tu salida anterior (corrígela, no la resumas):\n"
                    + raw_output
                )}
            ]

        raise ValueError(f"GPT devolvió subtags inválidos tras reintentos. Último error: {last_err}")


    # 4) Loop principal
    try:
        for file_idx, key in enumerate(xlsx_keys, start=1):
            fname = os.path.basename(key)
            local_xlsx_raw = os.path.join(local_conv, fname)

            print(f"\n[DOWNLOAD {file_idx}/{len(xlsx_keys)}] {s3_uri(bucket, key)}")
            s3_download_file(s3_client, bucket, key, local_xlsx_raw)

            # --- PREPROCESAR LOCALMENTE ANTES DE ETIQUETAR ---
            print(f"[PREPROC {file_idx}/{len(xlsx_keys)}] {local_xlsx_raw}")
            local_xlsx_preproc = preprocess_xlsx_local(
                local_raw_xlsx=local_xlsx_raw,
                preproc_script=preproc_script,
                preproc_extra_args=preproc_extra_args,
            )

            # --- ETIQUETAR usando el PREPROC ---
            print(f"[PROCESS {file_idx}/{len(xlsx_keys)}] (preproc) {local_xlsx_preproc}")

            df_input = pd.read_excel(local_xlsx_preproc)
            df_input = df_input.iloc[:, :3]  # indice, texto, tiempo
            calls = split_calls_by_index_reset(df_input)
            print(f"  -> Encontradas {len(calls)} llamadas en este archivo (preproc)")

            processed_master = get_processed_call_ids_from_master(master_local)
            processed_cache  = load_cache(local_xlsx_preproc)   # cache sobre el PREPROC
            already_done = processed_master.union(processed_cache)

            for i, call_df in enumerate(calls, start=1):
                call_id = f"{fname}__call{i}"   # estable por archivo original

                if call_id in already_done:
                    print(f"    - [SKIP] llamada {i}/{len(calls)} (ya procesada: {call_id})")
                    continue

                print(f"    - [RUN ] llamada {i}/{len(calls)} | call_id={call_id}")

                try:
                    df_lbl = label_call_with_gpt(call_df)

                    add_info = {
                        "call_id": call_id,
                        "source_file": fname,           # archivo original
                        "id_sponsor": str(id_sponsor),
                        "sponsor": sponsor,
                        "country": country,
                        "preproc_file": os.path.basename(local_xlsx_preproc),
                    }

                    append_to_master(df_lbl, master_path=master_local, add_info=add_info)

                    update_cache(local_xlsx_preproc, call_id)
                    already_done.add(call_id)

                except Exception as e:
                    print(f"\n[ERROR] Falló llamada {i}/{len(calls)} en archivo {file_idx}/{len(xlsx_keys)}")
                    print(f"        call_id={call_id}")
                    print(f"        motivo={e}")

                    upload_checkpoint(
                        s3_client=s3_client,
                        conn=conn,
                        bucket=bucket,
                        raw_prefix=raw_prefix,
                        sponsor_dir=sponsor_dir,
                        master_local=master_local,
                        id_sponsor=id_sponsor,
                        llamadas_totales=llamadas_totales,
                        reason="FAIL"
                    )
                    # cortamos para reintentar luego
                    return llamadas_totales

        # FIN: checkpoint final
        upload_checkpoint(
            s3_client=s3_client,
            conn=conn,
            bucket=bucket,
            raw_prefix=raw_prefix,
            sponsor_dir=sponsor_dir,
            master_local=master_local,
            id_sponsor=id_sponsor,
            llamadas_totales=llamadas_totales,
            reason="FINISH"
        )

        print("\n[FINISH] Proceso completado.")
        print("llamadas_totales (GPT):", llamadas_totales)
                # -------------------------
        # 6) DATASETPREP (enrich del master TSV)
        # -------------------------
        try:
            if dsprep is None:
                raise RuntimeError("No pude importar datasetprep.")

            out_dir = os.path.dirname(master_local)
            out_stem = os.path.splitext(os.path.basename(master_local))[0] + "_enriched"  # master_{sponsor}_enriched

            enriched_tsv, enriched_xlsx = dsprep.enrich_master_tsv(
                master_tsv_path=master_local,
                out_dir=out_dir,
                out_stem=out_stem,
                text_col="name",
                time_col="time",
                call_id_col="call_id",
                write_xlsx=True,
                write_tsv=True,
            )

            # subir enriched a S3/rawtraining
            if enriched_tsv and os.path.exists(enriched_tsv):
                key_tsv = f"{raw_prefix.rstrip('/')}/{os.path.basename(enriched_tsv)}"
                s3_upload_file(s3_client, enriched_tsv, bucket, key_tsv)
                print("[DATASETPREP] Enriched TSV subido:", s3_uri(bucket, key_tsv))

            if enriched_xlsx and os.path.exists(enriched_xlsx):
                key_xlsx = f"{raw_prefix.rstrip('/')}/{os.path.basename(enriched_xlsx)}"
                s3_upload_file(s3_client, enriched_xlsx, bucket, key_xlsx)
                print("[DATASETPREP] Enriched XLSX subido:", s3_uri(bucket, key_xlsx))

            print("[DATASETPREP] OK")

        except Exception as ee:
            print(f"[DATASETPREP][WARN] Falló datasetprep (master TSV quedó listo). Motivo: {ee}")

        return llamadas_totales
    
    
    except Exception as e:
        print(f"\n[EXCEPTION] Ocurrió un error inesperado durante el proceso: {e}")
        print("\n[STOP] Proceso interrumpido.")
        print("llamadas_totales (GPT):", llamadas_totales)
        return llamadas_totales


# ============================
# CLI
# ============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_sponsor", type=int, required=True)
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--bucket", type=str, default=BUCKET_DEFAULT)

    # NEW: preproc integration
    parser.add_argument(
        "--preproc_script",
        type=str,
        default=None,
        help="Ruta al script de preprocesamiento (default: preproc.py junto a este runner)."
    )
    parser.add_argument(
        "--preproc_args",
        type=str,
        default="",
        help="Args extra para preproc, como string. Ej: \"--no_presidio --stats\""
    )

    args = parser.parse_args()

    extra = args.preproc_args.strip().split() if args.preproc_args.strip() else None

    run_for_sponsor(
        args.id_sponsor,
        gpt_model=args.model,
        bucket=args.bucket,
        preproc_script=args.preproc_script,
        preproc_extra_args=extra,
    )
