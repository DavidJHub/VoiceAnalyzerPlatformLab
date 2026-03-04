import os
import io
import glob
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ============================
# CONFIGURACIÓN
# ============================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "No se encontró la variable OPENAI_API_KEY. "
        "Asegúrate de tener un archivo .env con OPENAI_API_KEY=tu_clave."
    )

client = OpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL = "gpt-4.1-mini"

MASTER_TSV = "ALL_LANG_DATA/Colombia/Bancolombia/master_bancol.tsv"

EXAMPLE_XLSX      = "ALL_LANG_DATA/Colombia/Bancolombia/ejemplo.xlsx"
CLASS_MATRIX_XLSX = "ALL_LANG_DATA/Colombia/Bancolombia/matriz.xlsx"

# ============================
# SUBTAGS (únicas etiquetas)
# ============================

# Conserva los subtags que ya venías usando
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
# CARGA DEL EJEMPLO Y MATRIZ
# ============================

def _norm_cols(df: pd.DataFrame):
    """Column names robustos: siempre str, strip, lower."""
    return [str(c).strip().lower() for c in df.columns]

def load_example_from_xlsx(path: str) -> str:
    """
    Lee un .xlsx de ejemplo etiquetado y lo convierte en un string lineal tipo:
    TEXTO SUBTAG TIEMPO TEXTO SUBTAG TIEMPO ...

    Estructura esperada:
    - columnas equivalentes: name/texto, subtag/cluster/etiqueta, time/tiempo
      o si no, usa las primeras 3 columnas.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de ejemplo: {path}")

    df = pd.read_excel(path)
    cols = _norm_cols(df)

    # texto
    if "name" in cols:
        text_col = df.columns[cols.index("name")]
    elif "texto" in cols:
        text_col = df.columns[cols.index("texto")]
    else:
        text_col = df.columns[0]

    # etiqueta (subtag)
    if "subtag" in cols:
        lab_col = df.columns[cols.index("subtag")]
    elif "cluster" in cols:
        lab_col = df.columns[cols.index("cluster")]
    elif "etiqueta" in cols:
        lab_col = df.columns[cols.index("etiqueta")]
    else:
        lab_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    # tiempo
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
    """
    Lee la matriz desde un .xlsx:
    columnas esperadas: name/texto y cluster/etiqueta/subtag
    """
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

    # limpiar
    df_out["name"] = df_out["name"].astype(str).str.replace("\t", " ").str.replace("\n", " ").str.strip()
    df_out["subtag"] = df_out["subtag"].astype(str).str.strip()
    df_out = df_out[
        (df_out["name"].str.lower() != "nan") &
        (df_out["subtag"].str.lower() != "nan") &
        (df_out["name"] != "") &
        (df_out["subtag"] != "")
    ].reset_index(drop=True)

    return df_out

TAGGED_EXAMPLE   = load_example_from_xlsx(EXAMPLE_XLSX)
CLASS_MATRIX_DF  = load_matrix_from_xlsx(CLASS_MATRIX_XLSX)

# ============================
# PROMPT BASE (solo SUBTAGS)
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


BASE_PROMPT = build_base_prompt(TAGGED_EXAMPLE, CLASS_MATRIX_DF)

# ============================
# FUNCIONES AUXILIARES
# ============================

def split_calls_by_index_reset(df):
    """
    Separa un DataFrame en múltiples llamadas usando reinicio de índice.
    Asume: col0=indice, col1=texto, col2=tiempo.
    """
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

def _normalize_subtag(s: str) -> str:
    s = str(s).strip().upper()
    # normalizaciones típicas (opcional)
    s = s.replace("TRATAMIENTO DE DATOS", "TRATAMIENTO DATOS")
    return s

def label_call_with_gpt(call_df):
    """
    Devuelve DataFrame con: name, subtag, time
    """
    call_text_block = build_call_text_for_prompt(call_df)

    messages = [
        {"role": "system", "content": "Eres un asistente experto en etiquetar llamadas de call center con SUBTAGS."},
        {"role": "user", "content": BASE_PROMPT + "\n\n" + call_text_block},
    ]

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0,
    )

    raw_output = response.choices[0].message.content.strip()

    # TSV sin headers: texto \t subtag \t tiempo
    tsv_buffer = io.StringIO(raw_output)
    df_labels = pd.read_csv(
        tsv_buffer,
        sep="\t",
        header=None,
        names=["texto", "subtag", "tiempo"],
        dtype=str
    )

    # Limpieza
    df_labels["texto"]  = df_labels["texto"].astype(str)
    df_labels["subtag"] = df_labels["subtag"].apply(_normalize_subtag)
    df_labels["tiempo"] = df_labels["tiempo"].astype(str).str.strip()

    # Validación dura de etiquetas (para detectar outputs raros rápido)
    bad = df_labels[~df_labels["subtag"].isin(SUBTAGS_SET)]
    if not bad.empty:
        raise ValueError(f"GPT devolvió subtags inválidos: {sorted(bad['subtag'].unique().tolist())}")

    df_labels = df_labels.rename(columns={"texto": "name", "tiempo": "time"})
    return df_labels

def append_to_master(df_new, master_path=MASTER_TSV, add_info=None):
    """
    df_new: name, subtag, time (+ extras)
    """
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

def get_processed_call_ids_from_master(master_path=MASTER_TSV):
    if not os.path.exists(master_path):
        return set()
    try:
        df = pd.read_csv(master_path, sep="\t", dtype=str)
    except Exception:
        return set()
    if "call_id" not in df.columns:
        return set()
    return set(df["call_id"].astype(str).unique())

def get_cache_path_for_xlsx(xlsx_path):
    return xlsx_path + ".cache"

def load_cache_for_file(xlsx_path):
    cache_path = get_cache_path_for_xlsx(xlsx_path)
    if not os.path.exists(cache_path):
        return set()
    out = set()
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            cid = line.strip()
            if cid:
                out.add(cid)
    return out

def update_cache_for_file(xlsx_path, call_id):
    cache_path = get_cache_path_for_xlsx(xlsx_path)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(str(call_id) + "\n")

# ============================
# PIPELINE PRINCIPAL
# ============================

def process_xlsx_file(xlsx_path, master_path=MASTER_TSV):
    """
    - Guarda incremental (por llamada)
    - Cache + maestro para reanudar
    """
    print(f"Procesando archivo: {xlsx_path}")
    df_input = pd.read_excel(xlsx_path)
    df_input = df_input.iloc[:, :3]  # indice, texto, tiempo

    calls = split_calls_by_index_reset(df_input)
    print(f"  -> Encontradas {len(calls)} llamadas en este archivo")

    processed_master = get_processed_call_ids_from_master(master_path)
    processed_cache  = load_cache_for_file(xlsx_path)
    already_done = processed_master.union(processed_cache)

    basename = os.path.basename(xlsx_path)

    for i, call_df in enumerate(calls, start=1):
        call_id = f"{basename}__call{i}"

        if call_id in already_done:
            print(f"    - Saltando llamada {i}/{len(calls)} (ya procesada: {call_id})")
            continue

        print(f"    - Etiquetando llamada {i}/{len(calls)} (call_id={call_id})...")

        try:
            df_lbl = label_call_with_gpt(call_df)
            df_lbl["call_id"] = call_id
            df_lbl["source_file"] = basename

            append_to_master(df_lbl, master_path=master_path)
            update_cache_for_file(xlsx_path, call_id)
            already_done.add(call_id)

        except Exception as e:
            print(f"    !! Error procesando llamada {i} ({call_id}): {e}")
            print("    Se detiene el procesamiento de este archivo para reanudar después desde caché.")
            break

    print(f"Finalizado procesamiento de: {xlsx_path}")

# ============================
# MAIN
# ============================

if __name__ == "__main__":
    xlsx_path = "ALL_LANG_DATA/Colombia/Bancolombia/processDump/topics_transcripts_convers_20251205.xlsx"
    process_xlsx_file(xlsx_path, master_path=MASTER_TSV)
