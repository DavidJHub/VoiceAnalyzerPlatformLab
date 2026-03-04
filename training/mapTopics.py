#!/usr/bin/env python
# build_train_matrix_mapped.py
import re, unicodedata, pandas as pd
from unidecode import unidecode

TOPIC_MAP_FILE   = "TOPIC_MAPPING.xlsx"
TRAIN_MATRIX_IN  = "ALL_LANG_DATA/TRAIN_MATRIX.xlsx"
TRAIN_MATRIX_OUT_XLSX = "TRAIN_MATRIX_MAPPED.xlsx"
TRAIN_MATRIX_OUT_CSV  = "TRAIN_MATRIX_MAPPED.csv"

# -----------------------------------------------------------------------------
# 1. Función de normalización robusta
# -----------------------------------------------------------------------------
def normalize(text: str) -> str:
    """
    • Pasa a minúsculas
    • Transforma caracteres acentuados a su base (á→a) con unidecode
    • Elimina caracteres no alfanuméricos (excepto espacios)
    • Colapsa espacios múltiples
    """
    text = str(text)
    text = text.replace("?", "")                      # quita signos “?” de mal encoding
    text = unidecode(text)                            # quita tildes/diéresis
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)          # solo letras, números y espacio
    text = re.sub(r"\s+", " ", text).strip()          # un solo espacio interior
    return text

# -----------------------------------------------------------------------------
# 2. Leer archivos
# -----------------------------------------------------------------------------
print("→ Cargando archivos…")
topic_map = pd.read_excel(TOPIC_MAP_FILE, engine="openpyxl", dtype=str).fillna("")
topic_map.columns = topic_map.columns.str.lower().str.strip()  # normaliza encabezados

train_df  = pd.read_excel(TRAIN_MATRIX_IN, engine="openpyxl", dtype=str).fillna("")
train_df.columns = train_df.columns.str.lower().str.strip()

# -----------------------------------------------------------------------------
# 3. Construir diccionario limpio topic_sp → cluster_general
# -----------------------------------------------------------------------------
topic_map["topic_sp_norm"] = topic_map["topic_sp"].apply(normalize)
topic_dict = dict(zip(topic_map["topic_sp_norm"], topic_map["cluster"].str.upper().str.strip()))

# -----------------------------------------------------------------------------
# 4. Añadir columna cluster_general a TRAIN_MATRIX
# -----------------------------------------------------------------------------
train_df["topic_sp_norm"] = train_df["cluster"].apply(normalize)  # ojo: 'cluster' aquí es topic_sp
train_df["cluster_general"] = train_df["topic_sp_norm"].map(topic_dict).fillna("")

# -----------------------------------------------------------------------------
# 5. Guardar resultados
# -----------------------------------------------------------------------------
cols_order = ["name", "cluster", "cluster_general", "s3_route", "pais", "sponsor"]
train_df[cols_order].to_excel(TRAIN_MATRIX_OUT_XLSX, index=False, engine="openpyxl")
train_df_unlabeled=train_df[train_df["cluster_general"]==""]

print(f"✅ Archivo mapeado guardado en:\n   • {TRAIN_MATRIX_OUT_XLSX}\n   • ")
