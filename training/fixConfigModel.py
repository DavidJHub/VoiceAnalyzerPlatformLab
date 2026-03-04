import os, json
import pandas as pd
from transformers import BertConfig, AutoTokenizer

MODEL_DIR = r"reentreno/Colombia/Bancolombia/model/model_output_bancol_subtag_timefeats"
BASE_MODEL = "dccuchile/bert-base-spanish-wwm-cased"

# 1) Cargar mapping de labels 
#    Debe ser un TSV con columnas: id, label  (o label, id)
mapping_candidates = [
    os.path.join(MODEL_DIR, "subtag_label_mapping.tsv"),
    os.path.join(MODEL_DIR, "label_mapping.tsv"),
    os.path.join(MODEL_DIR, "combo_label_mapping.tsv"),  # por si quedó uno viejo
]

mapping_path = None
for p in mapping_candidates:
    if os.path.exists(p):
        mapping_path = p
        break

if mapping_path is None:
    raise FileNotFoundError(
        f"No encuentro ningún mapping TSV en {MODEL_DIR}. "
        f"Crea uno (id<tab>label) desde tu pipeline de training."
    )

df_map = pd.read_csv(mapping_path, sep="\t")
cols = [c.strip().lower() for c in df_map.columns]
df_map.columns = cols

df_map = pd.read_csv(mapping_path, sep="\t")
df_map.columns = [c.strip().lower() for c in df_map.columns]
cols = list(df_map.columns)

if "id" in cols and "label" in cols:
    df_map = df_map.sort_values("id")
    id2label = {int(r["id"]): str(r["label"]) for _, r in df_map.iterrows()}

elif "id" in cols and "subtag" in cols:
    # ✅ TU CASO
    df_map = df_map.sort_values("id")
    id2label = {int(r["id"]): str(r["subtag"]).strip() for _, r in df_map.iterrows()}

elif "combo_label" in cols and "id" in cols:
    df_map = df_map.sort_values("id")
    id2label = {int(r["id"]): str(r["combo_label"]) for _, r in df_map.iterrows()}

else:
    raise ValueError(f"Mapping inesperado. Columnas encontradas: {cols}")

label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# 2) Crear un config BERT válido (con model_type=bert)
config = BertConfig.from_pretrained(BASE_MODEL)
config.num_labels = num_labels
config.id2label = id2label
config.label2id = label2id

# 3) Guardar config en MODEL_DIR
config.save_pretrained(MODEL_DIR)
print("OK: config.json guardado en", MODEL_DIR)

# 4) Guardar tokenizer en MODEL_DIR (si no existe)
tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True) if any(
    os.path.exists(os.path.join(MODEL_DIR, f)) for f in ["tokenizer.json", "vocab.txt", "tokenizer_config.json"]
) else AutoTokenizer.from_pretrained(BASE_MODEL)

tok.save_pretrained(MODEL_DIR)
print("OK: tokenizer guardado en", MODEL_DIR)
