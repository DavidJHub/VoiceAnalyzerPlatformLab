#!/usr/bin/env python
# robust_cluster_general_training.py
import re, unicodedata, numpy as np, pandas as pd, joblib, random
from collections import Counter
from tqdm import tqdm
from unidecode import unidecode
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from Levenshtein import distance as lev

# ---------- CONFIG ----------------------------------------------------- #
TMAP_XLSX   = "TOPIC_MAPPING.xlsx"
DATA_XLSX   = "TRAIN_MATRIX_MAPPED.xlsx"
OUT_XLSX    = "TRAIN_MATRIX_PRED.xlsx"
MODEL_FILE  = "cluster_general_model.joblib"
GENERAL_L   = ["SALUDO","OFRECIMIENTO","PRECIO","PERFILAMIENTO","MAC",
               "CONFIRMACION DATOS","PREGUNTA DE REFUERZO",
               "TERMINOS LEGALES","ATENCION","DESPEDIDA"]
EMB_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED, SAMP  = 42, 400
# ----------------------------------------------------------------------- #

def clean(t:str):
    t = unidecode(str(t)).lower()
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# -- 0. Cargar archivos -------------------------------------------------- #
df = pd.read_excel(DATA_XLSX, engine="openpyxl", dtype=str).fillna("")
df.columns = df.columns.str.lower().str.strip()
df["topic_sp_norm"] = df["cluster"].apply(clean)

tmap = pd.read_excel(TMAP_XLSX, dtype=str).fillna("")
tmap.columns = tmap.columns.str.lower().str.strip()
tmap["topic_sp_norm"] = tmap["topic_sp"].apply(clean)
tmap_valid = tmap[tmap["cluster"].isin(GENERAL_L)]

# -- 1. CONSOLIDAR VERDAD ------------------------------------------------ #
truth = {}  # topic_sp_norm -> cluster_general

# a) mapping externo gana
truth.update(dict(zip(tmap_valid["topic_sp_norm"], tmap_valid["cluster"])))

# b) mayoría interna de lo que quede
remaining = set(df["topic_sp_norm"]) - set(truth.keys())
for sp in remaining:
    cg_vals = df.loc[df["topic_sp_norm"]==sp,"cluster_general"]
    cg_vals = [c for c in cg_vals if c]           # no vacíos
    if cg_vals:
        most, cnt = Counter(cg_vals).most_common(1)[0]
        if cnt / len(cg_vals) > 0.5 and most in GENERAL_L:
            truth[sp] = most

# c) aún sin verdad => quedará a predicción
print(f"Etiquetas consolidadas manualmente: {len(truth):,}")

# --- Propagar verdad consolidada -----------------
mask_empty = df["cluster_general"] == ""
df.loc[mask_empty, "cluster_general"] = (
        df.loc[mask_empty, "topic_sp_norm"].map(truth).fillna(""))

# ---------------- DEDUPLICAR CORRECTAMENTE --------------------------
# 1) columna auxiliar: 0 si YA tiene etiqueta, 1 si está vacía
df["_vac"] = df["cluster_general"].eq("")

# 2) ordenar para que primero queden los NO vacíos
df = (df.sort_values("_vac")                     # 0 antes que 1
         .drop_duplicates(subset="topic_sp_norm", keep="first")
         .drop(columns="_vac")                   # limpia
         .reset_index(drop=True))

print(f"Después de deduplicar: {len(df):,} filas únicas de topic_sp")

# ---- split train / unlabeled  ----------------------------
train_df = df[df["cluster_general"] != ""].reset_index(drop=True)
unlab_df = df[df["cluster_general"] == ""].reset_index(drop=True)

# -- 4. FEATURES -------------------------------------------------------- #
embedder = SentenceTransformer(EMB_MODEL)
def feat(name, sp):
    txt = f"{name} {sp}"
    emb = embedder.encode([txt], normalize_embeddings=True)[0]
    levs = np.array([lev(clean(txt), g.lower())/max(len(txt),len(g)) for g in GENERAL_L])
    return np.hstack([emb, levs])

def build_matrix(sub):
    feats = [feat(n, c) for n,c in zip(sub["name"], sub["cluster"])]
    return np.vstack(feats)

X_train = build_matrix(train_df)
y_train = train_df["cluster_general"].values

# -- 5. MODELO ---------------------------------------------------------- #
pipe = Pipeline([
    ("sc", StandardScaler(with_mean=False)),
    ("clf", LogisticRegression(max_iter=1000,multi_class="multinomial",
                               n_jobs=-1,random_state=SEED))
])
pipe.fit(X_train, y_train)

# -- 6. Predicción segura ---------------------------------------------- #
if len(unlab_df):
    X_unl  = build_matrix(unlab_df)
    preds  = pipe.predict(X_unl)
    df.loc[unlab_df.index,"cluster_general"] = preds
    print(f"Asignadas {len(unlab_df):,} etiquetas nuevas.")

# -- 7. Guardar --------------------------------------------------------- #
df.to_excel(OUT_XLSX, index=False, engine="openpyxl")
joblib.dump({"pipe":pipe,"embedder":embedder,"labels":GENERAL_L}, MODEL_FILE)
print("✅ OUTPUT:", OUT_XLSX, "| MODEL:", MODEL_FILE)
