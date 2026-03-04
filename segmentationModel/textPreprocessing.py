#!/usr/bin/env python
import torch


import os
import pandas as pd
import numpy as np
import torch
import random
import spacy
from nltk.stem import SnowballStemmer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import TrainingArguments, IntervalStrategy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from datasets import Dataset
from segmentationModel.hyperTunning import optuna_hp_space

# ----------------- CONFIGURACIONES -----------------  #
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"  # Modelo preentrenado en español (BETO)
TRAIN_CSV = "ALL_LANG_DATA/etiquetas_llamadas_master_enriched.xlsx"
OUTPUT_DIR = "ALL_LANG_DATA/Colombia/Bancolombia/model_output_col_multitag"
SEED = 42
mode="FITTING"
import os
# Remove any env tokens so no Authorization header is sent
os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
os.environ.pop("HF_TOKEN", None)

# ----------------- SETEAR SEMILLA ----------------- #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ----------------- PREPROCESAMIENTO ----------------- #
# Cargar spaCy para lematización (modelo en español)

try:
    nlp = spacy.load("es_core_news_sm")
except Exception as e:
    raise RuntimeError("No se pudo cargar 'es_core_news_sm'. Ejecuta: python -m spacy download es_core_news_sm") from e

stemmer = SnowballStemmer("spanish")

def lemmatize_text(text: str) -> str:
    """Lematiza el texto usando spaCy."""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def stem_text(text: str) -> str:
    """Aplica stemming al texto utilizando SnowballStemmer."""
    words = text.lower().split()
    return " ".join([stemmer.stem(word) for word in words])

def preprocess_text(text: str) -> str:
    """
    Preprocesa el texto: elimina espacios extra, lematiza y aplica stemming.
    Este mismo preprocesamiento se aplicará tanto en entrenamiento como en inferencia.
    """
    try:
        text_clean = text.strip()
        text_lem = lemmatize_text(text_clean)
        text_stem = stem_text(text_lem)
        return text_stem
    except:
        return ""

# ----------------- CARGA DE DATOS ----------------- #
def load_data(csv_file: str) -> pd.DataFrame:
    """
    Carga los datos desde un CSV y verifica que contenga las columnas 'name' y 'cluster'.
    """
    try:
        df = pd.read_csv(csv_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding="latin-1")
    if "name" not in df.columns or "cluster" not in df.columns:
        raise ValueError("El CSV debe contener las columnas 'name' y 'cluster'.")
    return df

df = load_data(TRAIN_CSV)

# -----------------  CREACIÓN DEL MAPPING DE ETIQUETAS  ----------------- #
labels = sorted(df["cluster"].dropna().unique(), key=lambda x: str(x))
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}
df["label"] = df["cluster"].map(label2id)

# -----------------           DIVISIÓN DE DATOS         ----------------- #
train_df, val_df = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df["label"])

# ----------------- TOKENIZACIÓN (CON PREPROCESAMIENTO) ----------------- #
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=False)

def tokenize_function(examples):
    """
    Función de tokenización que aplica primero el preprocesamiento
    (lematización + stemming) a cada texto, y luego tokeniza.
    """
    # Aplicar preprocesamiento a cada texto de la columna "name"
    processed_texts = [preprocess_text(text) for text in examples["name"]]
    return tokenizer(processed_texts, truncation=True, padding="max_length", max_length=128)


train_dataset = Dataset.from_pandas(train_df[["name", "label"]])
val_dataset = Dataset.from_pandas(val_df[["name", "label"]])
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


#  -----------------  CONFIGURAR Y ENTRENAR EL MODELO  -----------------  #

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    use_auth_token=False
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
        "f1": f1_score(labels, predictions, average="weighted")
    }

if mode=="FITTING":
    training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=60,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=6e-7,
    warmup_ratio=0.1,
    weight_decay=0.01,

    eval_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.EPOCH,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,

    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=50,
    report_to="none",
    seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

if mode=="HYPERTUNING":
    training_args = TrainingArguments(
        output_dir="./optuna_runs",
        evaluation_strategy="epoch",
        save_strategy="no",          # evitamos miles de checkpoints
        logging_strategy="epoch",
        disable_tqdm=True,           # Optuna imprime su propio progreso
        fp16=True,
        report_to=["tensorboard"],   # opcional
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )




if __name__ == "__main__":
    if mode=="FITTING":
        trainer.train()
        eval_results = trainer.evaluate()
        print("Resultados de evaluación:", eval_results)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
    if mode=="HYPERTUNING":
        best_trial = trainer.hyperparameter_search(
            direction="maximize",        # buscamos mayor F1 (def. en compute_metrics)
            hp_space=optuna_hp_space,
            hp_name=lambda x: f"trial{str(x)}",
            n_trials=30,                 # o el nº que te permita tu GPU / tiempo
            backend="optuna",            # explícito para claridad
            compute_objective=lambda m: m["eval_f1"],  # métrica a optimizar
        )
        print("≈≈≈ MEJOR TRIAL ≈≈≈")
        print(best_trial)