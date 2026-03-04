#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import torch
import random
import spacy
from nltk.stem import SnowballStemmer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------- CONFIGURATIONS ----------------- #
MODEL_DIR = "./ALL_LANG_DATA/Colombia/Bancolombia/model_output_col_multitag"             # Directory of the previously trained model
FEEDBACK_CSV = "feedback_data.csv"         # CSV with feedback data
OUTPUT_DIR = "./model_output_feedback"     # Directory to save the updated model
SEED = 42

# ----------------- SET SEED ----------------- #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ----------------- PREPROCESSING ----------------- #
# Load spaCy Spanish model for lemmatization
try:
    nlp = spacy.load("es_core_news_sm")
except Exception as e:
    raise RuntimeError("Failed to load 'es_core_news_sm'. Run: python -m spacy download es_core_news_sm") from e

stemmer = SnowballStemmer("spanish")

def lemmatize_text(text: str) -> str:
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def stem_text(text: str) -> str:
    words = text.lower().split()
    return " ".join([stemmer.stem(word) for word in words])

def preprocess_text(text: str) -> str:
    """
    Preprocess the text: strip spaces, lemmatize, and apply stemming.
    This same preprocessing is used for training and inference.
    """
    text_clean = text.strip()
    text_lem = lemmatize_text(text_clean)
    text_stem = stem_text(text_lem)
    return text_stem

# ----------------- LOAD FEEDBACK DATA ----------------- #
def load_feedback_data(csv_file: str) -> pd.DataFrame:
    """
    Load feedback CSV. The CSV is expected to have at least:
      - 'text': the conversation or fragment.
      - 'feedback': a boolean indicating whether the model's predicted category is correct.
      - 'detected_cluster': the category detected by the model.
    """
    try:
        df = pd.read_csv(csv_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding="latin-1")
    required_columns = {'text', 'feedback', 'detected_cluster'}
    if not required_columns.issubset(set(df.columns)):
        raise ValueError(f"Feedback CSV must contain columns: {required_columns}")
    return df

feedback_df = load_feedback_data(FEEDBACK_CSV)

# ----------------- FILTER FOR POSITIVE FEEDBACK ----------------- #
# We only use samples where the feedback is True (i.e., the detected category was correct)
positive_feedback_df = feedback_df[feedback_df["feedback"] == True].copy()

# For these samples, we assume that the detected_cluster is the correct label.
positive_feedback_df.rename(columns={"detected_cluster": "cluster", "text": "name"}, inplace=True)
# Optionally, assign a sample weight (if you want to emphasize these examples)
positive_feedback_df["sample_weight"] = 1.0

# ----------------- CREATE LABEL MAPPING ----------------- #
# Load the existing mapping from the trained model configuration
# (If the model was saved with id2label in its config, we use it; otherwise, you may need to rebuild it.)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
id2label = model.config.id2label  # could be {0: "A", 1: "B", ...} or similar
# Build label2id from that mapping
label2id = {label: idx for idx, label in id2label.items()}
# Filter out only the feedback samples with clusters present in the model mapping.
positive_feedback_df = positive_feedback_df[positive_feedback_df["cluster"].isin(label2id.keys())]
# Map textual labels to numeric labels
positive_feedback_df["label"] = positive_feedback_df["cluster"].map(label2id)

# ----------------- TOKENIZATION ----------------- #

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

def tokenize_function(examples):
    processed_texts = [preprocess_text(text) for text in examples["name"]]
    tokenized = tokenizer(processed_texts,
                          truncation=True,
                          padding="max_length",
                          max_length=128)
    # Include sample weight if available
    if "sample_weight" in examples:
        tokenized["sample_weight"] = examples["sample_weight"]
    return tokenized

# Convert the positive feedback DataFrame to a Hugging Face Dataset
feedback_dataset = Dataset.from_pandas(positive_feedback_df[["name", "label", "sample_weight"]])
feedback_dataset = feedback_dataset.map(tokenize_function, batched=True)
feedback_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "sample_weight"])

# Optionally, split into training and validation subsets
train_dataset, val_dataset = feedback_dataset.train_test_split(test_size=0.1, seed=SEED).values()

# ----------------- CUSTOM LOSS FUNCTION WITH SAMPLE WEIGHTS (Standard cross-entropy in this case) ----------------- #
def custom_compute_loss(model, inputs, return_outputs=False):
    labels = inputs.get("label")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = torch.nn.CrossEntropyLoss()  # Using standard CE loss (weights could be added if desired)
    loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
    return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# ----------------- TRAINING ARGUMENTS ----------------- #
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Adjust batch size as needed
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    seed=SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    compute_loss=custom_compute_loss
)

# ----------------- FINE-TUNING ON FEEDBACK DATA ----------------- #
if __name__ == "__main__":
    trainer.train()
    eval_results = trainer.evaluate()
    print("Evaluation results on feedback data:", eval_results)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
