#!/usr/bin/env python
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
from nltk.stem import SnowballStemmer
import numpy as np
import re
from line_profiler import LineProfiler

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ----------------- CONFIG ----------------- #
MODEL_DIR   = "./model_output_best"
TEXT_COLUMN = "text"
WINDOW_SIZE = 16
STRIDE = 6
MAX_LENGTH = 12

# ----------------- PREPROCESSING ----------------- #
# spaCy Spanish model for lemmatization
try:
    nlp = spacy.load("es_core_news_sm")
except Exception as e:
    raise RuntimeError("Could not load 'es_core_news_sm'. Run: python -m spacy download es_core_news_sm") from e

stemmer = SnowballStemmer("spanish")

def lemmatize_text(text: str) -> str:
    """Lematizes the text using spaCy."""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


def stem_text(text: str) -> str:
    """Applies stemming to the text using SnowballStemmer."""
    words = text.lower().split()
    return " ".join([stemmer.stem(word) for word in words])

def preprocess_text(text: str) -> str:

    text_clean = text.strip()
    text_lem = lemmatize_text(text_clean)
    text_stem = stem_text(text_lem)
    return text_stem


# ----------------- LOAD MODEL & TOKENIZER ----------------- #
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
id2label = model.config.id2label


def build_word_timestamps(df_conversation, text_col="text"):
    """
    Expande la conversación a nivel de palabra. Para cada fila,
    divide el texto en palabras y reparte el rango [start, end]
    equitativamente entre esas palabras.

    Retorna:
       Una lista de tuplas (word, start_word, end_word).
    """
    word_timeline = []
    for _, row in df_conversation.iterrows():
        seg_text = str(row[text_col]).strip()
        seg_start = float(row["start"])
        seg_end = float(row["end"])
        words = seg_text.split()
        num_words = len(words)

        if num_words == 0:
            # Manejo de casos vacíos
            continue

        total_dur = seg_end - seg_start
        dur_per_word = total_dur / num_words

        current_start = seg_start
        for w in words:
            current_end = current_start + dur_per_word
            word_timeline.append((w, current_start, current_end))
            current_start = current_end

    return word_timeline

def classify_fragment(fragment: str, max_length: int = MAX_LENGTH) -> dict:
    """
    Preprocesa, tokeniza y clasifica un fragmento de texto con el modelo.
    Devuelve un dict con 'predicted_cluster' y las probabilidades para cada clase.
    """
    processed = preprocess_text(fragment)
    inputs = tokenizer(
        processed,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    predicted_id = int(np.argmax(probs))

    # Busca label según sea int o str
    if isinstance(id2label, dict):
        if predicted_id in id2label:
            predicted_label = id2label[predicted_id]
        elif str(predicted_id) in id2label:
            predicted_label = id2label[str(predicted_id)]
        else:
            raise KeyError(f"Label no encontrado para ID {predicted_id}.")
    else:
        predicted_label = id2label[predicted_id]

    result = {"predicted_cluster": predicted_label}
    for idx, prob in enumerate(probs):
        label = id2label[idx] if idx in id2label else str(idx)
        result[label] = prob
    return result

def unify_text(df: pd.DataFrame, text_col="text") -> str:
    """
    Concatena todo el texto de un DataFrame en un único string,
    ignorando saltos de línea y espacios extra.
    """
    combined_str = " ".join(str(x) for x in df[text_col].tolist())
    combined_str = combined_str.replace("\n", " ")
    combined_str = re.sub(r"\s+", " ", combined_str).strip()
    return combined_str


def classify_entire_conversation(df_conversation: pd.DataFrame,
                                 window_size: int,
                                 stride: int,
                                 max_length: int) -> pd.DataFrame:
    """
    1) Construye una lista (word, start, end) para toda la conversación.
    2) Aplica un sliding window de tamaño 'window_size' y paso 'stride'
       sobre los índices de esas palabras.
    3) Para cada ventana, usa classify_fragment() y define su start/end
       en base a la primera y última palabra de la ventana.
    """

    # (A) Expandir la conversación a nivel de palabra
    word_timeline = build_word_timestamps(df_conversation, text_col="text")
    # word_timeline es una lista: [(w1, s1, e1), (w2, s2, e2), ... ]

    if not word_timeline:
        # Si no hay palabras, retornamos algo vacío
        return pd.DataFrame(columns=["text", "start", "end", "predicted_cluster"])

    # (B) Preparar la lista de palabras para concatenar en fragmentos
    all_words = [wt[0] for wt in word_timeline]  # Solo la palabra
    total_words = len(all_words)

    fragments = []
    # Generamos las ventanas de palabras
    for i in range(0, total_words - window_size + 1, stride):
        # Obtenemos las palabras de la ventana
        window_words = all_words[i : i + window_size]

        # Determinamos el start/end a partir de la primera y última palabra
        start_window = word_timeline[i][1]                # tstart de la i-th palabra
        end_window   = word_timeline[i + window_size - 1][2]  # tend de la (i+window_size-1)-th palabra

        # Unimos las palabras para formar el texto del fragmento
        fragment_text = " ".join(window_words)

        fragments.append((fragment_text, start_window, end_window))

    # Si la conversación tiene menos palabras que 'window_size',
    # podemos añadir el fragmento completo (opcional)
    if total_words < window_size:
        fragment_text = " ".join(all_words)
        start_window = word_timeline[0][1]
        end_window   = word_timeline[-1][2]
        fragments.append((fragment_text, start_window, end_window))

    # (C) Clasificar cada fragmento
    results = []
    for fragment_text, fragment_start, fragment_end in fragments:
        classification = classify_fragment(fragment_text, max_length)
        classification["text"]  = fragment_text
        classification["start"] = fragment_start
        classification["end"]   = fragment_end
        results.append(classification)

    df_result = pd.DataFrame(results)

    # Identificar columnas de probabilidad
    prob_cols = [c for c in df_result.columns
                 if c not in ["text", "start", "end", "predicted_cluster"]]
    df_result.attrs["cluster_columns"] = prob_cols
    return df_result


def fit_csv_sliding_transformer(
    input_csv: str,
    output_csv: str,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    max_length: int = MAX_LENGTH
):
    """
    - Lee el CSV (debe contener columnas "text", "start", "end").
    - Si es una sola conversación, puedes llamar a classify_entire_conversation tal cual.
      Si son múltiples conversaciones, agrupa por algún ID de conversación.
    - Genera un CSV con los fragmentos y sus clasificaciones.
    """
    df = pd.read_csv(input_csv,encoding='utf-8')
    required = {"text", "start", "end"}
    if not required.issubset(df.columns):
        raise ValueError(f"El CSV debe contener columnas {required}.")

    # EJEMPLO 1: suponer que todo el CSV es una sola conversación
    classified_df = classify_entire_conversation(df, window_size, stride, max_length)
    classified_df["conversation_id"] = 0
    classified_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Predicciones guardadas en: {output_csv}")



import os
import shutil

def fitCSVConversations(input_folder, output_folder, window_size, stride, max_length):
    print(f'CALIFICANDO {input_folder}')
    
    base_folder = os.path.abspath(os.path.join(input_folder, os.pardir))
    isolated_folder = os.path.join(base_folder, "isolated")
    os.makedirs(isolated_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_csv = os.path.join(input_folder, filename)
            output_csv = os.path.join(output_folder, os.path.splitext(filename)[0] + ".csv")

            try:
                print(f"Processing {input_csv} -> {output_csv}")
                fit_csv_sliding_transformer(
                    input_csv=input_csv,
                    output_csv=output_csv,
                    window_size=window_size,
                    stride=stride,
                    max_length=max_length
                )
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
                # Obtener nombre base sin _transcript
                base_name = filename.replace("_transcript.csv", "")
                audio_file = os.path.join(base_folder, base_name + ".mp3")
                transcript_file = os.path.join(base_folder, filename)

                # Mover audio si existe
                if os.path.exists(audio_file):
                    shutil.move(audio_file, os.path.join(isolated_folder, os.path.basename(audio_file)))
                    print(f"Audio {audio_file} movido a {isolated_folder}")

                # Borrar transcripción si existe
                if os.path.exists(transcript_file):
                    os.remove(transcript_file)
                    print(f"Transcripción {transcript_file} eliminada")





if __name__ == "__main__":
    INPUT_FOLDER = "BANCOLSE_RAW"
    OUTPUT_FOLDER = "BANCOLSE_PREDICTED"

    #lp = LineProfiler()
    #lp.add_function(process_directory)
    #lp.enable()
    fitCSVConversations(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        max_length=MAX_LENGTH
    )
    #lp.disable()
    #lp.print_stats()