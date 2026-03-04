import re
import os
import pandas as pd
import unicodedata
from nltk.corpus import stopwords
import re, unicodedata, spacy
from spacy.lang.es.stop_words import STOP_WORDS

# Carga modelo spaCy en español (elige sm, md o lg)
nlp = spacy.load("es_core_news_sm", disable=["ner", "parser", "textcat"])

# Stop‑words: quitamos las que sí aportan sentido a preguntas/negaciones
CUSTOM_STOPWORDS = STOP_WORDS - {"no", "sí", "qué", "cómo", "dónde", "cuándo", "porqué", "por qué"}

QUESTION_TOK = "__question__"
EXCLAM_TOK   = "__exclaim__"


genMapping=[['multa asistencia', 'multiasistencia'],['mucha asistencia', 'multiasistencia']]
st_es=stopwords.words('spanish')

def normalize_text(text):
    """
    Normaliza el texto:
    - Convierte a minúsculas.
    - Elimina signos de puntuación.
    - Remueve acentos.
    - Elimina números si se decide ignorarlos.
    """
    # Convertir a minúsculas
    text = str(text).lower()

    # Eliminar signos de puntuación
    text = re.sub(r'[^\w\s]', '', text)

    # Remover acentos
    text = ''.join(
        (c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    )
    text = re.sub(r'\d+', '', text)

    return text



def preprocess_text(text: str, keep_case: bool = False) -> str:
    # 1) Unicode + espacios
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 2) Sustituir URLs, e‑mails, números >4 dígitos, emojis
    subs = [
        (r"https?://\S+|www\.\S+", "__url__"),
        (r"\b[\w\.-]+?@\w+?\.\w{2,4}\b", "__email__"),
        (r"\b\d{5,}\b", "__num__"),
        (r"[\U00010000-\U0010FFFF]", "__emoji__")  # rango Unicode emojis
    ]
    for patrón, token in subs:
        text = re.sub(patrón, token, text)

    # 3) Puntuación: convertimos ? ! … a tokens; quitamos el resto
    text = text.replace("¿", QUESTION_TOK).replace("?", QUESTION_TOK)
    text = text.replace("¡", EXCLAM_TOK).replace("!", EXCLAM_TOK)
    text = re.sub(r"[«»“”\"'(){}\[\],.;:]", " ", text)  # resto de puntuación

    # 4) Tokeniza + lematiza
    doc = nlp(text)
    lemmas = []
    for token in doc:
        lemma = token.lemma_.lower() if not keep_case else token.lemma_
        if lemma not in CUSTOM_STOPWORDS and not token.is_space:
            lemmas.append(lemma)

    return " ".join(lemmas)

def get_kws(string):
    phrases=re.split(',|\n', string)
    phrases = [s.strip() for s in phrases]
    return phrases

def word_count(text):
    return len(text.split())

def merge_words_into_sentences(words_df: pd.DataFrame, sentences_df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna palabras a frases usando un merge temporal (asof) y agrega por frase.
    Requiere columnas:
      words_df: start, end, word, confidence, speaker_confidence
      sentences_df: start, end (y opcional speaker, num_words, text)
    """
    s = sentences_df.copy().reset_index(drop=True)
    s["sentence_id"] = s.index

    w = words_df.copy()
    # Orden requerido para merge_asof
    s = s.sort_values("start")
    w = w.sort_values("start")

    # Asigna cada palabra a la frase con start más cercano por debajo (última frase iniciada)
    w2 = pd.merge_asof(
        w,
        s[["sentence_id", "start", "end"]],
        on="start",
        direction="backward",
        suffixes=("", "_sentence")
    )

    # solo palabras que realmente caen dentro del rango de la frase
    w2 = w2[w2["end"] <= w2["end_sentence"]]

    # Agregaciones por frase
    agg = w2.groupby("sentence_id").agg(
        words_list=("word", list),
        words_str=("word", lambda x: " ".join(x)),
        avg_confidence=("confidence", "mean"),
        avg_speaker_confidence=("speaker_confidence", "mean"),
    ).reset_index()

    # Merge de vuelta a sentences_df
    out = s.merge(agg, on="sentence_id", how="left")

    # Rellena frases sin palabras asignadas
    out["words_list"] = out["words_list"].apply(lambda x: x if isinstance(x, list) else [])
    out["words_str"] = out["words_str"].fillna("")
    return out.drop(columns=["sentence_id"])



def correctCommonTranscriptionMistakes(text,mapping=genMapping):
    for i in range(len(mapping)):
        text.replace(mapping[i][0], mapping[i][0])
    return text



def splitLongText(df, max_words):
    """
    Divide filas “largas” de una transcripción (a nivel frase/segmento) en dos partes más cortas,
    para producir ventanas de texto más adecuadas para un modelo de clasificación.

    Contexto típico:
    - Tienes un CSV de transcripción con columnas por segmento (start/end/speaker/text/words_list/words_str...).
    - Algunos segmentos son demasiado largos (muchas palabras) y el clasificador pierde precisión,
      se diluye el tema o excede límites de tokens.
    - Esta función corta esos segmentos en 2 subsegmentos con timestamps ajustados.

    Lógica de split:
    1) Para cada fila:
       - Obtiene `words_list` (lista de palabras del segmento).
       - Si viene como string, intenta convertirlo a lista (en el código original se usa eval).
       - Si el número de palabras > `max_words`, divide en dos mitades.
    2) Punto de corte:
       - Si `words_str` contiene signos de puntuación (. ! ? , ;), busca el signo más cercano al “centro”
         (mitad del string) y corta ahí para preservar coherencia lingüística.
       - Si no hay puntuación, corta por la mitad de la lista de palabras.
    3) Tiempos:
       - Calcula `mid_time` como el punto medio entre start y end.
       - Parte 1: [start, mid_time], Parte 2: [mid_time, end]
    4) Construye nuevas filas:
       - Recalcula `num_words` por cada parte.
       - Mantiene speaker y promedios de confianza.
       - Garantiza que `words_list` sea lista y `words_str` sea un string limpio.
    5) Si el segmento no excede `max_words`, lo deja igual (pero normaliza words_list/words_str).

    Args:
        df (pd.DataFrame):
            DataFrame de transcripción. Debe incluir, idealmente:
            - 'start', 'end', 'speaker'
            - 'words_list' (lista de palabras o string representando lista)
            - 'words_str' (string con palabras concatenadas)
            - 'avg_confidence', 'avg_speaker_confidence'
            - (opcional) 'text' (se reconstruye igualmente)
        max_words (int):
            Máximo permitido de palabras por segmento. Si se supera, la fila se divide en dos.
        output_path (str):
            Parámetro presente pero actualmente NO se usa para guardar.
            (La función solo retorna el DataFrame.)

    Returns:
        pd.DataFrame:
            Un nuevo DataFrame donde las filas largas fueron reemplazadas por dos filas más cortas.

    Notas/Limitaciones importantes:
        - Solo divide en 2 partes, aunque el segmento tenga 3x o 4x max_words.
        - Si words_list viene como string, usar eval es inseguro; es preferible ast.literal_eval.
        - El punto de corte por puntuación se calcula usando el índice de caracteres del string,
          pero la división de words_list se hace por la mitad; esto puede desalinear texto vs lista.
        - mid_time se asume mitad exacta del tiempo, no basado en tiempos reales de palabras.
    """
    new_rows = []

    for _, row in df.iterrows():
        words = row['words_list']

        # Convertir a lista si viene como string
        if isinstance(words, str):
            words = eval(words)  
        try:
           len(words)
           
        except:
            print(f"ERROR CONVERTING TO LIST: {row}")
            continue
        if len(words) > max_words:
            # Buscar el signo de puntuación más cercano a la mitad
            words_str = row['words_str']
            mid_index = len(words) // 2
            punctuation_indices = [m.start() for m in re.finditer(r'[.!?,;]', words_str)]

            if punctuation_indices:
                split_point = min(punctuation_indices, key=lambda x: abs(x - len(words_str) // 2))
                part1 = words_str[:split_point + 1].strip()
                part2 = words_str[split_point + 1:].strip()
            else:
                # Si no hay signos de puntuación, dividir en la palabra más cercana a la mitad
                part1 = ' '.join(words[:mid_index])
                part2 = ' '.join(words[mid_index:])

            # Calcular los nuevos tiempos
            start, end = row['start'], row['end']
            mid_time = start + (end - start) / 2

            # Crear dos nuevas filas con listas de palabras bien formateadas
            new_rows.append({
                'text': part1,
                'start': start,
                'end': mid_time,
                'speaker': row['speaker'],
                'num_words': len(part1.split()),
                'words_list': words[:mid_index],  # Asegurar que sigue siendo una lista
                'words_str': ' '.join(words[:mid_index]),  # Convertir a string limpio
                'avg_confidence': row['avg_confidence'],
                'avg_speaker_confidence': row['avg_speaker_confidence']
            })
            new_rows.append({
                'text': part2,
                'start': mid_time,
                'end': end,
                'speaker': row['speaker'],
                'num_words': len(part2.split()),
                'words_list': words[mid_index:],  # Asegurar que sigue siendo una lista
                'words_str': ' '.join(words[mid_index:]),  # Convertir a string limpio
                'avg_confidence': row['avg_confidence'],
                'avg_speaker_confidence': row['avg_speaker_confidence']
            })
        else:
            # Mantener la fila original si no supera el límite de palabras
            row_dict = row.to_dict()
            row_dict['words_str'] = ' '.join(words)  # Convertir a string bien formado
            row_dict['words_list'] = words  # Asegurar que sigue siendo una lista
            new_rows.append(row_dict)

    # Convertir a DataFrame y guardar
    new_df = pd.DataFrame(new_rows)
    return new_df


def splitConversations(input_dir, output_dir, max_words):

    # Recorrer todos los archivos CSV en el directorio de entrada
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            df = pd.read_csv(input_path)
            df_modified = splitLongText(df, max_words)
            df_modified.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Procesado: {file_name} -> Guardado en {output_path}")



def insertTopicTagsJson(macs_df, prices_df, transcript_dir):
    # Function to process each DataFrame
    def process_dataframe(df, key):
        for index, row in df.iterrows():
            #print("ARCHIVO ACTUAL")
            #print(row)
            json_filename = os.path.join(transcript_dir, row['file_name'].replace('.mp3', '_transcript_paragraphs.json'))
            #print(json_filename)
            weighted_start_time = row['start']+0.001
            if os.path.exists(json_filename):
                #print('HERE')
                with open(json_filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                updated = False
                for channel in data.get('results', {}).get('channels', []):
                    for paragraph in channel.get('alternatives', [])[0].get('paragraphs', []):
                        # Classify and add attributes at the paragraph level
                        if paragraph['start'] < weighted_start_time < paragraph['end']:
                            paragraph[key] = True
                            updated = True
                        for sentence in paragraph.get('sentences', []):
                            if 'volume_classification' in row and 'velocity_classification' in row:
                                if row['volume_classification']=='low' or row['velocity_classification']=='high':
                                    sentence['vol'] = row['volume_classification']
                                    sentence['vel'] = row['velocity_classification']
                                #print("VOLUMEN: "+ sentence['vol'] + " VELOCIDAD: "+ sentence['vel'])
                            if sentence['start'] < weighted_start_time < sentence['end']:
                                print(f'text for {key}: ' + sentence['text'])
                                sentence[key] = True
                                updated = True
                if updated:
                    with open(json_filename, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                    print(f"Updated {json_filename} with key {key}")
                else:
                    print(f"No update needed for {json_filename} for key {key}")

    # Process both DataFrames
    process_dataframe(macs_df, 'mac')
    process_dataframe(prices_df, 'price')