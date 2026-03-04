import glob
import json
import os
import re
import joblib

import numpy as np
import pandas as pd
from nltk import SnowballStemmer
from scipy.ndimage import gaussian_filter1d
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from Deprecated.TfIdfMultiModel import preprocess_text_alternative, classify_new_phrase_alternative, \
    get_confidence_score_for_topic, add_classification_to_row
from lang.VapLangUtils import normalize_text, get_kws, word_count, merge_words_into_sentences

pd.set_option('display.max_columns',30)
from setup.MatrixSetup import preprocess_and_extract_keywords, stop_words, matrix_setup, remove_connectors
from utils.VapFunctions import measure_volume_for_dataframe, measure_speed_classification,measure_rms_classification_pydub
from utils.VapUtils import jsonDecompose, get_data_from_name

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoTokenizer, AutoModel
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np


def jsonDecompose_sentences(file_path, output_dir):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Creado directorio de salida: {output_dir}")

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(f"Archivo cargado correctamente: {file_path}")

        if not isinstance(data, dict):
            raise ValueError("El contenido principal no es un diccionario.")

        metadata = data.get('metadata', {})
        channels = data.get('results', {}).get('channels', [])

        if not isinstance(channels, list):
            raise ValueError("Se espera que 'channels' sea una lista.")

        for channel_index, channel in enumerate(channels):
            alternatives = channel.get('alternatives', [])
            for alt_index, alternative in enumerate(alternatives):
                paragraphs_section = alternative.get('paragraphs', {})
                paragraphs_list = paragraphs_section.get('paragraphs', [])

                if isinstance(paragraphs_list, list):
                    print(f"Procesando 'paragraphs' en alternativa {alt_index} del canal {channel_index}")

                    paragraphs_data = {
                        "metadata": metadata,
                        "results": {
                            "channels": [{
                                "alternatives": [{
                                    "paragraphs": paragraphs_list
                                }]
                            }]

                        }
                    }

                    output_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_paragraphs.json"
                    output_filepath = os.path.join(output_dir, output_filename)
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(paragraphs_data, f, ensure_ascii=False, indent=4)
                    print(f"Archivo guardado: {output_filepath}")
                else:
                    print(f"'paragraphs' no es una lista en alternativa {alt_index}, archivo: {file_path}")

    except Exception as e:
        print(f"Error procesando el archivo {file_path}: {e}")

def process_json_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            _, words_df, sentences_df = jsonDecompose(file_path)
            mged=merge_words_into_sentences(words_df,sentences_df)
            if sentences_df is not None:
                try:
                    sentences_df['text']
                    excel_path = os.path.join(directory, f"{filename[:-5]}.xlsx")
                    mged.to_excel(excel_path, index=False)
                    #print(f"Saved {excel_path}")
                except:
                    empty_data = {
                                    "text": [' ',' '],
                                    "start": [0,0],
                                    "end": [0,0],
                                    "speaker": [0,0],
                                    "num_words": [0,0],
                                    "words_list": [0,0],
                                    "words_str": [0,0],
                                    "avg_confidence": [0,0],
                                    "avg_speaker_confidence": [0,0]
                                }
                    emp_dataframe=pd.DataFrame(empty_data)
                    excel_path = os.path.join(directory, f"{filename[:-5]}.xlsx")
                    emp_dataframe.to_excel(excel_path, index=False)
                    #print(f"Saved {excel_path}")



def process_json_files_json_output(directory):
    output_dir = os.path.join(directory, "transcript_sentences")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            paragraphs_data = jsonDecompose_sentences(file_path,output_dir)
            if paragraphs_data is not None:
                output_filepath = os.path.join(output_dir, f"{filename[:-5]}_paragraphs.json")
                with open(output_filepath, 'w', encoding='utf-8') as outfile:
                    json.dump(paragraphs_data, outfile, ensure_ascii=False, indent=4)
                print(f"Saved JSON to {output_filepath}")

def json_to_dataframe_memory(json_file_path):
    """Converts a single JSON file to a DataFrame, handling encoding issues and structure."""
    try:
        # Specify UTF-8 encoding to handle non-ASCII characters
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Safely access the 'paragraphs'
        paragraphs = data.get('results', {}).get('channels', [])[0].get('alternatives', [])[0].get('paragraphs', [])

        texts, start_times, end_times, num_words, speakers = [], [], [], [], []

        # Handle case where paragraphs is a dictionary
        if isinstance(paragraphs, dict):
            paragraphs = [paragraphs]  # Convert to list to handle uniformly

        # Ensure paragraphs is a list
        if not isinstance(paragraphs, list):
            raise TypeError(f"Expected list or dict for paragraphs but got {type(paragraphs)} in file {json_file_path}")

        for paragraph in paragraphs:
            sentences = paragraph.get('sentences', [])

            # Ensure sentences is a list
            if not isinstance(sentences, list):
                raise TypeError(f"Expected list for sentences but got {type(sentences)} in file {json_file_path}")

            for sentence in sentences:
                # Ensure each sentence is a dictionary
                if not isinstance(sentence, dict):
                    raise TypeError(
                        f"Expected dictionary for sentence but got {type(sentence)} in file {json_file_path}")

                texts.append(sentence.get('text', ''))
                start_times.append(sentence.get('start', None))
                end_times.append(sentence.get('end', None))
                num_words.append(paragraph.get('num_words', 0))
                speakers.append(paragraph.get('speaker', None))

        df = pd.DataFrame({
            'text': texts,
            'start': start_times,
            'end': end_times,
            'num_words': num_words,
            'speaker': speakers
        })

        return df

    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, KeyError) as e:
        print(f"Error processing file {json_file_path}: {e}")
        return None


def parse_json_directory_memory(directory_path):
    """Parses all JSON files in a directory and returns a list of DataFrames."""
    dataframes = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):  # Process only JSON files
            json_file_path = os.path.join(directory_path, filename)
            df = json_to_dataframe_memory(json_file_path)
            excel_path = os.path.join(directory_path, f"{filename[:-5]}.xlsx")
            df.to_excel(excel_path, index=False)
            #print(f"Saved {excel_path}")

    return dataframes

def normalize_classification_score(score, min_score, max_score):
    return (score - min_score) / (max_score - min_score) if max_score > min_score else score

def calculate_bow_score(processed_text, unique_keywords):
    return sum(1 for word in processed_text if word in unique_keywords)

def calculate_ngram_score(processed_text, unique_ngrams):
    text = ' '.join(processed_text)
    return sum(text.count(ngram) for ngram in unique_ngrams)


def smooth_array(data, sigma=2):
    """Smooths the data using a Gaussian filter.

    Args:
        data (np.array): The 1D array to be smoothed.
        sigma (float): The standard deviation for the Gaussian kernel.

    Returns:
        np.array: A smoothed version of the input data.
    """
    return gaussian_filter1d(data, sigma=sigma)


def calcular_actividad_norm(json_route):
    phrases_df = pd.DataFrame(jsonDecompose(json_route)[2])
    mm = np.max(phrases_df['end'])
    phrases_df['start_n'] = phrases_df['start'] / mm
    phrases_df['end_n'] = phrases_df['end'] / mm
    phrases_df['norm_duration'] = phrases_df['end_n'] - phrases_df['start_n']
    phrases_df['speed_n'] = phrases_df['num_words'] / phrases_df['norm_duration']
    phrases_df['speed_n'] = smooth_array(phrases_df['speed_n'])

    return phrases_df[['end_n', 'speed_n']]



def process_all_json(directory_path):
    json_files = glob.glob(os.path.join(directory_path, '*.json'))
    results = []
    for json_file in json_files:
        #print(json_file)
        try:
            result = calcular_actividad_norm(json_file)
            results.append(result)
        except:
            print(f'Audio vacío {json_file}')
            continue
    return results


def aggregate_and_average(results, num_windows=10):
    combined_df = pd.concat(results)
    combined_df['time_window'] = pd.cut(combined_df['end_n'], bins=num_windows, labels=False)
    averaged_results = combined_df.groupby('time_window')['speed_n'].mean().reset_index()

    return averaged_results

def process_directory_and_average(directory_path, num_windows=10):
    results = process_all_json(directory_path)
    averaged_results = aggregate_and_average(results, num_windows=num_windows)
    return averaged_results


def preprocess_with_stemming(text: str, stemmer) -> str:
    words = text.lower().split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)

# Function to generate embeddings using BETO
def generate_embeddings(text: str,tokenizer,embedding_model) -> np.ndarray:
    """
    Generate sentence embeddings using BETO.

    Parameters:
    - text (str): Input text to embed.

    Returns:
    - np.ndarray: Embedding vector for the input text.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings


def classify_conversation_df(df: pd.DataFrame,vectorizer,tokenizer,rf_classifier,stemmer,embedding_model, window_size: int = 15, stride: int = 7) -> pd.DataFrame:
    """
    Classify a DataFrame containing columns 'text', 'start', and 'end' by using a sliding window with overlap.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the conversation.
    - window_size (int): The number of words per sliding window.
    - stride (int): The step size for the sliding window.

    Returns:
    - pd.DataFrame: Original DataFrame with added columns for confidence scores and predicted cluster.
    """
    results = []

    for _, row in df.iterrows():
        text = row['text']
        start_time = row['start']
        end_time = row['end']

        # Split text into words and generate overlapping windows
        words = text.split()
        fragments = [
            " ".join(words[i:i + window_size])
            for i in range(0, len(words) - window_size + 1, stride)
        ]

        # Process each fragment
        for i, fragment in enumerate(fragments):
            stemmed_fragment = preprocess_with_stemming(fragment,stemmer)
            tfidf_vector = vectorizer.transform([stemmed_fragment]).toarray()
            embedding = generate_embeddings(fragment,tokenizer,embedding_model).reshape(1, -1)

            # Combine TF-IDF and embeddings
            combined_features = np.hstack([tfidf_vector, embedding])

            # Get probabilities for each cluster
            probabilities = rf_classifier.predict_proba(combined_features)[0]
            predicted_cluster = rf_classifier.classes_[np.argmax(probabilities)]

            # Store results with adjusted time windows
            fragment_start = start_time + (i * stride * (end_time - start_time) / len(words))
            fragment_end = fragment_start + (window_size * (end_time - start_time) / len(words))

            results.append({
                "text": fragment,
                "start": fragment_start,
                "end": fragment_end,
                "predicted_cluster": predicted_cluster,
                **{cluster: prob for cluster, prob in zip(rf_classifier.classes_, probabilities)}
            })

    # Create a DataFrame from the results
    classified_df = pd.DataFrame(results)

    # Normalize the probability columns
    prob_columns = [col for col in classified_df.columns if col not in ["text", "start", "end", "predicted_cluster"]]
    min_values = classified_df[prob_columns].min(axis=1)
    max_values = classified_df[prob_columns].max(axis=1)
    '''    classified_df[prob_columns] = classified_df[prob_columns].apply(
        lambda x: (x - min_values) / (max_values - min_values + 1e-8), axis=0
    )'''
    # Save the cluster column names for reference
    classified_df.attrs['cluster_columns'] = prob_columns
    return classified_df





def process_conversations_simple(filepath,vectorizer,tokenizer,rf_classifier,stemmer,embedding_model):
    test = pd.read_excel(filepath)
    ## FITEAR TEMA PRINCIPAL Y SCORE COMBINADO
    classified_conversation = classify_conversation_df(test,vectorizer,tokenizer,rf_classifier,stemmer,embedding_model, window_size=15)
    cluster_columns = classified_conversation.attrs['cluster_columns']
    classified_conversation['avg_confidence']= np.mean(classified_conversation[cluster_columns])
    classified_conversation['num_words'] = classified_conversation['text'].apply(word_count)
    classified_conversation['words_p_m'] = classified_conversation['num_words'] / np.abs(classified_conversation['end'] - classified_conversation['start']) * 60
    classified_conversation['confidence_score'] = classified_conversation.apply(lambda row: row[row['predicted_cluster']], axis=1)
    classified_conversation['file_name'] = os.path.basename(filepath).split('_transcript')[0] + '.mp3'
    classified_conversation['weight'] = classified_conversation['num_words'] * classified_conversation['confidence_score']

    price_filter= classified_conversation['precio']>=np.max(classified_conversation['precio'])*0.95
    mac_filter = classified_conversation['confirmacion'] >= np.max(classified_conversation['confirmacion']) * 0.95
    classified_conversation.loc[ price_filter,['predicted_cluster'] ] =  'precio'
    classified_conversation.loc[ mac_filter, ['predicted_cluster']] = 'confirmacion'

    classified_conversation = classified_conversation.sort_values('start')

    # Aggregate the results
    aggregated = classified_conversation.groupby('predicted_cluster').agg({
        'start': 'mean',
        'end': 'mean',
        'confidence_score': 'mean',
        'words_p_m': 'mean',
        'weight': 'mean',
        'num_words': 'mean',
    }).reset_index()
    aggregated['file_name'] = os.path.basename(filepath).split('_transcript')[0] + '.mp3'
    aggregated['centroid_start_time']=(aggregated['start']+aggregated['end'])/2


    return [classified_conversation, aggregated, np.max(test['end'])]


def process_directory_conversations_with_memory(directory,vectorizer,tokenizer,rf_classifier,stemmer,embedding_model):
    dataframes = []
    process_json_files(directory)
    process_json_files_json_output(directory)
    # Process main directory files
    files = [filename for filename in os.listdir(directory) if filename.endswith('.xlsx')]
    for filename in tqdm(files, desc=f'Processing files in {directory} directory'):
        filepath = os.path.join(directory, filename)
        try:
            processed_df = process_conversations_simple(filepath,vectorizer,tokenizer,rf_classifier,stemmer,embedding_model)
            dataframes.append(processed_df)
        except Exception as e:
            print(f"Error al procesar: {e}")
    memory_dir = os.path.join(directory, 'memory')
    parse_json_directory_memory(memory_dir)
    if os.path.exists(memory_dir):
        memory_files = [filename for filename in os.listdir(memory_dir) if filename.endswith('.xlsx')]
        for filename in tqdm(memory_files, desc="Processing files in /memory/ directory"):
            filepath = os.path.join(memory_dir, filename)
            try:
                processed_df = process_conversations_simple(filepath,vectorizer,tokenizer,rf_classifier,stemmer,embedding_model)
                dataframes.append(processed_df)
            except:
                print("error with memory file: " + filename)

    return dataframes


def rescale_centroids(df, tmo,max_time):
    """
    Rescale the centroid_start_time values to the TMO (average time call).

    Parameters:
    df (pd.DataFrame): DataFrame containing the centroid_start_time and TMO columns.
    tmo (float): The target TMO value to which the centroid_start_time should be rescaled.

    Returns:
    pd.DataFrame: DataFrame with rescaled centroid_start_time values.
    """
    # Assuming the total duration of the original data is the maximum value of centroid_start_time
    total_duration = max_time
    # Calculate the scaling factor
    scaling_factor = tmo / total_duration

    # Apply the scaling factor to the centroid_start_time values
    df['rescaled_centroid_start_time'] = df['centroid_start_time'] * scaling_factor

    return df


def get_all_transcripts(directorio):
    archivos_json = [archivo for archivo in os.listdir(directorio) if archivo.endswith('.json')]
    resultados_totales = pd.DataFrame({'id': [], 'file': [], 'transcript': [], 'confidence': [], 'conversation': [], 'speaker_order': [], 'TMO': [], 'agent_participation': []})
    counter = 0

    for archivo_json in archivos_json:
        ruta_completa = os.path.join(directorio, archivo_json)
        try:
            # Suponiendo que jsonDecompose devuelve transcript_df y sentences_df como DataFrames
            transcript_df, _, sentences_df = jsonDecompose(ruta_completa)

            # Añadiendo nuevas columnas con valores inicializados
            transcript_df = transcript_df.copy()
            transcript_df['id'] = counter
            transcript_df['file'] = archivo_json.split('_transcript')[0] + '.mp3'

            # Verificando y asegurando que solo se asignen valores individuales, no listas
            transcript_df.loc[0, 'conversation'] = ' '.join(sentences_df['text'].astype(str).tolist())
            transcript_df.loc[0, 'speaker_order'] = ','.join(sentences_df['speaker'].astype(str).tolist())
            transcript_df.loc[0, 'TMO'] = np.max(sentences_df['end']) / 60 if len(sentences_df) > 0 else 0
            transcript_df.loc[0, 'agent_participation'] = np.max([
                np.sum(sentences_df[sentences_df['speaker'] == 1]['num_words']) / np.sum(sentences_df['num_words']),
                np.sum(sentences_df[sentences_df['speaker'] == 0]['num_words']) / np.sum(sentences_df['num_words'])
            ]) if 'num_words' in sentences_df.columns and np.sum(sentences_df['num_words']) > 0 else 0

            # Concatenando resultados
            resultados_totales = pd.concat([resultados_totales, transcript_df], ignore_index=True)
            counter += 1

        except Exception as e:
            print(f"Error al procesar '{archivo_json}': {e}")
            counter += 1
            continue

    return resultados_totales


def get_all_transcripts_memory(directorio):
    archivos_excel = [archivo for archivo in os.listdir(directorio) if archivo.endswith('.xlsx')]
    resultados_totales = pd.DataFrame({'id': [],'file': [],'transcript': [], 'confidence': [], 'conversation': [], 'speaker_order': [], 'TMO':[], 'agent_participation':[] })
    counter=0
    for archivo_xlsx in archivos_excel:
        ruta_completa = os.path.join(directorio, archivo_xlsx)
        #print(ruta_completa)
        try:
            transcript_df = pd.read_excel(ruta_completa)  # Usando el tercer DataFrame retornado
            array_transcript_df_id=counter
            array_transcript_df_file=archivo_xlsx.split('_transcript')[0]+('.mp3')
            #print(transcript_df)
            #resultados = contar_palabras_prohibidas(sentences_df, speaker_id, palabras_prohibidas)
            #print(sentences_df['text'].array)
            array_transcript_df_conv=transcript_df['text'].array
            array_transcript_df_speaker_order=transcript_df['speaker'].array
            array_transcript_df_TMO=np.max(transcript_df['end'])/60
            array_transcript_df_AGENT_PART=np.max([ np.sum( transcript_df[transcript_df['speaker']==1]['num_words'])/np.sum(transcript_df['num_words']),
                                                            np.sum( transcript_df[transcript_df['speaker']==0]['num_words'])/np.sum(transcript_df['num_words'])])
            #print(array_transcript_df_id,array_transcript_df_file,array_transcript_df_conv,
            #      array_transcript_df_speaker_order,array_transcript_df_TMO,array_transcript_df_AGENT_PART)
            new_df=pd.DataFrame({'id': [counter],'file': [array_transcript_df_file],'transcript': [' '.join(array_transcript_df_conv)],
                                 'confidence': [0.95], 'conversation': [array_transcript_df_conv], 'speaker_order': [array_transcript_df_speaker_order],
                                 'TMO': [array_transcript_df_TMO], 'agent_participation': [array_transcript_df_AGENT_PART] })
            resultados_totales=pd.concat([resultados_totales, new_df], ignore_index=True)
            resultados_totales['archivo'] = archivo_xlsx
            counter+=1
            #print(f"Procesado: {archivo_json}")
        except Exception as e:
            print(f"Error al procesar '{archivo_xlsx}': {e}")
            counter+=1
            continue

    return resultados_totales


def calculate_general_statistics(dfs, tmo,max_times):
    """
    Calculate the statistics for each topic across multiple DataFrames.

    Parameters:
    dfs (list of pd.DataFrame): List of DataFrames containing the centroid_start_time and TMO columns.
    tmo (float): The target TMO value to which the centroid_start_time should be rescaled.

    Returns:
    pd.DataFrame: DataFrame with the statistics for each topic.
    """
    all_rescaled_dfs = []

    for i in range(len(dfs)):
        rescaled_df = rescale_centroids(dfs[i], tmo,max_times[i])
        all_rescaled_dfs.append(rescaled_df)

    # Combine all rescaled DataFrames
    combined_df = pd.concat(all_rescaled_dfs, ignore_index=True)

    # Calculate statistics for each topic
    #print(str(combined_df))
    statistics = combined_df.groupby('predicted_cluster').agg(
        mean_rescaled_centroid=('rescaled_centroid_start_time', 'mean'),
        std_rescaled_centroid =('rescaled_centroid_start_time', 'std'),
        min_rescaled_centroid =('rescaled_centroid_start_time', 'min'),
        max_rescaled_centroid=('rescaled_centroid_start_time', 'max'),
        section_score         =('confidence_score','mean'),
        topic_count = ('predicted_cluster', 'size')
    ).reset_index()

    return statistics


def count_keywords_in_text(keywords, target_text):
    normalized_text = normalize_text(target_text)
    normalized_text = remove_connectors(normalized_text)

    count = 0
    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)
        normalized_keyword = remove_connectors(normalized_keyword)
        if normalized_keyword in normalized_text:
            count += 1
    return count




def set_word_scoring(calls_df,topics_df):
    # Create lists of permitted and non-permitted words
    if 'KEYWORDS INFALTABLES' in topics_df.columns:
        topics_df['kws_array'] = topics_df['KEYWORDS INFALTABLES'].apply(get_kws)
        topics_df['kws_na'] = topics_df.dropna(subset=['KEYWORDS NO PERMITIDAS'])[
            'KEYWORDS NO PERMITIDAS'].apply(get_kws)
        topics_df['kws_sales'] = topics_df.dropna(subset=['KEYWORDS DE RECOMENDACIÓN'])[
            'KEYWORDS DE RECOMENDACIÓN'].apply(get_kws)

        all_must_keywords_series = topics_df['kws_array'].explode()
        all_musnt_keywords_series = topics_df['kws_na'].explode()
        all_must_keywords = list(set(all_must_keywords_series.dropna()))
        all_musnt_keywords = list(set(all_musnt_keywords_series.dropna()))
    else:
        all_must_keywords = topics_df[topics_df['permitida'] == 'Sí']['unique_top_keywords'].explode()
        all_musnt_keywords = topics_df[topics_df['permitida'] == 'No']['unique_top_keywords'].explode()
        all_must_keywords = list(set(all_must_keywords.dropna()))
        all_musnt_keywords = list(set(all_musnt_keywords.dropna()))
    calls_df['normalized'] = calls_df['transcript'].apply(normalize_text)
    calls_df['normalized'] = calls_df['normalized'].apply(remove_connectors)
    calls_df['count_must_have'] = calls_df['normalized'].apply(lambda x: count_keywords_in_text(all_must_keywords, x))
    calls_df['count_forbidden'] = calls_df['normalized'].apply(lambda x: count_keywords_in_text(all_musnt_keywords, x))
    calls_df['must_have_rate'] = calls_df['count_must_have'] / len(all_must_keywords) * 100
    if len(all_musnt_keywords) != 0:
        calls_df['forbidden_rate'] = calls_df['count_forbidden'] / (len(all_musnt_keywords)) * 100
    else:
        calls_df['forbidden_rate'] = 0
    return calls_df

def list_y_n_words(df):
    df['permitida'] = df['cluster'].apply(
        lambda x: 'No' if (x == 'no permitida' or x == 'nopermitida') else 'Sí')
    #print(df)
    if 'KEYWORDS INFALTABLES' in df.columns:
        df['kws_array'] = df['KEYWORDS INFALTABLES'].apply(get_kws)
        df['kws_na'] = df.dropna(subset=['KEYWORDS NO PERMITIDAS'])[
            'KEYWORDS NO PERMITIDAS'].apply(get_kws)
        all_must_keywords_series  = df['kws_array'].explode()
        all_musnt_keywords_series = df['kws_na'].explode()
        all_must_keywords = list(set(all_must_keywords_series.dropna()))
        all_musnt_keywords = list(set(all_musnt_keywords_series.dropna()))
    else:
        all_must_keywords = df[df['permitida'] == 'Sí']['name'].explode()
        all_musnt_keywords = df[df['permitida'] == 'No']['name'].explode()
        all_must_keywords = list(set(all_must_keywords.dropna()))
        all_musnt_keywords = list(set(all_musnt_keywords.dropna()))
    return all_must_keywords,all_musnt_keywords

def filter_dataframe_by_directory(dataframe, directory, file_column='file_name'):
    """
    Filters a DataFrame to include only rows where the filenames in the 'file_column'
    exist in the specified directory.

    Args:
    dataframe (pd.DataFrame): The DataFrame to filter.
    directory (str): The directory to check the files against.
    file_column (str): The column name in the DataFrame that contains the filenames. Default is 'file_name'.

    Returns:
    pd.DataFrame: A filtered DataFrame with rows where the 'file_column' value matches a file in the directory.
    """
    # Get the list of files in the directory
    directory_files = set(os.listdir(directory))
    print(directory_files)
    # Filter the dataframe based on whether the filename in 'file_column' exists in the directory
    print(dataframe[file_column])
    filtered_df = dataframe[dataframe[file_column].apply(lambda x: x in directory_files)]

    return filtered_df




def score_camp(campaign_id,campaign_directory,TMO,topics_combined_df,vectorizer_ngram,knn_preprocessed_alternative):
    model_name = 'dccuchile/bert-base-spanish-wwm-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding_model = AutoModel.from_pretrained(model_name)

    stemmer = SnowballStemmer("spanish")

    keywords_df = topics_combined_df

    # Create TF-IDF features and embeddings
    vectorizer = TfidfVectorizer()
    keywords_df['stemmed_name'] = keywords_df['name'].apply(lambda x: preprocess_with_stemming(x, stemmer))
    tfidf_features = vectorizer.fit_transform(keywords_df['stemmed_name']).toarray()
    keywords_df['embedding'] = keywords_df['name'].apply(lambda x: generate_embeddings(x,tokenizer,embedding_model))

    X_train = np.hstack([tfidf_features, np.vstack(keywords_df['embedding'].values)])
    y_train = keywords_df['cluster'].values

    cache_rf = f'{campaign_directory}/rf_classifier_{campaign_id}.joblib'

    if os.path.exists(cache_rf):
        print("Cargando resultados desde caché...")
        rf_classifier = joblib.load(cache_rf)
    else:
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_classifier.fit(X_train, y_train)

        joblib.dump(rf_classifier, f'{campaign_directory}/rf_classifier_{campaign_id}.joblib')


    # Definir el nombre del archivo de caché
    cache_file = f"{campaign_directory}/misc/cache_scores_{campaign_id}.pkl"

    # Verificar si el caché ya existe
    if os.path.exists(cache_file):
        print("Cargando resultados desde caché...")
        scores = joblib.load(cache_file)
    else:
        print("Procesando conversaciones y generando caché...")
        scores = process_directory_conversations_with_memory(campaign_directory,vectorizer,tokenizer,rf_classifier,stemmer,embedding_model)
        # Guardar los resultados en caché
        joblib.dump(scores, cache_file)
    print("TOTAL DE LLAMADAS A CALIFICAR: "+ str(len(scores)))
    statistics = calculate_general_statistics([scores[i][1] for i in range(len(scores))],TMO,[scores[i][2] for i in range(len(scores))])
    statistics.to_excel(campaign_directory + 'misc/statistics_pre.xlsx')
    MAT_a = get_all_transcripts(campaign_directory)
    #print("LLAMADAS CAL:" + str(MAT_a))
    MAT_M = get_all_transcripts_memory(campaign_directory + 'memory/')
    #print("LLAMADAS MEM:" + str(MAT_M))

    for df in [MAT_a, MAT_M]:
        df['price_score'] = 0
        df['mac_score'] = 0
        df['price_score_norm'] = 0
        df['mac_score_norm'] = 0
    MAT = pd.concat([MAT_a, MAT_M], axis=0)
    MAT.to_excel(campaign_directory + "misc/" + 'MAT.xlsx')
    #print("LLAMADAS TOT: " + str(MAT))
    # Create a new column to indicate if the word is permitted or not
    PERM,NOPERM=list_y_n_words(topics_combined_df)
    MAT['normalized'] = MAT['transcript'].apply(normalize_text)
    MAT['normalized'] = MAT['normalized'].apply(remove_connectors)
    MAT['count_must_have'] = MAT['normalized'].apply(lambda x: count_keywords_in_text(PERM, x))
    MAT['count_forbidden'] = MAT['normalized'].apply(lambda x: count_keywords_in_text(NOPERM, x))
    MAT['must_have_rate'] = MAT['count_must_have'] / len(PERM) * 100
    if len(NOPERM) != 0:
        MAT['forbidden_rate'] = MAT['count_forbidden'] / (len(NOPERM)) * 100
    else:
        MAT['forbidden_rate'] = 0
    # MAT['sales_rate']    =MAT['count_sales']/len(all_sales_keywords)*100

    MAT['score'] = np.sqrt(np.abs(
        (MAT['count_must_have'] - MAT['count_forbidden']) / (len(PERM) + len(NOPERM)))) * 10
    MAT['score'] = np.maximum(0, MAT['score'] + (MAT['TMO'] ** (2) - np.mean(MAT['TMO']) ** (2)) / (
                np.mean(MAT['TMO']) ** (3)))

    MAT[['DATE_TIME', 'CALL_ID', 'CLIENT_ID', 'LEAD_ID', 'PHONE']] = MAT['file'].apply(lambda x: pd.Series(get_data_from_name(x)))
    print("TOTAL DE LLAMADAS CALIFICADAS: " + str(len(scores)))
    topics_stats_convs = [scores[i][1] for i in range(len(scores))]
    #print("STATS DE CONVERSACIONES: "+ str(topics_stats_convs))
    #print("FRAGMENTOS DE TEMAS " + str(len(topics_stats_convs)))
    #print("FIRST  " + str(scores[0][1]))
    #print("LAST  " + str(scores[-1][1]))
    topics_stats_convs = pd.concat(topics_stats_convs, ignore_index=True)
    #print("VERIFICANDO: " + topics_stats_convs['file_name'][len(topics_stats_convs)-1])
    #print("VERIFICANDO: " + topics_stats_convs['file_name'][0])
    #print("FRAGMENTOS DE TEMAS " + str(len(topics_stats_convs)))
    topics_transcripts_convers = [scores[i][0] for i in range(len(scores))]
    topics_transcripts_convers = pd.concat(topics_transcripts_convers, ignore_index=False)


    #print("FRAGMENTOS DE TEMAS   DESPUES DE MEDIR VOLUMEN " + str(len(topics_stats_convs)))
    topics_stats_convs['velocity_classification'] = topics_stats_convs['words_p_m'].apply(measure_speed_classification)

    topics_stats_convs_scores = topics_stats_convs.groupby('file_name', as_index=False).agg({
        'confidence_score': 'mean'
    }).reset_index()

    MAT_complete = pd.merge(topics_stats_convs_scores, MAT, left_on='file_name', right_on='file', how='inner')

    MAT_complete.to_excel(campaign_directory + "misc/" + 'MAT_complete.xlsx')
    MAT_complete['confidence_score']=MAT_complete['confidence_score']**2
    max_score_price = np.max(topics_transcripts_convers['precio'])
    max_score_mac = np.max(topics_transcripts_convers['confirmacion'])
    for i in range(len(MAT_complete)):
        try:
            MAT_complete.loc[i,"price_score"] = np.max(scores[i][0]['precio'])
            MAT_complete.loc[i, "price_score_norm"] = np.max(scores[i][0]['precio']) / max_score_price
        except Exception as e:
            print('NO PRICE DETECTED')
            MAT_complete.loc[i, "price_score"] = 0
            MAT_complete.loc[i, "price_score_norm"] = 0
        try:
            MAT_complete.loc[i,"mac_score"] = np.max(scores[i][0]['confirmacion'])
            MAT_complete.loc[i, "mac_score_norm"] = np.max(scores[i][0]['confirmacion']) / max_score_mac
        except Exception as e:
            print('NO MAC DETECTED')
            MAT_complete.loc[i, "mac_score"] = 0
            MAT_complete.loc[i, "mac_score_norm"] = 0

    def normalizar_score(x, max_score=10):
        if x >= percentil_99:
            return max_score
        else:
            return (x / percentil_99) * max_score
    percentil_99 = MAT_complete['score'].quantile(0.99)

    MAT_complete['score'] = MAT_complete['score'].apply(normalizar_score)
    #print("MATRIZ COMPLETA")
    #print(MAT_complete)
    MAT_complete['price_score_norm']=MAT_complete['price_score_norm']**(1/2)
    MAT_complete['mac_score_norm']=MAT_complete['mac_score_norm']**(1/2)
    #print('CALIFICADAS: ' + str(MAT_complete))
    MAT_complete['score'] = MAT_complete['score']*MAT_complete['price_score_norm']**(1/2)

    MAT_complete['score'] = MAT_complete['score']*MAT_complete['mac_score_norm']**(1/2)
    MAT_complete['pen_factor']=1+(0.9-MAT_complete['agent_participation'])

    MAT_complete['score'] = MAT_complete['score']*MAT_complete['pen_factor']**(1/6)*(MAT_complete['confidence_score']/np.max(MAT_complete['confidence_score']))**(1/8)
    MAT_complete = MAT_complete.drop_duplicates(subset=['file_name'])
    #print('DROP: ' + str(MAT_complete))
    #print(MAT_complete['file_name'])
    MAT_complete.to_excel(campaign_directory + "misc/" + 'MAT_REGRADED.xlsx')
    MAT_CALLS_THIS_CAMPAIGN=MAT_complete
    #print('FINALES: ' + str(MAT_CALLS_THIS_CAMPAIGN))
    statistics['mean_rescaled_2_centroid'] = statistics['min_rescaled_centroid'] / np.max(
        statistics['max_rescaled_centroid'])
    MAT_CALLS_THIS_CAMPAIGN['LEAD_ID'] = MAT_CALLS_THIS_CAMPAIGN['LEAD_ID'].fillna("-1")
    MAT_CALLS_THIS_CAMPAIGN['LEAD_ID'] = MAT_CALLS_THIS_CAMPAIGN['LEAD_ID'].astype(str)

    MAT_complete_topics = pd.merge(topics_transcripts_convers, MAT_CALLS_THIS_CAMPAIGN[['id','file','transcript']], left_on='file_name',
                                   right_on='file', how='right')

    return MAT_CALLS_THIS_CAMPAIGN,statistics,MAT_complete_topics,NOPERM,topics_stats_convs_scores,topics_stats_convs

