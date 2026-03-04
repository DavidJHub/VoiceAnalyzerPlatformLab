import json
import os
import re
import librosa
import numpy as np
import pandas as pd
import unicodedata

from lang.VapLangUtils import merge_words_into_sentences


def get_data_from_name(name):
    warningg=0
    print(name)
    snippets=name.split('_')
    if len(snippets)>6 and (name.endswith('mp3') or name.endswith('wav')):
        if warningg!=0:
            print("WARNING: ALGUNOS AUDIOS TIENEN '_' EN EL PREFIJO, ESTO PUEDE CAUSAR ERRORES")
            warningg+=1
        snippets[-1].replace('-all','')
        return (snippets[-5], snippets[-4], snippets[-3], snippets[-2], snippets[-1])
    if len(snippets)==5 and (name.endswith('mp3') or name.endswith('wav')):
        snippets[-1].replace('-all','')
        return (snippets[-4], snippets[-3], snippets[-2], snippets[-1])
    if len(snippets)==6 and (name.endswith('mp3') or name.endswith('wav')):
        snippets[-1].replace('-all','')
        return (snippets[-5], snippets[-4], snippets[-3], snippets[-2], snippets[-1])
    else:
        try:
            snippets[5].replace('-all','')
            return (snippets[-5], snippets[-4], snippets[-3], snippets[-2], snippets[-1])
        except:
            snippets[3].replace('-all','')
            return (snippets[-5],snippets[-4],'0',snippets[-3],"0")
        


def jsonDecompose(route):
    """
    carga un archivo de transcripción .json
    Args:
        route (str): ruta del archivo JSON.

    Returns:
        transcript_df (Pandas DataFrame): DataFrame con los datos principales de la transcripción
        words_df      (Pandas DataFrame): DataFrame con las palabras en orden de aparición de la
                                          transcripción, el locutor dearizado y su métrica de confiabilidad
        transcript_df (Pandas DataFrame): DataFrame con las frases en orden de aparición con el
                                          locutor dearizado, la métrica de confiabilidad de locutor y
                                          el número de palabras por frase
    """
    try:
      with open(route, 'r', encoding='utf-8') as file:
          data = json.load(file)
      alternatives = data['results']['channels'][0]['alternatives'][0]
      transcript_df = pd.DataFrame([{'transcript': alternatives['transcript'], 'confidence': alternatives['confidence']}])
      words_data = alternatives.get('words', [])
      words_df = pd.DataFrame(words_data)
      phrases_df=pd.DataFrame(alternatives['paragraphs']).drop("transcript",axis=1)
      rows_paragraphs=[]
      for alternative in data['results']['channels'][0]['alternatives']:
          if 'paragraphs' in alternative and isinstance(alternative['paragraphs'], list):
              for sentence in alternative['paragraphs']:
                  row = {
                      'text': sentence['text'],
                      'start': sentence['start'],
                      'end': sentence['end'],
                      }
              rows_paragraphs.append(row)
    except Exception as e:
      print(f"Error al cargar '{route}' : {e}")
    transformed_data = []
    try:
      for _, row in phrases_df.iterrows():
          for sentence in row['paragraphs']['sentences']:
              transformed_data.append({
                  "text": sentence['text'],
                  "start": sentence['start'],
                  "end": sentence['end'],
                  "speaker": row['paragraphs']['speaker'],
                  "num_words": row['paragraphs']['num_words']
                  })
      sentences_df = pd.DataFrame(transformed_data).drop_duplicates()
    except Exception as e:
        print(f"Error al crear el dataframe de frases: {e}")
        return pd.DataFrame({"transcript": [],"confidence": []}),pd.DataFrame({"text": [],"start": [],"end": []}),pd.DataFrame({"text": [""],"start": [],"end": [],"speaker": [],"num_words": [],})
    return transcript_df,words_df,sentences_df

def rescale_centroids(df, tmo,max_time):
    """
    Rescale the centroid_start_time values to the TMO (average time call).

    Parameters:
    df (pd.DataFrame): DataFrame containing the centroid_start_time and TMO columns.
    tmo (float): The target TMO value to which the centroid_start_time should be rescaled.

    Returns:
    pd.DataFrame: DataFrame with rescaled centroid_start_time values.
    """
    total_duration = max_time
    scaling_factor = tmo / total_duration

    # Apply the scaling factor to the centroid_start_time values
    df['rescaled_centroid_start_time'] = df['centroid_start_time'] * scaling_factor

    return df


def calculate_total_audio_minutes(folder_path):
    total_seconds = 0

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)

        # Check if the current file is an audio file (you might adjust the extension check as needed)
        if file_path.lower().endswith(('.mp3', '.wav', '.flac')):
            # Load the audio file
            audio_length = librosa.get_duration(path=file_path)

            # Accumulate the total duration in seconds

            total_seconds += audio_length

    # Convert the total duration to minutes

    return total_seconds/60


def rename_prome_files(directory):
    # Ensure the directory path ends with a slash or backslash if needed
    # For example: directory = '/path/to/dir/'
    # Otherwise, you can just join paths using os.path.join.
    for filename in os.listdir(directory):
        print(f'FIXING PREFIX {filename}')
        if filename.startswith('PROME_2_'):
            print(f'FIXING PREFIX {filename}')
            old_path = os.path.join(directory, filename)
            new_filename = 'PROME2_' + filename[len('PROME_2_'):]  # Replace the prefix
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed {filename} to {new_filename}")


def insertTopicTagsJson(df, transcript_dir):
    """
    For each row in df, this function opens the corresponding transcript JSON file,
    checks whether the row's time_centroid_macs or time_centroid_prices falls
    inside any paragraph/sentence, and if it does, sets paragraph['mac'] = True
    or paragraph['price'] = True (likewise in the relevant sentence).

    The DataFrame 'df' must have columns:
      - file_name
      - time_centroid_macs
      - time_centroid_prices
    Optionally, if it has 'volume_classification' or 'velocity_classification',
    these will also be inserted into the matching sentences.

    transcript_dir is the directory containing files named like:
       <original mp3 name without extension>_transcript_paragraphs.json
    """

    # Group the rows by file_name so we only open/write each JSON once
    grouped = df.groupby('file_name', dropna=False)

    for file_name, group_rows in grouped:
        # Build the path to the JSON transcript
        json_filename = os.path.join(
            transcript_dir,
            file_name.replace('.mp3', '_transcript_paragraphs.json')
        )

        if not os.path.exists(json_filename):
            print(f"JSON file not found for {file_name}: {json_filename}")
            continue

        # Load the transcript JSON
        with open(json_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        updated = False

        # For each row corresponding to this file
        for _, row in group_rows.iterrows():
            # We read the time centroids
            mac_time = row.get('time_centroid_macs', np.nan)
            price_time = row.get('time_centroid_prices', np.nan)

            vol_class = row.get('volume_classification', None)
            vel_class = row.get('velocity_classification', None)

            # Go through the channels/paragraphs/sentences
            for channel in data.get('results', {}).get('channels', []):
                alt_list = channel.get('alternatives', [])
                if not alt_list:
                    continue

                paragraphs = alt_list[0].get('paragraphs', [])
                for paragraph in paragraphs:
                    p_start = paragraph.get('start', 0)
                    p_end = paragraph.get('end', 0)

                    # 1) If time_centroid_macs falls in the paragraph range
                    if pd.notna(mac_time) and (p_start <= mac_time <= p_end):
                        paragraph['mac'] = True
                        updated = True

                    # 2) If time_centroid_prices falls in the paragraph range
                    if pd.notna(price_time) and (p_start <= price_time <= p_end):
                        paragraph['price'] = True
                        updated = True

                    # Also update the sentences within this paragraph
                    for sentence in paragraph.get('sentences', []):
                        s_start = sentence.get('start', 0)
                        s_end = sentence.get('end', 0)

                        # If row has volume/velocity classification, add it
                        if vol_class or vel_class:
                            # Example: attach them at the sentence level
                            if vol_class:
                                sentence['vol'] = vol_class
                            if vel_class:
                                sentence['vel'] = vel_class

                        if pd.notna(mac_time) and (s_start <= mac_time <= s_end):
                            sentence['mac'] = True
                            sentence['highlight_error'] = "MAC MODULADO/MAC INCORRECTO"
                            # Uncomment to see what text gets flagged:
                            # print(f"MAC tag -> sentence: {sentence['text']}")
                            updated = True

                        if pd.notna(price_time) and (s_start <= price_time <= s_end):
                            sentence['price'] = True
                            sentence['highlight_error'] = "PRECIO MODULADO/PRECIO INCORRECTO"
                            # Uncomment to see what text gets flagged:
                            # print(f"PRICE tag -> sentence: {sentence['text']}")
                            updated = True

        # If we made any changes, write the JSON back
        if updated:
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Updated {json_filename} with Mac/Price tags.")
        else:
            print(f"No tag updates needed for {json_filename}.")



'''def insertTopicTagsJson(macs_df, prices_df, transcript_dir):
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
    process_dataframe(prices_df, 'price')'''

def merge_keep_right(df1, df2, on, how='inner'):
    # Identify the overlapping columns (excluding the 'on' key)
    overlapping_columns = df1.columns.intersection(df2.columns).tolist()
    overlapping_columns.remove(on)  

    # Drop the overlapping columns from df1
    df1_dropped = df1.drop(columns=overlapping_columns)

    # Perform the merge, keeping only the right columns where duplicates exist
    merged_df = pd.merge(df1_dropped, df2, on=on, how=how)

    return merged_df


def jsonDecomposeSentencesHighlight(file_path, output_dir, keywords_good, keywords_bad=None):
    if keywords_bad is None:
        keywords_bad = []

    def _normalize_keywords(keywords):
        norm = []
        for kw in keywords:
            if kw is None:
                continue
            try:
                kw_str = str(kw)
            except Exception:
                continue
            norm.append(kw_str)
        return norm

    keywords_good = _normalize_keywords(keywords_good)
    keywords_bad  = _normalize_keywords(keywords_bad)

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Creado directorio de salida: {output_dir}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Archivo cargado correctamente: {file_path}")

        if not isinstance(data, dict):
            raise ValueError("El contenido principal no es un diccionario.")

        metadata = data.get("metadata", {})
        channels = data.get("results", {}).get("channels", [])

        if not isinstance(channels, list):
            raise ValueError("Se espera que 'channels' sea una lista.")

        for channel_index, channel in enumerate(channels):
            alternatives = channel.get("alternatives", [])
            if not isinstance(alternatives, list):
                continue

            for alt_index, alternative in enumerate(alternatives):
                # --- IGUAL que jsonDecomposeSentences ---
                paragraphs_section = alternative.get("paragraphs", {})
                paragraphs_list = paragraphs_section.get("paragraphs", [])

                if not isinstance(paragraphs_list, list):
                    print(f"'paragraphs' no es una lista en alternativa {alt_index}, archivo: {file_path}")
                    continue

                print(f"Procesando 'paragraphs' en alternativa {alt_index} del canal {channel_index}")

                # --- Features extra: marcar keywords en cada sentence ---
                for paragraph in paragraphs_list:
                    if not isinstance(paragraph, dict):
                        continue

                    sentences = paragraph.get("sentences", [])
                    if not isinstance(sentences, list):
                        continue

                    for sentence in sentences:
                        if not isinstance(sentence, dict):
                            continue

                        text_raw = sentence.get("text", "")
                        try:
                            text = "" if text_raw is None else str(text_raw)
                        except Exception:
                            text = ""

                        found_good = [kw for kw in keywords_good if kw in text]
                        found_bad  = [kw for kw in keywords_bad  if kw in text]

                        if found_good:
                            sentence["unmissable"] = list(dict.fromkeys(found_good))
                        else:
                            # opcional: si quieres limpiar cuando no hay
                            sentence.pop("unmissable", None)

                        if found_bad:
                            sentence["forbidden"] = list(dict.fromkeys(found_bad))
                        else:
                            sentence.pop("forbidden", None)

                # --- Salida igual que jsonDecomposeSentences ---
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

                with open(output_filepath, "w", encoding="utf-8") as f:
                    json.dump(paragraphs_data, f, ensure_ascii=False, indent=4)

                print(f"Archivo guardado: {output_filepath}")

    except Exception as e:
        print(f"Error procesando el archivo {file_path}: {e}")

def getTranscriptParagraphsJson(directory):
    output_dir = os.path.join(directory, "transcript_sentences")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            paragraphs_data = jsonDecomposeSentences(file_path, output_dir)
            if paragraphs_data is not None:
                output_filepath = os.path.join(output_dir, f"{filename[:-5]}_paragraphs.json")
                with open(output_filepath, 'w', encoding='utf-8') as outfile:
                    json.dump(paragraphs_data, outfile, ensure_ascii=False, indent=4)
                print(f"Saved JSON to {output_filepath}")


def getTranscriptParagraphsJsonHighlights(directory, keywords_good, keywords_bad):
    output_dir = os.path.join(directory, "transcript_sentences")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            paragraphs_data = jsonDecomposeSentencesHighlight(file_path, output_dir, keywords_good, keywords_bad)
            if paragraphs_data is not None:
                output_filepath = os.path.join(directory, f"{filename[:-5]}_paragraphs.json")
                with open(output_filepath, 'w', encoding='utf-8') as outfile:
                    json.dump(paragraphs_data, outfile, ensure_ascii=False, indent=4)
                print(f"Saved JSON to {output_filepath}")


def sort_by_variance(df, column):
    """
    Ordena las filas de un DataFrame por la varianza de una columna específica.

    Parámetros:
    df (pandas.DataFrame): El DataFrame que contiene los datos.
    column (str): El nombre de la columna en la que se calculará la varianza.

    Retorna:
    pandas.DataFrame: Un DataFrame con todas las filas, ordenadas por la varianza en la columna especificada.
    """
    # Verificar que los valores en la columna sean listas o arrays numéricos
    df['variance'] = df[column].apply(lambda x: np.var(x) if isinstance(x, (list, np.ndarray)) else 0)

    # Ordenar el DataFrame por la varianza en orden descendente
    df_sorted = df.sort_values(by='variance', ascending=False)

    # Eliminar la columna de varianza antes de devolver el resultado
    df_sorted = df_sorted.drop(columns=['variance'])

    return df_sorted



def list_files_to_dataframe(directory_path):
    # List all files in the specified directory
    files = os.listdir(directory_path)
    # Create a DataFrame with the file names
    df = pd.DataFrame(files, columns=["file_name"])
    return df


def get_data_types(input_tuple):
    """
    Toma una tupla como entrada y devuelve una nueva tupla con los tipos de datos
    correspondientes a cada elemento de la tupla de entrada.

    :param input_tuple: Tupla con elementos de cualquier tipo.
    :return: Tupla con los tipos de datos de cada elemento en la tupla de entrada.
    """
    return tuple(type(element) for element in input_tuple)


def read_csv_with_error_handling(filename):
  df = pd.DataFrame()  # Create an empty DataFrame to store valid rows

  with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
      try:
        row = line.strip().split(',')  # Assuming CSV is comma-separated
        df = df.append(pd.Series(row), ignore_index=True)
      except UnicodeDecodeError:
        print(f"Error decoding line: {line}")
        # Optionally, log the error or handle it differently

  return df

def jsonDecomposeSentences(file_path, output_dir):
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


def jsonTranscriptionToCsv(directory: str, directoryOutput: str) -> None:
    """
    Convierte transcripciones en formato JSON (por archivo) a CSV a nivel de frases,
    enriqueciendo cada frase con las palabras que caen dentro de su ventana temporal
    y métricas promedio de confianza.

    Flujo por archivo:
      1) Lee el JSON con `jsonDecompose` -> obtiene `words_df` y `sentences_df`.
      2) Fusiona palabras dentro de cada frase con `merge_words_into_sentences`.
      3) Guarda como CSV en `directoryOutput` con el mismo nombre base del JSON.

    Manejo de errores:
      - Si el dataframe de frases no existe o no tiene la columna `text`, se crea un
        CSV de “relleno” con columnas estándar para evitar que procesos aguas abajo fallen.

    Args:
        directory (str): Carpeta donde están los archivos .json.
        directoryOutput (str): Carpeta destino donde se escribirán los .csv.

    Returns:
        None
    """
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            _, words_df, sentences_df = jsonDecompose(file_path)
            if sentences_df is not None:
                try:
                    mged=merge_words_into_sentences(words_df,sentences_df)
                    sentences_df['text']
                    excel_path = os.path.join(directoryOutput, f"{filename[:-5]}.csv")
                    mged.to_csv(excel_path, index=False,encoding='utf-8')
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
                    csv_path = os.path.join(directoryOutput, f"{filename[:-5]}.csv")
                    emp_dataframe.to_csv(csv_path, index=False)
                    #print(f"Saved {excel_path}")



def memoryJsonToDataframe(json_file_path):
    """Converts a single JSON file to a DataFrame."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        paragraphs = data.get('results', {}).get('channels', [])[0].get('alternatives', [])[0].get('paragraphs', [])
        texts, start_times, end_times, num_words, speakers = [], [], [], [], []
        if isinstance(paragraphs, dict):
            paragraphs = [paragraphs]  # Convert to list to handle uniformly
        if not isinstance(paragraphs, list):
            raise TypeError(f"Expected list or dict for paragraphs but got {type(paragraphs)} in file {json_file_path}")
        for paragraph in paragraphs:
            sentences = paragraph.get('sentences', [])
            if not isinstance(sentences, list):
                raise TypeError(f"Expected list for sentences but got {type(sentences)} in file {json_file_path}")
            for sentence in sentences:
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


def loadJsonMemory(directory_path):
    """Parses all JSON files in a directory and returns a list of DataFrames."""
    dataframes = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):  # Process only JSON files
            json_file_path = os.path.join(directory_path, filename)
            df = memoryJsonToDataframe(json_file_path)
            excel_path = os.path.join(directory_path, f"{filename[:-5]}.xlsx")
            df.to_excel(excel_path, index=False)
            #print(f"Saved {excel_path}")

    return dataframes


def _sec_to_min_sec(segundos):
    """
    Convierte segundos a un float con forma MM.SS
    Ej: 75  -> 1.15  (1 min, 15 s)
        9   -> 0.09
        125 -> 2.05
    Nota: esto NO es minutos en base 10, es solo una representación numérica.
    """
    neg = segundos < 0
    s = int(round(abs(segundos)))

    m = s // 60
    ss = s % 60

    val = m + ss / 100.0
    return -val if neg else val