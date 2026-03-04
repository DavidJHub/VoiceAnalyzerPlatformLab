import logging
import os
import re
from datetime import datetime

import librosa
import numpy as np
from pydub import AudioSegment
from scipy.stats import entropy
from soundfile import SoundFile as AudioFile
from tqdm import tqdm


def get_latest_directories(bucket, s3_path,s3_client):
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_path, Delimiter='/')

    if 'CommonPrefixes' not in response:
        return []
    date_pattern_1 = re.compile(r'^\d{4}-\d{2}-\d{2}$')  # Formato yyyy-mm-dd
    date_pattern_2 = re.compile(r'^\d{8}$')  # Formato yyyymmdd

    directorios = []

    for obj in response['CommonPrefixes']:
        folder_name = obj['Prefix'].strip('/').split('/')[-1]
        if date_pattern_1.match(folder_name):
            # Convertir a datetime para ordenar después
            directorios.append((folder_name, datetime.strptime(folder_name, '%Y-%m-%d')))
        elif date_pattern_2.match(folder_name):
            directorios.append((folder_name, datetime.strptime(folder_name, '%Y%m%d')))
    directorios.sort(key=lambda x: x[0], reverse=True)

    rutas_completas = [f's3://{bucket}/{s3_path}{directorio[0]}/' for directorio in directorios]

    return rutas_completas

def clean_column_blank_regex(df, column_name):
    """
    Elimina todos los tipos de espacios en blanco, incluyendo tabulaciones,
    saltos de línea y espacios Unicode no visibles en la columna especificada.
    """
    # Define un patrón que incluye todos los tipos de espacios en blanco visibles y no visibles
    whitespace_pattern = r'[\s\u00A0\u2000-\u200B\u202F\u205F\u3000]+'

    df[column_name] = df[column_name].apply(
        lambda x: re.sub(whitespace_pattern, '', x) if isinstance(x, str) else x
    )
    return df

def get_campaign_parameters(campaign_path, df):
    """
    Obtiene los parámetros de la campaña especificada por el usuario.
    """
    mapping_camps = clean_column_blank_regex(df, 'path')
    mapping_camps_expanded = mapping_camps.assign(path=mapping_camps['path'].str.split(',')).explode('path')
    camp_data = mapping_camps_expanded[mapping_camps_expanded['path'] == campaign_path].reset_index()
    camp_data = clean_column_blank_regex(camp_data, 'sponsor')
    if not camp_data.empty:
        return camp_data.iloc[0]
    else:
        logging.error(f"Campaña con ID {campaign_path} no encontrada en el inventario.")
        return None
    
def measure_rms_classification_pydub(audio_file, start_time, duration=3):
    """
    Mide el RMS de un archivo de audio MP3 usando pydub y lo clasifica.
    """
    try:
        start_ms = start_time * 1000
        duration_ms = duration * 1000

        audio = AudioSegment.from_file(audio_file)[start_ms:start_ms + duration_ms]
        samples = np.array(audio.get_array_of_samples())

        if len(samples) == 0:
            raise ValueError("El audio cargado está vacío.")

        rms = np.sqrt(np.mean(np.square(samples)))   # Normalización para int16
        return rms

    except FileNotFoundError:
        print(f"Archivo no encontrado: {audio_file}")
        return None, "Error: Archivo no encontrado"
    except ValueError as ve:
        print(f"Error en el valor: {ve}")
        return None, f"Error: {ve}"
    except Exception as e:
        print(f"Error inesperado: {e}")
        return None, f"Error inesperado: {e}"


def measureDbClassification(audio_file, start_time, duration=3):
    """
    Mide el nivel de volumen en decibelios (dBFS) de un archivo de audio,
    tomando un fragmento centrado en 'start_time' con longitud 'duration'.
    Retorna el valor dBFS y una clasificación (bajo/medio/alto).

    Parámetros:
    -----------
    audio_file : str
        Ruta del archivo de audio (e.g. 'audio.mp3').
    start_time : float
        Segundo en el que se centrará el fragmento de medición.
    duration   : float, opcional
        Duración (en segundos) del fragmento a medir. Por defecto 3 segundos.

    Returns:
    --------
    (float, str)
        Un tuple (db_value, classification), donde 'db_value' es el nivel en dBFS,
        y 'classification' es la etiqueta de volumen ('bajo', 'medio' o 'alto').
    """
    try:
        half_duration = duration / 2
        start_ms = int(max((start_time - half_duration), 0) * 1000)
        end_ms = int((start_time + half_duration) * 1000)

        audio_segment = AudioSegment.from_file(audio_file)[start_ms:end_ms]
        if len(audio_segment) == 0:
            raise ValueError("El fragmento de audio cargado está vacío.")

        db_value = audio_segment.dBFS

        if db_value < -45:
            classification = "bajo"
        elif -45 <= db_value < -20:
            classification = "medio"
        else:
            classification = "alto"

        return db_value, classification

    except FileNotFoundError:
        print(f"Archivo no encontrado: {audio_file}")
        return None, "Error: Archivo no encontrado"
    except ValueError as ve:
        return None, f"Error: {ve}"
    except Exception as e:
        return None, f"Error inesperado: {e}"

def measureDbAplitude(directory, df,time_column,suffix, duration=5):
    """
    Mide el volumen (en decibelios) y la clasificación del volumen
    para cada fila de df, basándose en un archivo de audio almacenado
    en 'directory + row["file_name"]'.
    Se agregan columnas 'volume_db' y 'volume_classification'.

    Parámetros
    ----------
    directory : str
        Ruta al directorio donde se ubican los archivos de audio.
    df : pd.DataFrame
        DataFrame que contiene al menos la columna 'file_name'.
        Idealmente también otras columnas como 'start' y 'end'
        para calcular el instante de medición.
    duration : float, default=3
        Duración en segundos del fragmento que se tomará
        para medir el volumen.

    Returns
    -------
    pd.DataFrame
        El mismo DataFrame de entrada, con dos columnas nuevas:
        'volume_db' y 'volume_classification'.
    """
    mp='_'+suffix
    df['volume_db'+mp] = None
    df['volume_classification'+mp] = None

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Measuring dB Volume"):
        file_path = os.path.join(directory, row['file_name'])

        db_value, classification = measureDbClassification(
            audio_file=file_path,
            start_time=row[time_column],
            duration=duration
        )

        # Guardamos los resultados
        df.at[index, 'volume_db'+mp] = db_value
        df.at[index, 'volume_classification'+mp] = classification

    return df



def measure_volume_for_dataframe(directory, df2):
    df2['volume_rms'] = None
    df2['volume_classification'] = None

    for index, row in tqdm(df2.iterrows(), total=df2.shape[0], desc="Measuring volume"):
        file_name = directory + row['file_name']
        centroid_start_time = (row['end'] + row['start']) / 2
        rms = measure_rms_classification_pydub(file_name, centroid_start_time)
        df2.at[index, 'volume_rms'] = rms
    mean_rms = np.mean(df2['volume_rms'])
    threshold = mean_rms * 0.2

    for index, row in tqdm(df2.iterrows(), total=df2.shape[0], desc="Classifying volume"):
        if row['volume_rms'] >= threshold:
            vc = 'high'
        elif row['volume_rms'] <= threshold:
            vc = 'low'
        else:
            vc = 'mid'
        df2.at[index, 'volume_classification'] = vc

    return df2


def measure_speed_classification(average_speed):
    if average_speed < 120:
        speed_classification = 'low'
    elif 120 <= average_speed < 250:
        speed_classification = 'mid'
    else:
        speed_classification = 'high'

    return speed_classification


def kl_div(audio_path1, audio_path2, sr=8000):
    """
    Calculate the Kullback-Leibler divergence between two audio files' frequency distributions.

    Parameters:
    audio_path1 (str): Path to the first audio file.
    audio_path2 (str): Path to the second audio file.
    sr (int): Sampling rate to use for loading audio files. Default is 8000 Hz.

    Returns:
    float: The Kullback-Leibler divergence between the frequency distributions of the two audio files.
    """
    def compute_frequency_distribution(audio_path):
        with AudioFile(audio_path) as f:
            audio = f.read(dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        stft = np.abs(librosa.stft(audio ))
        freq_distribution = np.sum(stft, axis=1)
        if np.sum(freq_distribution) > 0:
            freq_distribution /= np.sum(freq_distribution)

        return freq_distribution

    def normalize_kl_divergence(p, q):
        """
        Calculate normalized Kullback-Leibler divergence between 0 and 1.
        """
        kl_div = entropy(p, q)
        max_entropy = np.log(len(p))
        normalized_kl_div = kl_div / max_entropy
        return normalized_kl_div

    p = compute_frequency_distribution(audio_path1)
    q = compute_frequency_distribution(audio_path2)
    min_length = min(len(p), len(q))
    freq_dist1 = p[:min_length]
    freq_dist2 = q[:min_length]
    return normalize_kl_divergence(freq_dist1, freq_dist2)


def calculate_folder_kld(unprocessed_dir, processed_dir):
    kld_values = []

    def is_mp3(filename):
        return filename.lower().endswith('.mp3')

    unprocessed_files = set([f for f in os.listdir(unprocessed_dir) if is_mp3(f)])
    processed_files = set([f for f in os.listdir(processed_dir) if is_mp3(f)])

    common_files = list(unprocessed_files.intersection(processed_files))
    common_files.sort(key=lambda x: os.stat(os.path.join(unprocessed_dir, x)).st_mtime, reverse=True)
    recent_common_files = common_files[:3]

    for filename in recent_common_files:
        unprocessed_path = os.path.join(unprocessed_dir, filename)
        processed_path = os.path.join(processed_dir, filename)
        kld = kl_div(unprocessed_path, processed_path)
        kld_values.append(kld)
    if kld_values:
        average_kld = np.mean(kld_values)
    else:
        average_kld = None
    return average_kld