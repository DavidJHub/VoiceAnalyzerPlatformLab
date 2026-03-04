path_campaña = ('3334')


import os
from itertools import cycle
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import scipy
import time
import psutil
import glob
from collections import Counter
import ast
from datetime import datetime
from multiprocessing import Pool


# AUDIO PROCESSING

import librosa
from pedalboard.io import AudioFile
from pedalboard import *
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment
import scipy.signal as signal
import pyloudnorm as pyln
from pedalboard import Pedalboard, Gain, Limiter, Compressor, NoiseGate, LowShelfFilter
from scipy.io.wavfile import write, read
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter, spectrogram, fftconvolve
from scipy import signal
from numpy.fft import fft, ifft,rfft, fftfreq
#import pyflac


#VISUALIZATION



# TOPIC DETECTION AND REGEX

import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import string
import re
from nltk.corpus import stopwords
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# DATA OUTPUT AND DB CONNECTION

from openai import OpenAI
import random

import boto3
import tempfile
import mysql.connector
import logging
from deepgram import DeepgramClient, PrerecordedOptions
from httpx import Timeout
from decimal import Decimal


# ANALYSIS

from sklearn.preprocessing import MinMaxScaler
from nltk.stem import SnowballStemmer
import Levenshtein
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy


from deepgram import DeepgramClient, PrerecordedOptions

DEEPGRAM_API_KEY = '750f1b6741886c1c2e2194946a5f2676bd42db4f'


import nest_asyncio
nest_asyncio.apply()

nltk.download('punkt')
nltk.download('stopwords')

st_es=stopwords.words('spanish')
# Load the Spanish language model
#nlp = spacy.load('es_core_news_sm')




stop_words = set(stopwords.words('spanish'))

stemmer = SnowballStemmer("spanish")

common_spanish_stopwords = list({
    'de', 'la', 'en', 'con', 'el', 'y', 'a', 'que', 'los', 'del',
    'se', 'por', 'las', 'un', 'para', 'o', 'es', 'una', 'al','si'
})+list(stop_words)


import mysql.connector
import inspect

print("Module imported successfully")
print(inspect.getfile(mysql.connector))


def connection_mysql(nombrebd, user, pwd, host):
    # Configurar el logger
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)

    try:
        # Establecer la conexión con la base de datos
        conexion = mysql.connector.connect(
            host=host,
            user=user,
            password=pwd,
            database=nombrebd
        )

        # Crear un cursor para ejecutar consultas
        cursor = conexion.cursor()

        return conexion, cursor
    except mysql.connector.Error as err:
        logger.error(f"Error al realizar la conexión con la base de datos: {err}")
        return None, None

# Detalles de conexión
nombrebd = "aihub_bd"
usuario = "admindb"
contraseña = "VAPigs2024.*"
host = "vapdb.cjq4ek6ygqif.us-east-1.rds.amazonaws.com"

PORT = 3306
DATABASE = "aihub_bd"
USERNAME = "admindb"
PASSWORD = "VAPigs2024.*"
HOST = "vapdb.cjq4ek6ygqif.us-east-1.rds.amazonaws.com"


# Conectar a la base de datos
conexion, cursor = connection_mysql(nombrebd, usuario, contraseña, host)

query = f"""
    SELECT *
    FROM marketing_campaigns
"""

if cursor:
    cursor.execute(query)
    column_names = [i[0] for i in cursor.description]
    data = cursor.fetchall()
    df_inventory = pd.DataFrame(data, columns=column_names)

    # Cerrar cursor y conexión
    cursor.close()
    conexion.close()
else:
    print("No se pudo establecer la conexión con la base de datos.")


def conectar():
    return mysql.connector.connect(
        host=HOST,
        port=PORT,
        user=USERNAME,
        passwd=PASSWORD,
        db=DATABASE
    )

def get_path_from_campaign(campaign, df):
    #print(campaign)
    row = df[df['path'] == campaign]
    #print('TROUBLESHOOT:'+str(row))
    if not row.empty:
        route = row.iloc[0]['s3']
        print(f'Downloading files from {route}')
        route = route.split('s3iahub.igs/')[1]
        return route, row.iloc[0]['path'], row.iloc[0]['sponsor']
    else:
        #raise ValueError("Campaña no encontrada")
        return None,None,None



def clean_column_blank_regex(df, column_name):
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', '', x) if isinstance(x, str) else x)
    return df


def download_files_with_prefix(campaign, df, s3_client, bucket_name, local_output_folder):
    try:
        s3_path, prefix, sponsor = get_path_from_campaign(campaign, df)
        print(f'Calculated S3 path: {s3_path}')
        print(f'Prefix: {prefix}')

        objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_path)
        print(f'S3 objects: {objects}')
        if not os.path.exists(local_output_folder):
            os.makedirs(local_output_folder)

        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        audio_files = [obj['Key'] for obj in objects.get('Contents', []) if obj['Key'].endswith('.mp3') and prefix in obj['Key']]
        print(f'Audio files to download: {audio_files}')

        for audio_file in audio_files:
            local_file_path = os.path.join(local_output_folder, os.path.basename(audio_file))
            s3_client.download_file(bucket_name, audio_file, local_file_path)
            print(f"Archivo descargado: {local_file_path}")
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f'Error downloading files: {e}')


def download_files_from_s3(campaign, bucket_name, prefix, download_path, prefix_camp_param=None, sponsor_param=None):
    s3 = boto3.client('s3',
                      aws_access_key_id='AKIA47CRVCSZH6PHSC52',
                      aws_secret_access_key='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',
                      region_name='us-east-1')
    bucket_name = bucket_name
    s3_path, prefix_camp, sponsor = get_path_from_campaign(campaign, mapping_camps)

    if prefix_camp is None:
        prefix_camp = prefix_camp_param

    print('PREFIX:', prefix_camp)
    print('SPONSOR:', sponsor)

    # Crear el directorio de descarga si no existe
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            size = obj['Size']
            if key.endswith('.mp3') and prefix_camp in key and size > 100 * 1024:
                file_name = os.path.basename(key)
                file_path = os.path.join(download_path, file_name)
                print(f"Downloading {key} to {file_path}")
                s3.download_file(bucket_name, key, file_path)


def download_afectadas_from_s3(campaign, bucket_name, prefix, download_path, prefix_camp_param=None, sponsor_param=None):
    s3 = boto3.client('s3',
                      aws_access_key_id='AKIA47CRVCSZH6PHSC52',
                      aws_secret_access_key='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',
                      region_name='us-east-1')
    bucket_name = bucket_name
    s3_path, prefix_camp, sponsor = get_path_from_campaign(campaign, mapping_camps)

    if prefix_camp is None:
        prefix_camp = prefix_camp_param

    print('PREFIX:', prefix_camp)
    print('SPONSOR:', sponsor)

    # Crear el directorio de descarga si no existe
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            size = obj['Size']
            if key.endswith('.mp3') and prefix_camp in key and size < 150 * 1024:
                file_name = os.path.basename(key)
                file_path = os.path.join(download_path, file_name)
                print(f"Downloading {key} to {file_path}")
                s3.download_file(bucket_name, key, file_path)



def get_last_modified_directory(bucket_name, prefix):
    # Initialize a session using Amazon S3
    s3 = boto3.client('s3',
                      aws_access_key_id='AKIA47CRVCSZH6PHSC52',
                      aws_secret_access_key='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',
                      region_name='us-east-1')

    # Ensure the prefix ends with a '/'
    if not prefix.endswith('/'):
        prefix += '/'

    last_modified_dir = prefix
    while True:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=last_modified_dir, Delimiter='/')

        directories = response.get('CommonPrefixes', [])

        if not directories:
            print(f"No more directories found. Last directory: {last_modified_dir}")
            return f"s3://{bucket_name}/{last_modified_dir}"

        last_modified_time = datetime.min.replace(tzinfo=timezone.utc)  # Make datetime.min offset-aware
        next_last_modified_dir = None

        for directory in directories:
            dir_prefix = directory['Prefix']

            dir_response = s3.list_objects_v2(Bucket=bucket_name, Prefix=dir_prefix)
            if 'Contents' in dir_response:
                #print(f"Contents of {dir_prefix}: {dir_response['Contents']}")

                dir_last_modified = max(obj['LastModified'] for obj in dir_response['Contents'])

                if dir_last_modified > last_modified_time:
                    last_modified_time = dir_last_modified
                    next_last_modified_dir = dir_prefix

        if next_last_modified_dir is None:
            print(f"No more directories with contents found. Last directory: {last_modified_dir}")
            return f"s3://{bucket_name}/{last_modified_dir}"

        last_modified_dir = next_last_modified_dir





def get_latest_directory(bucket_name, prefix):
    # Initialize a session using Amazon S3
    s3 = boto3.client('s3',
                      aws_access_key_id='AKIA47CRVCSZH6PHSC52',
                      aws_secret_access_key='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',
                      region_name='us-east-1')

    # Ensure the prefix ends with a '/'
    if not prefix.endswith('/'):
        prefix += '/'

    # List the objects with the specified prefix, using Delimiter to only fetch directories
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')

    directories = response.get('CommonPrefixes', [])

    if not directories:
        print(f"No directories found under the prefix: {prefix}")
        return None

    # Extract the dates from directory names and find the latest one
    latest_dir = None
    latest_date = None

    for directory in directories:
        dir_prefix = directory['Prefix'].rstrip('/')

        # Extract the date part (assuming the directory format is YYYYMMDD/)
        try:
            dir_date = datetime.strptime(dir_prefix.split('/')[-1], '%Y%m%d')
        except ValueError:
            print(f"Skipping non-date directory: {dir_prefix}")
            continue

        if latest_date is None or dir_date > latest_date:
            latest_date = dir_date
            latest_dir = dir_prefix

    if latest_dir:
        return f"s3://{bucket_name}/{latest_dir}/"
    else:
        print(f"No valid date directories found under the prefix: {prefix}")
        return None



def list_all_elements(bucket_name, prefix):
    s3 = boto3.client('s3',
                      aws_access_key_id='AKIA47CRVCSZH6PHSC52',
                      aws_secret_access_key='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',
                      region_name='us-east-1')

    # Initialize paginator to handle large number of objects
    paginator = s3.get_paginator('list_objects_v2')

    # Create a dictionary to store files and directories
    elements = {
        'files': [],
        'directories': []
    }

    # Iterate through all the pages of results
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        # Check if 'Contents' is in the response
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                # Check if the key represents a directory
                if key.endswith('/'):
                    elements['directories'].append(key)
                else:
                    elements['files'].append(key)

    return elements


def match_csv(campaign_name, csv_files):
    # Initialize the best match and minimum distance
    best_match = None
    min_distance = float('inf')

    # Calculate the Levenshtein distance for each file name
    for file_name in csv_files:
        # Remove the file extension for comparison
        base_name = os.path.splitext(file_name)[0]

        # Calculate the distance
        distance = Levenshtein.distance(campaign_name, base_name)
        print(f"Distance between '{campaign_name}' and '{base_name}': {distance}")

        # Update the best match if this distance is the smallest
        if distance < min_distance:
            min_distance = distance
            best_match = file_name

    return best_match


def obtener_directorios_cronologicos(bucket, prefix):
    s3 = boto3.client('s3',
                      aws_access_key_id='AKIA47CRVCSZH6PHSC52',
                      aws_secret_access_key='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',
                      region_name='us-east-1')
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')

    if 'CommonPrefixes' not in response:
        return []
    format = 0
    # Expresión regular para extraer fechas con el formato yyyy-mm-dd
    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
    date_pattern_2 = re.compile(r'\d{8}')
    directorios = []
    for obj in response['CommonPrefixes']:
        folder_name = obj['Prefix'].strip('/').split('/')[-1]
        if date_pattern.match(folder_name):
            directorios.append(folder_name)
            format = 0
        if date_pattern_2.match(folder_name):
            directorios.append(folder_name)
            format = 1

    # Ordenar las fechas en orden descendente
    if format == 0:
        directorios.sort(key=lambda date: datetime.strptime(date, '%Y-%m-%d'), reverse=True)
        rutas_completas = [f's3://{bucket}/{prefix}{directorio}/' for directorio in directorios]
    if format == 1:
        directorios.sort(key=lambda date: datetime.strptime(date, '%Y%m%d'), reverse=True)
        rutas_completas = [f's3://{bucket}/{prefix}{directorio}/' for directorio in directorios]
    return rutas_completas


def list_all_objects(bucket_name, prefix):
    s3 = boto3.client('s3',
                      aws_access_key_id='AKIA47CRVCSZH6PHSC52',
                      aws_secret_access_key='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',
                      region_name='us-east-1')


    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    all_keys = []
    for page in pages:
        for obj in page.get('Contents', []):
            all_keys.append(obj['Key'])

    return all_keys


def download_mat_from_s3(bucket_name, prefix, download_path):
    s3 = boto3.client('s3',
                      aws_access_key_id='AKIA47CRVCSZH6PHSC52',
                      aws_secret_access_key='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',
                      region_name='us-east-1')

    # Normalize the prefix
    prefix = prefix
    print(f"Normalized prefix: {prefix}")

    # Create the download directory if it doesn't exist
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # List all objects under the prefix for debugging
    all_objects = list_all_objects(bucket_name, prefix)
    #print(f"All objects under prefix '{prefix}': {all_objects}")

    # Check if any CSV files are found and download them
    found_files = False
    for key in all_objects:
        if key.endswith('.csv'):
            found_files = True
            file_name = os.path.basename(key)
            file_path = os.path.join(download_path, file_name)
            print(f"Downloading {key} to {file_path}")
            s3.download_file(bucket_name, key, file_path)

    if not found_files:
        print("No CSV files found in the specified prefix.")

def normalize_s3_route(route):
    segments_1,segment_2,_ = re.split(r'/', route)

    normalized_segments = segments_1.capitalize()

    normalized_route = normalized_segments + '/' + segment_2 + '/'
    return normalized_route



mapping_camps = clean_column_blank_regex(df_inventory, 'path')

mapping_camps['path']=mapping_camps['path'].apply(str)

mapping_camps_expanded = mapping_camps.assign(path=mapping_camps['path'].str.split(',')).explode('path')

CAMP_CODE= path_campaña

mat_location='MATRIZ_'+CAMP_CODE
CAMP_DATA=mapping_camps_expanded[mapping_camps_expanded['path']==CAMP_CODE].reset_index()

CAMP_DATA=clean_column_blank_regex(CAMP_DATA,'sponsor')

camp_data_campaign=CAMP_DATA['campaign'][0]

PATHS=mapping_camps[mapping_camps['campaign']==camp_data_campaign]['path'].to_numpy()
PATHS

sponsor=CAMP_DATA['sponsor'][0]
camp_id=CAMP_DATA['campaign_id'][0]

bucket_name_matrix='matrices.aihub'

prefix_matrix=CAMP_DATA['country'][0].replace(' ','')+'/'+CAMP_DATA['sponsor'][0].replace(' ','')+'/'
prefix_memory= CAMP_DATA['country'][0].replace(' ','')+'/'+CAMP_DATA['sponsor'][0].replace(' ','')+'/'

bucket_name = 's3iahub.igs'
prefix=CAMP_DATA['s3'][0].split(bucket_name+'/')[1]
print(prefix)
#download_prefix=get_last_modified_directory(bucket_name, prefix)

DATE_FOLDER= obtener_directorios_cronologicos(bucket_name, prefix)
print(DATE_FOLDER)
last=DATE_FOLDER[0].split(bucket_name+'/')[1]
last

download_mat_from_s3(bucket_name_matrix, prefix_matrix,mat_location )
prefix_matrix

download_prefix=last
memory_dirs=DATE_FOLDER[3:]

PATHS=PATHS[0].split(',')

id_vicidial=CAMP_CODE
download_path = sponsor.capitalize()+'/'

for i in PATHS:
    download_files_from_s3(id_vicidial,bucket_name, download_prefix, download_path,i+'_',i)

def delete_small_mp3_files(directory, size_limit_kb=100):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if file_path.endswith('.mp3') and os.path.isfile(file_path):
            file_size_kb = os.path.getsize(file_path) / 1024

            if file_size_kb < size_limit_kb:
                print(f"Deleting {filename} (Size: {file_size_kb:.2f} KB)")
                os.remove(file_path)

delete_small_mp3_files(download_path)

for i in PATHS:
    download_afectadas_from_s3(id_vicidial,bucket_name, download_prefix, download_path+'/isolated/',i+'_',i)
    #download_files_from_s3(id_vicidial,bucket_name, download_prefix, download_path,i+'_',i)


def count_mp3_files(directory):
    """Count the number of MP3 files in the specified directory, excluding subdirectories."""
    total_mp3_files = sum(1 for entry in os.listdir(directory)
                          if os.path.isfile(os.path.join(directory, entry)) and entry.lower().endswith('.mp3'))
    return total_mp3_files


unread_n=count_mp3_files(download_path+'/isolated/')

import os
import boto3
from datetime import datetime, timezone, timedelta
import warnings


def obtener_directorios_cronologicos(bucket, prefix):
    s3 = boto3.client('s3',
                      aws_access_key_id='AKIA47CRVCSZH6PHSC52',
                      aws_secret_access_key='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',
                      region_name='us-east-1')
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')

    if 'CommonPrefixes' not in response:
        return []

    # Expresión regular para extraer fechas con el formato yyyy-mm-dd
    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')

    directorios = []
    for obj in response['CommonPrefixes']:
        folder_name = obj['Prefix'].strip('/').split('/')[-1]
        if date_pattern.match(folder_name):
            directorios.append(folder_name)

    # Ordenar las fechas en orden descendente
    directorios.sort(key=lambda date: datetime.strptime(date, '%Y-%m-%d'), reverse=True)

    # Agregar el prefijo para obtener las rutas completas
    rutas_completas = [f's3://{bucket}/{prefix}{directorio}/' for directorio in directorios]

    return rutas_completas


def download_memory(bucket, rutas, fecha_limite=None, num_archivos=100, ruta_local='.'):
    s3 = boto3.client('s3',
                      aws_access_key_id='AKIA47CRVCSZH6PHSC52',
                      aws_secret_access_key='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',
                      region_name='us-east-1')

    memory_path = os.path.join(download_path, 'memory')
    if not os.path.exists(memory_path):
        os.makedirs(memory_path)
    try:
        if fecha_limite is None:
            fecha_limite = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        fecha_limite_dt = datetime.strptime(fecha_limite, '%Y-%m-%d')
    except:
        if fecha_limite is None:
            fecha_limite = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        fecha_limite_dt = datetime.strptime(fecha_limite, '%Y%m%d')
    archivos_descargados = 0

    for ruta in rutas:
        directorio_fecha = ruta.split('/')[-2]
        try:
            directorio_fecha_dt = datetime.strptime(directorio_fecha, '%Y-%m-%d')
        except:
            directorio_fecha_dt = datetime.strptime(directorio_fecha, '%Y%m%d')
        if directorio_fecha_dt <= fecha_limite_dt:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=f'{ruta.split(bucket + "/")[1]}')
            if 'Contents' in response:
                for obj in response['Contents']:
                    if archivos_descargados >= num_archivos:
                        return
                    archivo_key = obj['Key']
                    if archivo_key.endswith('.mp3') and obj[
                        'Size'] > 102400:  # Solo descargar archivos mayores a 100 KB
                        s3.download_file(bucket, archivo_key, f'{ruta_local}/{archivo_key.split("/")[-1]}')
                        archivos_descargados += 1

    if archivos_descargados < num_archivos:
        warnings.warn(
            f"No se pudieron descargar los {num_archivos} archivos solicitados. Solo se descargaron {archivos_descargados} archivos.")


def download_memory_json(bucket_json, rutas, fecha_limite=None, num_archivos=100, ruta_local='.'):
    s3 = boto3.client('s3',
                      aws_access_key_id='AKIA47CRVCSZH6PHSC52',
                      aws_secret_access_key='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',
                      region_name='us-east-1')
    try:
        if fecha_limite is None:
            fecha_limite = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        fecha_limite_dt = datetime.strptime(fecha_limite, '%Y-%m-%d')
    except:
        if fecha_limite is None:
            fecha_limite = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        fecha_limite_dt = datetime.strptime(fecha_limite, '%Y%m%d')
    archivos_descargados = 0

    for ruta in rutas:
        directorio_fecha = ruta.split('/')[-2]
        print(directorio_fecha)
        try:
            directorio_fecha_dt = datetime.strptime(directorio_fecha, '%Y-%m-%d')
        except:
            directorio_fecha_dt = datetime.strptime(directorio_fecha, '%Y%m%d')
            print('format: ' + str(directorio_fecha_dt))

        if directorio_fecha_dt <= fecha_limite_dt:
            print(prefix)
            response = s3.list_objects_v2(Bucket=bucket_json,
                                          Prefix=f'{prefix_memory + "transcript_sentences/" + ruta.split("s3iahub.igs" + "/")[1].split("/")[2]}')
            if 'Contents' in response:
                for obj in response['Contents']:
                    if archivos_descargados >= num_archivos:
                        return
                    archivo_key = obj['Key']
                    print(archivo_key)
                    if archivo_key.endswith('.json'):
                        s3.download_file(bucket_json, archivo_key, f'{ruta_local}/{archivo_key.split("/")[-1]}')
                        archivos_descargados += 1

    if archivos_descargados < num_archivos:
        warnings.warn(
            f"No se pudieron descargar los {num_archivos} archivos solicitados. Solo se descargaron {archivos_descargados} archivos.")


def download_memory_json(bucket_json, rutas, fecha_limite=None, num_archivos=100,ruta_local='.'):
    s3 = boto3.client('s3',
                      aws_access_key_id='AKIA47CRVCSZH6PHSC52',
                      aws_secret_access_key='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',
                      region_name='us-east-1')
    try:
        if fecha_limite is None:
            fecha_limite = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        fecha_limite_dt = datetime.strptime(fecha_limite, '%Y-%m-%d')
    except:
        if fecha_limite is None:
            fecha_limite = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        fecha_limite_dt = datetime.strptime(fecha_limite, '%Y%m%d')
    archivos_descargados = 0

    for ruta in rutas:
        directorio_fecha = ruta.split('/')[-2]
        print(directorio_fecha)
        try:
            directorio_fecha_dt = datetime.strptime(directorio_fecha, '%Y-%m-%d')
        except:
            directorio_fecha_dt = datetime.strptime(directorio_fecha, '%Y%m%d')
            print('format: ' +str(directorio_fecha_dt))

        if directorio_fecha_dt <= fecha_limite_dt:
            print(prefix)
            response = s3.list_objects_v2(Bucket=bucket_json, Prefix=f'{prefix_memory+"transcript_sentences/"+ruta.split("s3iahub.igs" + "/")[1].split("/")[2]}')
            if 'Contents' in response:
                for obj in response['Contents']:
                    if archivos_descargados >= num_archivos:
                        return
                    archivo_key = obj['Key']
                    print(archivo_key)
                    if archivo_key.endswith('.json'):
                        s3.download_file(bucket_json, archivo_key, f'{ruta_local}/{archivo_key.split("/")[-1]}')
                        archivos_descargados += 1

    if archivos_descargados < num_archivos:
        warnings.warn(f"No se pudieron descargar los {num_archivos} archivos solicitados. Solo se descargaron {archivos_descargados} archivos.")

download_memory(bucket_name, memory_dirs,  num_archivos=100 ,ruta_local=download_path+'/memory/')
download_memory_json('documentos.aihub',memory_dirs,num_archivos=100, ruta_local=download_path+'/memory/')
