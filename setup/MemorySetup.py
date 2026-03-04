import os
import warnings

import boto3

from database.dbConfig import generate_s3_client
from utils.VapFunctions import get_latest_directories

from utils.campaignMetrics import count_local_files


def download_memory_audios(campaign_id, s3_path, bucket_name, local_output_folder, n,days_ago):
    """
    Descarga los archivos de audio del S3 desde el penúltimo directorio hacia atrás hasta tener al menos n audios.
    Los audios se descargan en la ruta especificada + 'memory/'.
    """
    s3_client = generate_s3_client()
    prefix = s3_path.rstrip('/') + '/'
    directories = get_latest_directories(bucket_name, prefix, s3_client)

    if len(directories) < 2:
        print("No hay suficientes directorios para descargar audios de memoria.")
        return

    print("TOTAL AUDIOS A DESCARGAR: " + str(n))
    local_mp3_count = count_local_files(local_output_folder,'.mp3')

    memory_folder = os.path.join(local_output_folder, 'memory')
    if not os.path.exists(memory_folder):
        os.makedirs(memory_folder)

    current_audio_count = local_mp3_count+count_local_files(memory_folder,'.mp3')
    if current_audio_count >= n:
        print(f"Ya hay al menos {n} audios en el directorio de memoria.")
        return

    for directory in directories[days_ago+1:]:  # Comenzar desde el penúltimo directorio
        route = directory.split(bucket_name + '/')[1]
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=route)
        for page in pages:
            for obj in page.get('Contents', []):
                if current_audio_count >= n:
                    print(f"Se han descargado al menos {n} audios en el directorio de memoria.")
                    return
                key = obj['Key']
                size = obj['Size']
                if key.endswith('.mp3') and campaign_id in key and size > 100 * 1024:
                    file_name = os.path.basename(key)
                    file_path = os.path.join(memory_folder, file_name)
                    s3_client.download_file(bucket_name, key, file_path)
                    current_audio_count += 1
                    #print(f"Archivo descargado: {file_path}")
    if current_audio_count < n:
        warnings.warn(f"No se pudieron descargar los {n} archivos solicitados. Solo se descargaron {current_audio_count} archivos.")

def download_memory_json(bucket_json,campaign_id,country,sponsor, local_output_folder, days_ago=0,num_archivos=20):
    """
    Descarga los archivos JSON relacionados con los archivos MP3 del directorio de memoria.
    """
    s3_client = generate_s3_client()

    prefix = country + '/' + sponsor + '/' + "transcript_sentences/"
    print("RUTA DE LOS TRANSCRIPTOS EN MEMORIA:" + prefix)
    directories = get_latest_directories(bucket_json, prefix, s3_client)
    memory_folder = local_output_folder + '/memory/'
    archivos_descargados = count_local_files(memory_folder,'.json')

    if not os.path.exists(memory_folder):
        os.makedirs(memory_folder)
    # Listar todos los archivos MP3 en el directorio de memoria
    for directory in directories[days_ago+1:]:
        directorio_fecha = directory.split('/')[-2]
        #print(directorio_fecha)
        #print(prefix + "transcript_sentences/" + directorio_fecha)
        response = s3_client.list_objects_v2(Bucket=bucket_json,
                                      Prefix=f'{prefix  + directorio_fecha}')
        if 'Contents' in response:
            for obj in response['Contents']:
                if archivos_descargados >= num_archivos:
                    return
                archivo_key = obj['Key']
                #print(archivo_key)
                if archivo_key.endswith('.json') and campaign_id in archivo_key:
                    s3_client.download_file(bucket_json, archivo_key, f'{memory_folder}/{archivo_key.split("/")[-1]}')
                    archivos_descargados += 1

    if archivos_descargados < num_archivos:
        warnings.warn(
            f"No se pudieron descargar los {num_archivos} archivos solicitados. Solo se descargaron {archivos_descargados} archivos.")

