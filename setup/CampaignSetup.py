import os
import logging

import json
import pandas as pd

from audio.ConcatCalls import concatenate_audios_main
import database.dbConfig as dbcfg
from setup.MemorySetup import download_memory_json
from database.S3Loader import renombrar_archivos_s3, download_audio_files, download_audio_files_fixed_route
from utils.VapFunctions import  clean_column_blank_regex, get_campaign_parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def obtener_inventario():
    """
    Obtiene el inventario de campañas desde la base de datos.
    """
    conexion = None
    cursor = None
    try:
        conexion = dbcfg.conectar(HOST=dbcfg.HOST_DB_VAP,
                                  DATABASE=dbcfg.DB_NAME_VAP,
                                  USERNAME=dbcfg.USER_DB_VAP,
                                  PASSWORD=dbcfg.PASSWORD_DB_VAP)
        cursor = conexion.cursor()
        if cursor:
            query = "SELECT * FROM marketing_campaigns"
            cursor.execute(query)
            column_names = [i[0] for i in cursor.description]
            data = cursor.fetchall()
            df_inventory = pd.DataFrame(data, columns=column_names)
            return clean_column_blank_regex(df_inventory, 'path')
        else:
            logger.warning("No se pudo obtener el cursor de la base de datos.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error al obtener inventario: {e}")
        return pd.DataFrame()
    finally:
        if cursor:
            cursor.close()
        if conexion:
            conexion.close()


import boto3


def fix_audio_filenames_s3(
    bucket: str,
    prefix: str = "",
    dry_run: bool = True,
    s3: boto3.client = None,
) -> None:
    """
    Verifica y corrige nombres de archivos de audio en S3.

    Esperado: [TAG]_[FECHAHORA]_[LEADID]_[EPOCH]_[CLIENTID]_[PHONE].mp3|.wav
              → 5 guiones bajos = 6 partes.
    Si encuentra exactamente 4 guiones bajos (5 partes) añade '_000'
    antes de la extensión.

    Parámetros
    ----------
    bucket : str
        Nombre del bucket S3.
    prefix : str, opcional
        Prefijo/ruta de los objetos a revisar.
    dry_run : bool, opcional
        Si True, sólo informa; si False, copia-renombra y elimina el original.
    s3 : boto3.client, opcional
        Cliente S3 ya creado; si None se genera con boto3.client('s3').
    """
    print(f"Revisando archivos en {prefix}...")
    if s3 is None:
        s3 = dbcfg.generate_s3_client()
        print(s3)
    print(f"Bucket: {bucket}, Prefijo: {prefix}, Dry run: {dry_run}")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Ignorar carpetas simuladas y extensiones no deseadas
            if key.lower().endswith((".mp3", ".wav")):
                dirname, filename = os.path.split(key)
                name, ext = os.path.splitext(filename)
                parts = name.split("_")
                if len(parts) == 5:          # ← sólo 4 guiones bajos
                    corrected_name = f"{name}_000{ext}"
                    new_key = f"{dirname}/{corrected_name}" if dirname else corrected_name

                    if dry_run:
                        print(f"[DRY-RUN]  {key}  →  {new_key}")
                    else:
                        # Copiar al nuevo nombre y borrar el antiguo
                        s3.copy_object(
                            Bucket=bucket,
                            CopySource={"Bucket": bucket, "Key": key},
                            Key=new_key,
                        )
                        s3.delete_object(Bucket=bucket, Key=key)
                        print(f"Renombrado: {key}  →  {new_key}")
                elif len(parts) == 4:          # ← sólo 4 guiones bajos
                    corrected_name = f"{name}_000_000{ext}"
                    new_key = f"{dirname}/{corrected_name}" if dirname else corrected_name

                    if dry_run:
                        print(f"[DRY-RUN]  {key}  →  {new_key}")
                    else:
                        # Copiar al nuevo nombre y borrar el antiguo
                        s3.copy_object(
                            Bucket=bucket,
                            CopySource={"Bucket": bucket, "Key": key},
                            Key=new_key,
                        )
                        s3.delete_object(Bucket=bucket, Key=key)
                        print(f"Renombrado: {key}  →  {new_key}")

                elif len(parts) != 6:
                    # Informativo: casos con estructura inesperada
                    print(f"[WARN] Nombre irregular ({len(parts)-1} guiones bajos): {key}")

def move_small_audios(download_path, size_limit_kb=100):
    """
    Mueve los archivos de audio con un tamaño menor a size_limit_kb a la carpeta "isolated".
    """
    audio_files = [f for f in os.listdir(download_path) if (f.endswith('.mp3') or f.endswith('.wav'))]
    isolated_folder = os.path.join(download_path, 'isolated')
    if not os.path.exists(isolated_folder):
        os.makedirs(isolated_folder)
    for file in audio_files:
        file_path = os.path.join(download_path, file)
        if os.path.getsize(file_path) < size_limit_kb * 1024:
            try:
                os.rename(file_path, os.path.join(isolated_folder, file))
                print(f" {file_path} Tomado como UNREAD, aislando...")
            except Exception as e:
                try:
                    os.remove(file_path)
                    print(f"Archivo {file} eliminado debido a un error.")
                except Exception as delete_error:
                    print(f"Error al intentar eliminar el archivo {file}: {delete_error}")

def save_dict_as_json(data, file_path):
    """
    Saves a given dictionary as a JSON file to the specified file path.

    :param data: Dictionary to be saved as JSON.
    :param file_path: Path where the JSON file will be saved.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Archivo JSON dummy creado: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

default_json_structure = {
    "metadata": {
        "transaction_key": "",
        "request_id": "",
        "sha256": "",
        "created": "",
        "duration": 0.0,
        "channels": 0,
        "models": [],
        "model_info": {}
    },
    "results": {
        "channels": [
            {
                "alternatives": [
                    {
                        "transcript": " ",
                        "confidence": 0.0,
                        "words": [
                            {
                                "word": " ",
                                "start": 0.0,
                                "end": 0.0,
                                "confidence": 0.0,
                                "punctuated_word": "",
                                "speaker": 0,
                                "speaker_confidence": 0.0
                            }
                        ],
                        "paragraphs": {
                            "transcript": " ",
                            "paragraphs": [
                                {
                                    "sentences": [
                                        {
                                            "text": ". .",
                                            "start": 0,
                                            "end": 0
                                        }
                                    ],
                                    "start": 0,
                                    "end": 0,
                                    "num_words":0,
                                    "speaker": 0
                                },
                            ]
                        }
                    }
                ]
            }
        ]
    }
}

def setDefaultJsonStructure():
    """
    Sets the default JSON structure for transcripts.
    """
    return default_json_structure

def tagCleanning(PREFIX):
    NEWPREFIX = None
    print("ENTRADA:{}".format(PREFIX))
    print(len(PREFIX.split("_")))
    if len(PREFIX.split("_"))>2:
        print(PREFIX.split("_")[0])
        print(PREFIX.split("_")[1])
        NEWPREFIX = PREFIX.split("_")[0] +  PREFIX.split("_")[1] + "_" 
        print(NEWPREFIX)
    else:
        NEWPREFIX = PREFIX
    return NEWPREFIX


def campaign_setup(path_campania,mapping_camps_expanded,campaign_parameters,days_ago=0,oparam1=None):
    """
    Realiza el setup de la campaña recibiendo los parámetros de ejecución del main y 
    retorna los parámetros de la campaña: Sigue los pasos mostrados a continuación:
    1. Extraer los parámetros de la campaña a partir del PREFIX.
    2. Renombra los archivos de campañas defectuosas si aplica.
    3. Descarga los archivos de audio del S3.
    4. Descarga los archivos de transcripciones para memoria del S3.
    5. Aisla los audios vacíos (menores a 100 KB) en la carpeta "isolated".
    """
    S3_PATH = campaign_parameters['s3'].split('s3iahub.igs' + '/')[1]
    print("tag normal: "+str(tagCleanning(oparam1)==oparam1))

    if not (tagCleanning(oparam1)==oparam1):     # Renombrar archivos si aplica
            print("RENOMBRANDO ARCHIVOS...")
            renombrar_archivos_s3(S3_PATH,'s3iahub.igs',oparam1,tagCleanning(oparam1),days_ago)
            path_campania=tagCleanning(oparam1)
    campaign_directory = os.path.join(_PROJECT_ROOT, 'process', path_campania) + '/'
    print(campaign_parameters)

    if campaign_parameters is not None:        # GENERACION DE DIRECTORIOS EN LOCAL PARA PROCESAMIENTO
        matrix_path = campaign_directory + 'misc/'
        if not os.path.exists(matrix_path):
            os.makedirs(matrix_path)
        save_dict_as_json(default_json_structure, matrix_path + "empty_transcript.json")
        reconstruct_folder =campaign_directory+"/"+path_campania +'_RECONS' + "/"
        if not os.path.exists(reconstruct_folder):
            os.makedirs(reconstruct_folder)
        memory_folder =campaign_directory+"/"+path_campania +'_RECONS' + "/" + 'memory'+ "/"
        if not os.path.exists(memory_folder):
            os.makedirs(memory_folder)

        # PARÁMETROS FIJOS DE CAMPAÑA

        SPONSOR = campaign_parameters['sponsor']
        COUNTRY= campaign_parameters['country']
        CAMPAIGN = campaign_parameters['campaign']
        PATHS = mapping_camps_expanded[mapping_camps_expanded['campaign'] == CAMPAIGN]['path'].to_numpy()
        for i in PATHS:
            route=download_audio_files(i,S3_PATH,SPONSOR, 's3iahub.igs', campaign_directory, days_ago)
            print("DESCARGANDO MEMORIA DE TRANSCRIPCIONES")
            download_memory_json('documentos.aihub',path_campania, COUNTRY,
                                    SPONSOR, campaign_directory+path_campania+'_RECONS',days_ago, num_archivos=0)
        audioList,_ = concatenate_audios_main(campaign_directory)
        print("AISLANDO AUDIOS VACÍOS")
        move_small_audios(campaign_directory, size_limit_kb=120)
    return campaign_parameters,route,audioList


def campaign_setup_manual_route(path_campania,mapping_camps_expanded,campaign_parameters,days_ago=0,oparam1=None):
    """
    Realiza el setup de la campaña recibiendo los parámetros de ejecución del main y
    retorna los parámetros de la campaña: Sigue los pasos mostrados a continuación:
    1. Obtener el inventario de campañas desde la base de datos.
    2. Extraer los parámetros de la campaña a partir del PREFIX.
    3. Renombra los archivos de campañas defectuosas si aplica.
    4. Descarga los archivos de audio del S3.
    5. Descarga los archivos de transcripciones para memoria del S3.
    6. Aisla los audios vacíos (menores a 120 KB) en la carpeta "isolated".
    """
    campaign_directory = os.path.join(_PROJECT_ROOT, 'process', path_campania) + '/'
    logger.info(f"Parámetros de campaña: {campaign_parameters}")
    if campaign_parameters is not None:
        matrix_path = campaign_directory + 'misc/'
        save_dict_as_json(default_json_structure, matrix_path + "empty_transcript.json")
        S3_PATH = campaign_parameters['s3'].split('s3iahub.igs' + '/')[1]
        reconstruct_folder =campaign_directory+"/"+path_campania +'_RECONS' + "/"
        if not os.path.exists(reconstruct_folder):
            os.makedirs(reconstruct_folder)
        memory_folder =campaign_directory+"/"+path_campania +'_RECONS' + "/" + 'memory'+ "/"
        if not os.path.exists(memory_folder):
            os.makedirs(memory_folder)
        SPONSOR = campaign_parameters['sponsor']
        COUNTRY= campaign_parameters['country']
        camp_data_campaign = campaign_parameters['campaign']
        PATHS = mapping_camps_expanded[mapping_camps_expanded['campaign'] == camp_data_campaign]['path'].to_numpy()
        for i in PATHS:
            route=download_audio_files_fixed_route(i,S3_PATH,SPONSOR, 's3iahub.igs', campaign_directory)
        #rename_prome_files(path_campania)
        audioList,_ = concatenate_audios_main(campaign_directory)
        print("AISLANDO AUDIOS VACÍOS")
        move_small_audios(campaign_directory, size_limit_kb=120)
    return campaign_parameters,route,audioList