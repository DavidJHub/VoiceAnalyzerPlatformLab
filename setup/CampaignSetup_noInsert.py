import os
import logging

import json
import mysql.connector
import pandas as pd

from audio.ConcatCalls import concatenate_audios_main
from setup.MemorySetup import download_memory_json
from database.S3Loader import renombrar_archivos_s3, download_audio_files_fixed_route,download_transcripts_files_fixed_route
from utils.VapFunctions import  clean_column_blank_regex, get_campaign_parameters

# Configuraciones iniciales
logging.basicConfig(level=logging.ERROR)

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

def obtener_inventario():
    """
    Obtiene el inventario de campañas desde la base de datos.
    """
    conexion,cursor = connection_mysql('aihub_bd', 'admindb', 'VAPigs2024.*', 'vapdb.cjq4ek6ygqif.us-east-1.rds.amazonaws.com')
    if cursor:
        query = "SELECT * FROM marketing_campaigns"
        cursor.execute(query)
        column_names = [i[0] for i in cursor.description]
        data = cursor.fetchall()
        df_inventory = pd.DataFrame(data, columns=column_names)
        conexion.close()
        cursor.close()
        return df_inventory

        # Cerrar cursor y conexión
        cursor.close()
        conexion.close()
    else:
        return pd.DataFrame()




def move_small_audios(download_path, size_limit_kb=100):
    """
    Mueve los archivos de audio con un tamaño menor a size_limit_kb a una carpeta llamada "isolated".
    """
    audio_files = [f for f in os.listdir(download_path) if f.endswith('.mp3')]
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




def campaign_setup(path_campania,route):
    """
    Realiza el setup de la campaña y retorna los parámetros de la campaña.
    """
    df_inventory = obtener_inventario()
    if df_inventory.empty:
        print("No se pudo obtener el inventario.")
        return None
    else:
        mapping_camps = clean_column_blank_regex(df_inventory, 'path')
        mapping_camps_expanded = mapping_camps.assign(path=mapping_camps['path'].str.split(',')).explode('path')
        campaign_parameters = get_campaign_parameters(path_campania, mapping_camps_expanded)
        campaign_directory = 'process/' + path_campania + '/'
        print(campaign_parameters)
        if campaign_parameters is not None:
            matrix_path = campaign_directory + 'misc/'
            #print(f"Parámetros de la campaña {path_campaña}:{campaign_parameters}")
            if not os.path.exists(matrix_path):
                os.makedirs(matrix_path)
            save_dict_as_json(default_json_structure, matrix_path + "empty_transcript.json")
            S3_PATH = campaign_parameters['s3'].split('s3iahub.igs' + '/')[1]
            print("DESCARGANDO DE " + S3_PATH)
            memory_folder = os.path.join(campaign_directory+path_campania+'_RECONS', 'memory')
            if not os.path.exists(memory_folder):
                os.makedirs(memory_folder)
            SPONSOR = campaign_parameters['sponsor']
            COUNTRY= campaign_parameters['country']
            dRoute = COUNTRY + '/'+ 'recuperadas'+ '/' + "Serfinanza" + '/'
            tRoute = COUNTRY + '/'+ 'recuperadas'+ '/transcriptos/' + SPONSOR + '/'

            camp_data_campaign = campaign_parameters['campaign']
            PATHS = mapping_camps_expanded[mapping_camps_expanded['campaign'] == camp_data_campaign]['path'].to_numpy()

            for i in PATHS:
                route=download_audio_files_fixed_route(i,dRoute,SPONSOR, 's3iahub.igs', campaign_directory,0)
                print("DESCARGANDO TRANSCRIPTOS ")
                #route=download_transcripts_files_fixed_route(i,tRoute,SPONSOR, 's3iahub.igs', campaign_directory,0)
            #rename_prome_files(path_campania)
            #concatenate_audios_main(campaign_directory)
            print("AISLANDO AUDIOS VACÍOS")
            move_small_audios(campaign_directory, size_limit_kb=100)
        return campaign_parameters,route



