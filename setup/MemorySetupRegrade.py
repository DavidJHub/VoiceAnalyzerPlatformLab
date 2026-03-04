import os

from database.dbConfig import generate_s3_client
from utils.VapFunctions import get_latest_directories
from utils.campaignMetrics import count_local_files

def download_memory_json(bucket_json,campaign_id,country,sponsor, local_output_folder):
    """
    Descarga los archivos JSON relacionados con los archivos MP3 del directorio de memoria.
    """
    s3_client = generate_s3_client()

    prefix = country + '/' + sponsor + '/' + "transcript_sentences/"
    print("RUTA DE LOS TRANSCRIPTOS EN MEMORIA:" + prefix)
    directories = get_latest_directories(bucket_json, prefix, s3_client)
    memory_folder = os.path.join(local_output_folder, 'memory')
    archivos_descargados = count_local_files(memory_folder,'.json')

    if not os.path.exists(memory_folder):
        os.makedirs(memory_folder)
    # Listar todos los archivos MP3 en el directorio de memoria
    for directory in directories[20:]:
        directorio_fecha = directory.split('/')[-2]
        #print(directorio_fecha)
        #print(prefix + "transcript_sentences/" + directorio_fecha)
        response = s3_client.list_objects_v2(Bucket=bucket_json,
                                      Prefix=f'{prefix  + directorio_fecha}')
        if 'Contents' in response:
            for obj in response['Contents']:
                archivo_key = obj['Key']
                #print(archivo_key)
                if archivo_key.endswith('.json') and campaign_id in archivo_key:
                    s3_client.download_file(bucket_json, archivo_key, f'{memory_folder}/{archivo_key.split("/")[-1]}')
                    archivos_descargados += 1