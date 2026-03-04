import os
from tqdm import tqdm
from pydub import AudioSegment
from utils.campaignMetrics import count_local_files
from utils.VapFunctions import get_latest_directories
import database.dbConfig as dbcfg
from utils.campaignMetrics import count_s3_audio_files



def _normalize_audio_base(filename: str) -> str:
    """
    Normaliza un nombre de audio para comparar por 'base key':
    - usa basename
    - lower
    - quita extensión (.wav/.mp3)
    - remueve '-concat' y variantes comunes
    """
    base = os.path.basename(filename).lower()
    stem, _ = os.path.splitext(base)

    # Si SOLO quieres '-concat', deja solo esa línea.
    for token in ["-concat", "_concat", " concat", "-concatenated", "_concatenated"]:
        stem = stem.replace(token, "")

    return stem


def download_audio_files(campaign_id, s3_path, sponsor, bucket_name, local_output_folder, days_ago=0):
    """
    Descarga audios (.mp3/.wav) desde S3 para el directorio más reciente (o days_ago).

    NUEVA REGLA (base key):
    - Si ya hay archivos locales:
      si el audio en S3 (.wav o .mp3) tiene el mismo 'base key' normalizado
      (sin extensión y sin '-concat') que cualquier audio local (.wav o .mp3),
      entonces NO se descarga.
    """
    s3_client = dbcfg.generate_s3_client()
    directory = get_latest_directories(bucket_name, s3_path, s3_client)[days_ago]
    route = directory.split(bucket_name + '/')[1]
    print('LA RUTA DE DESCARGA EN S3 ES: ' + route)
    print('PREFIX:', campaign_id)
    print('SPONSOR:', sponsor)

    if not directory:
        print("No se pudieron obtener directorios válidos.")
        return route

    if not os.path.exists(local_output_folder):
        os.makedirs(local_output_folder)

    s3_audio_files = count_s3_audio_files(bucket_name, campaign_id, route, s3_client)
    print("TOTAL AUDIOS A DESCARGAR: " + str(s3_audio_files))

    local_mp3_count = count_local_files(local_output_folder, '.mp3')
    local_wav_count = count_local_files(local_output_folder, '.wav')
    local_total = local_mp3_count + local_wav_count
    print(f"AUDIOS YA PRESENTES LOCALMENTE: {local_total}")

    # --- índice de bases normalizadas de TODOS los audios locales ---
    local_bases = set()
    if local_total > 0:
        for fname in os.listdir(local_output_folder):
            if fname.lower().endswith((".wav", ".mp3")):
                local_bases.add(_normalize_audio_base(fname))

    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=route)

    audio_keys = []
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if (key.endswith('.mp3') or key.endswith('.wav')) and campaign_id in key:
                audio_keys.append(key)

    skipped = 0
    for key in tqdm(audio_keys, desc="Descargando audios ", unit=" archivo"):
        file_name = os.path.basename(key)
        ext = os.path.splitext(file_name)[1].lower()

        # --- DEDUP por base key, independiente de extensión ---
        if local_total > 0 and ext in (".wav", ".mp3"):
            base_s3 = _normalize_audio_base(file_name)
            if base_s3 in local_bases:
                skipped += 1
                continue

        file_path = os.path.join(local_output_folder, file_name)
        s3_client.download_file(bucket_name, key, file_path)

    if skipped:
        print(f"Se saltaron {skipped} audios por coincidencia de base key (sin extensión y sin '-concat') con archivos locales.")

    return route

def renombrar_archivos_s3(s3_path, bucket_name, old_prefix, new_prefix, days_ago):
    """
    Renombra archivos dentro de un directorio en un bucket de S3 si cumplen con un prefijo específico.
    Por ejemplo, si un archivo comienza con `old_prefix`, se renombrará para que comience con `new_prefix`.

    :param bucket_name: Nombre del bucket de S3
    :param directorio: Directorio dentro del bucket donde se encuentran los archivos (por ejemplo 'carpeta/subcarpeta/')
    :param old_prefix: Prefijo viejo que se busca en el nombre de los archivos (por ejemplo 'PROME_2_')
    :param new_prefix: Nuevo prefijo con el cual se renombrarán los archivos (por ejemplo 'PROME2_')
    """
    s3_client = dbcfg.generate_s3_client()

    directorio = get_latest_directories(bucket_name, s3_path, s3_client)[days_ago]
    route = directorio.split(bucket_name + '/')[1]
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=route)
    all_objs = []
    for page in pages:
        for obj in page.get('Contents', []):
            all_objs.append(obj)
    for obj in tqdm(all_objs, desc="Renombrando archivos", unit="archivo"):
        old_key = obj['Key']
        relative_name = old_key[len(route):] if directorio else old_key
        if relative_name.startswith(old_prefix):
            new_name = new_prefix + relative_name[len(old_prefix):]
            new_key = route + new_name if route else new_name
            # print(f"Renombrando {old_key} a {new_key}...")
            s3_client.copy_object(Bucket=bucket_name, CopySource={'Bucket': bucket_name, 'Key': old_key}, Key=new_key)
            s3_client.delete_object(Bucket=bucket_name, Key=old_key)





def cargar_audios_procesados_a_s3(directorio, bucket_name, ruta_s3):
    """
    Carga archivos con extensión .json desde un directorio a un bucket de S3.

    :param directorio: Directorio local donde están los archivos .json
    :param bucket_name: Nombre del bucket de S3
    :param ruta_s3: Ruta dentro del bucket de S3 donde se cargarán los archivos
    """
    s3_client = dbcfg.generate_s3_client()
    directory = get_latest_directories(bucket_name, ruta_s3, s3_client)[0]
    route = directory.split(bucket_name + '/')[1]

    archivos = [archivo for archivo in os.listdir(directorio) if (archivo.endswith('.mp3') or archivo.endswith('.wav'))]

    for archivo in tqdm(archivos, desc="Cargando audios a S3", unit="archivo"):
        archivo_local = os.path.join(directorio, archivo)
        ruta_completa_s3 = os.path.join(ruta_s3, archivo)
        ruta_completa_s3 = ruta_completa_s3.replace(os.path.sep, '/')

        try:
            s3_client.upload_file(archivo_local, bucket_name, ruta_completa_s3)
            # print(f'Archivo {archivo} cargado exitosamente a {ruta_completa_s3}')
        except Exception as e:
            print(f'Error al cargar {archivo}: {e}')


def cargar_unread_a_s3(directorio, bucket_name, ruta_s3):
    """
    Carga archivos con extensión .json desde un directorio a un bucket de S3.

    :param directorio: Directorio local donde están los archivos .json
    :param bucket_name: Nombre del bucket de S3
    :param ruta_s3: Ruta dentro del bucket de S3 donde se cargarán los archivos
    """
    s3_client = dbcfg.generate_s3_client()

    for archivo in os.listdir(directorio + 'isolated/'):
        if archivo.endswith('.mp3') or archivo.endswith('.wav'):
            archivo_local = os.path.join(directorio + 'isolated/', archivo)
            ruta_completa_s3 = os.path.join(ruta_s3, archivo)
            ruta_completa_s3 = ruta_completa_s3.replace(os.path.sep, '/')

            try:
                s3_client.upload_file(archivo_local, bucket_name, ruta_completa_s3)
                print(f'Archivo {archivo} cargado exitosamente a {ruta_completa_s3}')
            except Exception as e:
                print(f'Error al cargar {archivo}: {e}')


def cargar_archivos_json_a_s3(directorio, bucket_name, ruta_s3):
    """
    Carga archivos con extensión .json desde un directorio a un bucket de S3.

    :param directorio: Directorio local donde están los archivos .json
    :param bucket_name: Nombre del bucket de S3
    :param ruta_s3: Ruta dentro del bucket de S3 donde se cargarán los archivos
    """
    s3_client = dbcfg.generate_s3_client()

    for archivo in os.listdir(directorio):
        if archivo.endswith('.json'):
            archivo_local = os.path.join(directorio, archivo)
            ruta_completa_s3 = os.path.join(ruta_s3, archivo)
            ruta_completa_s3 = ruta_completa_s3.replace(os.path.sep, '/')

            try:
                s3_client.upload_file(archivo_local, bucket_name, ruta_completa_s3)
                #print(f'Archivo {archivo} cargado exitosamente a {ruta_completa_s3}')
            except Exception as e:
                print(f'Error al cargar {archivo}: {e}')

###---------------------------------------------------------------------------------------------------------------------
###______________________________________FUNCIÓN DE CARGA REJECTED CALLS A S3_________________________________________
###---------------------------------------------------------------------------------------------------------------------

def cargar_excel_a_s3(path_excel, bucket_name, ruta_s3):
    """
    Carga un archivo Excel (.xlsx o .xls) a un bucket de S3.

    :param path_excel: Ruta local completa al archivo Excel (ej: process/.../misc/result_rejected.xlsx)
    :param bucket_name: Nombre del bucket de S3
    :param ruta_s3: Prefijo/ruta dentro del bucket donde se cargará (ej: Colombia/Sponsor/2025-01-01/misc/)
                   Nota: puede terminar en / o no; da igual.
    """
    s3_client = dbcfg.generate_s3_client()

    if not os.path.isfile(path_excel):
        raise FileNotFoundError(f"No existe el archivo Excel en: {path_excel}")

    if not (path_excel.lower().endswith(".xlsx") or path_excel.lower().endswith(".xls")):
        raise ValueError(f"El archivo no parece ser Excel (.xlsx/.xls): {path_excel}")

    archivo = os.path.basename(path_excel)# nombre del archivo, como en el loop de jsons
    ruta_completa_s3 = os.path.join(ruta_s3, archivo)
    ruta_completa_s3 = ruta_completa_s3.replace(os.path.sep, '/')

    try:
        s3_client.upload_file(path_excel, bucket_name, ruta_completa_s3)
        print(f'Archivo {archivo} cargado exitosamente a {ruta_completa_s3}')
    except Exception as e:
        print(f'Error al cargar {archivo}: {e}')

###---------------------------------------------------------------------------------------------------------------------
###______________________________________FIN DE FUNCIÓN DE CARGA REJECTED CALLS A S3_________________________________________
###---------------------------------------------------------------------------------------------------------------------


def cargar_audios_concat_a_s3(directorio, bucket_name, ruta_s3):
    """
    Carga archivos con extensión .json desde un directorio a un bucket de S3.

    :param directorio: Directorio local donde están los archivos .json
    :param bucket_name: Nombre del bucket de S3
    :param ruta_s3: Ruta dentro del bucket de S3 donde se cargarán los archivos
    """
    s3_client = dbcfg.generate_s3_client()

    for archivo in os.listdir(directorio):
        if (archivo.endswith('.mp3') or archivo.endswith('.wav')):
            archivo_local = os.path.join(directorio, archivo)
            ruta_completa_s3 = os.path.join(ruta_s3, archivo)
            ruta_completa_s3 = ruta_completa_s3.replace(os.path.sep, '/')

            try:
                s3_client.upload_file(archivo_local, bucket_name, ruta_completa_s3)
                #print(f'Archivo {archivo} cargado exitosamente a {ruta_completa_s3}')
            except Exception as e:
                print(f'Error al cargar {archivo}: {e}')



def download_grading_matrix(bucket_name, mat_path, local_output_folder):
    s3 = dbcfg.generate_s3_client()
    prefix = mat_path.split(bucket_name + '/')[1]
    matrix_folder = os.path.join(local_output_folder, 'misc')
    matrix_path_gen = matrix_folder + "/gen/"
    print("RUTA DE MATRICES EN S3:" + prefix)
    if not os.path.exists(matrix_folder):
        os.makedirs(matrix_folder)
    if not os.path.exists(matrix_path_gen):
        os.makedirs(matrix_path_gen)
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    all_keys = []
    for page in pages:
        for obj in page.get('Contents', []):
            all_keys.append(obj['Key'])
    
    found_files = False
    for key in all_keys:
        if key.endswith('dictionary.csv'):
            file_name = os.path.basename(key)
            file_path = os.path.join(matrix_path_gen, file_name)
            print(f"Downloading {key} to {file_path}")
            s3.download_file(bucket_name, key, file_path)
        elif key.endswith('.csv'):
            found_files = True
            file_name = os.path.basename(key)
            file_path = os.path.join(matrix_folder, file_name)
            print(f"Downloading {key} to {file_path}")
            s3.download_file(bucket_name, key, file_path)
        elif key.endswith('.xlsx') and ("Matriz" or "MATRIZ" in key): 
            found_files = True
            file_name = os.path.basename(key)
            file_path = os.path.join(matrix_folder, file_name)
            print(f"Downloading {key} to {file_path}")
            s3.download_file(bucket_name, key, file_path)
    if not found_files:
        print("No CSV files found in the specified prefix.")



def download_guion(bucket_name, mat_path, local_output_folder):
    s3 = dbcfg.generate_s3_client()
    prefix = mat_path.split(bucket_name + '/')[1]
    guion_folder = os.path.join(local_output_folder, 'misc')
    guion_path_gen = guion_folder + "/gen/"
    print("RUTA DE GUION EN S3:" + prefix)
    if not os.path.exists(guion_path_gen):
        os.makedirs(guion_path_gen)
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    all_keys = []
    for page in pages:
        for obj in page.get('Contents', []):
            all_keys.append(obj['Key'])
    
    found_files = False
    for key in all_keys:
        if key.endswith('.pdf') and ("GUION" or "Guion" in key.upper()): 
            found_files = True
            file_name = os.path.basename(key)
            file_path = os.path.join(guion_folder, file_name)
            print(f"Downloading {key} to {file_path}")
            s3.download_file(bucket_name, key, file_path)
    if not found_files:
        print("No PDF files found in the specified prefix.")


def compress_audio_files(input_dir, output_dir, target_bitrate="256k"):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            # Define the full path to the input and output files
            input_filepath = os.path.join(input_dir, filename)
            output_filepath = os.path.join(output_dir, filename)

            # Load the audio file
            audio = AudioSegment.from_file(input_filepath)

            # Export the audio file with compression
            audio.export(output_filepath, format="mp3", bitrate=target_bitrate)
            print(f"Compressed and saved: {output_filepath}")


def download_audio_files_fixed_route(campaign_id,route,sponsor, bucket_name, local_output_folder):
    """
    Descarga los archivos de audio del S3 que correspondan a los directorios ordenados de fecha más reciente a más antigua,
    separando las llamadas que pesen menos de 100kb en un subdirectorio '/isolated'.
    """
    s3_client = dbcfg.generate_s3_client()
    directory = route
    print('LA RUTA DE DESCARGA EN S3 ES: '+ route)
    print('PREFIX:', campaign_id)
    print('SPONSOR:', sponsor)
    if not os.path.exists(local_output_folder):
        os.makedirs(local_output_folder)

    s3_mp3_count = count_s3_audio_files(bucket_name,campaign_id, route, s3_client)
    print("TOTAL AUDIOS A DESCARGAR: " + str(s3_mp3_count))
    if str(s3_mp3_count)==0:
        raise Exception("No hay audios en s3 con este prefijo. Terminando proceso...")
    local_mp3_count = count_local_files(local_output_folder,'.mp3')
    local_wav_count = count_local_files(local_output_folder,'.wav')
    print(local_mp3_count)
    print(local_wav_count)
    paginator = s3_client.get_paginator('list_objects_v2')

    pages = paginator.paginate(Bucket=bucket_name, Prefix=route)
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.mp3') or key.endswith('.wav') and campaign_id in key:
                file_name = os.path.basename(key)
                file_path = os.path.join(local_output_folder, file_name)
                s3_client.download_file(bucket_name, key, file_path)
                print(f"Archivo descargado: {file_path}")
    return route



def download_transcripts_files_fixed_route(campaign_id,s3_path,sponsor, bucket_name, local_output_folder,days_ago):
    """
    Descarga los archivos de audio del S3 que correspondan a los directorios ordenados de fecha más reciente a más antigua,
    separando las llamadas que pesen menos de 100kb en un subdirectorio '/isolated'.
    """

    s3_client = dbcfg.generate_s3_client()
    directory = get_latest_directories(bucket_name, s3_path, s3_client)[days_ago]
    route=directory.split(bucket_name+'/')[1]
    print('LA RUTA DE DESCARGA EN S3 ES: '+ route)
    print('PREFIX:', campaign_id)
    print('SPONSOR:', sponsor)

    if not os.path.exists(local_output_folder):
        os.makedirs(local_output_folder)

    s3_audio_files = count_s3_audio_files(bucket_name,campaign_id, route, s3_client)
    print("TOTAL AUDIOS A DESCARGAR: " + str(s3_audio_files))
    if str(s3_audio_files)==0:
        raise Exception("No hay audios en s3 con este prefijo. Terminando proceso...")
    local_mp3_count = count_local_files(local_output_folder,'.mp3')
    local_wav_count = count_local_files(local_output_folder,'.wav')
    print(local_mp3_count + local_wav_count)
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=route)
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.json') and campaign_id in key:
                file_name = os.path.basename(key)
                file_path = os.path.join(local_output_folder, file_name)
                s3_client.download_file(bucket_name, key, file_path)
                print(f"Archivo descargado: {file_path}")
    return route

if __name__ == "__main__":
    #renombrar_archivos_s3(nombrebd, usuario, contraseña, host, ruta, local_output_folder)
    campaign_id = "ALLIZ_"
    route= "Colombia/Allianz/2025-07-24/"
    sponsor="Allianz"
    bucket_name="s3iahub.igs"
    local_output_folder = "C:/Users/Petya_/MEGAPROJECTS/VAP_RELEASE/data/Allianz/2025-07-24/"
    download_audio_files_fixed_route(campaign_id,route,sponsor, bucket_name, local_output_folder)