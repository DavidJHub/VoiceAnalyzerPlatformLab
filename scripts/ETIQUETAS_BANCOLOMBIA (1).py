import boto3
import re

from database.dbConfig import generate_s3_client

# Configuración de AWS

BUCKET_NAME = 's3iahub.igs'
PREFIX = 'Colombia/Bancolombia/2025-05-19/'

# Conexión al cliente de S3
s3 = generate_s3_client()

# Contadores globales
field2_counter = 1
field4_counter = 1

def reconstruct_filename(filename):
    global field2_counter, field4_counter

    # Caso 1: Etiquetas con 6 campos
    pattern_6_fields = r'(BANCOL|BANCOLFI|BANCOLST)_(\d{12})_(\d+)_(\d+)_(\d+)_(\d+)-all\.mp3'
    match_6 = re.match(pattern_6_fields, filename)
    if match_6:
        prefix, date, field1, field2, field3, field4 = match_6.groups()
        year = "20" + date[:2]
        month = date[2:4]
        day = date[4:6]
        hour = date[6:8]
        minute = date[8:10]
        second = date[10:12]
        new_date = f"{year}{month}{day}-{hour}{minute}{second}"
        return f"{prefix}_{new_date}_{field1}_{field2}_{field3}_{field4}-all.mp3"

    #   Caso 2: Etiquetas con 5 campos
    pattern_5_fields = r'(BANCOL|BANCOLFI|BANCOLST)_(\d{8}-\d{6})_(\d+)_(\d+)_(\d+)-all\.mp3'
    match_5 = re.match(pattern_5_fields, filename)
    if match_5:
        prefix, date, field1, field2, field3 = match_5.groups()
        year = date[:4]
        month = date[4:6]
        day = date[6:8]
        hour = date[9:11]
        minute = date[11:13]
        second = date[13:15]
        new_date = f"{year}{month}{day}-{hour}{minute}{second}"
        # Agregar el 6º campo como un contador
        new_field4 = field4_counter
        field4_counter += 1
        return f"{prefix}_{new_date}_{field1}_{field2}_{field3}_{new_field4}-all.mp3"

    # Caso 3: Etiquetas con 4 campos
    pattern_4_fields = r'(BANCOL|BANCOLFI|BANCOLST)_(\d{8}-\d{6})_(\d+)_(\d+)-all\.mp3'
    match_4 = re.match(pattern_4_fields, filename)
    if match_4:
        prefix, date, field1, field3 = match_4.groups()
        year = date[:4]
        month = date[4:6]
        day = date[6:8]
        hour = date[9:11]
        minute = date[11:13]
        second = date[13:15]
        new_date = f"{year}{month}{day}-{hour}{minute}{second}"
        # Agregar el 4º campo (field2) y el 6º campo como contadores
        new_field2 = field2_counter
        new_field4 = field4_counter
        field2_counter += 1
        field4_counter += 1
        return f"{prefix}_{new_date}_{field1}_{new_field2}_{field3}_{new_field4}-all.mp3"

    # Si no coincide con ningún patrón, devolver None
    return None

def process_files():
    # Listar los objetos en el bucket y prefijo especificado
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
    if 'Contents' in response:
        for obj in response['Contents']:
            original_filename = obj['Key'].split('/')[-1]  # Obtener solo el nombre del archivo
            new_filename = reconstruct_filename(original_filename)
            if new_filename:
                print(f"Original: {original_filename} -> Nuevo: {new_filename}")
                # Renombrar el archivo en S3
                s3.copy_object(
                    Bucket=BUCKET_NAME,
                    CopySource={'Bucket': BUCKET_NAME, 'Key': obj['Key']},
                    Key=f"{PREFIX}{new_filename}"
                )
                s3.delete_object(Bucket=BUCKET_NAME, Key=obj['Key'])
            else:
                print(f"Archivo no coincide con ningún patrón: {original_filename}")
    else:
        print("No se encontraron archivos en la carpeta especificada.")

if __name__ == "__main__":
    process_files()