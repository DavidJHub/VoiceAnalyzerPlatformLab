# RETORNAR NUM ARCHIVOS, NUM ARCHIVOS DESPUES DE CONCATENAR, NUM ARCHIVOS PROCESADOS, TMO,
import os
def count_s3_audio_files(bucket_name,campaign_id, prefix, s3_client):
    """
    Cuenta el número de archivos MP3 en un bucket de S3 con un prefijo específico.
    """
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    total_files = 0
    for page in pages:
        for obj in page.get('Contents', []):
            if (obj['Key'].endswith('.mp3') or obj['Key'].endswith('.wav')) and campaign_id in obj['Key']:
                total_files += 1
    return total_files

def count_local_files(directory,extension='.mp3'):
    """
    Cuenta el número de archivos MP3 en el directorio especificado, excluyendo subdirectorios.
    """
    total_mp3_files = sum(1 for entry in os.listdir(directory)
                          if os.path.isfile(os.path.join(directory, entry)) and entry.lower().endswith(extension))
    return total_mp3_files
