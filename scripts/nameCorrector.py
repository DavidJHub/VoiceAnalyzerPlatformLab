import boto3
from botocore.exceptions import ClientError
import os

from database.dbConfig import generate_s3_client

def renombrar_archivos_recursivo(
    bucket_name: str,
    base_prefix: str,
    old_prefix: str,
    new_prefix: str,
    aws_profile: str | None = None,
    region_name: str | None = "us-east-1"
):
    """
    Recorre recursivamente todos los objetos que empiecen por `base_prefix`
    (p. ej. 'carpeta/' o 'carpeta/subcarpeta/') y renombra los que
    tengan `old_prefix` al inicio del nombre de archivo.
    """

    # 1. Cliente S3 (mejor usar perfiles/roles que claves embebidas)
    s3_client = generate_s3_client()

    # 2. Paginador sobre el prefijo base
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=base_prefix):
        for obj in page.get("Contents", []):
            old_key = obj["Key"]                   # p.ej. 'carpeta/a/b/PROME_2_ejemplo.csv'
            filename = os.path.basename(old_key)   # p.ej. 'PROME_2_ejemplo.csv'

            if filename.startswith(old_prefix):
                # Nuevo nombre de archivo
                new_filename = new_prefix + filename[len(old_prefix):]
                # Construimos la key final manteniendo el path original
                new_key = old_key.split('/'+old_prefix)[0] + '/' + new_filename

                try:
                    # Copiar con el nombre nuevo
                    s3_client.copy_object(
                        Bucket=bucket_name,
                        CopySource={"Bucket": bucket_name, "Key": old_key},
                        Key=new_key
                    )
                    # Eliminar el objeto viejo
                    s3_client.delete_object(Bucket=bucket_name, Key=old_key)
                    print(f"✅ {old_key}  ➜  {new_key}")
                except ClientError as e:
                    print(f"⚠️  Error renombrando {old_key}: {e}")


def renombrar_archivos_recursivo(
    bucket_name: str,
    base_prefix: str,
    old_prefix: str,
    new_prefix: str,
    aws_profile: str | None = None,
    region_name: str | None = "us-east-1"
):
    """
    Recorre recursivamente todos los objetos que empiecen por `base_prefix`
    (p. ej. 'carpeta/' o 'carpeta/subcarpeta/') y renombra los que
    tengan `old_prefix` al inicio del nombre de archivo.
    """

    # 1. Cliente S3 (mejor usar perfiles/roles que claves embebidas)
    s3_client = generate_s3_client()

    # 2. Paginador sobre el prefijo base
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=base_prefix):
        for obj in page.get("Contents", []):
            old_key = obj["Key"]                   # p.ej. 'carpeta/a/b/PROME_2_ejemplo.csv'
            filename = os.path.basename(old_key)   # p.ej. 'PROME_2_ejemplo.csv'

            if filename.startswith(old_prefix):
                # Nuevo nombre de archivo
                new_filename = new_prefix + filename[len(old_prefix):]
                # Construimos la key final manteniendo el path original
                new_key = old_key.split('/'+old_prefix)[0] + '/' + new_filename

                try:
                    # Copiar con el nombre nuevo
                    s3_client.copy_object(
                        Bucket=bucket_name,
                        CopySource={"Bucket": bucket_name, "Key": old_key},
                        Key=new_key
                    )
                    s3_client.delete_object(Bucket=bucket_name, Key=old_key)
                    print(f"✅ {old_key}  ➜  {new_key}")
                except ClientError as e:
                    print(f"⚠️  Error renombrando {old_key}: {e}")

