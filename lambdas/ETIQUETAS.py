import boto3
import re

# Configuración de AWS
AWS_ACCESS_KEY = 'AKIA47CRVCSZMOPVDDNN'
AWS_SECRET_KEY = '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'
AWS_REGION = 'us-east-1'  # Cambia según tu región
BUCKET_NAME = 's3iahub.igs'

# Conexión al cliente de S3
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def get_recent_folder(bucket_name, prefix, index=0):
    """
    Obtiene la carpeta más reciente o la segunda más reciente según el índice.
    Maneja carpetas con y sin guion en el nombre.
    """
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    if 'CommonPrefixes' in response:
        # Obtener las carpetas
        folders = [prefix['Prefix'] for prefix in response['CommonPrefixes']]

        # Extraer fechas de las carpetas y manejar guiones
        def extract_date(folder_name):
            # Intentar extraer fecha con guion (formato YYYY-MM-DD)
            match_with_dash = re.search(r'(\d{4}-\d{2}-\d{2})', folder_name)
            if match_with_dash:
                return match_with_dash.group(1)
            # Intentar extraer fecha sin guion (formato YYYYMMDD)
            match_without_dash = re.search(r'(\d{8})', folder_name)
            if match_without_dash:
                # Convertir a formato YYYY-MM-DD para ordenar correctamente
                raw_date = match_without_dash.group(1)
                return f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            return None

        # Filtrar carpetas válidas con fechas extraídas
        folders_with_dates = [(folder, extract_date(folder)) for folder in folders]
        folders_with_dates = [(folder, date) for folder, date in folders_with_dates if date]

        # Ordenar carpetas por fecha en orden descendente
        folders_with_dates.sort(key=lambda x: x[1], reverse=True)

        # Seleccionar la carpeta según el índice
        if index < len(folders_with_dates):
            return folders_with_dates[index][0]
        else:
            raise IndexError(f"No hay suficientes carpetas para el índice {index}.")
    else:
        raise ValueError("No se encontraron carpetas en el bucket.")

def reconstruct_filename(filename, prefix_to_add):
    """
    Reconstruye el nombre del archivo agregando un prefijo adicional.
    Si el prefijo ya contiene el texto adicional, no se procesa.
    """
    # Expresión regular para capturar los campos de la etiqueta
    pattern = r'(\w+)_(\d{8}-\d{6})_(\d+)_(\d+)_(\d+)_(\d+)-all\.mp3'
    match = re.match(pattern, filename)
    if match:
        prefix, date, field1, field2, field3, field4 = match.groups()
        
        # Verificar si el prefijo ya contiene el texto adicional
        if prefix.endswith(prefix_to_add):
            print(f"El archivo ya tiene el prefijo '{prefix_to_add}', no se procesa: {filename}")
            return None
        
        # Agregar el prefijo adicional
        new_prefix = f"{prefix}{prefix_to_add}"
        
        # Reconstruir el nombre del archivo
        new_filename = f"{new_prefix}_{date}_{field1}_{field2}_{field3}_{field4}-all.mp3"
        return new_filename
    return None

def process_files(prefix, prefix_to_add, index=0):
    """
    Procesa los archivos en la carpeta seleccionada según el índice y el prefijo.
    """
    # Obtener la carpeta más reciente según el índice
    recent_folder = get_recent_folder(BUCKET_NAME, prefix, index)
    print(f"Procesando archivos en la carpeta: {recent_folder}")

    # Listar los objetos en el bucket y prefijo especificado
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=recent_folder)
    if 'Contents' in response:
        for obj in response['Contents']:
            original_filename = obj['Key'].split('/')[-1]  # Obtener solo el nombre del archivo
            new_filename = reconstruct_filename(original_filename, prefix_to_add)
            if new_filename:
                print(f"Original: {original_filename} -> Nuevo: {new_filename}")
                # Renombrar el archivo en S3
                s3.copy_object(
                    Bucket=BUCKET_NAME,
                    CopySource={'Bucket': BUCKET_NAME, 'Key': obj['Key']},
                    Key=f"{recent_folder}{new_filename}"
                )
                s3.delete_object(Bucket=BUCKET_NAME, Key=obj['Key'])
    else:
        print("No se encontraron archivos en la carpeta especificada.")

if __name__ == "__main__":
    # Procesar el directorio de El Salvador/DAVIVIENDA/
    process_files(prefix='El Salvador/DAVIVIENDA/', prefix_to_add='SV', index=0)
    
    # Procesar el directorio de Honduras/Banrural/
    process_files(prefix='Honduras/Banrural/', prefix_to_add='HD', index=0)