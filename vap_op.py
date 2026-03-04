import boto3
import datetime
import pandas as pd
import os

# Configuración de AWS
AWS_ACCESS_KEY = 'AKIA47CRVCSZMOPVDDNN'
AWS_SECRET_KEY = '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'
AWS_REGION = 'us-east-1'
BUCKET_NAME = 'catvap'

# Conexión al cliente de S3
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# Carga la tabla de referencia (PLANTA) para mapear extensiones a cédulas
PLANTA_FILE = 'PLANTA_ACTUALIZADA.xlsx'  # Planta
planta_df = pd.read_excel(PLANTA_FILE, dtype={'EXTENSIÓN': str, 'CÉDULA': str})

def map_extension_to_cedula(extension):
    """Mapea la extensión al número de cédula usando la tabla PLANTA"""
    row = planta_df.loc[planta_df['EXTENSIÓN'] == extension]
    if not row.empty:
        cedula = row.iloc[0]['CÉDULA']
        if pd.notna(cedula):  # Verifica si la cédula no es NaN
            print(f"Extensión {extension} mapeada a cédula {cedula}")
            return cedula
        else:
            print(f"Extensión {extension} tiene cédula NaN. Asignando '1234'.")
    else:
        print(f"Extensión {extension} no encontrada. Asignando '1234'.")
    return '1234'  # Valor por defecto si no se encuentra la extensión o la cédula es NaN

def convert_epoch_to_datetime(epoch):
    """Convierte el epoch a formato de fecha y hora (aaammdd-hhmmss)"""
    dt = datetime.datetime.fromtimestamp(float(epoch))
    return dt.strftime('%Y%m%d-%H%M%S')

def get_recent_folder(bucket_name, index):
    """Obtiene la carpeta más reciente o según el índice especificado"""
    try:
        # Lista los objetos en el bucket
        response = s3.list_objects_v2(Bucket=bucket_name, Delimiter='/')
        if 'CommonPrefixes' in response:
            # Extrae las carpetas y las ordena por fecha
            folders = [prefix['Prefix'] for prefix in response['CommonPrefixes']]
            folders.sort(reverse=True)  # Orden descendente (más reciente primero)

            if index < len(folders):
                selected_folder = folders[index]
                print(f"Carpeta seleccionada: {selected_folder}")
                return selected_folder
            else:
                print(f"Índice fuera de rango. Hay {len(folders)} carpetas disponibles.")
                return None
        else:
            print("No se encontraron carpetas en el bucket.")
            return None
    except Exception as e:
        print(f"Error obteniendo carpetas: {e}")
        return None

def rename_files_in_s3(bucket_name, prefix):
    """Renombra los archivos directamente en S3"""
    try:
        # Lista los objetos en el bucket con el prefijo especificado
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            print(f"Archivos en el directorio '{prefix}':")
            for obj in response['Contents']:
                original_key = obj['Key']
                if original_key.endswith('.wav'):
                    # Extrae los datos del nombre del archivo
                    filename = os.path.basename(original_key)  # solo el nombre
                    parts = filename.split('-')
                    if len(parts) < 6:
                        print(f"Formato inválido: {filename}")
                        continue

                    epoch = parts[-1].replace('.wav', '')
                    extension = parts[-2]
                    phone_number = parts[-3]
                    q_code = parts[2]  # Q14015

                    # Convierte el epoch a fecha y hora
                    datetime_str = convert_epoch_to_datetime(epoch)

                    # Mapea la extensión al número de cédula
                    cedula = map_extension_to_cedula(extension)

                    # Construye el nuevo nombre del archivo
                    new_name = f"{q_code}_{datetime_str}_{phone_number}_{cedula}_{extension}.wav"
                    new_key = f"{prefix}{new_name}"  # Incluye el prefijo original

                    # Renombra el archivo en S3
                    s3.copy_object(
                        Bucket=bucket_name,
                        CopySource={'Bucket': bucket_name, 'Key': original_key},
                        Key=new_key
                    )
                    s3.delete_object(Bucket=bucket_name, Key=original_key)
                    print(f"Renombrado en S3: {original_key} → {new_key}")
        else:
            print(f"No se encontraron archivos en el directorio '{prefix}'.")
    except Exception as e:
        print(f"Error renombrando archivos en S3: {e}")

# Ejemplo de uso
if __name__ == "__main__":
    # Índice de carpeta (0 para la más reciente, 1 para la segunda más reciente, etc.)
    folder_index = 0  # 0 mas reciente - 1 segunda mas reciente

    # Obtiene la carpeta según el índice
    selected_folder = get_recent_folder(BUCKET_NAME, folder_index)

    if selected_folder:
        # Renombra los archivos directamente en S3
        rename_files_in_s3(BUCKET_NAME, selected_folder)