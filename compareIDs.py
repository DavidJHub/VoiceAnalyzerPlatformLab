import boto3
import pandas as pd

from database.dbConfig import generate_s3_client
import database.dbConfig as dbcfg

# Parámetros de configuración
BUCKET_NAME = 'cacai'
S3_PREFIX_1 = 'Colombia/Bancolombia/Abril'  # termina con / si es un "directorio"
S3_PREFIX_2 = 'Colombia/Bancolombia/Mayo'  # termina con / si es un "directorio"
EXCEL_PATH = 'lead_id.xlsx'  # ruta local al archivo
ID_COLUMN = 'lead_id'  # nombre de la columna con los IDs

s3 = generate_s3_client()

df = pd.read_excel(EXCEL_PATH)
ids = df[ID_COLUMN].astype(str).tolist()

response_1 = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_PREFIX_1)
s3_keys_abril = [obj['Key'] for obj in response_1.get('Contents', [])]
response_2 = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_PREFIX_2)
s3_keys_mayo = [obj['Key'] for obj in response_2.get('Contents', [])]

s3_keys=s3_keys_abril + s3_keys_mayo

def check_id_in_s3(id_value):
    return any(id_value in key for key in s3_keys)

df['exists_in_s3'] = df[ID_COLUMN].astype(str).apply(check_id_in_s3)

print(df[['lead_id', 'exists_in_s3']])
print("RESULTADOS VERDADEROS: {}".format(len(df[df['exists_in_s3'] == True])))
print("RESULTADOS FALSOS: {}".format(len(df[df['exists_in_s3'] == False])))

conexion = dbcfg.conectar(HOST=dbcfg.HOST_DB_VAP,  
                                DATABASE=dbcfg.DB_NAME_VAP,  
                                USERNAME=dbcfg.USER_DB_VAP,  
                                PASSWORD=dbcfg.PASSWORD_DB_VAP)


cursor = conexion.cursor()

cursor.execute("SELECT lead_id FROM agent_audio_data WHERE campaign_id IN (71, 79)")
lead_ids_db = set(str(row[0]) for row in cursor.fetchall())

df['exists_in_db'] = df[ID_COLUMN].apply(lambda x: x in lead_ids_db)

cursor.close()
conexion.close()

print(df[[ID_COLUMN, 'exists_in_db']])
df.to_excel('resultados_comparacion_db.xlsx', index=False)