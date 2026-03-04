import pandas as pd
import re
import os
import shutil
import unicodedata

# Paths de archivos y carpetas
archivo_asesores = "PLANTA.xlsx" #archivo de planta de asesores, relacion extension y cedula
archivo_llamadas = "LLAMADAS VAP ALLIANZ.xlsx" #archivo donde estan los registros de llamadas con el relacion a extension
carpeta_entrada = "LLAMADAS ALLIANZ VAP"         # Donde están los .mp3 originales
carpeta_salida = "AUDIOS_RENOMBRADOS" # Carpeta donde se guardarán los archivos renombrados

# Crear carpeta de salida si no existe
os.makedirs(carpeta_salida, exist_ok=True)

# Cargar archivos
asesores_df = pd.read_excel(archivo_asesores)
llamadas_df = pd.read_excel(archivo_llamadas)

# Mapeo de extensión a cédula
ext_to_cedula = asesores_df.set_index('EXTENSIÓN')['CÉDULA'].astype(str).to_dict()

# Limpiar el nombre del servicio
def limpiar_servicio(servicio):
    # Eliminar tildes
    servicio = unicodedata.normalize('NFD', servicio)
    servicio = ''.join(c for c in servicio if unicodedata.category(c) != 'Mn')
    # Eliminar "Hogar -" y contenido entre paréntesis
    servicio = re.sub(r"Hogar\s*-\s*", "", servicio, flags=re.IGNORECASE)
    servicio = re.sub(r"\(.*?\)", "", servicio)
    return servicio.strip().upper()

# Generar nombre nuevo y mover archivo si existe
def procesar_audio(row):
    llamada = row['LLAMADA']
    fecha = row['FechaAsistencia']
    # Elimina el "-1 y el -2" del CódigoAsistencia si está presente
    codigo_asistencia = str(row['CódigoAsistencia']).replace("/", "-").replace("-1", "").replace("-2", "")
    extension = row['EXTENSION']
    servicio = limpiar_servicio(row['Servicio'])
    
    fecha_str = fecha.strftime("%Y%m%d-%H%M%S")

    tipo_servicio = servicio  # Este será el tipo de servicio

    cedula = ext_to_cedula.get(int(extension), "SIN_CEDULA")

    nuevo_nombre = f"ALLIZ_{fecha_str}_{codigo_asistencia}_{tipo_servicio}_{cedula}_{extension}.mp3" # Formato del nuevo nombre

    nombre_original = llamada + ".mp3"
    ruta_origen = os.path.join(carpeta_entrada, nombre_original)
    ruta_destino = os.path.join(carpeta_salida, nuevo_nombre)

    if os.path.exists(ruta_origen):
        shutil.copy(ruta_origen, ruta_destino)
        print(f"Copiado: {nuevo_nombre}")
    else:
        print(f"No encontrado: {nombre_original}")

# Procesar todos los registros
llamadas_df.apply(procesar_audio, axis=1)
