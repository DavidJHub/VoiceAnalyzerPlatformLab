import os
import pandas as pd

# Ruta base
base_path = "process/"

# Lista para guardar los DataFrames encontrados
dataframes = []

# Recorrer los subdirectorios dentro de "process/"
for root, dirs, files in os.walk(base_path):
    if "misc" in dirs:  # Verifica si existe la carpeta "misc"
        file_path = os.path.join(root, "misc", "result_rejected.xlsx")
        if os.path.exists(file_path):
            try:
                df = pd.read_excel(file_path)
                dataframes.append(df)
                print(f"Archivo encontrado y cargado: {file_path}")
            except Exception as e:
                print(f"Error leyendo {file_path}: {e}")

# Concatenar todos los DataFrames si hay al menos uno
if dataframes:
    # Unirlos alineando columnas (rellena con NaN si falta alguna columna en algún archivo)
    df_final = pd.concat(dataframes, ignore_index=True, sort=False)

    # Guardar en un solo archivo
    output_file = "REPORTE_0820.xlsx"
    df_final.to_excel(output_file, index=False)
    print(f"Reporte generado: {output_file} con {len(df_final)} filas.")
else:
    print("No se encontraron archivos df_concatenado_afectadas.xlsx en los subdirectorios.")
