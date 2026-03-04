import os
import random

def generate_random_id(length):
    """Genera un ID numérico aleatorio de la longitud especificada."""
    return ''.join([str(random.randint(0, 9)) for _ in range(length)])

def rename_files(directory):
    """
    Renombra los archivos en el directorio dado al formato deseado.

    Formato final:
    ALLIZ_YYYYMMDD-HHMMSS_1234568_1234567891_80123123_3201234123.mp3

    Parámetros:
        directory (str): Ruta al directorio donde se encuentran los archivos.
    """
    for filename in os.listdir(directory):
        # Verifica que el archivo tenga extensión .mp3
        if filename.endswith(".mp3"):
            try:
                # Extraer la parte de fecha y hora del nombre original
                parts = filename.split('-')
                if len(parts) >= 3:
                    date_time = parts[1] + '-' + parts[2]

                    # Generar IDs genéricos aleatorios
                    random_ids = [generate_random_id(length) for length in [7, 10, 8, 10]]

                    # Crear el nuevo nombre del archivo
                    new_filename = f"ALLIZ_{date_time}_{'_'.join(random_ids)}.mp3"

                    # Renombrar el archivo
                    original_path = os.path.join(directory, filename)
                    new_path = os.path.join(directory, new_filename)
                    os.rename(original_path, new_path)

                    print(f"Renombrado: {filename} -> {new_filename}")
                else:
                    print(f"Formato de nombre inesperado: {filename}")
            except Exception as e:
                print(f"Error al renombrar {filename}: {e}")

# Ejemplo de uso:
# Proporciona el directorio donde se encuentran los archivos.
# Asegúrate de que el directorio existe y contiene archivos mp3.

def main():
    """Función principal para ejecutar el script."""
    # Proporciona el directorio donde se encuentran los archivos
    print('Algo')
    directory_path = "../Muestra de llamadas Allianz-20250125T034433Z-001/Muestra de llamadas Allianz"
    if os.path.isdir(directory_path):
        rename_files(directory_path)
    else:
        print("La ruta proporcionada no es un directorio válido.")

if __name__ == "__main__":
    main()
