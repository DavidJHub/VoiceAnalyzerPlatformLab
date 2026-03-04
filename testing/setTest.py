
import subprocess # Import main.py to access the main function
import sys
import torch
from database.S3Loader import renombrar_archivos_s3
import database.dbConfig as dbcfg

NEW_PATH = "DMUADCM_"
OLD_PATH = "DMULFCTA_"
S3_PATH = "Colombia/Davivienda/"
S3_BUCKET = "s3iahub.igs"

def set_test_data(param1, param2=0):
    """
    Set test data for the VAP system.
    """

    # Rename files in the selected directory
    renombrar_archivos_s3(S3_PATH,S3_BUCKET, OLD_PATH, param1,param2)
    
    print(f"Test data set in directory: {S3_PATH}")

if __name__ == '__main__':

    param1 = NEW_PATH
    param2 = 0

    print("INICIANDO PROCESO PARA "+ param1 +" CON DELAY "+ str(param2))
    print("CUDA disponible:", torch.cuda.is_available())

    print("Dispositivo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    set_test_data(param1,param2)

    param2 = str(param2)  # Ensure param2 is a string for subprocess compatibility
    subprocess.run([sys.executable, "main.py", param1, param2,'dry_run'])
    print("FINALIZADO PROCESO PARA " + param1 + " CON DELAY " + str(param2) + " EXITOSAMENTE.")