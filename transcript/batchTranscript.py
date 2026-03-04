import os
import json
from tqdm import tqdm
from transcript.transcriptAudio import transcribe_with_retry
from utils.VapUtils import jsonDecompose
from setup.CampaignSetup import setDefaultJsonStructure

def batchTranscript(audio_files,campaign_directory_input,campaign_directory_output,retry_attempts,OPTION_TRANSCRIPT_ENGINE):
    for filename in tqdm(audio_files, desc="Transcribiendo audios", unit="archivo") :
            nombre_input = campaign_directory_input + "/" + filename
            last_transcribed = None  
            nombre_archivo_sin_extension = filename.split("/")[-1].split(".")[0]
            if OPTION_TRANSCRIPT_ENGINE == "DEEPGRAM":
                nombre_output = campaign_directory_output + "/" + nombre_archivo_sin_extension + "_transcript.json"
                if not os.path.exists(nombre_output):
                    transcribe_with_retry(nombre_input, nombre_output, retry_attempts)
                    last_transcribed = nombre_input  

                if os.path.exists(nombre_output):
                    if os.path.getsize(nombre_output) < 4000:
                        transcribe_with_retry(nombre_input, nombre_output, retry_attempts)
                        last_transcribed = nombre_input
                    else:
                        continue  # Si el archivo ya existe y es válido, continuar con el siguiente archivo

                if os.path.exists(nombre_output) and os.path.getsize(nombre_output) > 4000:
                    last_transcribed = filename  # Guardar el último archivo válido
                #print(f'Verifying transcript {output_file}')

                    # ---------------------------------------------------------------
                    # PARSEAR Y DESCOMPONER JSON
                    # ---------------------------------------------------------------
                try:
                    jsonDecompose(nombre_output)
                except (Exception,) as e:
                    print(f"{nombre_output} unsolvable, using default empty file")
                    data = setDefaultJsonStructure()
                    file = open(nombre_output, "w")
                    json.dump(data, file, indent=4)
                    file.close()