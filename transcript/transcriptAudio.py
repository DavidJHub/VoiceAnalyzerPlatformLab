import json
try:
    from deepgram import DeepgramClient, PrerecordedOptions  
except ImportError:
    from deepgram import Deepgram as DeepgramClient          
    PrerecordedOptions = dict  
import transcript.transcriptConfig as tcfg

def CallDeepgram_transcript(audiofile,outputfile,model=tcfg.model):
    deepgram = DeepgramClient(tcfg.DEEPGRAM_API_KEY)
    with open(audiofile, 'rb') as buffer_data:
        payload = { 'buffer': buffer_data }


        options = PrerecordedOptions(
            punctuate=True, model=model, language=tcfg.language,smart_format=True,diarize=True,
        )

        #print('Requesting transcript...')

        response = deepgram.listen.prerecorded.v('1').transcribe_file(payload, options)
        response_json = response.to_json(indent=4)
        response_dict = json.loads(response.to_json())

        response_json = json.dumps(response_dict, ensure_ascii=False, indent=4)
        output_filename=outputfile
        #output_filename = dr + "transcript_{}_{}-noise.json".format(data_path.split("/")[-2].split(" ")[1],model)
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(response_json)
        #print(f"Transcript saved to {output_filename}")

def transcribe_with_retry(ruta_completa, output_file, retry_attempts=tcfg.timeout_attempts):
    attempt = 0
    while attempt < retry_attempts:
        try:
            CallDeepgram_transcript(ruta_completa, output_file)
            return
        except Exception as e:
            attempt += 1
            print(f"Error details: {e}")
            print(f"TimeoutError encountered. Attempt {attempt} of {retry_attempts}. Retrying...")
    print(f"Failed to transcribe {ruta_completa} after {retry_attempts} attempts.")
