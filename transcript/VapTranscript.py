import json

try:
    from deepgram import DeepgramClient, PrerecordedOptions  # works if SDK v3 is installed
except ImportError:
    from deepgram import Deepgram as DeepgramClient          # SDK v2 alias
    PrerecordedOptions = dict  

retry_attempts = 3  # Número de reintentos

OPTION_TRANSCRIPT_ENGINE="DEEPGRAM"



def CallDeepgram_transcript(audiofile,outputfile,model="nova-2"):
    deepgram = DeepgramClient('65e2df0efe4c7390470ff7d2936bc047a6a1e899')
    with open(audiofile, 'rb') as buffer_data:
        payload = { 'buffer': buffer_data }


        options = PrerecordedOptions(
            punctuate=True, model=model, language="es-419",smart_format=True,diarize=True,
        )

        print('Requesting transcript...')

        response = deepgram.listen.prerecorded.v('1').transcribe_file(payload, options)
        response_json = response.to_json(indent=4)
        response_dict = json.loads(response.to_json())

        response_json = json.dumps(response_dict, ensure_ascii=False, indent=4)
        output_filename=outputfile
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(response_json)
        print(f"Transcript saved to {output_filename}")

def transcribe_with_retry(ruta_completa, output_file, retry_attempts):
    attempt = 0
    while attempt < retry_attempts:
        try:
            CallDeepgram_transcript(ruta_completa, output_file)
            return
        except:
            attempt += 1
            print(f"TimeoutError encountered. Attempt {attempt} of {retry_attempts}. Retrying...")
    print(f"Failed to transcribe {ruta_completa} after {retry_attempts} attempts.")

################################################################################
#################################  VAP ENGINE  #################################
################################################################################

import wave
import json
import numpy as np
import librosa
import scipy.signal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from vosk import Model, KaldiRecognizer
import time
import soundfile as sf


def transcribe_audio(audio_path, model):
    """
    Transcribe an audio file using Vosk with Spanish model.
    """
    if audio_path.endswith('.mp3') or audio_path.endswith('.wav'):
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        wav_path = audio_path.replace('.mp3', '.wav')
        sf.write(wav_path, y, sr)
        audio_path = wav_path
    else:
        1
    wf = wave.open(audio_path, "rb")

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        return ""

    rec = KaldiRecognizer(model, wf.getframerate())

    transcript = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            transcript += result.get('text', '') + " "

    # Get final result
    result = json.loads(rec.FinalResult())
    transcript += result.get('text', '')

    return transcript.strip()


def extract_features(audio_path):
    """
    Extract MFCC features from audio for speaker diarization.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    y = librosa.effects.preemphasis(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    return mfcc.T, sr, len(y)


def classify_speakers(features, num_speakers=2):
    """
    Classify speakers in an audio using GMM.
    """
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    gmm = GaussianMixture(n_components=num_speakers, covariance_type='diag', random_state=42)
    gmm.fit(features_normalized)

    speaker_labels = gmm.predict(features_normalized)

    return speaker_labels


def apply_rules(speaker_labels, features, sr):
    """
    Apply custom rules to improve speaker diarization.
    """
    frame_duration = len(features) / len(speaker_labels) / sr
    improved_labels = speaker_labels.copy()
    amplitude_threshold = np.percentile(np.abs(features), 80)
    silence_threshold = 0.05

    for i in range(1, len(speaker_labels) - 1):
        if speaker_labels[i] != speaker_labels[i - 1]:
            # Rule 1: Check if the speaker change is followed by brief silence
            if np.mean(np.abs(features[i])) < silence_threshold:
                improved_labels[i] = improved_labels[i - 1]

            if frame_duration < 3.0:
                if speaker_labels[i] != speaker_labels[i + 1] and np.mean(np.abs(features[i])) > amplitude_threshold:
                    improved_labels[i] = speaker_labels[i - 1]

    return improved_labels


def diarize_audio(audio_path, model):
    """
    Perform transcription and diarization for an audio file.
    """
    transcription, cut_time = transcribe_audio(audio_path, model)

    if not transcription:
        print("No transcription available.")
        return "", [], 0, 0, cut_time

    features, sr, audio_length, cut_time = extract_features(audio_path)
    speaker_labels = classify_speakers(features)

    improved_labels = apply_rules(speaker_labels, features, sr)

    return transcription, improved_labels, sr, audio_length, cut_time





def match_speakers_to_audio(audio_length, sr, speaker_labels, cut_time, frame_size=4000):
    """
    Match speaker labels to the original audio to determine who is speaking and when.
    """
    frame_duration = frame_size / sr
    num_frames = len(speaker_labels)
    adjusted_frame_duration = audio_length / num_frames / sr

    speaker_segments = []
    for i in range(num_frames):
        start_time = i * adjusted_frame_duration + cut_time
        end_time = (i + 1) * adjusted_frame_duration + cut_time
        speaker_segments.append((speaker_labels[i], start_time, end_time))

    return speaker_segments


