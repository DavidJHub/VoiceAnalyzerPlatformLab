import wave
import json
import numpy as np
import librosa
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from vosk import KaldiRecognizer
import soundfile as sf


def transcribe_audio(audio_path, model):
    """
    Transcribe an audio file using Vosk with Spanish model.
    """
    if audio_path.endswith('.mp3'):
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
    silence_threshold = 0.3

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
    transcription = transcribe_audio(audio_path, model)

    if not transcription:
        print("No transcription available.")
        return "", [], 0, 0

    features, sr, audio_length = extract_features(audio_path)
    speaker_labels = classify_speakers(features)

    improved_labels = apply_rules(speaker_labels, features, sr)

    return transcription, improved_labels, sr, audio_length


def match_speakers_to_audio(audio_length, sr, speaker_labels, frame_size=4000):
    """
    Match speaker labels to the original audio to determine who is speaking and when.
    """
    frame_duration = frame_size / sr
    num_frames = len(speaker_labels)
    adjusted_frame_duration = audio_length / num_frames / sr

    speaker_segments = []
    for i in range(num_frames):
        start_time = i * adjusted_frame_duration
        end_time = (i + 1) * adjusted_frame_duration
        speaker_segments.append((speaker_labels[i], start_time, end_time))

    return speaker_segments


def transcribe_audio_with_timestamps(audio_path, model):
    """
    Transcribe an audio file using Vosk with Spanish model and obtain timestamps for each word.
    """
    if audio_path.endswith('.mp3'):
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        wav_path = audio_path.replace('.mp3', '.wav')
        sf.write(wav_path, y, sr)
        audio_path = wav_path
    else:
        1

    wf = wave.open(audio_path, "rb")

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        return pd.DataFrame()

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    words_info = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if 'result' in result:
                for word in result['result']:
                    words_info.append({
                        'word': word['word'],
                        'start': word['start'],
                        'end': word['end']
                    })

    # Get final result
    result = json.loads(rec.FinalResult())
    if 'result' in result:
        for word in result['result']:
            words_info.append({
                'word': word['word'],
                'start': word['start'],
                'end': word['end']
            })

    df_transcription = pd.DataFrame(words_info)
    return df_transcription


def apply_speaker_smoothing(df, window_size=3):
    """
    Apply Gaussian smoothing to the speaker column to reduce noise in speaker diarization.
    """
    if 'speaker' not in df.columns:
        print("Speaker column not found in the DataFrame.")
        return df

    speaker_series = df['speaker'].fillna(method='ffill').fillna(method='bfill')
    smoothed_speaker_series = gaussian_filter1d(speaker_series.astype(float), sigma=window_size, mode='nearest')
    df['smoothed_speaker'] = np.round(smoothed_speaker_series).astype(int)
    return df