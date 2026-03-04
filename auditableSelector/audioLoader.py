import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def get_mel_features(audio_path, n_mels=128):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB, sr


def get_mel_features(audio_path, n_mels=128, n_fft=512, hop_length=512):
    """
    Computes the mel spectrogram (in dB) for the given audio.

    Returns:
        S_dB: Mel spectrogram in dB (shape: [n_mels, n_frames]).
        sr: Sampling rate.
    """
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                       n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB, sr


def measure_spectral_flatness(audio_path, n_fft=512, hop_length=512):
    """
    Computes spectral flatness for each frame.

    Returns:
        flatness: 1D array (n_frames,) of spectral flatness values.
        sr: Sampling rate.
    """
    y, sr = librosa.load(audio_path, sr=None)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
    return flatness, sr


def visualize_spectral_flatness(flatness,sr, hop_length=512):



    times = librosa.frames_to_time(np.arange(flatness.shape[1]), sr=sr, hop_length=hop_length)

    plt.figure(figsize=(10, 4))
    plt.plot(times, flatness[0], label='Spectral Flatness')
    plt.xlabel('Time (s)')
    plt.ylabel('Flatness')
    plt.title('Spectral Flatness Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    audio_path = "AUDIOS/BANPOPC_20250325-131008_2177893_1742926208_1034659369_3133140360-all.mp3"  
    mel_features, sample_rate = get_mel_features(audio_path, n_mels=40)
    flatness, flatness_mean, sr = measure_spectral_flatness(audio_path)
    print("Spectral flatness (per frame) shape:", flatness.shape)
    print("Mean spectral flatness:", flatness_mean)
    print("Mel Features Shape:", mel_features.shape)
    visualize_spectral_flatness(flatness,sr)