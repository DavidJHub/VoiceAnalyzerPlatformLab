import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans



def get_snr(audio_path, n_fft=512, hop_length=32):
    """
    Computes the SNR (in dB) over time using the segmentation mask.

    The average noise energy is estimated from the frames classified as non-speech.
    SNR per frame is calculated as:

        SNR = 10 * log10(frame_energy / noise_energy)

    Parameters:
        audio_path (str): Path to the audio file.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for frame analysis.

    Returns:
        mask (np.ndarray): Binary array with speech (1) and non-speech (0) frames.
        snr (np.ndarray): Array of SNR values (dB) for each frame.
        sr (int): Sampling rate of the audio.
    """
    mask, rms, sr = segment_speech(audio_path, n_fft, hop_length)

    # Estimate noise energy from the frames classified as non-speech.
    noise_frames = rms[mask == 0]
    # Avoid division by zero; if no noise frames are found, use a small epsilon.
    if len(noise_frames) == 0:
        noise_energy = np.finfo(float).eps
    else:
        noise_energy = np.mean(noise_frames)
        if noise_energy == 0:
            noise_energy = np.finfo(float).eps

    # Compute SNR in dB for each frame: 10 * log10(signal_energy / noise_energy)
    snr = 10 * np.log10(rms / noise_energy)

    return mask, snr, sr


# Example usage:
if __name__ == "__main__":
    audio_path = "data/Allianz/2025-07-24/ALLIZ_20250601-110200_1263821_TRASLADO_1001053776_133.mp3"  # Replace with your audio file path

    # Obtain segmentation mask and SNR over time
    mask, snr, sr = get_snr(audio_path)

    print("Segmentation mask (1=speech, 0=non-speech):")
    print(mask)
    print("\nSNR over time (dB):")
    print(snr)

    times = librosa.frames_to_time(np.arange(len(snr)), sr=sr, hop_length=32)
    plt.figure(figsize=(10, 4))
    plt.plot(times, snr, label='SNR (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('SNR (dB)')
    plt.title('SNR Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()

    y, _ = librosa.load(audio_path, sr=sr)
    times_rms = librosa.frames_to_time(np.arange(len(mask)), sr=sr, hop_length=32)
    plt.figure(figsize=(10, 4))
    plt.plot(times_rms, mask, label='Speech Mask', color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Speech (1) / Non-Speech (0)')
    plt.title('Speech Activity Segmentation')
    plt.legend()
    plt.tight_layout()
    plt.show()
