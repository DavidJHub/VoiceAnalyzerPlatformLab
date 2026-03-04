import numpy as np

import librosa.display
import matplotlib.pyplot as plt


from auditableSelector.audioLoader import get_mel_features, measure_spectral_flatness
from auditableSelector.speechSegmentation import get_snr


def assess_audio_quality(mel_spec_db, flatness, snr, mask,
                         mel_threshold=-40, snr_threshold=10, flatness_threshold=0.2):
    """
    Assesses audio quality based on three metrics computed over conversation (speech) sections only:
      - Average mel energy (in dB)
      - Average SNR (in dB)
      - Average spectral flatness

    Parameters:
        mel_spec_db (np.ndarray): Mel spectrogram in dB (shape: [n_mels, n_frames]).
        flatness (np.ndarray): 1D array of spectral flatness values (n_frames,).
        snr (np.ndarray): 1D array of SNR values (n_frames,).
        mask (np.ndarray): Binary speech mask (1: speech, 0: non-speech) for each frame.
        mel_threshold (float): Threshold for average mel energy.
        snr_threshold (float): Threshold for average SNR.
        flatness_threshold (float): Threshold for average spectral flatness.

    Returns:
        is_bad (bool): True if audio is deemed bad quality.
        quality_flags (list): List of reasons why the audio was flagged.
        metrics (dict): Dictionary of computed average metrics over conversation sections.
    """
    # Average mel energy per frame (average over mel bins)
    mel_per_frame = np.mean(mel_spec_db, axis=0)  # shape: (n_frames,)
    conversation_mel = mel_per_frame[mask == 1]
    avg_mel = np.mean(conversation_mel) if len(conversation_mel) > 0 else np.min(mel_per_frame)

    # Average spectral flatness computed over speech frames
    conversation_flatness = flatness[mask == 1]
    avg_flatness = np.mean(conversation_flatness) if len(conversation_flatness) > 0 else np.max(flatness)

    # Average SNR over conversation frames (ignore non-finite values)
    conversation_snr = snr[mask == 1]
    valid_snr = conversation_snr[np.isfinite(conversation_snr)]
    avg_snr = np.mean(valid_snr) if len(valid_snr) > 0 else np.min(snr)

    quality_flags = []
    if avg_snr < snr_threshold:
        quality_flags.append(f"Low SNR (avg: {avg_snr:.2f} dB < {snr_threshold} dB)")
    if avg_flatness > flatness_threshold:
        quality_flags.append(f"High spectral flatness (avg: {avg_flatness:.2f} > {flatness_threshold})")
    if avg_mel < mel_threshold:
        quality_flags.append(f"Low average mel energy (avg: {avg_mel:.2f} dB < {mel_threshold} dB)")

    is_bad = len(quality_flags) > 1
    metrics = {"avg_mel": avg_mel, "avg_snr": avg_snr, "avg_flatness": avg_flatness}
    return is_bad, quality_flags, metrics


def classify_audio_quality(audio_path, n_fft=512, hop_length=512, n_mels=128,
                           mel_threshold=-45, snr_threshold=5, flatness_threshold=0.1):
    """
    Given an audio file, computes the mel spectrogram, spectral flatness, and SNR over conversation (speech)
    segments, then assesses the quality. Returns 1 if the audio is classified as bad quality,
    else returns 0.

    Parameters:
        audio_path (str): Path to the audio file.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for frame analysis.
        n_mels (int): Number of mel bands.
        mel_threshold (float): Mel energy threshold (dB) for quality assessment.
        snr_threshold (float): SNR threshold (dB) for quality assessment.
        flatness_threshold (float): Spectral flatness threshold for quality assessment.

    Returns:
        int: 1 if bad quality, 0 otherwise.
    """
    try:
        mel_spec_db, sr = get_mel_features(audio_path, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

        flatness, _ = measure_spectral_flatness(audio_path, n_fft=n_fft, hop_length=hop_length)

        mask, snr, _ = get_snr(audio_path, n_fft=n_fft, hop_length=hop_length)

        is_bad, quality_flags, metrics = assess_audio_quality(mel_spec_db, flatness, snr, mask,
                                                              mel_threshold=mel_threshold,
                                                              snr_threshold=snr_threshold,
                                                              flatness_threshold=flatness_threshold)


        return 1 if is_bad else 0
    except Exception as e:
        return 0


if __name__ == "__main__":
    audio_path = "AUDIOS/BANPOPC_20250325-131008_2177893_1742926208_1034659369_3133140360-all.mp3"  # Replace with your audio file path

    n_fft = 512
    hop_length = 512
    n_mels = 128

    mel_spec_db, sr = get_mel_features(audio_path, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

    flatness, _ = measure_spectral_flatness(audio_path, n_fft=n_fft, hop_length=hop_length)

    mask, snr, _ = get_snr(audio_path, n_fft=n_fft, hop_length=hop_length)

    is_bad, quality_flags, metrics = assess_audio_quality(mel_spec_db, flatness, snr, mask,
                                                          mel_threshold=-45, snr_threshold=5, flatness_threshold=0.1)

    print("Audio Quality  (over conversation sections):")
    print("--------------------------------------------")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")
    if is_bad:
        print("Flagged as BAD quality due to:")
        for flag in quality_flags:
            print(" -", flag)
    else:
        print("Audio quality is GOOD.")

    # (Optional) Visualization of the speech mask over mel energy:
    times = librosa.frames_to_time(np.arange(mel_spec_db.shape[1]), sr=sr, hop_length=hop_length)

    mel_per_frame = np.mean(mel_spec_db, axis=0)


    plt.figure(figsize=(10, 4))
    plt.plot(times, mel_per_frame, label='Mel Energy (dB)', alpha=0.7)
    plt.plot(times, mask * np.max(mel_per_frame), label='Speech Mask', color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (dB)')
    plt.title('Mel Energy with Speech Segmentation Mask')
    plt.legend()
    plt.tight_layout()
    plt.show()