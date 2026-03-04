import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np


def extract_activity_regions_with_features(
    audio_path,
    n_components=2,
    frame_length=512,
    hop_length=512,
    n_mfcc=13
):
    """
    Loads an audio file, extracts multiple acoustic features, then fits a GMM
    to classify frames as 'active' (speech) or 'inactive' (non-speech).
    Returns a sample-level mask (1 = activity, 0 = no activity) and sample rate.
    """
    y, sr = librosa.load(audio_path, sr=8000, mono=True)

    # Compute various features
    mfcc_feats = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                      hop_length=hop_length,
                                      n_fft=frame_length)
    chroma_feats = librosa.feature.chroma_stft(y=y, sr=sr,
                                               hop_length=hop_length,
                                               n_fft=frame_length)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                      hop_length=hop_length,
                                                      n_fft=frame_length)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                      hop_length=hop_length,
                                                      n_fft=frame_length,
                                                      n_bands=6,
                                                      fmin=librosa.note_to_hz('C2'))

    # Transpose => frames as rows
    mfcc_feats = mfcc_feats.T          # shape (n_frames, n_mfcc)
    chroma_feats = chroma_feats.T      # shape (n_frames, 12)
    spec_centroid = spec_centroid.T    # shape (n_frames, 1)
    spec_contrast = spec_contrast.T    # shape (n_frames, 7)

    # Align length by min frames
    n_frames = min(
        mfcc_feats.shape[0],
        chroma_feats.shape[0],
        spec_centroid.shape[0],
        spec_contrast.shape[0]
    )
    mfcc_feats = mfcc_feats[:n_frames, :]
    chroma_feats = chroma_feats[:n_frames, :]
    spec_centroid = spec_centroid[:n_frames, :]
    spec_contrast = spec_contrast[:n_frames, :]

    # Stack horizontally
    X = np.hstack([mfcc_feats, chroma_feats, spec_centroid, spec_contrast])

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)

    # Pick cluster with larger sum-of-means as "voice"
    cluster_means = gmm.means_
    sum_of_means = np.sum(cluster_means, axis=1)
    active_cluster = np.argmax(sum_of_means)

    # Frame-level mask (True/False)
    frame_mask = (labels == active_cluster)

    # Convert frame-level mask to sample-level
    mask = np.zeros(len(y), dtype=int)
    for i in range(n_frames):
        start_sample = i * hop_length
        end_sample = min(start_sample + frame_length, len(y))
        if frame_mask[i]:
            mask[start_sample:end_sample] = 1

    return y, sr, mask


def plot_mask_and_energy(audio_path,
                         frame_length=512,
                         hop_length=256):
    """
    Loads audio, computes:
      - The GMM-based speech activity mask.
      - The per-frame RMS (for energy).
    Then plots both in a two-subplot figure:
      1) normalized RMS vs. time,
      2) the mask (0/1) vs. time.
    """
    # 1) Get mask
    y, sr, mask = extract_activity_regions_with_features(
        audio_path,
        frame_length=frame_length,
        hop_length=hop_length
    )
    # 2) Compute RMS (frame-level)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    # Frame times
    frame_times = librosa.frames_to_time(
        frames=np.arange(len(rms)), sr=sr,
        hop_length=hop_length, n_fft=frame_length
    )
    # Normalize RMS for plotting
    rms_norm = rms / (np.max(rms) + 1e-10)

    # 3) Build a frame-level mask from the sample-level mask
    #    (so we can align it with 'frame_times').
    n_frames = len(rms_norm)
    frame_mask = np.zeros(n_frames, dtype=int)
    for i in range(n_frames):
        start_sample = i * hop_length
        end_sample = min(start_sample + frame_length, len(y))
        # If more than half the samples in that frame are mask=1,
        # consider the whole frame "active."
        if np.mean(mask[start_sample:end_sample]) > 0.5:
            frame_mask[i] = 1

    # 4) Plot results
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6))

    # Top subplot: Normalized RMS
    axs[0].plot(frame_times, rms_norm, label='Normalized RMS')
    axs[0].set_ylabel("Amplitude (norm)")
    axs[0].set_title("Energy (RMS) over time")
    axs[0].legend(loc='upper right')

    # Bottom subplot: Mask (0/1)
    axs[1].step(frame_times, frame_mask, where='post', label='Mask (GMM)')
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].set_ylabel("Activity Mask")
    axs[1].set_xlabel("Time (s)")
    axs[1].legend(loc='upper right')

    plt.suptitle(f"Activity Regions for: {audio_path}")
    plt.show()



def separate_features_by_activity(df, mask_column="smoothed_mask"):
    """
    Separates a DataFrame with frame-level features into two DataFrames:
      - active_df: frames with mask==1 (speech/activity)
      - inactive_df: frames with mask==0 (silence/noise)
    """
    active_df = df[df[mask_column] == 1].copy()
    inactive_df = df[df[mask_column] == 0].copy()
    return active_df, inactive_df




def smooth_mask(mask, sr, hop_length, smoothing_time=0.3, threshold=0.3):
    """
    Smooths a binary mask (0/1) using a moving average filter.

    Parameters:
      mask : np.ndarray
          A 1D binary array (e.g., frame-level voice mask).
      sr : int
          Sample rate of the audio.
      hop_length : int
          Hop length used to compute the mask frames.
      smoothing_time : float
          Window length in seconds to smooth the mask (default 0.5 sec).
      threshold : float
          Threshold to binarize the smoothed signal (default 0.5).

    Returns:
      smoothed_mask : np.ndarray
          A 1D binary array after smoothing.
    """
    # Compute window size in number of frames
    window_size = int(np.round(smoothing_time * sr / hop_length))
    if window_size < 1:
        window_size = 1

    # Uniform moving average: convolve with a kernel of ones, then normalize.
    kernel = np.ones(window_size) / window_size
    smooth = np.convolve(mask, kernel, mode='same')

    # Binarize the smoothed result based on the threshold.
    smoothed_mask = (smooth >= threshold).astype(int)
    return smoothed_mask

# ----------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    audio_file = "AUDIOS/BANPOPC_20250325-112830_2184002_1742920110_1001298682_3122061782-all.mp3"
    plot_mask_and_energy(audio_file)
