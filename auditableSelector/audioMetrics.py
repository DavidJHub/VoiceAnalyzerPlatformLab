import numpy as np

def compute_quality_metrics(active_df, inactive_df):
    """
    Computes summary statistics from the two DataFrames and returns a dictionary of quality metrics.

    Metrics computed (if available in the DataFrame):
      - Mean and standard deviation of RMS in active vs. inactive frames.
      - SNR in dB computed as: 20 * log10(mean_active_rms / mean_inactive_rms)
      - For each additional feature column (if present), the mean and std in both distributions.

    Parameters
    ----------
    active_df : pd.DataFrame
        Frame-level features for active (speech) segments.
    inactive_df : pd.DataFrame
        Frame-level features for inactive (noise/silence) segments.

    Returns
    -------
    metrics : dict
        Dictionary of computed quality metrics.
    """
    metrics = {}
    if len(active_df) > 0 and len(inactive_df) > 0:
        active_rms = active_df["rms"]
        inactive_rms = inactive_df["rms"]
        metrics["active_mean_rms"] = active_rms.mean()
        metrics["active_std_rms"] = active_rms.std()
        metrics["inactive_mean_rms"] = inactive_rms.mean()
        metrics["inactive_std_rms"] = inactive_rms.std()
        snr = 20 * np.log10(active_rms.mean() / (inactive_rms.mean() + 1e-10))
        metrics["SNR_dB"] = snr

    # For any additional features (e.g., MFCCs, spectral centroids) you might have in the DataFrame,
    # compute their mean and standard deviation in the active and inactive sets.
    # Here we exclude common columns used for time and mask.
    excluded = {"time_s", "smoothed_mask", "mask_value", "rms", "rms_norm"}
    feature_cols = [col for col in active_df.columns if col not in excluded]
    for col in feature_cols:
        metrics[f"active_{col}_mean"] = active_df[col].mean()
        metrics[f"active_{col}_std"] = active_df[col].std()
        metrics[f"inactive_{col}_mean"] = inactive_df[col].mean()
        metrics[f"inactive_{col}_std"] = inactive_df[col].std()

    return metrics

def compute_quality_features(y, sr):
    """
    Computes several global audio quality features from the waveform.

    Returns a dictionary with:
      - global_rms: overall root mean square value.
      - global_zcr_mean: mean zero crossing rate.
      - spectral_centroid_mean: mean spectral centroid.
      - spectral_flatness_mean: mean spectral flatness.
      - dynamic_range: difference between 95th and 5th percentiles of the absolute amplitude.

    Parameters
    ----------
    y : np.ndarray
        Audio waveform (mono).
    sr : int
        Sample rate.

    Returns
    -------
    features : dict
        Dictionary of computed global quality features.
    """
    features = {}
    # Overall energy (RMS)
    features["global_rms"] = np.sqrt(np.mean(y ** 2))

    # Zero Crossing Rate (frame-level average)
    zcr = librosa.feature.zero_crossing_rate(y)
    features["global_zcr_mean"] = zcr.mean()

    # Spectral Centroid (mean)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = spec_centroid.mean()

    # Spectral Flatness
    spec_flatness = librosa.feature.spectral_flatness(y=y)
    features["spectral_flatness_mean"] = spec_flatness.mean()

    # Dynamic Range: difference between 95th and 5th percentile of absolute amplitudes
    dynamic_range = np.percentile(np.abs(y), 95) - np.percentile(np.abs(y), 5)
    features["dynamic_range"] = dynamic_range

    return features