import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import librosa
import soundfile as sf
import tempfile

from gmmActivity import (
    extract_activity_regions_with_features,
    plot_mask_and_energy,
    smooth_mask,
    separate_features_by_activity
)

from audioMetrics import compute_quality_metrics
# ------------------------
# 1) Utility functions
# ------------------------

def measure_saturation(audio, threshold=0.95):
    """Returns % of samples whose abs value >= threshold."""
    if len(audio) == 0:
        return 0.0
    num_saturated = np.sum(np.abs(audio) >= threshold)
    return (num_saturated / len(audio)) * 100


def measure_snr(audio, sr, top_db=60, frame_length=512, hop_length=512):
    """
    Approximate SNR by splitting voice vs. noise via librosa.effects.split
    and computing RMS(voice) / RMS(noise).
    Returns dict with frame_times, frame_rms, voice_mask, and snr_db (scalar).
    """
    rms_each_frame = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]
    frame_times = librosa.frames_to_time(
        np.arange(len(rms_each_frame)), sr=sr, hop_length=hop_length, n_fft=frame_length
    )
    intervals_voice = librosa.effects.split(
        audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )

    voice_mask = np.zeros(len(rms_each_frame), dtype=bool)
    for start, end in intervals_voice:
        start_t = start / sr
        end_t = end / sr
        in_interval = (frame_times >= start_t) & (frame_times < end_t)
        voice_mask[in_interval] = True

    # Compute approximate global SNR
    if np.any(voice_mask):
        rms_voice = np.mean(rms_each_frame[voice_mask])
    else:
        rms_voice = 1e-10
    noise_mask = ~voice_mask
    if np.any(noise_mask):
        rms_noise = np.mean(rms_each_frame[noise_mask])
    else:
        rms_noise = 1e-10
    snr_db = 20.0 * np.log10(rms_voice / (rms_noise + 1e-10))

    return {
        "frame_times": frame_times,
        "frame_rms": rms_each_frame,
        "voice_mask": voice_mask,
        "snr_db": snr_db
    }


def measure_activity(audio, sr, energy_threshold=0.01, frame_length=512, hop_length=512):
    """
    Measures RMS per frame, normalizes, and checks if >= threshold => 'active.'
    Returns dict with frame_times, frame_rms, active_mask, and activity_ratio.
    """
    rms_each_frame = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]
    frame_times = librosa.frames_to_time(
        np.arange(len(rms_each_frame)), sr=sr, hop_length=hop_length, n_fft=frame_length
    )
    max_rms = np.max(rms_each_frame) if len(rms_each_frame) > 0 else 1e-10
    rms_norm = rms_each_frame / (max_rms + 1e-10)
    active_mask = rms_norm >= energy_threshold
    activity_ratio = np.mean(active_mask)
    return {
        "frame_times": frame_times,
        "frame_rms": rms_each_frame,
        "active_mask": active_mask,
        "activity_ratio": activity_ratio
    }


def measure_energy(audio, sr, frame_length=512, hop_length=512):
    """
    Returns per-frame RMS and sum-of-squares as arrays,
    plus frame_times for plotting.
    """
    rms_each_frame = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]
    frame_times = librosa.frames_to_time(
        np.arange(len(rms_each_frame)), sr=sr, hop_length=hop_length, n_fft=frame_length
    )
    # sum-of-squares per frame ~ (rms^2)*frame_length
    energy_sumsq = (rms_each_frame ** 2) * frame_length
    return {
        "frame_times": frame_times,
        "energy_rms": rms_each_frame,
        "energy_sumsq": energy_sumsq
    }


# ------------------------
# 2) Streamlit App
# ------------------------
st.title("Interactive Audio Analysis")

uploaded_file = st.file_uploader("Upload a WAV/FLAC/MP3 file", type=["wav", "flac", "mp3"])

if uploaded_file is not None:
    with st.spinner("Loading & processing..."):
        # Save the uploaded file to a temporary file so that gmmActivity functions (which expect a file path) can work.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # Load audio using soundfile or librosa as fallback
        try:
            audio, sr = sf.read(temp_path)
        except:
            audio, sr = librosa.load(temp_path, sr=None, mono=True)

        # Normalize audio to [-1, 1]
        peak = np.max(np.abs(audio))
        if peak > 1e-9:
            audio = audio / peak

        # Calculate global metrics
        sat_percent  = measure_saturation(audio, 0.95)
        snr_data     = measure_snr(audio, sr)
        act_data     = measure_activity(audio, sr, energy_threshold=0.1)
        eng_data     = measure_energy(audio, sr)

        # Use the imported function from gmmActivity.py to extract the activity mask.
        # extract_activity_regions_with_features returns y, sr, and a sample-level mask.
        y_gmm, sr_gmm, sample_mask = extract_activity_regions_with_features(temp_path, frame_length=512, hop_length=512)

        # Compute frame-level RMS for plotting (using the y from gmm function)
        rms = librosa.feature.rms(y=y_gmm, frame_length=512, hop_length=512)[0]
        frame_times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=512, n_fft=512
        )

        # Convert the sample-level mask (length of y_gmm) to a frame-level mask.
        n_frames = len(rms)
        frame_mask = np.zeros(n_frames, dtype=int)
        for i in range(n_frames):
            start_sample = i * 512
            end_sample = min(start_sample + 512, len(y_gmm))
            # If more than half the samples in the frame are active, mark the frame as active.
            if np.mean(sample_mask[start_sample:end_sample]) > 0.5:
                frame_mask[i] = 1

        duration_sec = len(y_gmm) / sr

        # Build a DataFrame with time, RMS, and GMM mask values.
        df = pd.DataFrame({
            "time_s": frame_times,
            "rms": rms,
            "gmm_voice_mask": frame_mask
        })
        df["smoothed_mask"] = smooth_mask(df["gmm_voice_mask"].values, sr=8000, hop_length=512, smoothing_time=0.5,
                                          threshold=0.5)
        df["rms_norm"] = df["rms"] / (df["rms"].max() + 1e-10)

    # Audio player
    st.audio(uploaded_file, format="audio/wav")

    # Global metrics display
    st.subheader("Global Metrics")
    col1, col2  = st.columns(2)
    col1.metric("Saturation (%)", f"{sat_percent:.2f}")
    col2.metric("Approx. SNR (dB)", f"{snr_data['snr_db']:.2f}")

    # Time range selector: allow user to zoom in on a specific section.
    st.subheader("Time Range Selector")
    time_range = st.slider(
        "Select time range in seconds",
        min_value=0.0,
        max_value=float(duration_sec),
        value=(0.0, float(duration_sec)),
        step=0.1
    )
    df_filtered = df[(df["time_s"] >= time_range[0]) & (df["time_s"] <= time_range[1])]

    # Plot 1: RMS-based Analysis
    st.subheader("RMS-based Analysis")
    df_melted = df_filtered.melt(
        id_vars=["time_s"],
        value_vars=["rms_norm"],
        var_name="Metric",
        value_name="Value"
    )
    chart_rms = (
        alt.Chart(df_melted)
        .mark_line(color="blue")
        .encode(
            x=alt.X("time_s", title="Time (s)"),
            y=alt.Y("Value", title="Normalized RMS", scale=alt.Scale(domain=[0, 1])),
            tooltip=["time_s", "Value"]
        )
        .interactive()
        .properties(width=700, height=300)
    )
    st.altair_chart(chart_rms, use_container_width=True)

    # Plot 2: GMM Voice Mask vs. Time
    st.subheader("GMM Voice Mask")

    mask_plot_data = df_filtered[["time_s", "smoothed_mask"]].copy()
    mask_plot_data["mask_value"] = mask_plot_data["smoothed_mask"]

    chart_mask = (
        alt.Chart(mask_plot_data)
        .mark_area(opacity=0.5, color="red")
        .encode(
            x=alt.X("time_s", title="Time (s)"),
            y=alt.Y("mask_value", title="Voice (1) / Silence (0)", scale=alt.Scale(domain=[0, 1])),
        )
        .interactive()
        .properties(width=700, height=200)
    )
    st.altair_chart(chart_mask, use_container_width=True)
    st.write("GMM frames labeled as voice:", df["gmm_voice_mask"].sum(), "/", len(df))

    active_df, inactive_df=separate_features_by_activity(df_filtered)
    Mtcs = compute_quality_metrics(active_df, inactive_df)

    st.subheader("Activity Metrics")
    col1, col2, col3, col4, col5= st.columns(5)

    col1.metric("active mean rms ", f"{Mtcs['active_mean_rms']:.2f}")
    col2.metric("active std rms ", f"{Mtcs['active_std_rms']:.2f}")
    col3.metric("inactive mean rms ", f"{Mtcs['inactive_mean_rms']:.2f}")
    col4.metric("inactive std rms ", f"{Mtcs['inactive_std_rms']:.2f}")
    col5.metric("SNR ", f"{Mtcs['SNR_dB']:.2f}")

    st.success("Analysis complete!")