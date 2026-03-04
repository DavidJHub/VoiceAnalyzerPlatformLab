
import numpy as np
import scipy.signal as sps
import librosa
import scipy.signal as sps
import soundfile as sf
from audio.vadAsrUtils import _apply_pedalboard_chain, _leveler
from audio.measureActivity import vad_energy_adaptive_array
import librosa

def fade_in(y, sr, ms=30):
    n = int(sr * ms / 1000)
    if n <= 1:
        return y
    w = np.linspace(0.0, 1.0, n, dtype=y.dtype)
    y2 = y.copy()
    y2[:n] *= w
    return y2

def bandlimit_telco(y, sr, hp_hz=150.0, lp_hz=3400.0):
    y = y - float(np.mean(y))
    nyq = 0.5 * sr
    hp = min(max(20.0, hp_hz), nyq*0.95)
    lp = min(max(200.0, lp_hz), nyq*0.95)
    b_hp, a_hp = sps.butter(2, hp, btype="highpass", fs=sr)
    y = sps.filtfilt(b_hp, a_hp, y)
    b_lp, a_lp = sps.butter(4, lp, btype="lowpass", fs=sr)
    y = sps.filtfilt(b_lp, a_lp, y)
    return y.astype(np.float32)

def merge_by_gap(segments, gap_ms=150):
    if not segments:
        return []
    gap = gap_ms / 1000.0
    segs = sorted(segments)
    out = [list(segs[0])]
    for a, b in segs[1:]:
        if a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]

def drop_short(segments, min_ms=200):
    m = min_ms / 1000.0
    return [(a,b) for a,b in segments if (b-a) >= m]

def pad_segments(segments, pad_ms=100, max_t=None):
    p = pad_ms/1000.0
    out=[]
    for a,b in segments:
        a2 = max(0.0, a-p)
        b2 = b+p
        if max_t is not None:
            b2 = min(max_t, b2)
        out.append((a2,b2))
    # re-merge after padding
    return merge_by_gap(out, gap_ms=0)

def preprocess_for_asr_diar_vad(
    y, sr=8000,
    # telco cleaning
    hp_hz=150.0, lp_hz=3400.0,
    target_lufs=-23.0, ceiling_db=-1.0,
    # pedalboard compression/limiting (usa tus helpers)
    comp_thresh_db=-26.0, comp_ratio=2.5, comp_attack_ms=7.0, comp_release_ms=220.0,
    makeup_gain_db=0.0, limit_thresh_db=-1.5, limit_release_ms=60.0,
    # VAD energy params (ya los tienes)
    vad_params=None,
    # merge params
    gap_ms=150, min_seg_ms=200, pad_ms=100,
    # output
    out_sr=16000,
):
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32, copy=False)

    # A) limpieza estable
    y1 = fade_in(y, sr, ms=30)
    y2 = bandlimit_telco(y1, sr, hp_hz=hp_hz, lp_hz=lp_hz)
    y3 = _leveler(y2, sr, mode="lufs", target_lufs=target_lufs, limiter_ceiling_db=ceiling_db)

    # comp+limiter (sin gate)
    y4 = _apply_pedalboard_chain(
        y3, sr,
        hp_hz=hp_hz, lp_hz=lp_hz,
        gate_thresh_db=None, gate_ratio=1.0,  # desactivado
        gate_attack_ms=0.0, gate_release_ms=0.0,
        comp_thresh_db=comp_thresh_db, comp_ratio=comp_ratio,
        comp_attack_ms=comp_attack_ms, comp_release_ms=comp_release_ms,
        makeup_gain_db=makeup_gain_db,
        limit_thresh_db=limit_thresh_db, limit_release_ms=limit_release_ms
    )

    y_clean8k = np.clip(y4, -1.0, 1.0).astype(np.float32)

    # B) audio para ASR/diar (16k)
    y_asr = y_clean8k
    sr_asr = sr
    if out_sr is not None and out_sr != sr:
        y_asr = librosa.resample(y_asr, orig_sr=sr, target_sr=out_sr).astype(np.float32)
        sr_asr = int(out_sr)

    # C) VAD (sobre limpio 8k o 16k: yo prefiero 8k por consistencia temporal con tu VAD)
    if vad_params is None:
        vad_params = dict(
            win_ms=25, hop_ms=10,
            target_rms_dbfs=-24, p_percentile=35, delta_db=4,
            noise_floor_pct=10, hysteresis_db=2,
            local_win_ms=750, local_mix=0.6,
            min_run_frames=6, smooth_ms=350, pad_ms=0
        )

    seg_energy = vad_energy_adaptive_array(y_clean8k, sr=sr, returns="segments", **vad_params)

    # WebRTC VAD (opcional, para fusionar)
    # Si quieres usarlo: reaprovecha tu bloque webrtc (pero afuera, más limpio).
    # Por simplicidad aquí asumimos solo energy; te recomiendo OR con WebRTC si tienes muchos FN.

    # Post-proc segmentos: merge gaps + drop short + pad
    seg = merge_by_gap(seg_energy, gap_ms=gap_ms)
    seg = drop_short(seg, min_ms=min_seg_ms)
    seg = pad_segments(seg, pad_ms=pad_ms, max_t=len(y_clean8k)/sr)

    # D) métricas
    peak_dbfs = 20*np.log10(np.max(np.abs(y_clean8k))+1e-12)
    rms = np.sqrt(np.mean(y_clean8k*y_clean8k)+1e-12)
    rms_dbfs = 20*np.log10(rms+1e-12)
    clip_rate = float(np.mean(np.abs(y_clean8k) > 0.99))

    # noise floor y snr proxy (RMS por frame en banda 300-3400)
    frame_len = int(0.025*sr)
    hop = int(0.010*sr)
    rms_frames = librosa.feature.rms(y=y_clean8k, frame_length=frame_len, hop_length=hop)[0]
    rms_db = 20*np.log10(rms_frames + 1e-12)
    noise_floor_db = float(np.percentile(rms_db, 10))
    speech_floor_db = float(np.percentile(rms_db, 90))
    snr_proxy = float(speech_floor_db - noise_floor_db)

    total_t = len(y_clean8k)/sr
    speech_t = float(sum((b-a) for a,b in seg))
    speech_ratio = float(speech_t / total_t) if total_t > 0 else 0.0

    meta = dict(
        sr_in=sr, sr_asr=sr_asr,
        peak_dbfs=float(peak_dbfs),
        rms_dbfs=float(rms_dbfs),
        crest_db=float(peak_dbfs - rms_dbfs),
        clip_rate=clip_rate,
        noise_floor_db=noise_floor_db,
        snr_proxy=snr_proxy,
        speech_ratio=speech_ratio,
        n_segments=len(seg),
    )

    return {
        "y_clean": y_clean8k,     # para VAD/segmentación/quality
        "sr_clean": sr,
        "y_asr": y_asr,           # para ASR/diarización
        "sr_asr": sr_asr,
        "segments": seg,          # segmentos finales
        "meta": meta
    }
