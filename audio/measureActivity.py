import librosa, numpy as np
from scipy.ndimage import uniform_filter1d
import librosa, librosa.display, numpy as np

try:
    from .constants import DEFAULT_HOP_LENGTH, DEFAULT_N_FFT
except ImportError:  # pragma: no cover - compatibility with alternate package names
    try:
        from audio.constants import DEFAULT_HOP_LENGTH, DEFAULT_N_FFT  # type: ignore
    except ImportError:  # pragma: no cover - final fallback for script usage
        from constants import DEFAULT_HOP_LENGTH, DEFAULT_N_FFT  # type: ignore



def merge_by_gap(segments, gap_ms=120):
    if not segments:
        return []
    gap = gap_ms / 1000.0
    segs = sorted(segments)
    merged = [list(segs[0])]
    for a, b in segs[1:]:
        if a <= merged[-1][1] + gap:   # solape o gap pequeño
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [tuple(x) for x in merged]

def vad_energy_adaptive(path, win_ms=25, hop_ms=10, p_percentile=60, delta_db=2,
                        min_run_frames=3, smooth_ms=500,returns="segments"):
    y, sr = librosa.load(path, sr=8_000, mono=True)
    hop = DEFAULT_HOP_LENGTH
    win = DEFAULT_N_FFT
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    thr = np.percentile(rms_db, p_percentile) + delta_db
    act_mask = rms_db >= thr

    from itertools import groupby
    act_mask_enforced = np.zeros_like(act_mask, dtype=bool)
    idx = 0
    for val, grp in groupby(act_mask):
        length = len(list(grp))
        if val and length >= min_run_frames:
            act_mask_enforced[idx:idx+length] = True
        idx += length

    # Smooth with moving average window (half-second)
    smooth_frames = int(smooth_ms / hop_ms)
    prob = uniform_filter1d(act_mask_enforced.astype(float), size=smooth_frames)
    final_mask = prob >= 0.5          # 0.5 ↔ mayoria

    # Convert to segments
    segments = []
    t = np.arange(len(final_mask)) * hop / sr
    current = None
    for i, flag in enumerate(final_mask):
        if flag and current is None:
            current = [t[i], None]
        elif not flag and current is not None:
            current[1] = t[i]
            segments.append(tuple(current))
            current = None
    if current is not None:
        segments.append((current[0], t[-1]))
    if returns == "mask":
        return final_mask
    elif returns == "segments":
        return segments





from itertools import groupby
from typing import List, Tuple, Union

def vad_energy_adaptive_array(
    y: np.ndarray,
    sr: int,
    win_ms: int = 25,
    hop_ms: int = 5,
    p_percentile: float = 35,
    delta_db: float = 3,
    min_run_frames: int = 3,
    smooth_ms: int = 200,
    returns: str = "segments",
    # --- mejoras de sensibilidad / robustez ---
    normalize: bool = True,
    target_rms_dbfs: float = -23.0,   # objetivo de nivel global (dBFS)
    noise_floor_pct: float = 10.0,    # percentil para estimar piso de ruido (dB)
    hysteresis_db: float = 1.5,       # histéresis: thr_on - thr_off
    local_win_ms: int = 500,         # mediana local de dB para umbral local
    local_mix: float = 0.7,           # mezcla entre umbral global y local (0..1)
    pad_ms: int = 110                  # relleno a los segmentos detectados
):
    """
    VAD (Voice Activity Detection) adaptativo basado en energía con normalización global,
    piso de ruido, histéresis y umbral local rodante.

    Parámetros nuevos clave
    -----------------------
    normalize : bool
        Si True, normaliza el nivel global a target_rms_dbfs (mejora sensibilidad).
    target_rms_dbfs : float
        Nivel RMS objetivo en dBFS tras normalizar (p.ej. -25 dBFS).
    noise_floor_pct : float
        Percentil (bajo) para estimar piso de ruido en dB.
    hysteresis_db : float
        Diferencia entre umbral de encendido (thr_on) y apagado (thr_off).
    local_win_ms : int
        Ventana para mediana local de dB, para umbral local adaptativo.
    local_mix : float
        0=solo umbral global; 1=solo umbral local; intermedio mezcla ambos.
    pad_ms : int
        Relleno que se añade a cada segmento al inicio y al fin.
    """
    eps = 1e-10

    # --- 0) Normalización global de nivel (opcional) ---
    if normalize:
        # RMS global actual en dBFS
        rms_global = np.sqrt(np.mean(y**2) + eps)
        rms_dbfs = 20 * np.log10(rms_global + eps)
        gain_db = target_rms_dbfs - rms_dbfs
        gain = 10**(gain_db / 20.0)
        y = y * gain

    # --- 1) Configuración de frames ---
    hop = int(sr * hop_ms / 1000)
    win = int(sr * win_ms / 1000)
    if hop <= 0: hop = 1
    if win < 2: win = 2

    # --- 2) Energía RMS por frame en dB ---
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True)[0]
    rms_db = librosa.amplitude_to_db(rms + eps, ref=np.max)

    # --- 3) Umbral global dinámico + piso de ruido ---
    #     - umbral base por percentil alto
    thr_global = np.percentile(rms_db, p_percentile) + delta_db
    #     - piso de ruido por percentil bajo (para evitar que el thr quede demasiado alto)
    noise_floor = np.percentile(rms_db, noise_floor_pct)
    #     - no dejes que el umbral global baje por debajo del piso + margen pequeño
    thr_global = np.maximum(thr_global, noise_floor + 1.0)

    # --- 4) Umbral local: mediana rodante en dB ---
    local_win_frames = max(1, int(local_win_ms / hop_ms))
    # mediana aproximada: usa filtro de media sobre dB suavizados como aproximación estable
    # (si necesitas una mediana verdadera, puedes implementar una ventana deslizante con np.median por bloques)
    rms_db_smooth = uniform_filter1d(rms_db, size=local_win_frames, mode='nearest')
    thr_local = rms_db_smooth + delta_db
    thr_local = np.maximum(thr_local, noise_floor + 1.0)

    # Mezcla global-local
    thr_mix = (1 - local_mix) * thr_global + local_mix * thr_local

    # --- 5) Histéresis: umbral de encendido/apagado ---
    thr_on = thr_mix
    thr_off = thr_mix - hysteresis_db

    # Estado con histéresis
    active = np.zeros_like(rms_db, dtype=bool)
    is_on = False
    for i, e in enumerate(rms_db):
        if not is_on:
            if e >= thr_on[i]:
                is_on = True
        else:
            if e < thr_off[i]:
                is_on = False
        active[i] = is_on

    # --- 6) Enforce runs mínimos ---
    act_mask_enforced = np.zeros_like(active, dtype=bool)
    idx = 0
    for val, grp in groupby(active):
        length = len(list(grp))
        if val and length >= min_run_frames:
            act_mask_enforced[idx:idx+length] = True
        idx += length

    # --- 7) Suavizado final (probabilidad por filtro uniforme) ---
    smooth_frames = max(1, int(smooth_ms / hop_ms))
    prob = uniform_filter1d(act_mask_enforced.astype(float), size=smooth_frames, mode='nearest')
    final_mask = prob >= 0.5

    if returns == "mask":
        return final_mask

    # --- 8) A segmentos (con padding) ---
    t = np.arange(len(final_mask)) * hop / sr
    segments = []
    current = None
    for i, flag in enumerate(final_mask):
        if flag and current is None:
            current = [t[i], None]
        elif not flag and current is not None:
            current[1] = t[i]
            segments.append(tuple(current))
            current = None
    if current is not None:
        segments.append((current[0], t[-1] + hop / sr))

    # padding
    pad = pad_ms / 1000.0
    # fusionar solapes tras padding
    padded = []
    for (a, b) in segments:
        a = max(0.0, a - pad)
        b = b + pad
        if not padded:
            padded.append([a, b])
        else:
            if a <= padded[-1][1]:
                padded[-1][1] = max(padded[-1][1], b)
            else:
                padded.append([a, b])
    return [tuple(x) for x in padded]

if __name__ == "__main__":
    audio_test_path="data/Allianz/2025-07-24/ALLIZ_20250601-210200_1263943_CERRAJERIA_1007595176_129.mp3"
    segments = vad_energy_adaptive(audio_test_path, win_ms=25, hop_ms=50, p_percentile=10, delta_db=6,
                                   min_run_frames=3, smooth_ms=3_000)
        # 2. Graficar usando esos segmentos
    print(segments)  

    y, sr = librosa.load(audio_test_path, sr=None, mono=True)   


