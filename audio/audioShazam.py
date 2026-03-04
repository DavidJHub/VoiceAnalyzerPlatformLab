import glob
import json
import os
from typing import Any, Callable, Iterable, List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import librosa, soundfile as sf
from scipy.signal import butter, filtfilt, find_peaks

# ---------- Parámetros ----------
SR = 8000
HP = 1500.0   # Hz
LP = 3800.0   # Hz
PREEMPH = 0.97
NMS_MERGE_SEC = 0.02            # fusionar detecciones separadas < 20 ms
THRESH = 0.7                    # umbral NCC; calibra 0.65–0.8
MIN_DISTANCE_SEC = 0.01         # picos separados al menos 10 ms


def bandpass(y, sr, low, high, order=4):
    ny = 0.5*sr
    b, a = butter(order, [low/ny, high/ny], btype='band')
    return filtfilt(b, a, y)

def preemphasis(y, a=0.97):
    y2 = np.empty_like(y)
    y2[0] = y[0]
    y2[1:] = y[1:] - a*y[:-1]
    return y2

def whiten(y):
    y = y - np.mean(y)
    std = np.std(y) + 1e-10
    return y / std

def preprocess(y, sr=SR):
    y = preemphasis(y, PREEMPH)
    y = bandpass(y, sr, HP, LP)
    return whiten(y)

def ncc_fft(x, h):
    """
    NCC entre x y plantilla h (h ya preprocesada).
    Devuelve arreglo de puntuaciones (len ~ len(x)-len(h)+1).
    Implementación eficiente con FFT y normalización por energía local.
    """
    n = len(x); m = len(h)
    if m > n:
        return np.array([])

    # correlación (x * h_rev)
    H = np.fft.rfft(h[::-1], n=n)
    X = np.fft.rfft(x, n=n)
    corr = np.fft.irfft(X * H, n=n)
    corr = corr[m-1:n]  # parte válida

    # energía local de x (ventana m) con suma acumulada
    x2 = x**2
    csum = np.cumsum(np.concatenate(([0.0], x2)))
    win_energy = csum[m:] - csum[:-m]  # longitud n-m+1

    # normalización
    denom = np.sqrt(win_energy) * (np.linalg.norm(h) + 1e-10)
    ncc = corr / (denom + 1e-10)
    return ncc

def merge_peaks(times, min_sep):
    times = np.array(sorted(times))
    if len(times) == 0: return []
    merged = [times[0]]
    for t in times[1:]:
        if t - merged[-1] >= min_sep:
            merged.append(t)
    return merged

# ---------- Pipeline ----------
def build_template(template_path, sr=SR):
    y, _ = librosa.load(template_path, sr=sr, mono=True)
    y = preprocess(y, sr)
    # recorta silencio por si la plantilla viene con márgenes grandes
    if len(y) > 0:
        thr = 0.1*np.max(np.abs(y))
        nz = np.where(np.abs(y) > thr)[0]
        if nz.size >= 1:
            y = y[max(0, nz[0]-int(0.002*sr)) : min(len(y), nz[-1]+int(0.002*sr))]
    # normaliza a norma 1 para estabilidad
    y = y / (np.linalg.norm(y) + 1e-10)
    return y

def detect_signature(audio_path, template_wave, sr=SR,
                     thresh=THRESH, min_distance_sec=MIN_DISTANCE_SEC):
    x, _ = librosa.load(audio_path, sr=sr, mono=True)
    x = preprocess(x, sr)

    ncc = ncc_fft(x, template_wave)
    if ncc.size == 0:
        return [], ncc
    # picos por umbral y distancia mínima
    distance = max(1, int(min_distance_sec * sr))
    peaks, _ = find_peaks(ncc, height=thresh, distance=distance)
    # tiempos (centro de la ventana)
    m = len(template_wave)
    times = (peaks + m//2) / sr
    # NMS temporal (merge cercano)
    times = merge_peaks(times, NMS_MERGE_SEC)
    return times, ncc


def _adaptive_thresh(ncc, base_thresh=0.94):
    import numpy as np
    med = np.median(ncc)
    mad = np.median(np.abs(ncc - med)) + 1e-8
    th_mad = med + 6.0 * mad
    p995 = np.quantile(ncc, 0.995)
    return max(base_thresh, th_mad, p995 + 0.02)

def _cosine_sim(a, b):
    import numpy as np
    a = a.astype(np.float32, copy=False); b = b.astype(np.float32, copy=False)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return -1.0
    return float(np.dot(a, b) / (na * nb))

def _local_snr_db(x, i, m, pad=3):
    import numpy as np
    L = len(x)
    i0 = max(0, i - pad*m)
    i1 = min(L, i + pad*m)
    # exclude the core segment from the "noise" window
    n1 = x[i0:max(i0, i - m)]
    n2 = x[min(i + 2*m, i1):i1]
    noise = np.concatenate([n1, n2]) if n1.size + n2.size > 0 else x[max(0, i - 10*m):i]
    seg = x[i:i+m]
    rms = lambda z: (np.sqrt(np.mean(z*z)) + 1e-10)
    return 20.0 * np.log10(rms(seg) / rms(noise)) if noise.size else 0.0

def zncc_fft(x: np.ndarray, t: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    ZNCC por FFT (zero-mean normalized cross-correlation).
    Devuelve una curva aprox en [-1, 1] (con cuidado numérico).
    """
    x = x.astype(np.float32, copy=False)
    t = t.astype(np.float32, copy=False)
    m = len(t)
    if m < 2 or len(x) < m:
        return np.array([], dtype=np.float32)

    # --- Template: cero-mean y norma 1 ---
    t0 = t - t.mean()
    t_norm = np.sqrt(np.sum(t0 * t0) + eps)
    t0 = t0 / t_norm

    # --- Sumas móviles de x y x^2 (para mean/std por ventana) ---
    # conv con ones en FFT o en time; aquí en time (suficientemente rápido para 8k y m~8000)
    w = np.ones(m, dtype=np.float32)
    sum_x  = np.convolve(x, w, mode="valid")
    sum_x2 = np.convolve(x * x, w, mode="valid")

    mean_x = sum_x / m
    var_x  = (sum_x2 / m) - (mean_x * mean_x)
    var_x  = np.maximum(var_x, 0.0)
    std_x  = np.sqrt(var_x + eps)

    # --- Correlación cruzada x con template (t0 invertido) por FFT con padding ---
    n = len(x) + m - 1
    N = 1 << (n - 1).bit_length()  # siguiente potencia de 2
    X = np.fft.rfft(x, n=N)
    T = np.fft.rfft(t0[::-1], n=N)
    corr_full = np.fft.irfft(X * T, n=N)[:n].astype(np.float32)

    # para modo "valid": alineación donde template cae completamente dentro de x
    corr_valid = corr_full[m-1 : m-1 + (len(x)-m+1)]

    # --- ZNCC: como t0 es cero-mean y norma 1, basta dividir por std_x*sqrt(m) ---
    # (porque std_x es por muestra; energía ventana ~ std_x*sqrt(m) si zero-mean)
    zncc = corr_valid / (std_x * np.sqrt(m) + eps)

    # opcional: anular donde std muy baja (silencio)
    zncc[std_x < 1e-3] = 0.0

def detect_signature_from_array(
    y: np.ndarray,
    sr: int,
    template_wave: np.ndarray,
    *,
    thresh: float,
    min_distance_sec: float,
    nms_merge_sec: float,
    preprocess_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ncc_fft: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    merge_peaks: Callable[[np.ndarray, float], np.ndarray] = None,
    debug: bool =True
) -> Tuple[List[float], np.ndarray]:
    """
    Detecta la 'firma' en un audio ya cargado.

    Parámetros
    ----------
    y : np.ndarray
        Señal mono.
    sr : int
        Sample rate de y (debe coincidir con template_wave).
    template_wave : np.ndarray
        Plantilla a correlacionar (misma tasa sr).
    thresh : float
        Umbral de altura para los picos de NCC.
    min_distance_sec : float
        Distancia mínima entre picos (en segundos).
    nms_merge_sec : float
        Ventana de 'merge' para NMS temporal (en segundos).
    preprocess_fn : callable | None
        Función opcional de preprocesado sobre y antes de correlacionar.
        Firma esperada: preprocess_fn(y, sr) -> y_proc
    ncc_fft : callable
        Función de correlación (NCC) en frecuencia: ncc_fft(x, template) -> np.ndarray
    merge_peaks : callable
        Función para fusionar picos cercanos: merge_peaks(times, nms_merge_sec) -> times_merged

    Retorna
    -------
    times : list[float]
        Tiempos (s) de los picos detectados (centro de la ventana).
    ncc : np.ndarray
        Curva de correlación normalizada.
    """
    assert ncc_fft is not None, "Debes pasar ncc_fft como función."
    assert merge_peaks is not None, "Debes pasar merge_peaks como función."
    x = y.astype(np.float32, copy=False)
    if preprocess_fn is not None:
        x = preprocess_fn(x, sr)

    ncc = ncc_fft(x, template_wave)
    if ncc is None or ncc.size == 0:
        return [], np.array([], dtype=np.float32)

    distance = max(1, int(min_distance_sec * sr))
    peaks, props = find_peaks(ncc, height=thresh, distance=distance)

    m = len(template_wave)
    times = (peaks + m // 2) / float(sr)
    if times.size:
        times = merge_peaks(times, nms_merge_sec)
    else:
        times = np.array([], dtype=float)
    if debug:
        return list(map(float, times)), ncc
    else:
        return list(map(float, times))