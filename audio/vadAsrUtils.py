import numpy as np
import pyloudnorm as pyln
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, NoiseGate, Compressor, Limiter, Gain
import scipy.signal as sps
import soundfile as sf


def align_mask_to_length(mask: np.ndarray, target_len: int, threshold: float = 0.5) -> np.ndarray:
    """Resample/pad a binary mask so it matches ``target_len``.

    Parameters
    ----------
    mask:
        Vector con valores binarios o probabilidades.
    target_len:
        Largo deseado del vector final.
    threshold:1
        Umbral usado al volver a binarizar si ``mask`` no es binario.
    """

    target_len = int(max(0, target_len))
    if target_len == 0:
        return np.zeros(0, dtype=np.uint8)

    mask = np.asarray(mask)
    if mask.ndim == 0:
        mask = mask.reshape(1)

    if mask.size == 0:
        return np.zeros(target_len, dtype=np.uint8)

    if mask.size == target_len:
        return mask.astype(np.uint8)

    x_old = np.linspace(0, target_len - 1, num=mask.size)
    x_new = np.arange(target_len)
    interp = np.interp(x_new, x_old, mask.astype(float))
    return (interp >= float(threshold)).astype(np.uint8)

def _db_to_lin(db): return 10.0**(db/20.0)


def _leveler(y, sr, mode="lufs", target_lufs=-18.0, limiter_ceiling_db=-1.0):
    """
    Normaliza sonoridad global y aplica un limitador simple de pico.
    mode: 'lufs' (requiere pyloudnorm) o 'peak' (normaliza pico).
    """
    y = y.astype(float)
    normalized = False
    if mode == "lufs":
            meter = pyln.Meter(sr)
            loud = meter.integrated_loudness(y)
            y = pyln.normalize.loudness(y, loud, target_lufs)
            normalized = True
    if not normalized:
        # Fallback: normaliza a pico -1 dBFS (o al ceiling)
        # (si quieres RMS target, puedes añadirlo aquí)
        peak = np.max(np.abs(y)) + 1e-12
        tgt = _db_to_lin(limiter_ceiling_db)  # e.g., -1 dBFS
        if peak > 0:
            y = y * (tgt / peak)

    # Limitador suave de seguridad al ceiling (por si algún transitorio pasó)
    peak = np.max(np.abs(y)) + 1e-12
    ceiling = _db_to_lin(limiter_ceiling_db)
    if peak > ceiling:
        y = y * (ceiling / peak)
    return np.clip(y, -1.0, 1.0)

def _apply_pedalboard_chain(y, sr,
                            hp_hz=120.0, lp_hz=6000.0,
                            gate_thresh_db=None, gate_ratio=3.0, gate_attack_ms=8.0, gate_release_ms=120.0,
                            comp_thresh_db=-28.0, comp_ratio=5.0, comp_attack_ms=10.0, comp_release_ms=100.0,
                            makeup_gain_db=0.0, limit_thresh_db=-2.0, limit_release_ms=60.0):



    # asegurar frecuencias válidas
    nyq = 0.5 * sr
    hp_hz = max(20.0, min(hp_hz, nyq * 0.95))
    lp_hz = None if lp_hz is None else max(200.0, min(lp_hz, nyq * 0.95))

    chain = []
    chain.append(HighpassFilter(cutoff_frequency_hz=hp_hz))
    if lp_hz is not None:
        chain.append(LowpassFilter(cutoff_frequency_hz=lp_hz))
    # NoiseGate (si gate_thresh_db es None, lo autocalibramos fuera)
    if gate_thresh_db is not None:
        chain.append(NoiseGate(threshold_db=gate_thresh_db, ratio=gate_ratio,
                               attack_ms=gate_attack_ms, release_ms=gate_release_ms))
    chain += [
        Compressor(threshold_db=comp_thresh_db, ratio=comp_ratio,
                   attack_ms=comp_attack_ms, release_ms=comp_release_ms),
        Gain(gain_db=makeup_gain_db),
        Limiter(threshold_db=limit_thresh_db, release_ms=limit_release_ms),
    ]
    pb = Pedalboard(chain)
    y32 = y.astype(np.float32, copy=False)
    return pb(y32, sr)


