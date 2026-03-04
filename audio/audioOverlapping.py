import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d
from typing import List, Tuple, Optional, Dict, Any

def segments_to_frame_mask(segments, sr, hop_length, n_frames):
    """segments: [(start,end),...] en segundos → mask bool de largo n_frames."""
    mask = np.zeros(n_frames, dtype=bool)
    for a, b in segments:
        if b <= a:
            continue
        i0 = int(np.floor(a * sr / hop_length))
        i1 = int(np.ceil (b * sr / hop_length))
        i0 = max(0, i0)
        i1 = min(n_frames, i1)
        if i1 > i0:
            mask[i0:i1] = True
    return mask



def _robust_z(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    med = np.median(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    if iqr < eps:
        iqr = np.std(x) + eps
    return (x - med) / (iqr + eps)

def _runs_to_segments(mask: np.ndarray, times: np.ndarray, min_run_frames: int) -> List[Tuple[float, float]]:
    if mask.size == 0:
        return []
    mask = mask.astype(bool)
    # runs
    padded = np.concatenate([[False], mask, [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends   = np.where(d == -1)[0]
    segs = []
    for s, e in zip(starts, ends):
        if (e - s) >= min_run_frames:
            a = float(times[s])
            b = float(times[e-1])  # último frame dentro del run
            # sumar 1 hop para cerrar más “realista”
            segs.append((a, float(times[min(e, len(times)-1)] if e < len(times) else times[-1])))
    return segs

def _merge_by_gap(segs: List[Tuple[float,float]], gap_ms: float = 150.0) -> List[Tuple[float,float]]:
    if not segs:
        return []
    gap = gap_ms / 1000.0
    segs = sorted(segs)
    out = [list(segs[0])]
    for a, b in segs[1:]:
        if a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]

def detect_overlap_clear_fast_ranked(
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int = 256,
    hop_length: int = 80,
    fmax: float = 3400.0,
    vad_segments: Optional[List[Tuple[float,float]]] = None,
    smooth_ms: float = 250.0,
    min_run_ms: float = 160.0,
    merge_gap_ms: float = 120.0,
    pad_ms: float = 80.0,
    k_iqr: float = 2.0,
    min_abs_thr: float = 0.8,
    weights: Optional[Dict[str,float]] = None,
    # ranking
    rank_by: str = "p95",   # "p95" | "mean" | "max"
    ascending: bool = False,
    return_debug: bool = False
) -> Any:
    """
    Retorna segmentos con overlap 'claro' pero ORDENADOS por fuerza de overlap (score).
    Cada elemento incluye start/end/duration y scores (mean, p95, max).
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or y.size < n_fft:
        out = []
        return (out, {}) if return_debug else out

    # STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann", center=True)
    mag = np.abs(S)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    keep = freqs <= float(min(fmax, sr/2))
    mag = mag[keep, :]
    freqs = freqs[keep]

    n_frames = mag.shape[1]
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
    hop_sec = hop_length / float(sr)

    # VAD gating
    if vad_segments is not None:
        vad_mask = segments_to_frame_mask(vad_segments, sr, hop_length, n_frames)
    else:
        vad_mask = np.ones(n_frames, dtype=bool)

    eps = 1e-10

    # Features
    thr = np.percentile(mag, 75, axis=0)
    richness = (mag > thr[None, :]).sum(axis=0).astype(np.float32)

    log_mag = np.log(mag + eps)
    flatness = np.exp(log_mag.mean(axis=0)) / (mag.mean(axis=0) + eps)

    mag_sum = mag.sum(axis=0) + eps
    mu = (freqs[:, None] * mag).sum(axis=0) / mag_sum
    spread = np.sqrt((((freqs[:, None] - mu[None, :])**2) * mag).sum(axis=0) / mag_sum).astype(np.float32)

    crest = (mag.max(axis=0) / (mag.mean(axis=0) + eps)).astype(np.float32)

    z_rich  = _robust_z(richness)
    z_flat  = _robust_z(flatness)
    z_spre  = _robust_z(spread)
    z_crest = _robust_z(crest)

    if weights is None:
        weights = {"rich": 1.3, "flat": 1.0, "spread": 1.0, "inv_crest": 0.8}

    score = (
        weights["rich"] * z_rich +
        weights["flat"] * z_flat +
        weights["spread"] * z_spre +
        weights["inv_crest"] * (-z_crest)
    ).astype(np.float32)

    score = np.where(vad_mask, score, 0.0)

    # Suavizado
    smooth_frames = max(1, int(round((smooth_ms/1000.0) / hop_sec)))
    score_s = uniform_filter1d(score, size=smooth_frames, mode="nearest")

    vv = score_s[vad_mask]
    if vv.size == 0:
        out = []
        return (out, {}) if return_debug else out

    med = float(np.median(vv))
    iqr = float(np.percentile(vv, 75) - np.percentile(vv, 25))
    thr_s = max(min_abs_thr, med + k_iqr * max(iqr, 1e-6))

    mask = score_s >= thr_s

    # Runs mínimos
    min_run_frames = max(1, int(round((min_run_ms/1000.0) / hop_sec)))

    padded = np.concatenate([[False], mask, [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends   = np.where(d == -1)[0]

    # Segmentos + scoring por segmento
    segs_scored = []
    for s, e in zip(starts, ends):
        if (e - s) < min_run_frames:
            continue
        a = float(times[s])
        b = float(times[e-1] + hop_sec)
        vals = score_s[s:e]
        seg = {
            "start": a,
            "end": b,
            "duration": float(b - a),
            "score_mean": float(np.mean(vals)),
            "score_p95": float(np.percentile(vals, 95)),
            "score_max": float(np.max(vals)),
        }
        segs_scored.append(seg)

    # Merge gaps (en tiempo) + pad, pero conservando scores:
    # (para speed: merge solo por tiempo y recomputa score en merged usando mask/score_s)
    # Aquí lo haremos simple: primero merge tiempos, luego recalculamos score en merged.
    merged_times = _merge_by_gap([(s["start"], s["end"]) for s in segs_scored], gap_ms=merge_gap_ms)
    p = pad_ms / 1000.0
    merged_times = [(max(0.0, a-p), min(times[-1] + hop_sec, b+p)) for a,b in merged_times]
    merged_times = _merge_by_gap(merged_times, gap_ms=0.0)

    # Re-score de merged segments usando score_s
    final = []
    for a,b in merged_times:
        i0 = int(np.floor(a / hop_sec))
        i1 = int(np.ceil (b / hop_sec))
        i0 = max(0, i0); i1 = min(len(score_s), i1)
        if i1 <= i0:
            continue
        vals = score_s[i0:i1]
        final.append({
            "start": float(a),
            "end": float(b),
            "duration": float(b-a),
            "score_mean": float(np.mean(vals)),
            "score_p95": float(np.percentile(vals, 95)),
            "score_max": float(np.max(vals)),
        })

    # Ranking
    key = {"duration":"duration","p95":"score_p95", "mean":"score_mean", "max":"score_max"}[rank_by]
    final.sort(key=lambda d: d[key], reverse=(not ascending))

    if not return_debug:
        return final

    dbg = {
        "times": times,
        "vad_mask": vad_mask,
        "score_s": score_s,
        "thr": thr_s,
        "mask": mask,
    }
    return final, dbg