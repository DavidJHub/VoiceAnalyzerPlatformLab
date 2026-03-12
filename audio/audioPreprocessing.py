import glob
import pandas as pd
import soundfile as sf
from audio.audioOverlapping import detect_overlap_segments, detect_overlap_spectral
from audio.audioShazam import detect_signature_from_array,ncc_fft, merge_peaks
import librosa
from audio.audioPrepUtils import preprocess_audio_for_vad, align_mask_to_length
from audio.audioStageSegmentators import cut_dial_start
import os
import numpy as np
from typing import Iterable, Dict, Any, Optional, List, Tuple
from audio.measureActivity import vad_energy_adaptive_array

# ---- asume que ya tienes definidas/importadas:
# cut_dial_start(y, sr, audio_path=None, **cut_kwargs)
# preprocess_audio_for_vad(y, sr, **preproc_kwargs) -> dict con key "y"
# ncc_fft(x, template)  y  merge_peaks(times, nms_merge_sec)



def list_audio_files(input_dir: str,
                     pattern: str = "**/*",
                     exts: Iterable[str] = (".wav", ".mp3", ".flac", ".ogg", ".m4a")) -> List[str]:
    return sorted([
        p for p in glob.glob(os.path.join(input_dir, pattern), recursive=True)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts
    ])


def windowed_db(
    y: np.ndarray, sr: int, window_sec: float = 3.0, hop_sec: Optional[float] = None, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula el volumen en dBFS por ventanas (~3 s por defecto).
    Devuelve (db_values, t_starts), donde t_starts son los inicios de cada ventana.
    Por defecto usa ventanas no solapadas (hop = window).
    """
    if hop_sec is None:
        hop_sec = window_sec
    win = max(1, int(round(window_sec * sr)))
    hop = max(1, int(round(hop_sec * sr)))
    if y.ndim > 1:
        y = y.mean(axis=1)

    values = []
    t_starts = []
    for start in range(0, len(y) - win + 1, hop):
        seg = y[start:start + win]
        rms = np.sqrt(np.mean(seg**2) + eps)
        db = 20.0 * np.log10(rms + eps)  # dBFS (ref=1.0 a full scale)
        values.append(db)
        t_starts.append(start / sr)

    return np.array(values, dtype=float), np.array(t_starts, dtype=float)


def main_process_batch(
    input_dir: str,
    output_dir: str,
    *,
    # plantilla para detect_signature (opcional)
    template_path: Optional[str] = None,
    template_resample_to: Optional[int] = None,  # si None, usa SR nativo de la plantilla
    # kwargs de cada paso:
    cut_kwargs: Optional[Dict[str, Any]] = None,
    detect_kwargs: Optional[Dict[str, Any]] = None,   # requiere: thresh, min_distance_sec, nms_merge_sec, ncc_fft, merge_peaks
    preproc_kwargs: Optional[Dict[str, Any]] = None,
    # opciones generales
    pattern: str = "**/*",
    exts: Iterable[str] = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
    preserve_subdirs: bool = True,
    overwrite: bool = True,
    mono: bool = True,
    force_wav_out: bool = True,
    verbose: bool = True,
    # ventanas de volumen (~3 s) para array["vol_3sec_window"]
    vol_window_sec: float = 3.0,
    vol_hop_sec: Optional[float] = None,
    summary_excel_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Flujo por archivo (SR fijo a 8000 Hz):
      load_audio (resample si hace falta) -> cut_dial_start -> detect_signature -> preprocess_audio_for_vad
      -> (VAD) -> construir y_active/y_inactive + sus tiempos absolutos (sin sanitize/merge de segmentos)
      -> detect_overlap_segments (legacy, basado en energía/centroides)
      -> detect_overlap_spectral (sin diarización, riqueza espectral)
      -> save_audio_to
    """
    TARGET_SR = 8000

    cut_kwargs     = cut_kwargs or {}
    detect_kwargs  = detect_kwargs or {}
    preproc_kwargs = preproc_kwargs or {}

    os.makedirs(output_dir, exist_ok=True)
    files = list_audio_files(input_dir, pattern=pattern, exts=exts)
    if verbose:
        print(f"[main] {len(files)} archivos en '{input_dir}'.")

    # 1) Cargar plantilla una sola vez (si se usa)
    template_wave = None
    template_sr = None
    if template_path:
        template_wave, template_sr = librosa.load(template_path, sr=template_resample_to, mono=True)

    # 2) Cache de plantillas re-muestreadas por sr del audio
    tmpl_cache: Dict[int, np.ndarray] = {}

    def _build_active_inactive_with_time(
        y_arr: np.ndarray,
        sr: int,
        segments_sec: List[Tuple[float, float]],
        t0_offset_sec: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Versión liviana: asume segments_sec ya ordenados, sin solapes y dentro de [0, n/sr].
        NO hay sanitize/merge/ordenamiento. Solo convierte a índices y compone activo/inactivo.
        """
        n = len(y_arr)

        # convierte segmentos a índices
        seg_idx: List[Tuple[int, int]] = []
        for s, e in segments_sec:
            a = int(round(s * sr))
            b = int(round(e * sr))
            if b > a:
                a = max(0, a)
                b = min(n, b)
                if b > a:
                    seg_idx.append((a, b))

        # ACTIVO
        active_chunks, t_active_chunks = [], []
        for a, b in seg_idx:
            active_chunks.append(y_arr[a:b])
            t_active_chunks.append((np.arange(a, b, dtype=np.int64) / sr) + float(t0_offset_sec))
        y_active = (np.concatenate(active_chunks).astype(y_arr.dtype, copy=False)
                    if active_chunks else np.array([], dtype=y_arr.dtype))
        t_active = (np.concatenate(t_active_chunks).astype(np.float64, copy=False)
                    if t_active_chunks else np.array([], dtype=np.float64))

        # INACTIVO (= complemento directo usando los bordes de seg_idx)
        inactive_chunks, t_inactive_chunks = [], []
        last_end = 0
        for a, b in seg_idx:
            if a > last_end:
                inactive_chunks.append(y_arr[last_end:a])
                t_inactive_chunks.append((np.arange(last_end, a, dtype=np.int64) / sr) + float(t0_offset_sec))
            last_end = b
        if last_end < n:
            inactive_chunks.append(y_arr[last_end:n])
            t_inactive_chunks.append((np.arange(last_end, n, dtype=np.int64) / sr) + float(t0_offset_sec))

        y_inactive = (np.concatenate(inactive_chunks).astype(y_arr.dtype, copy=False)
                      if inactive_chunks else np.array([], dtype=y_arr.dtype))
        t_inactive = (np.concatenate(t_inactive_chunks).astype(np.float64, copy=False)
                      if t_inactive_chunks else np.array([], dtype=np.float64))
        return y_active, y_inactive, t_active, t_inactive

    rows: List[Dict[str, Any]] = []

    for i, in_path in enumerate(files, start=1):
        base = os.path.basename(in_path)
        try:
            # 0) forzar SR=8000 si hace falta
            y, sr = librosa.load(in_path, sr=None, mono=mono)
            if sr != TARGET_SR:
                y = librosa.resample(y.astype(np.float32, copy=False), orig_sr=sr, target_sr=TARGET_SR)
                sr = TARGET_SR
            else:
                y = y.astype(np.float32, copy=False)

            # 1) cut_dial_start (sin escribir)
            y_cut, dial_time = cut_dial_start(y, sr, **cut_kwargs)  # dial_time: offset al original (s) o None

            # 2) detect_signature (sobre y_cut, SIN preprocesar)
            times = []
            if template_wave is not None:
                if sr not in tmpl_cache:
                    if template_sr is not None and template_sr != sr:
                        tmpl_cache[sr] = librosa.resample(
                            template_wave.astype(np.float32, copy=False),
                            orig_sr=template_sr,
                            target_sr=sr
                        )
                    else:
                        tmpl_cache[sr] = template_wave.astype(np.float32, copy=False)

                times, _ = detect_signature_from_array(
                    y_cut, sr, tmpl_cache[sr],
                    thresh=0.9,
                    min_distance_sec=0.4,
                    nms_merge_sec=detect_kwargs["nms_merge_sec"],
                    ncc_fft=detect_kwargs["ncc_fft"],
                    merge_peaks=detect_kwargs["merge_peaks"],
                )
            hangup_n_detections = len(times)
            hangup_signaturetime = (times[-1] if hangup_n_detections > 0 else None)

            # 3) preprocess_audio_for_vad (sobre y_cut)
            feats = preprocess_audio_for_vad(y_cut, sr, **preproc_kwargs)
            y_proc = feats["y"]

            # 4) VAD → activity_segments y vad_mask
            if sr:
                default_hop_ms = (feats.get("hop_length", 128) / float(sr)) * 1_000.0
                default_win_ms = (feats.get("n_fft", 512) / float(sr)) * 1_000.0
            else:
                default_hop_ms = 10.0
                default_win_ms = 25.0

            vad_hop_ms = float(feats.get("vad_hop_ms", default_hop_ms))
            vad_win_ms = float(feats.get("vad_win_ms", default_win_ms))

            activity_segments = vad_energy_adaptive_array(
                y_proc, sr, returns="segments", hop_ms=vad_hop_ms, win_ms=vad_win_ms
            )
            vad_mask = vad_energy_adaptive_array(
                y_proc, sr, returns="mask", hop_ms=vad_hop_ms, win_ms=vad_win_ms
            )

            log_rms = feats.get("log_rms")
            target_len = int(len(log_rms)) if isinstance(log_rms, np.ndarray) else 0
            vad_mask = align_mask_to_length(vad_mask, target_len)
            feats["vad_energy_mask"] = vad_mask

            # 5) y_active / y_inactive y sus tiempos ABSOLUTOS (audio original)
            t0 = float(dial_time) if (dial_time is not None) else 0.0
            y_active, y_inactive, t_active, t_inactive = _build_active_inactive_with_time(
                y_proc, sr, activity_segments, t0_offset_sec=t0
            )

            # 6A) Overlapping legacy (energía/centroides) → detect_overlap_segments
            overlap_times_rel, overlap_centroids = detect_overlap_segments(
                y_proc, sr,
                vad_mask=vad_mask, restrict_to_vad=True,
                returns="centroids",
                score_thresh=0.95, smooth_ms=250, min_run_ms=120
            )
            if overlap_times_rel is None or np.size(overlap_times_rel) == 0:
                overlap_times_abs = np.array([], dtype=float)
            else:
                overlap_times_abs = np.asarray(overlap_times_rel, dtype=float) + t0

            # 6B) Overlapping espectral (sin diarización) → detect_overlap_spectral
            #    Se usan n_fft / hop_length coherentes con feats
            n_fft = int(feats.get("n_fft", 512))
            hop_length = int(feats.get("hop_length", 128))

            overlap_spec_times, overlap_spec_centroids, richness_score, overlap_spec_mask = detect_overlap_spectral(
                y_proc,
                sr,
                n_fft=n_fft,
                hop_length=hop_length,
                active_percentile=70.0,
                richness_factor=1.3,
                smooth_ms=250.0,
                min_run_ms=120.0,
                vad_mask=vad_mask
            )

            # Métrica global: % de frames marcados como overlap espectral
            if overlap_spec_mask.size > 0:
                overlap_spectral_rate = float(overlap_spec_mask.mean())
            else:
                overlap_spectral_rate = 0.0

            # 7) Save audio procesado
            rel_dir = os.path.relpath(os.path.dirname(in_path), input_dir) if preserve_subdirs else ""
            out_dir_for_file = os.path.join(output_dir, rel_dir)
            os.makedirs(out_dir_for_file, exist_ok=True)

            stem, ext = os.path.splitext(base)
            out_name = f"{stem}.wav" if force_wav_out else f"{stem}{ext}"
            out_path = os.path.join(out_dir_for_file, out_name)

            if overwrite or (not os.path.exists(out_path)):
                # Anti-clip mínimo
                mx = float(np.max(np.abs(y_proc))) if y_proc.size else 0.0
                if mx > 0.999:
                    y_proc = (y_proc / mx) * 0.999
                sf.write(out_path, y_proc, sr)

            # 8) Volumen por ventanas (~3 s) en el audio procesado
            vol_db, t_starts = windowed_db(y_proc, sr, window_sec=vol_window_sec, hop_sec=vol_hop_sec)

            # 9) Fila del DF
            rows.append({
                "file_name": base,
                "dialtime": None if dial_time is None else float(dial_time),
                "sr": int(sr),
                "hangup_n_detections": int(hangup_n_detections),
                "times": list(map(float, times)),
                "hangup_signaturetime": None if hangup_signaturetime is None else float(hangup_signaturetime),
                "feats[y]": y_proc,
                "array[vol_3sec_window]": vol_db,
                "array[call_time]": t_starts,
                "array[y_active]": y_active,
                "array[y_inactive]": y_inactive,
                "array[y_active_time]": t_active,
                "array[y_inactive_time]": t_inactive,
                "array[overlap_times]": overlap_times_abs,
                "array[overlap_centroids]": overlap_centroids,
                # Nuevas métricas espectrales
                "array[overlap_spectral_times]": overlap_spec_times,
                "array[overlap_spectral_score]": richness_score,
                "array[overlap_spectral_mask]": overlap_spec_mask,
                "overlap_spectral_rate": overlap_spectral_rate,
            })

            if verbose and (i % 10 == 0 or i == len(files)):
                print(f"[main] {i}/{len(files)} procesado → detecciones={hangup_n_detections}")

        except Exception as e:
            if verbose:
                print(f"[ERROR] {base}: {e}")
            rows.append({
                "file_name": base,
                "dialtime": np.nan,
                "sr": np.nan,
                "hangup_n_detections": 0,
                "times": [],
                "hangup_signaturetime": np.nan,
                "feats[y]": np.array([], dtype=np.float32),
                "array[vol_3sec_window]": np.array([], dtype=float),
                "array[call_time]": np.array([], dtype=float),
                "array[y_active]": np.array([], dtype=np.float32),
                "array[y_inactive]": np.array([], dtype=np.float32),
                "array[y_active_time]": np.array([], dtype=float),
                "array[y_inactive_time]": np.array([], dtype=float),
                "array[overlap_times]": np.array([], dtype=float),
                "array[overlap_centroids]": np.array([], dtype=float),
                "array[overlap_spectral_times]": np.array([], dtype=float),
                "array[overlap_spectral_score]": np.array([], dtype=float),
                "array[overlap_spectral_mask]": np.array([], dtype=bool),
                "overlap_spectral_rate": np.nan,
            })

    # Construye DataFrame
    df = pd.DataFrame(rows, columns=[
        "file_name",
        "dialtime",
        "sr",
        "hangup_n_detections",
        "times",
        "hangup_signaturetime",
        "feats[y]",
        "array[vol_3sec_window]",
        "array[call_time]",
        "array[y_active]",
        "array[y_inactive]",
        "array[y_active_time]",
        "array[y_inactive_time]",
        "array[overlap_times]",
        "array[overlap_centroids]",
        "array[overlap_spectral_times]",
        "array[overlap_spectral_score]",
        "array[overlap_spectral_mask]",
        "overlap_spectral_rate",
    ])

    if verbose:
        ok = (df["feats[y]"].apply(lambda a: isinstance(a, np.ndarray) and a.size > 0)).sum()
        print(f"[main] Listo. {ok}/{len(df)} con audio procesado escrito en '{output_dir}'.")

    if summary_excel_path:
        df.to_excel(summary_excel_path, index=False)

    return df


if __name__ == "__main__":
    hangup_signature = r"process/test/sign/fragment_signature1.wav" 
    cut_kwargs = dict(band=(420,440), slice_seconds=40.0, intensity_ratio=0.7, min_duration_s=0.3)
    detect_kwargs = dict(
        thresh=0.5,
        min_distance_sec=0.4,
        nms_merge_sec=0.25,
        ncc_fft=ncc_fft,         
        merge_peaks=merge_peaks,
    )
    preproc_kwargs = dict(
        enable_leveler=True, leveler_mode="lufs", target_lufs=-18.0, limiter_ceiling_db=-2.0,
        use_pedalboard=True, pb_hp_hz=120.0, pb_lp_hz=3800.0, pb_gate_thresh_db=None,
        pb_gate_ratio=4.0, pb_gate_attack_ms=8.0, pb_gate_release_ms=240.0,
        pb_comp_thresh_db=-24.0, pb_comp_ratio=4.0, pb_comp_attack_ms=10.0, pb_comp_release_ms=240.0,
        use_pcen=True, pcen_percentile=90.0, pcen_time_constant=0.05, pcen_gain=0.95,
        use_pitch_gate=True, use_webrtcvad=False, vad_aggressiveness=2,
    )
    _ = main_process_batch(
        input_dir="process/test",
        output_dir="process/test/processed",
        template_path=hangup_signature,
        template_resample_to=8000,
        cut_kwargs=cut_kwargs,
        detect_kwargs=detect_kwargs,
        preproc_kwargs=preproc_kwargs,
        preserve_subdirs=True,
        force_wav_out=True,
        verbose=True,
        summary_excel_path="process/test/audio_outputs_test.xlsx",
    )