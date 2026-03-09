import glob
from matplotlib import pyplot as plt
import pandas as pd
import soundfile as sf
from audio.audioOverlapping import detect_overlap_clear_fast_ranked
from audio.audioShazam import detect_signature_from_array,ncc_fft,zncc_fft, merge_peaks
import librosa
from audio.audioStageSegmentators import cut_dial_start, split_activity_vs_background
import os
import numpy as np
from typing import Iterable, Dict, Any, Optional, List, Tuple
from audio.audioVadAsr import preprocess_for_asr_diar_vad
from audio.measureActivity import vad_energy_adaptive_array,merge_by_gap
from utils.VapUtils import jsonDecompose



def audioOutputWpm(
    audio_outputs: pd.DataFrame,
    df_windows: pd.DataFrame,
    transcripts_dir: str,
    file_col: str = "file_name",
    window_sec: int = 1,
    hop_sec: int = 1,
    vol_agg: str = "mean",
    json_glob: str = "*_transcript.json",
):
    """
    1) Reads sentences from transcript JSONs at:
         results -> channels[] -> alternatives[] -> sentences[]
       Each sentence has: text, start, end.
       Word count = len(text.split()).
    2) For each window in df_windows (file_name, time_window, vol_window) computes WPM:
         Sentences whose midpoint (start+end)/2 falls in [time_window, time_window+window_sec)
         contribute their word count.  wpm = total_words / window_sec * 60
    3) Adds 'wpm' column to df_windows (in-place copy).
    Returns:
      (df_windows_enriched, audio_outputs_with_match_key)
    """
    import json as _json

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _json_key(path: str) -> str:
        return os.path.basename(path).split("_transcript")[0]

    def _audio_key(fname: str) -> str:
        return os.path.basename(fname).split(".")[0]

    def _parse_sentences(path: str) -> List[Dict[str, Any]]:
        """Extract sentences list from Deepgram-style transcript JSON."""
        with open(path, "r", encoding="utf-8") as fh:
            data = _json.load(fh)
        out: List[Dict[str, Any]] = []
        try:
            channels = data["results"]["channels"]
        except (KeyError, TypeError):
            return out
        for ch in channels:
            for alt in ch.get("alternatives", []):
                for s in alt.get("sentences", []):
                    text = s.get("text", "")
                    try:
                        start = float(s["start"])
                        end   = float(s["end"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    words = len(text.split())
                    if words > 0 and end > start:
                        out.append({"words": words, "start": start, "end": end,
                                    "mid": (start + end) * 0.5})
        return out

    # ------------------------------------------------------------------
    # 1) Load sentences keyed by transcript stem
    # ------------------------------------------------------------------
    transcript_paths = sorted(glob.glob(os.path.join(transcripts_dir, json_glob)))

    sentences_by_key: Dict[str, List[Dict[str, Any]]] = {}
    ingest_errors: List[Dict[str, str]] = []

    for p in transcript_paths:
        try:
            sentences_by_key[_json_key(p)] = _parse_sentences(p)
        except Exception as exc:
            ingest_errors.append({"path": p, "error": str(exc)})

    # ------------------------------------------------------------------
    # 2) Add match_key to audio_outputs (non-destructive copy)
    # ------------------------------------------------------------------
    out = audio_outputs.copy()
    out["match_key"] = out[file_col].apply(_audio_key)

    # ------------------------------------------------------------------
    # 3) Enrich df_windows with wpm
    # ------------------------------------------------------------------
    df_win = df_windows.copy()
    df_win["wpm"] = 0.0

    files_with_transcript = 0

    for file_name, grp in df_win.groupby("file_name", sort=False):
        key = _audio_key(file_name)
        sentences = sentences_by_key.get(key)
        if not sentences:
            continue
        files_with_transcript += 1

        for idx in grp.index:
            t_start = df_win.at[idx, "time_window"]
            t_end   = t_start + window_sec
            words_in_window = sum(
                s["words"] for s in sentences
                if t_start <= s["mid"] < t_end
            )
            df_win.at[idx, "wpm"] = words_in_window * 60.0 / window_sec

    report = {
        "transcripts_found":     len(transcript_paths),
        "transcripts_ingested":  len(sentences_by_key),
        "ingest_errors":         ingest_errors,
        "windows_total":         len(df_win),
        "files_with_transcript": files_with_transcript,
    }

    return df_win, out



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
    # opciones generales
    exts: Iterable[str] = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
    preserve_subdirs: bool = True,
    overwrite: bool = True,
    mono: bool = True,
    force_wav_out: bool = True,
    verbose: bool = True,
    # ventanas de volumen (~3 s) para array["vol_3sec_window"]
    vol_window_sec: float = 3.0,
    vol_hop_sec: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flujo por archivo (SR fijo a 8000 Hz):
      load_audio (resample si hace falta) -> cut_dial_start -> detect_signature -> preprocess_audio_for_vad
      -> (VAD) -> construir y_active/y_inactive + sus tiempos absolutos (sin sanitize/merge de segmentos)
      -> detect_overlap_segments (legacy, basado en energía/centroides)
      -> detect_overlap_spectral (sin diarización, riqueza espectral)
      -> save_audio_to

    Retorna:
      df_audio    — un registro por archivo con métricas agregadas (sin arrays de ventanas).
      df_windows  — un registro por ventana con columnas: file_name, time_window, vol_window.
    """
    TARGET_SR = 8000



    os.makedirs(output_dir, exist_ok=True)
    files = list_audio_files(input_dir, exts=exts)
    if verbose:
        print(f"[main] {len(files)} archivos en '{input_dir}'.")

    # 1) Cargar plantilla una sola vez (si se usa)
    template_wave = None
    template_sr = None
    if template_path:
        template_wave, template_sr = librosa.load(template_path, sr=template_resample_to, mono=True)

    # 2) Cache de plantillas re-muestreadas por sr del audio
    tmpl_cache: Dict[int, np.ndarray] = {}

    rows: List[Dict[str, Any]] = []
    window_rows: List[Dict[str, Any]] = []

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
            y_cut, dial_time = cut_dial_start(y, sr)  # dial_time: offset al original (s) o None

            # 2) detect_signature (sobre y_cut, SIN preprocesar)
            times_det = []
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

                times_det,ncc = detect_signature_from_array(
                                y_cut, sr, template_wave,
                                thresh=0.5,
                                min_distance_sec=2,
                                nms_merge_sec=0.5,
                                ncc_fft=zncc_fft,         
                                merge_peaks=merge_peaks,
                                debug=False,
                            )
            hangup_n_detections = len(times_det)
            hangup_signaturetime = (times_det[-1] if hangup_n_detections > 0 else None)

            # 3) preprocess_for_asr_diar_vad (sobre y_cut)
            y_comp=preprocess_for_asr_diar_vad(
                        y_cut, sr=8000,
                        hp_hz=150.0, lp_hz=3400.0,
                        target_lufs=-23.0, ceiling_db=-1.0,
                        comp_thresh_db=-26.0, comp_ratio=2.5, comp_attack_ms=7.0, comp_release_ms=220.0,
                        makeup_gain_db=0.0, limit_thresh_db=-1.5, limit_release_ms=60.0,
                        vad_params=None,
                        gap_ms=150, min_seg_ms=200, pad_ms=100,
                        out_sr=16000,
                    )

            # 4) VAD → activity_segments y vad_mask

            activity_segments = vad_energy_adaptive_array(y_comp['y_clean'],sr=8000, win_ms=25, hop_ms=10,target_rms_dbfs=-24, p_percentile=35, delta_db=4,
                                   noise_floor_pct=10,hysteresis_db=2,local_win_ms=750,local_mix=0.6,
                                   min_run_frames=4, smooth_ms=2_00,pad_ms=120,returns= "segments")
            activity_segments = merge_by_gap(activity_segments, gap_ms=120)  # merge cercano para evitar fragmentación excesiva

            # 5) y_active / y_inactive y sus tiempos ABSOLUTOS (audio original)
            try:
                overlaps= detect_overlap_clear_fast_ranked(
                                    y_comp['y_clean'], sr,
                                    vad_segments=activity_segments,
                                    n_fft=512, hop_length=160, fmax=3400,
                                    rank_by="duration",
                                    smooth_ms=512,
                                    min_run_ms=125,
                                    merge_gap_ms=60,
                                    pad_ms=80,
                                    k_iqr=2.25,
                                    min_abs_thr=0.8,
                                )
            except Exception as e:
                if verbose:
                    print(f"[WARN] NO OVERLAPS en '{base}': {e}")
                overlaps = []

            top_overlap = overlaps[:1] if overlaps else []
            time_overlap = top_overlap[0]['start'] if top_overlap else None
            duration_overlap = top_overlap[0]['duration'] if top_overlap else None
            score_overlap = top_overlap[0]['score_p95'] if top_overlap else None
            num_overlaps = len(overlaps)

            overlap_tot_time = sum(i['end']-i['start'] for i in overlaps)
            speech_tot_time  = sum(b-a for a,b in activity_segments)
            overlap_ratio_in_speech = overlap_tot_time / speech_tot_time if speech_tot_time > 1e-6 else 0.0
            # 7) Save audio procesado
            rel_dir = os.path.relpath(os.path.dirname(in_path), input_dir) if preserve_subdirs else ""
            out_dir_for_file = os.path.join(output_dir, rel_dir)
            os.makedirs(out_dir_for_file, exist_ok=True)

            stem, ext = os.path.splitext(base)
            out_name = f"{stem}.wav" if force_wav_out else f"{stem}{ext}"
            out_path = os.path.join(out_dir_for_file, out_name)
            if overwrite or (not os.path.exists(out_path)):
                print(f"[main] Escribiendo procesado a '{out_path}'")
                sf.write(out_path, y_comp['y_clean'], sr)

            # 8) Volumen por ventanas (~3 s) en el audio procesado
            vol_db, t_starts = windowed_db(y_comp['y_clean'], sr, window_sec=1, hop_sec=vol_hop_sec)
            for t_w, v_w in zip(t_starts.tolist(), vol_db.tolist()):
                window_rows.append({"file_name": base, "time_window": t_w, "vol_window": v_w})

            # 9) Fila del DF principal
            rows.append({
                "file_name": base,
                "dialtime": None if dial_time is None else float(dial_time),
                "sr": int(sr),
                "hangup_n_detections": int(hangup_n_detections),
                "times_det": times_det,
                "hangup_signaturetime": None if hangup_signaturetime is None else float(hangup_signaturetime),
                "time_overlap": time_overlap,
                "duration_overlap": duration_overlap,
                "score_overlap": score_overlap,
                "num_overlaps": num_overlaps,
                "overlap_total_time": overlap_tot_time,
                "overlap_ratio_in_speech": overlap_ratio_in_speech,
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
                "times_det": [],
                "hangup_signaturetime": np.nan,
                "time_overlap":  np.nan,
                "duration_overlap":  np.nan,
                "score_overlap":  np.nan,
                "num_overlaps":  np.nan,
                "overlap_total_time":  np.nan,
                "overlap_ratio_in_speech":  np.nan,
            })

    # Construye DataFrames
    df_audio = pd.DataFrame(rows, columns=[
                "file_name",
                "dialtime",
                "sr",
                "hangup_n_detections",
                "times_det",
                "hangup_signaturetime",
                "time_overlap",
                "duration_overlap",
                "score_overlap",
                "num_overlaps",
                "overlap_total_time",
                "overlap_ratio_in_speech",
    ])

    df_windows = pd.DataFrame(window_rows, columns=["file_name", "time_window", "vol_window"])

    if verbose:
        print(f"[main] Listo. {len(df_audio)} archivos procesados en '{output_dir}'.")

    return df_audio, df_windows


if __name__ == "__main__":
    hangup_signature = r"process/test/sign/fragment_signature1.wav" 
    preproc_kwargs = dict(
        enable_leveler=True, leveler_mode="lufs", target_lufs=-23.0, limiter_ceiling_db=-2.0,
        use_pedalboard=True, pb_hp_hz=120.0, pb_lp_hz=3800.0, pb_gate_thresh_db=None,
        pb_gate_ratio=4.0, pb_gate_attack_ms=8.0, pb_gate_release_ms=240.0,
        pb_comp_thresh_db=-24.0, pb_comp_ratio=4.0, pb_comp_attack_ms=10.0, pb_comp_release_ms=240.0,
        use_pcen=True, pcen_percentile=90.0, pcen_time_constant=0.05, pcen_gain=0.95,
        use_pitch_gate=True, use_webrtcvad=False, vad_aggressiveness=2,
    )
    _, _windows = main_process_batch(
        input_dir="process/test",
        output_dir="process/test/processed",
        template_path=hangup_signature,
        template_resample_to=8000,
        preserve_subdirs=True,
        force_wav_out=True,
        verbose=True,
    )