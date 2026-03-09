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
    vol_agg: str = "mean",              # "mean" o "median"
    json_glob: str = "*_transcript.json",
):
    """
    Pipeline completo:
      1) Ingesta de transcritos JSON desde transcripts_dir
      2) Crea match-key:
           - json_key = json_filename.split('_transcript')[0]
           - audio_key = file_name.split('.')[0]
      3) Cruce con audio_outputs
      4) Calcula y agrega (usando df_windows con columnas file_name/time_window/vol_window):
           - times_5s
           - vols_5s
           - wpm_5s
    Retorna:
      out_df, report_dict
    """

    # ----------------------------
    # Helpers
    # ----------------------------
    def json_key_from_filename(fname: str) -> str:
        # "....-all_transcript.json" -> "....-all"
        base = os.path.basename(fname)
        return base.split("_transcript")[0]

    def audio_key_from_file_name(fname: str) -> str:
        # "....-all.mp3" -> "....-all"
        base = os.path.basename(fname)
        return base.split(".")[0]

    def aggregate_signal_1s_to_5s(vols_1s: np.ndarray, T: int):
        starts = np.arange(0, T - window_sec + 1, hop_sec, dtype=int)
        if vol_agg == "median":
            v5 = np.array([np.nanmedian(vols_1s[s:s+window_sec]) for s in starts], dtype=float)
        else:
            v5 = np.array([np.nanmean(vols_1s[s:s+window_sec]) for s in starts], dtype=float)
        t5 = (starts + window_sec).astype(int)  # fin de ventana
        return t5, v5

    def wpm_5s_from_words(words_df: pd.DataFrame, T: int):
        """
        Regla discreta:
          t_mid=(start+end)/2
          bin_end=ceil(t_mid)
          counts[bin_end]++
        Luego agrega counts por ventanas de window_sec.
        """
        counts_1s = np.zeros(T + 1, dtype=int)

        if words_df is not None and len(words_df):
            starts = pd.to_numeric(words_df["start"], errors="coerce").to_numpy()
            ends   = pd.to_numeric(words_df["end"],   errors="coerce").to_numpy()
            m = np.isfinite(starts) & np.isfinite(ends)
            starts, ends = starts[m], ends[m]

            swap = starts > ends
            if np.any(swap):
                starts[swap], ends[swap] = ends[swap], starts[swap]

            t_mid = (starts + ends) / 2.0
            bin_end = np.ceil(t_mid).astype(int)

            # filtrar fuera de rango (NO clip)
            bin_end = bin_end[(bin_end >= 0) & (bin_end <= T)]
            if len(bin_end):
                counts_1s += np.bincount(bin_end, minlength=T + 1).astype(int)

            # requisito: counts[0]=0
            if counts_1s[0] > 0 and T >= 1:
                counts_1s[1] += counts_1s[0]
                counts_1s[0] = 0

        starts_win = np.arange(0, T - window_sec + 1, hop_sec, dtype=int)
        words_5s = np.array([counts_1s[s:s+window_sec].sum() for s in starts_win], dtype=float)
        wpm_5s = words_5s * 60.0 / window_sec
        t5 = (starts_win + window_sec).astype(int)
        return t5, wpm_5s

    # ----------------------------
    # 1) Ingesta de transcritos -> dict key -> words_df
    # ----------------------------
    transcript_paths = sorted(glob.glob(os.path.join(transcripts_dir, json_glob)))

    words_by_key = {}
    ingest_errors = []

    for p in transcript_paths:
        try:
            _, words_df, _ = jsonDecompose(p)
            key = json_key_from_filename(p)
            words_by_key[key] = words_df
        except Exception as e:
            ingest_errors.append({"path": p, "error": str(e)})

    # Índice rápido: file_name -> arrays de ventanas (ordenados por tiempo)
    windows_by_file: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for fname, grp in df_windows.groupby("file_name", sort=False):
        grp_sorted = grp.sort_values("time_window")
        windows_by_file[fname] = (
            grp_sorted["time_window"].to_numpy(dtype=float),
            grp_sorted["vol_window"].to_numpy(dtype=float),
        )

    # ----------------------------
    # 2) Cruce + 3) Cálculo 5s
    # ----------------------------
    out = audio_outputs.copy()
    out["match_key"] = out[file_col].apply(audio_key_from_file_name)

    out["times_5s"] = None
    out["vols_5s"]  = None
    out["wpm_5s"]   = None

    missing_transcript = []
    computed = 0

    for idx, row in out.iterrows():
        key = row["match_key"]
        words_df = words_by_key.get(key)
        if words_df is None:
            missing_transcript.append(key)
            continue

        win_entry = windows_by_file.get(row[file_col])
        if win_entry is None:
            times_1s = np.array([], dtype=float)
            vols_1s  = np.array([], dtype=float)
        else:
            times_1s, vols_1s = win_entry

        try:
            T = int(times_1s[-1]) if len(times_1s) > 0 else 300
        except (ValueError, IndexError):
            T = 300

        # volumen 5s (reusa vols)
        t5_vol, v5 = aggregate_signal_1s_to_5s(vols_1s, T)

        # wpm 5s (misma grilla por T)
        t5_wpm, wpm5 = wpm_5s_from_words(words_df, T)

        # sanity: deben coincidir los endpoints
        if not np.array_equal(t5_vol, t5_wpm):
            raise ValueError(f"{row[file_col]}: times_5s de volumen y wpm no coinciden")

        out.at[idx, "times_5s"] = t5_vol.tolist()
        out.at[idx, "vols_5s"]  = v5.tolist()
        out.at[idx, "wpm_5s"]   = wpm5.tolist()
        computed += 1

    report = {
        "transcripts_found": len(transcript_paths),
        "transcripts_ingested": len(words_by_key),
        "ingest_errors": ingest_errors,  # lista de dicts
        "rows_total": len(out),
        "rows_computed": computed,
        "rows_missing_transcript": len(missing_transcript),
        "missing_transcript_keys_sample": sorted(set(missing_transcript))[:20],
    }

    return out, report



def list_audio_files(input_dir: str,
                     pattern: str = "**/*",
                     exts: Iterable[str] = (".wav", ".mp3", ".flac", ".ogg", ".m4a")) -> List[str]:
    return sorted([
        p for p in glob.glob(os.path.join(input_dir, pattern), recursive=False)
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
                "y_proc": y_comp['y_clean'],
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
                "y_proc": np.array([], dtype=np.float32),
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
                "y_proc",
                "time_overlap",
                "duration_overlap",
                "score_overlap",
                "num_overlaps",
                "overlap_total_time",
                "overlap_ratio_in_speech",
    ])

    df_windows = pd.DataFrame(window_rows, columns=["file_name", "time_window", "vol_window"])

    if verbose:
        ok = (df_audio["y_proc"].apply(lambda a: isinstance(a, np.ndarray) and a.size > 0)).sum()
        print(f"[main] Listo. {ok}/{len(df_audio)} con audio procesado escrito en '{output_dir}'.")

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