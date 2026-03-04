from pathlib import Path
from audio.audioPreprocessing import main_process_batch
from audio.audioShazam import merge_peaks, ncc_fft
from pyannote.audio import Pipeline
from pydub import AudioSegment

# ======================================================================
#  UTILIDADES AUDIO (las mismas que ya tenías)
# ======================================================================

def silence_like(source: AudioSegment, duration_ms: int) -> AudioSegment:
    return (
        AudioSegment.silent(duration=duration_ms, frame_rate=source.frame_rate)
        .set_channels(source.channels)
        .set_sample_width(source.sample_width)
    )


def pad_to_length(seg: AudioSegment, target_len: int) -> AudioSegment:
    """Rellena con silencio hasta target_len (ms)."""
    if len(seg) >= target_len:
        return seg
    pad_ms = target_len - len(seg)
    silence = (
        AudioSegment.silent(duration=pad_ms, frame_rate=seg.frame_rate)
        .set_sample_width(seg.sample_width)
        .set_channels(seg.channels)
    )
    return seg + silence


def export_stereo_from_two_mono(
    left: AudioSegment,
    right: AudioSegment,
    out_path: str
) -> None:
    """
    Crea un WAV estéreo con 'left' en canal izquierdo
    y 'right' en canal derecho.
    """
    # Asegurar mono
    if left.channels != 1:
        left = left.set_channels(1)
    if right.channels != 1:
        right = right.set_channels(1)

    # Alinear frame_rate y sample_width
    if left.frame_rate != right.frame_rate:
        right = right.set_frame_rate(left.frame_rate)
    if left.sample_width != right.sample_width:
        right = right.set_sample_width(left.sample_width)

    # Igualar duración
    max_len = max(len(left), len(right))
    left_p = pad_to_length(left, max_len)
    right_p = pad_to_length(right, max_len)

    # Construir estéreo
    stereo = AudioSegment.from_mono_audiosegments(left_p, right_p)
    stereo.export(out_path, format="wav")
    print("Stereo guardado:", out_path, "duración (ms):", len(stereo))


def export_compressed_mp3(
    wav_path_in: str,
    mp3_path_out: str,
    bitrate: str = "192k"
) -> None:
    """
    Comprime un archivo WAV (mono o estéreo) a MP3.
    - Normaliza a 44.1 kHz y 16 bits.
    """
    audio = AudioSegment.from_file(wav_path_in)
    audio = audio.set_frame_rate(44100).set_sample_width(2)  # 16 bits
    audio.export(mp3_path_out, format="mp3", bitrate=bitrate)
    print(f"MP3 comprimido guardado en: {mp3_path_out}")


# ======================================================================
#  DIARIZACIÓN DE UN SOLO ARCHIVO
# ======================================================================

def diarize_and_build_tracks(
    pipeline: Pipeline,
    wav_path: Path,
    out_root: Path,
    do_mp3: bool = True,
    mp3_bitrate: str = "128k"
):
    """
    - Corre diarización con pyannote.
    - Exporta pistas mono por speaker.
    - Crea estéreo caller_left / agent_right + MP3.
    """
    wav_path = wav_path.resolve()
    print(f"\nProcesando archivo: {wav_path}")

    # --- 1. Diarizar ---
    diar = pipeline(str(wav_path), num_speakers=2)

    # --- 2. Cargar audio completo y preparar pistas ---
    wav = AudioSegment.from_file(str(wav_path))
    duration_ms = len(wav)
    tracks = {label: silence_like(wav, duration_ms) for label in diar.labels()}

    # --- 3. Superponer intervenciones ---
    for turn, _, speaker in diar.itertracks(yield_label=True):
        start_ms = int(round(turn.start * 1000))
        end_ms   = int(round(turn.end   * 1000))

        start_ms = max(0, min(start_ms, duration_ms))
        end_ms   = max(0, min(end_ms, duration_ms))
        if end_ms <= start_ms:
            continue

        segment = wav[start_ms:end_ms]
        tracks[speaker] = tracks[speaker].overlay(segment, position=start_ms)

    # Carpeta de salida
    out_dir = out_root / wav_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 4. Exportar pistas mono ---
    mono_paths = {}
    for spk, audio in tracks.items():
        out_mono = out_dir / f"{wav_path.stem}_{spk}.wav"
        audio.export(str(out_mono), format="wav")
        mono_paths[spk] = out_mono
        print("Pista guardada:", out_mono, "duración (ms):", len(audio), "== original:", duration_ms)

    # --- 5. Construir estéreo y respetar nombre + sufijo -stereo ---
    labels_sorted = sorted(tracks.keys())
    if len(labels_sorted) >= 2:
        caller_label = labels_sorted[0]
        agent_label  = labels_sorted[1]
    else:
        caller_label = agent_label = labels_sorted[0]

    caller_audio = AudioSegment.from_file(str(mono_paths[caller_label]))
    agent_audio  = AudioSegment.from_file(str(mono_paths[agent_label]))

    # 👇 AQUÍ CAMBIA EL NOMBRE: mismo stem + "-stereo"
    base_stem = wav_path.stem  # p.ej. CARDIFCR_...-all
    stereo_wav_path = out_dir / f"{base_stem}-stereo.wav"
    stereo_mp3_path = out_dir / f"{base_stem}-stereo.mp3"

    export_stereo_from_two_mono(caller_audio, agent_audio, str(stereo_wav_path))

    if do_mp3:
        export_compressed_mp3(
            wav_path_in=str(stereo_wav_path),
            mp3_path_out=str(stereo_mp3_path),
            bitrate=mp3_bitrate,
        )
        print("Stereo MP3 final:", stereo_mp3_path)



# ======================================================================
#  PIPELINE POR BATCH: PREPROCESAR + DIARIZAR TODO
# ======================================================================

def run_full_batch_pipeline(
    campaign_directory: str,
    processed_output_directory: str,
    hangup_signature: str,
    summary_excel_name: str = "audio_outputs_test.xlsx",
    mp3_bitrate: str = "128k"
):
    """
    1) Preprocesa todos los audios en campaign_directory con main_process_batch.
    2) Recorre processed_output_directory y corre diarización + export estéreo/MP3
       para cada WAV generado.
    """

    campaign_dir = Path(campaign_directory).resolve()
    processed_dir = Path(processed_output_directory).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    # =======================================================
    # 1) PREPROCESAMIENTO DE AUDIO (tu main_process_batch)
    # =======================================================
    # IMPORTANTE: asumo que main_process_batch, ncc_fft, merge_peaks
    # ya están definidos/importados en tu proyecto.

    cut_kwargs = dict(
        band=(420, 440),
        slice_seconds=40.0,
        intensity_ratio=0.65,
        min_duration_s=0.35
    )

    detect_kwargs = dict(
        thresh=0.5,
        min_distance_sec=0.4,
        nms_merge_sec=0.25,
        ncc_fft=ncc_fft,          # tus funciones
        merge_peaks=merge_peaks,
    )

    preproc_kwargs = dict(
        enable_leveler=True, leveler_mode="lufs", target_lufs=-18.0, limiter_ceiling_db=-2.0,
        use_pedalboard=True, pb_hp_hz=120.0, pb_lp_hz=6000.0, pb_gate_thresh_db=None,
        pb_gate_ratio=3.0, pb_gate_attack_ms=8.0, pb_gate_release_ms=120.0,
        pb_comp_thresh_db=-24.0, pb_comp_ratio=3.0, pb_comp_attack_ms=10.0, pb_comp_release_ms=100.0,
        use_pcen=True, pcen_percentile=90.0, pcen_time_constant=0.05, pcen_gain=0.95,
        use_pitch_gate=True, use_webrtcvad=True, vad_aggressiveness=2,
    )

    summary_excel_path = str(campaign_dir / "misc" / summary_excel_name)
    (campaign_dir / "misc").mkdir(exist_ok=True)

    print("=== Preprocesando batch de audio ===")
    audioData = main_process_batch(
        input_dir=str(campaign_dir),
        output_dir=str(processed_dir),
        template_path=hangup_signature,
        template_resample_to=8000,
        cut_kwargs=cut_kwargs,
        detect_kwargs=detect_kwargs,
        preproc_kwargs=preproc_kwargs,
        preserve_subdirs=True,
        force_wav_out=True,
        verbose=True,
        summary_excel_path=summary_excel_path,
    )
    # Puedes usar audioData si te da metadata; aquí no lo usamos directamente.

    # =======================================================
    # 2) DIARIZACIÓN EN BATCH SOBRE LOS WAV PROCESADOS
    # =======================================================
    print("=== Cargando modelo de diarización de pyannote ===")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token=True  # asumes HF_TOKEN en el entorno
    )

    # Carpeta raíz para resultados de diarización (puede ser = processed_dir/diarized)
    diarized_root = processed_dir / "diarized"
    diarized_root.mkdir(parents=True, exist_ok=True)

    # Recorremos todos los WAV producidos por el preprocesamiento
    print(f"=== Iniciando diarización sobre WAV en: {processed_dir} ===")
    for wav_path in processed_dir.rglob("*.wav"):
        # Si quieres filtrar solo archivos -all.wav, usa:
        # if not wav_path.name.endswith("-all.wav"): continue
        diarize_and_build_tracks(
            pipeline=pipeline,
            wav_path=wav_path,
            out_root=diarized_root,
            do_mp3=True,
            mp3_bitrate=mp3_bitrate,
        )

    print("=== Pipeline de batch finalizado ===")


# ======================================================================
# EJEMPLO DE USO
# ======================================================================

if __name__ == "__main__":
    # Directorio donde están los audios originales (batch)
    campaign_directory = "process/Audios_Cardif"  
    # Directorio donde se guardarán los audios preprocesados
    processed_output_directory = "process/Audios_Cardif/processed" 

    hangup_signature = (
        "data/hangout/"
        "BANCOLBI_20250910-180407_2863174_1757545447_1032414142_3204579981-all_fragment_signature1.wav"
    )

    run_full_batch_pipeline(
        campaign_directory=campaign_directory,
        processed_output_directory=processed_output_directory,
        hangup_signature=hangup_signature,
        summary_excel_name="audio_outputs_test.xlsx",
        mp3_bitrate="128k",
    )
