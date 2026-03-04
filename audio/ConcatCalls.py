import os, traceback
import pandas as pd
from pydub import AudioSegment
from pydub.utils import which

def find_audio_concat_groups(download_path: str) -> pd.DataFrame:
    """
    Replica la lógica de agrupación/orden de concatenate_audios_main,
    pero solo devuelve un DataFrame con:
      - file_name (salida esperada -concat.mp3)
      - lead_id
      - num_concat (cantidad de audios a concatenar)
    """

    # 1) Reúne .mp3 y .wav
    audio_files = [
        f for f in os.listdir(download_path)
        if f.lower().endswith((".mp3", ".wav"))
    ]
    if not audio_files:
        return pd.DataFrame(columns=["file_name", "lead_id", "num_concat"])

    # 2) Extrae info del nombre (misma lógica)
    def extract_info(fname: str) -> dict:
        parts = fname.split("_")
        if len(parts) == 4:
            return dict(archivo=fname, fecha=parts[1], lead_id=parts[2])
        if len(parts) == 5:
            return dict(archivo=fname, fecha=parts[1], lead_id=parts[2])
        elif len(parts) == 6:
            return dict(archivo=fname, fecha=parts[1], lead_id=parts[2])
        else:
            raise ValueError(f"Nombre inesperado: {fname}")

    df = pd.DataFrame([extract_info(f) for f in audio_files])

    # 3) Solo lead_id con >=2 archivos, orden cronológico por "fecha"
    grupos = (
        df.groupby("lead_id")
          .filter(lambda g: len(g) > 1)
          .sort_values("fecha")
          .groupby("lead_id")["archivo"]
          .apply(list)
    )

    if grupos.empty:
        return pd.DataFrame(columns=["file_name", "lead_id", "num_concat"])

    # 4) Construye la salida esperada, igual que tu result_name
    rows = []
    for lead_id, files in grupos.items():
        result_name = f"{os.path.splitext(files[-1])[0]}-concat.mp3"
        rows.append({
            "file_name": result_name,
            "lead_id": lead_id,
            "num_concat": len(files),
        })

    return pd.DataFrame(rows, columns=["file_name", "lead_id", "num_concat"])



def concatenate_audios_main(download_path: str):
    """
    Concatena audios por lead_id (>=2), borra inputs si todo sale bien,
    y RETORNA:
      - df_groups: DataFrame con (file_name, lead_id, num_concat, status)
      - concatenated_inputs: lista de dicts con audios únicos concatenados:
            [{"lead_id":..., "output_file":..., "num_concat":..., "input_files":[...]}]
    """

    # -- 0. Confirmar que FFmpeg existe --
    AudioSegment.converter = which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
    if not os.path.isfile(AudioSegment.converter):
        raise RuntimeError(
            f"FFmpeg no encontrado en {AudioSegment.converter}. "
            "Instálalo o ajusta la ruta."
        )

    # -- 1. Reúne .mp3 y .wav --
    audio_files = [
        f for f in os.listdir(download_path)
        if f.lower().endswith((".mp3", ".wav"))
    ]
    if not audio_files:
        print("No hay audios mp3/wav en la carpeta.")
        df_empty = pd.DataFrame(columns=["file_name", "lead_id", "num_concat", "status"])
        return df_empty, []

    # -- 2. Extrae info del nombre --
    def extract_info(fname: str) -> dict:
        parts = fname.split("_")
        if len(parts) == 4:
            return dict(archivo=fname, fecha=parts[1], lead_id=parts[2], idagent=parts[3])
        if len(parts) == 5:
            return dict(archivo=fname, fecha=parts[1], lead_id=parts[2], idagent=parts[3], idclient=parts[4])
        if len(parts) == 6:
            return dict(archivo=fname, fecha=parts[1], lead_id=parts[2], idcall=parts[3], idclient=parts[4], phone=parts[5])
        raise ValueError(f"Nombre inesperado: {fname}")

    df = pd.DataFrame([extract_info(f) for f in audio_files])

    # -- 3. Obtén sólo lead_id con ≥2 archivos --
    grupos = (
        df.groupby("lead_id")
          .filter(lambda g: len(g) > 1)
          .sort_values("fecha")           # orden cronológico
          .groupby("lead_id")["archivo"]
          .apply(list)
    )

    if grupos.empty:
        print("No hay grupos con más de un audio para concatenar.")
        df_empty = pd.DataFrame(columns=["file_name", "lead_id", "num_concat", "status"])
        return df_empty, []

    # -- 4. Helpers --
    def concat(files, out_path):
        concat_audio = AudioSegment.empty()
        for f in files:
            ruta = os.path.join(download_path, f)
            if not os.path.isfile(ruta):
                raise FileNotFoundError(ruta)
            ext = os.path.splitext(f)[1][1:].lower()
            concat_audio += AudioSegment.from_file(ruta, format=ext)
        concat_audio.export(out_path, format="mp3")

    rows = []                  # para DataFrame de grupos (incluye num_concat)
    concatenated_inputs = []   # lista de audios únicos concatenados (inputs)

    for lead_id, files in grupos.items():
        result_name = f"{os.path.splitext(files[-1])[0]}-concat.mp3"
        out_path = os.path.join(download_path, result_name)

        # fila base del df de grupos
        row = {
            "file_name": result_name,
            "lead_id": lead_id,
            "num_concat": len(files),
            "status": None,
        }

        if os.path.exists(out_path):
            print(f"[SKIP] {out_path} ya existe.")
            row["status"] = "skipped_exists"
            rows.append(row)
            continue

        try:
            print(f"[INFO] Concatenando {len(files)} audios → {result_name}")
            concat(files, out_path)
        except Exception:
            print(f"[ERROR] Falló lead_id {lead_id}")
            traceback.print_exc()
            row["status"] = "failed"
            rows.append(row)
            continue  # no borres nada si falla
        else:
            # éxito: registra inputs únicos de este grupo
            concatenated_inputs.append({
                "lead_id": lead_id,
                "output_file": result_name,
                "num_concat": len(files),
                "input_files": files[:]   # copia
            })

            # borra inputs
            for f in files:
                try:
                    os.remove(os.path.join(download_path, f))
                except OSError as e:
                    print(f"[WARN] No se pudo borrar {f}: {e}")

            row["status"] = "concatenated"
            rows.append(row)

    df_groups = pd.DataFrame(rows, columns=["file_name", "lead_id", "num_concat", "status"])
    print("Proceso terminado.")
    return df_groups, concatenated_inputs