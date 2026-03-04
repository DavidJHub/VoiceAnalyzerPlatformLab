import os
from typing import List, Optional

# Ajusta estos imports a tu proyecto
from utils.VapUtils import jsonDecomposeSentencesHighlight
from database.S3Loader import cargar_archivos_json_a_s3


import os
from typing import List, Optional, Dict, Any, Tuple

import database.dbConfig as dbcfg

# Ajusta imports a tu repo
# from tu_modulo import jsonDecomposeSentencesHighlight, cargar_archivos_json_a_s3


# -----------------------------
# Helpers FS
# -----------------------------
def list_subdirs(root_process: str) -> List[str]:
    root_process = os.path.abspath(root_process)
    if not os.path.isdir(root_process):
        raise FileNotFoundError(f"No existe el directorio root: {root_process}")
    subdirs = []
    for name in os.listdir(root_process):
        full = os.path.join(root_process, name)
        if os.path.isdir(full):
            subdirs.append(full)
    subdirs.sort()
    return subdirs


def list_jsons_in_dir(dir_path: str) -> List[str]:
    files = []
    for name in os.listdir(dir_path):
        if name.lower().endswith(".json"):
            files.append(os.path.join(dir_path, name))
    files.sort()
    return files


# -----------------------------
# DB helpers (MySQL-like)
# -----------------------------
def _fetchone_dict(cursor) -> Optional[Dict[str, Any]]:
    row = cursor.fetchone()
    if not row:
        return None
    # cursor puede devolver dict o tuple según config
    if isinstance(row, dict):
        return row
    # si es tuple, mapeamos con cursor.description
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


def _fetchall_dict(cursor) -> List[Dict[str, Any]]:
    rows = cursor.fetchall() or []
    if not rows:
        return []
    if isinstance(rows[0], dict):
        return rows
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in rows]


def get_campaign_info_for_subdir(conexion, subdir_name: str) -> Optional[Dict[str, Any]]:
    """
    Busca en marketing_campaigns una fila donde path contenga el subdir_name.
    Retorna {country, sponsor, campaign_id, path} o None si no encuentra.
    """
    q = """
        SELECT country, sponsor, campaign_id, path
        FROM marketing_campaigns
        WHERE LOCATE(%s, path) > 0
        LIMIT 5
    """
    # LIMIT 5 para detectar ambigüedad (múltiples matches)
    cur = conexion.cursor()
    cur.execute(q, (subdir_name,))
    rows = _fetchall_dict(cur)
    cur.close()

    if not rows:
        return None

    if len(rows) > 1:
        # Si hay varias coincidencias, tomamos la primera pero avisamos.
        # Puedes cambiar la regla aquí (p. ej. priorizar match exacto por token)
        print(f"[WARN] Subdir '{subdir_name}' matchea {len(rows)} filas en marketing_campaigns. "
              f"Usando la primera (campaign_id={rows[0].get('campaign_id')}).")

    return rows[0]


def get_folder_date_for_campaign(conexion, campaign_id: Any) -> Optional[str]:
    """
    Busca folder_date en vap_status por campaign_id.
    Si hay varias filas, tomamos la más reciente por folder_date (si es comparable).
    Ajusta ORDER BY si tienes otra columna de recencia.
    """
    q = """
        SELECT folder_date
        FROM vap_status
        WHERE campaign_id = %s
        ORDER BY folder_date DESC
        LIMIT 1
    """
    cur = conexion.cursor()
    cur.execute(q, (campaign_id,))
    row = _fetchone_dict(cur)
    cur.close()
    if not row:
        return None
    return row.get("folder_date")


def build_s3_prefix(country: str, sponsor: str, folder_date: str) -> str:
    # documentos.aihub  country/sponsor/transcript_sentences/folder_date/
    # (bucket se pasa aparte)
    country = str(country).strip().strip("/")
    sponsor = str(sponsor).strip().strip("/")
    folder_date = str(folder_date).strip().strip("/")
    return f"{country}/{sponsor}/transcript_sentences/{folder_date}/"


# -----------------------------
# Main batch
# -----------------------------
def run_full_automation(
    root_process: str = "process",
    *,
    output_subdir: str = "transcript_sentences",
    keywords_good: Optional[List[str]] = None,
    keywords_bad: Optional[List[str]] = None,
    bucket: str = "documentos.aihub",
) -> None:
    if keywords_good is None:
        keywords_good = []
    if keywords_bad is None:
        keywords_bad = []

    # 1) Conexión DB
    print("[INFO] Conectando a DB...")
    conexion = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )
    print("[OK] Conectado.")

    root_process_abs = os.path.abspath(root_process)
    subdirs = list_subdirs(root_process_abs)

    if not subdirs:
        print(f"[WARN] No hay subdirectorios dentro de: {root_process_abs}")
        return

    print(f"[INFO] Root: {root_process_abs}")
    print(f"[INFO] Subdirectorios encontrados: {len(subdirs)}")

    for campaign_dir in subdirs:
        subdir_name = os.path.basename(campaign_dir.rstrip("\\/"))
        print("\n" + "=" * 90)
        print(f"[INFO] Subdir: {subdir_name}")

        # 2) Consultar marketing_campaigns por path que contenga subdir_name
        info = get_campaign_info_for_subdir(conexion, subdir_name)
        if not info:
            print(f"[WARN] No se encontró match en marketing_campaigns para '{subdir_name}'. Se omite upload.")
            continue

        country = info.get("country")
        sponsor = info.get("sponsor")
        campaign_id = info.get("campaign_id")

        if country is None or sponsor is None or campaign_id is None:
            print(f"[ERROR] Fila incompleta en marketing_campaigns para '{subdir_name}': {info}. Se omite.")
            continue

        # 3) Consultar folder_date en vap_status con campaign_id
        folder_date = get_folder_date_for_campaign(conexion, campaign_id)
        if not folder_date:
            print(f"[WARN] No se encontró folder_date en vap_status para campaign_id={campaign_id}. Se omite upload.")
            continue

        s3_prefix = build_s3_prefix(country, sponsor, folder_date)
        print(f"[INFO] DB -> country={country}, sponsor={sponsor}, campaign_id={campaign_id}, folder_date={folder_date}")
        print(f"[INFO] S3 destino -> bucket={bucket}, prefix={s3_prefix}")

        # 4) Procesar JSONs locales: process/SUBDIR/*.json -> process/SUBDIR/transcript_sentences/
        json_files = list_jsons_in_dir(campaign_dir)
        if not json_files:
            print(f"[WARN] No hay JSONs en {campaign_dir}. Se omite.")
            continue

        output_dir = os.path.join(campaign_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        print(f"[INFO] JSONs a procesar: {len(json_files)}")
        print(f"[INFO] Output dir: {output_dir}")

        for fp in json_files:
            try:
                jsonDecomposeSentencesHighlight(
                    file_path=fp,
                    output_dir=output_dir,
                    keywords_good=keywords_good,
                    keywords_bad=keywords_bad,
                )
            except Exception as e:
                print(f"[ERROR] Highlight falló en {fp}: {e}")

        # 5) Subir output_dir a S3 en prefix calculado
        try:
            cargar_archivos_json_a_s3(output_dir + "/", bucket, s3_prefix)
            print("[OK] Upload completado.")
        except Exception as e:
            print(f"[ERROR] Upload falló para '{subdir_name}': {e}")

    # Cierre DB
    try:
        conexion.close()
    except Exception:
        pass
    print("\n[OK] Automatización completa.")


if __name__ == "__main__":
    ROOT_PROCESS = r"process"  # o ruta absoluta
    BUCKET = "documentos.aihub"

    # Keywords (ajusta a lo tuyo)
    KEYWORDS_GOOD = ["integral", "hola", "buenos días"]
    KEYWORDS_BAD = []

    run_full_automation(
        ROOT_PROCESS,
        bucket=BUCKET,
        keywords_good=KEYWORDS_GOOD,
        keywords_bad=KEYWORDS_BAD,
    )
