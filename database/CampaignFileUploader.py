"""
CampaignFileUploader
====================
Sube el archivo ``topics_transcripts_convers.xlsx`` generado en ``misc/``
durante el procesamiento de una campaña hacia la ruta correspondiente en S3,
usando la información de ``marketing_campaigns`` en la base de datos.

Diseñado para ser invocado al final del pipeline de VAP (main.py) de forma
automática, eliminando la necesidad de ejecutar el script manual.
"""

import logging
import os
from datetime import datetime
from typing import Optional, Tuple

import database.dbConfig as dbcfg

TARGET_FILENAME = "topics_transcripts_convers.xlsx"

log = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _find_target_file(campaign_directory: str) -> Optional[str]:
    """Devuelve la ruta al archivo objetivo en ``misc/`` o ``None``."""
    target = os.path.join(campaign_directory, "misc", TARGET_FILENAME)
    return target if os.path.isfile(target) else None


def _get_s3_base_path(conn, prefix_name: str) -> Optional[str]:
    """
    Busca en ``marketing_campaigns`` la columna ``s3`` cuya columna ``path``
    contenga el *prefix_name* (con o sin trailing ``_``).
    """
    candidates = [prefix_name]
    normalized = prefix_name.rstrip("_")
    if normalized != prefix_name:
        candidates.append(normalized)

    sql = """
        SELECT s3, path
        FROM marketing_campaigns
        WHERE FIND_IN_SET(%s, REPLACE(path, ' ', '')) > 0
        LIMIT 1
    """

    cursor = conn.cursor()
    try:
        for candidate in candidates:
            log.info("Buscando en BD prefijo dentro de path: %s", candidate)
            cursor.execute(sql, (candidate,))
            row = cursor.fetchone()
            if row:
                s3_value, path_value = row
                log.info(
                    "Coincidencia encontrada. candidate=%s | path_db=%s | s3=%s",
                    candidate, path_value, s3_value,
                )
                return s3_value
    finally:
        cursor.close()

    return None


def _build_destination_key(
    db_s3_path: str,
    final_filename: str,
    bucket_override: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Construye ``(bucket, key)`` a partir de la ruta S3 almacenada en BD.

    Si se proporciona *bucket_override* se usa ese bucket en lugar del que
    aparece en la URI de la BD.
    """
    normalized = db_s3_path.strip()

    if normalized.startswith("s3://"):
        normalized = normalized[5:]

    first_slash = normalized.find("/")
    if first_slash == -1:
        raise ValueError(f"Ruta S3 inválida en BD: {db_s3_path}")

    bucket = bucket_override or normalized[:first_slash]
    base_path = normalized[first_slash + 1:].strip("/")
    key = f"{base_path}/conversations/{final_filename}"
    return bucket, key


def _s3_file_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey"):
            return False
        raise


# ── función principal ────────────────────────────────────────────────────────

def upload_campaign_topics_file(
    campaign_directory: str,
    prefix: str,
    conn=None,
    date_str: Optional[str] = None,
    bucket_override: Optional[str] = None,
) -> dict:
    """
    Sube ``topics_transcripts_convers.xlsx`` de la campaña a S3.

    Parámetros
    ----------
    campaign_directory : str
        Ruta local del directorio de la campaña (ej. ``process/BANC_/``).
    prefix : str
        Prefijo / nombre de la campaña (ej. ``BANC_``).
    conn : mysql.connector.connection, optional
        Conexión a la BD VAP. Si es ``None`` se crea una nueva.
    date_str : str, optional
        Fecha para el nombre del archivo destino (``YYYYMMDD``).
        Por defecto usa la fecha actual.
    bucket_override : str, optional
        Si se desea forzar un bucket diferente al almacenado en BD.

    Retorna
    -------
    dict
        ``{"prefix", "status", "message", "s3_uri"}``
        donde *status* puede ser ``UPLOADED``, ``SKIPPED`` o ``ERROR``.
    """
    result = {
        "prefix": prefix,
        "status": "PENDING",
        "message": "",
        "s3_uri": "",
    }

    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    close_conn = False
    try:
        # 1. Buscar archivo local
        local_file = _find_target_file(campaign_directory)
        if not local_file:
            result["status"] = "SKIPPED"
            result["message"] = (
                f"No se encontró {TARGET_FILENAME} en "
                f"{os.path.join(campaign_directory, 'misc')}"
            )
            print(f"[CampaignFileUploader] {result['message']}")
            return result

        # 2. Obtener ruta base S3 desde BD
        if conn is None:
            conn = dbcfg.conectar(
                HOST=dbcfg.HOST_DB_VAP,
                DATABASE=dbcfg.DB_NAME_VAP,
                USERNAME=dbcfg.USER_DB_VAP,
                PASSWORD=dbcfg.PASSWORD_DB_VAP,
            )
            close_conn = True

        db_s3_path = _get_s3_base_path(conn, prefix)
        if not db_s3_path:
            result["status"] = "SKIPPED"
            result["message"] = (
                f"No se encontró prefijo {prefix} dentro de "
                "marketing_campaigns.path"
            )
            print(f"[CampaignFileUploader] {result['message']}")
            return result

        # 3. Construir nombre destino con fecha
        safe_prefix = prefix.rstrip("_")
        final_name = f"topics_transcripts_convers_{safe_prefix}_{date_str}.xlsx"
        bucket, key = _build_destination_key(
            db_s3_path, final_name, bucket_override
        )

        # 4. Verificar si ya existe en S3
        s3_client = dbcfg.generate_s3_client()

        if _s3_file_exists(s3_client, bucket, key):
            result["status"] = "SKIPPED"
            result["message"] = "El archivo ya existe en S3"
            result["s3_uri"] = f"s3://{bucket}/{key}"
            print(
                f"[CampaignFileUploader] Archivo ya existe, se omite: "
                f"s3://{bucket}/{key}"
            )
            return result

        # 5. Subir
        print(f"[CampaignFileUploader] Subiendo {local_file} → s3://{bucket}/{key}")
        s3_client.upload_file(local_file, bucket, key)

        result["status"] = "UPLOADED"
        result["message"] = f"Archivo subido correctamente desde {local_file}"
        result["s3_uri"] = f"s3://{bucket}/{key}"
        print(f"[CampaignFileUploader] {result['message']}")

    except Exception as e:
        result["status"] = "ERROR"
        result["message"] = str(e)
        print(f"[CampaignFileUploader][ERROR] prefix={prefix}: {e}")
    finally:
        if close_conn and conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    return result
