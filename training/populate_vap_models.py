"""
populate_vap_models.py
======================
Recorre un bucket de S3 buscando modelos de segmentación entrenados y los
registra en la tabla ``vap_models`` de la base de datos VAP.

Estructura esperada en S3
--------------------------
    {bucket}/{pais}/{sponsor}/model/{dir_con_model_}/
    {bucket}/{pais}/{sponsor}/model/{dir_con_model_}/time_priors_subtag.json

    • Se busca cualquier "directorio" (prefijo S3) cuyo nombre empiece con
      "model_" dentro del subnivel /model/ de cada sponsor.
    • Se toma la fecha del objeto config.json dentro del directorio como
      upload_date (proxy de cuándo se subió el modelo).

Lógica de sponsor_id
---------------------
    Se cruza el segmento {sponsor} de la ruta S3 con el campo
    ``marketing_campaigns.sponsor`` (sin diferenciar mayúsculas/minúsculas
    ni espacios). Si hay varias filas con el mismo nombre, se toma la
    primera (ORDER BY id ASC).

Duplicados
----------
    Antes de insertar se verifica si ya existe una fila con el mismo
    ``model_route``. Si existe, se omite (no se actualiza).

Uso
---
    python populate_vap_models.py --bucket aihubmodelos
    python populate_vap_models.py --bucket aihubmodelos --dry-run
    python populate_vap_models.py --bucket aihubmodelos --prefix Colombia/
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Ajustar el path para importar módulos del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database.dbConfig as dbcfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
_MODEL_SUBDIR        = "model"          # subnivel fijo entre sponsor/ y model_xxx/
_MODEL_DIR_PREFIX    = "model_"         # los directorios de modelo empiezan así
_TIME_PRIORS_FILE    = "time_priors_subtag.json"
_CONFIG_SENTINEL     = "config.json"    # archivo usado para detectar upload_date


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _list_all_keys(s3_client, bucket: str, prefix: str = "") -> list:
    """Devuelve todos los object keys bajo bucket/prefix usando paginación."""
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj)   # dict con Key, LastModified, Size, …
    return keys


def _discover_models(s3_client, bucket: str, prefix: str = "") -> list[dict]:
    """
    Recorre el bucket y devuelve una lista de dicts con la info de cada modelo
    encontrado.

    Cada dict tiene:
        pais, sponsor_name, model_dir_name, model_route,
        time_priors_route (o None), upload_date (datetime UTC)
    """
    logger.info("Listando objetos en s3://%s/%s …", bucket, prefix)
    all_objects = _list_all_keys(s3_client, bucket, prefix)
    logger.info("Total de objetos encontrados: %d", len(all_objects))

    # Indexar por key para lookup rápido de fechas
    key_meta: dict[str, dict] = {obj["Key"]: obj for obj in all_objects}

    # Recolectar prefijos únicos de nivel modelo:
    #   {pais}/{sponsor}/model/{model_dir}/
    # El key tiene al menos 4 componentes tras dividir por "/"
    model_prefixes: dict[str, dict] = {}   # model_prefix -> info dict

    for obj in all_objects:
        key: str = obj["Key"]
        parts = key.split("/")
        # Necesitamos al menos: pais / sponsor / "model" / model_dir / archivo
        if len(parts) < 5:
            continue

        pais        = parts[0]
        sponsor_seg = parts[1]
        subdir      = parts[2]
        model_dir   = parts[3]

        if subdir != _MODEL_SUBDIR:
            continue
        if not model_dir.startswith(_MODEL_DIR_PREFIX):
            continue

        model_prefix = f"{pais}/{sponsor_seg}/{subdir}/{model_dir}/"
        if model_prefix not in model_prefixes:
            model_route = f"{bucket}/{model_prefix.rstrip('/')}"
            model_prefixes[model_prefix] = {
                "pais":           pais,
                "sponsor_name":   sponsor_seg,
                "model_dir_name": model_dir,
                "model_route":    model_route,
                "model_prefix":   model_prefix,
                "upload_date":    None,
                "time_priors_route": None,
            }

    if not model_prefixes:
        logger.warning("No se encontraron directorios de modelo en s3://%s/%s", bucket, prefix)
        return []

    # Para cada modelo, determinar upload_date y time_priors_route
    for mp, info in model_prefixes.items():
        # upload_date = LastModified de config.json (o del primer objeto si no existe)
        config_key = f"{mp}{_CONFIG_SENTINEL}"
        if config_key in key_meta:
            info["upload_date"] = key_meta[config_key]["LastModified"]
        else:
            # Buscar cualquier objeto dentro de este prefijo
            candidates = [
                key_meta[k]["LastModified"]
                for k in key_meta
                if k.startswith(mp)
            ]
            info["upload_date"] = min(candidates) if candidates else datetime.now(tz=timezone.utc)

        # time_priors_route
        tp_key = f"{mp}{_TIME_PRIORS_FILE}"
        if tp_key in key_meta:
            info["time_priors_route"] = f"{bucket}/{tp_key}"

    logger.info("Modelos detectados: %d", len(model_prefixes))
    return list(model_prefixes.values())


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_sponsor_id(conn, sponsor_name: str) -> Optional[int]:
    """
    Busca el sponsor_id en marketing_campaigns cuyo campo sponsor coincida
    (case-insensitive, ignorando espacios extra) con sponsor_name.

    Se ordena por updated_at DESC para tomar el registro activo más reciente,
    descartando filas con borrado lógico o total que suelen quedar primero si
    se ordenara por id ASC.

    Devuelve None si no se encuentra.
    """
    sql = """
        SELECT id
        FROM   marketing_campaigns
        WHERE  LOWER(REPLACE(sponsor, ' ', '')) = LOWER(REPLACE(%s, ' ', ''))
        ORDER  BY updated_at DESC
        LIMIT  1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (sponsor_name,))
        row = cur.fetchone()
    return int(row[0]) if row else None


def _model_route_exists(conn, model_route: str) -> bool:
    """Devuelve True si ya existe una fila con ese model_route."""
    sql = "SELECT 1 FROM vap_models WHERE model_route = %s LIMIT 1"
    with conn.cursor() as cur:
        cur.execute(sql, (model_route,))
        return cur.fetchone() is not None


def _insert_model(conn, row: dict, dry_run: bool) -> bool:
    """
    Inserta una fila en vap_models.  Devuelve True si se insertó (o habría
    insertado en dry_run).
    """
    sql = """
        INSERT INTO vap_models
            (sponsor_id, model_route, time_priors_route, model_name, upload_date, tested)
        VALUES
            (%s, %s, %s, %s, %s, 0)
    """
    params = (
        row["sponsor_id"],
        row["model_route"],
        row.get("time_priors_route"),
        row["model_dir_name"],
        row["upload_date"],
    )

    if dry_run:
        logger.info("[DRY-RUN] INSERT vap_models: sponsor_id=%s | route=%s | name=%s | date=%s",
                    row["sponsor_id"], row["model_route"], row["model_dir_name"], row["upload_date"])
        return True

    with conn.cursor() as cur:
        cur.execute(sql, params)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def populate(bucket: str, prefix: str = "", dry_run: bool = False) -> None:
    # 1. Conectar a S3 y BD
    s3  = dbcfg.generate_s3_client()
    conn = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP,
    )

    # 2. Descubrir modelos en S3
    models = _discover_models(s3, bucket, prefix)
    if not models:
        logger.info("Nada que insertar.")
        return

    inserted = skipped_dup = skipped_no_sponsor = 0

    for info in models:
        model_route  = info["model_route"]
        sponsor_name = info["sponsor_name"]

        # 3. Buscar sponsor_id en la BD
        sponsor_id = _get_sponsor_id(conn, sponsor_name)
        if sponsor_id is None:
            logger.warning(
                "SKIP — sponsor '%s' no encontrado en marketing_campaigns "
                "(ruta: %s)", sponsor_name, model_route,
            )
            skipped_no_sponsor += 1
            continue

        info["sponsor_id"] = sponsor_id

        # 4. Verificar duplicado
        if _model_route_exists(conn, model_route):
            logger.info("SKIP — ya existe en vap_models: %s", model_route)
            skipped_dup += 1
            continue

        # 5. Insertar
        _insert_model(conn, info, dry_run)
        logger.info(
            "OK — sponsor_id=%d | model=%s | time_priors=%s | date=%s",
            sponsor_id, model_route,
            info.get("time_priors_route") or "N/A",
            info["upload_date"],
        )
        inserted += 1

    conn.close()
    logger.info(
        "Resumen: %d insertados | %d duplicados omitidos | %d sin sponsor",
        inserted, skipped_dup, skipped_no_sponsor,
    )
    if dry_run and inserted:
        logger.info("(dry-run activado: ninguna fila fue realmente insertada)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Poblar vap_models escaneando modelos en S3.",
    )
    parser.add_argument(
        "--bucket", required=True,
        help="Nombre del bucket S3 (ej. aihubmodelos)",
    )
    parser.add_argument(
        "--prefix", default="",
        help="Prefijo opcional para limitar el escaneo (ej. 'Colombia/')",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Muestra qué se insertaría sin modificar la BD",
    )
    args = parser.parse_args()

    populate(
        bucket  = args.bucket,
        prefix  = args.prefix,
        dry_run = args.dry_run,
    )
