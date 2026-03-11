"""
ModelSelector — resuelve el mejor modelo de segmentación disponible para un sponsor.

Prioridad de selección
----------------------
1. Modelo más reciente con ``tested=1`` registrado en ``vap_models`` para el sponsor.
2. Si no hay ninguno marcado como tested, el modelo más reciente sin importar estado.
3. Fallback global: variables de entorno ``TEXT_MODEL_DIR`` / ``TIME_PRIORS_JSON``.

Caché local
-----------
Los modelos se descargan desde S3 la primera vez y quedan en ``MODEL_CACHE_DIR``
(env var, por defecto ``"model_cache"``).  Las ejecuciones siguientes reutilizan el
directorio local sin volver a contactar S3.

Estructura de rutas S3 esperada
--------------------------------
    model_route = "{bucket}/{pais}/{sponsor}/{model_dir}"
    Ej.          "aihubmodelos/Colombia/Bancolombia/model_output_col_multitag_v3"

El archivo ``time_priors.json`` puede estar:
    * Dentro del directorio del modelo (``model_route/time_priors.json``), ó
    * En una ruta independiente indicada en ``vap_models.time_priors_route``.
"""

import logging
import os
from typing import Optional, Tuple

import database.dbConfig as dbcfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
_DEFAULT_CACHE_ROOT: str = os.getenv("MODEL_CACHE_DIR", "model_cache")
_TIME_PRIORS_FILENAME: str = "time_priors.json"


# ---------------------------------------------------------------------------
# Consulta a vap_models
# ---------------------------------------------------------------------------

def _query_best_model(conn, sponsor_id: int) -> Optional[dict]:
    """
    Devuelve el mejor registro de ``vap_models`` para el sponsor.

    Criterio: primero los ``tested=1``, luego los más recientes por ``upload_date``.
    Devuelve ``None`` si no existe ningún modelo para ese sponsor.
    """
    sql = """
        SELECT
            id,
            sponsor_id,
            model_route,
            time_priors_route,
            model_name,
            upload_date,
            tested
        FROM  vap_models
        WHERE sponsor_id = %s
        ORDER BY tested DESC, upload_date DESC
        LIMIT 1
    """
    try:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(sql, (int(sponsor_id),))
            return cur.fetchone()
    except Exception as exc:
        logger.error(
            "[ModelSelector] Error al consultar vap_models para "
            "sponsor_id=%s: %s", sponsor_id, exc
        )
        return None


# ---------------------------------------------------------------------------
# Parsing de rutas S3
# ---------------------------------------------------------------------------

def _parse_s3_route(model_route: str) -> Tuple[str, str]:
    """
    Convierte ``"bucket/key/..."`` o ``"s3://bucket/key/..."`` en ``(bucket, key)``.

    Raises
    ------
    ValueError si el formato no es válido.
    """
    route = model_route.strip()
    if route.startswith("s3://"):
        route = route[len("s3://"):]
    parts = route.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"model_route inválida: '{model_route}'. "
            "Formato esperado: 'bucket/pais/sponsor/modelo' o 's3://bucket/pais/sponsor/modelo'."
        )
    return parts[0], parts[1]


# ---------------------------------------------------------------------------
# Caché local
# ---------------------------------------------------------------------------

def _local_cache_path(model_route: str, cache_root: str) -> str:
    """
    Mapea una ``model_route`` S3 a un directorio local bajo ``cache_root``.

    Ejemplo:
        model_route = "aihubmodelos/Colombia/Bancolombia/model_v3"
        → "<cache_root>/aihubmodelos/Colombia/Bancolombia/model_v3"
    """
    # Normalizar: quitar prefijo s3:// si lo hubiera
    route = model_route.strip()
    if route.startswith("s3://"):
        route = route[len("s3://"):]
    # Convertir separadores
    relative = route.replace("/", os.sep)
    return os.path.join(cache_root, relative)


def _model_is_cached(local_dir: str) -> bool:
    """
    Devuelve ``True`` si ``local_dir`` parece un directorio de modelo válido,
    es decir, contiene al menos ``config.json``.
    """
    return (
        os.path.isdir(local_dir)
        and os.path.exists(os.path.join(local_dir, "config.json"))
    )


# ---------------------------------------------------------------------------
# Descarga desde S3
# ---------------------------------------------------------------------------

def _download_model_from_s3(bucket: str, key: str, local_dir: str) -> None:
    """
    Descarga todos los objetos bajo ``s3://bucket/key/`` en ``local_dir``.

    Raises
    ------
    FileNotFoundError si el prefijo S3 está vacío.
    """
    s3 = dbcfg.generate_s3_client()
    prefix = key.rstrip("/") + "/"

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    downloaded = 0
    for page in pages:
        for obj in page.get("Contents", []):
            obj_key = obj["Key"]
            relative = obj_key[len(prefix):]
            if not relative:          # objeto raíz vacío, ignorar
                continue
            local_file = os.path.join(local_dir, relative.replace("/", os.sep))
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            logger.info("Descargando s3://%s/%s → %s", bucket, obj_key, local_file)
            s3.download_file(bucket, obj_key, local_file)
            downloaded += 1

    if downloaded == 0:
        raise FileNotFoundError(
            f"No se encontraron archivos en s3://{bucket}/{prefix}. "
            "Verifica que model_route sea correcto y que el modelo esté subido a S3."
        )
    logger.info(
        "[ModelSelector] Descarga completada: %d archivos → %s", downloaded, local_dir
    )


# ---------------------------------------------------------------------------
# Fallback global
# ---------------------------------------------------------------------------

def _fallback_model() -> Tuple[Optional[str], Optional[str]]:
    """Devuelve (TEXT_MODEL_DIR, TIME_PRIORS_JSON) desde variables de entorno."""
    return os.getenv("TEXT_MODEL_DIR"), os.getenv("TIME_PRIORS_JSON")


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def resolve_model_for_sponsor(
    conn,
    sponsor_id: int,
    local_cache_root: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resuelve el mejor modelo disponible para un sponsor y devuelve sus rutas locales.

    Parámetros
    ----------
    conn : conexión MySQL activa (dbConfig.conectar)
    sponsor_id : ID numérico del sponsor
    local_cache_root : directorio raíz de caché local (por defecto MODEL_CACHE_DIR env var)

    Retorna
    -------
    (local_model_dir, local_time_priors_json)
        Ambas son rutas locales listas para usar.
        ``local_time_priors_json`` puede ser ``None`` si no se encuentra el archivo.

    Comportamiento de fallback
    --------------------------
    Si no existe modelo en ``vap_models`` o la descarga falla, se devuelven los
    valores de las variables de entorno ``TEXT_MODEL_DIR`` / ``TIME_PRIORS_JSON``.
    """
    cache_root = local_cache_root or _DEFAULT_CACHE_ROOT

    # ------------------------------------------------------------------ #
    # 1. Consultar vap_models
    # ------------------------------------------------------------------ #
    row = _query_best_model(conn, sponsor_id)

    if row is None:
        logger.info(
            "[ModelSelector] Sin modelo personalizado para sponsor_id=%s. "
            "Usando fallback global (TEXT_MODEL_DIR).",
            sponsor_id,
        )
        return _fallback_model()

    model_route       = row["model_route"]
    time_priors_route = row.get("time_priors_route")   # puede ser None
    model_name        = row["model_name"]
    upload_date       = row["upload_date"]
    tested            = bool(row["tested"])

    logger.info(
        "[ModelSelector] Modelo seleccionado para sponsor_id=%s: "
        "'%s'  (upload=%s, tested=%s)",
        sponsor_id, model_name, upload_date, tested,
    )

    # ------------------------------------------------------------------ #
    # 2. Resolver directorio local del modelo
    # ------------------------------------------------------------------ #
    local_model_dir = _local_cache_path(model_route, cache_root)

    if not _model_is_cached(local_model_dir):
        logger.info(
            "[ModelSelector] Modelo no encontrado en caché local (%s). "
            "Descargando desde S3...",
            local_model_dir,
        )
        try:
            bucket, key = _parse_s3_route(model_route)
            os.makedirs(local_model_dir, exist_ok=True)
            _download_model_from_s3(bucket, key, local_model_dir)
        except Exception as exc:
            logger.error(
                "[ModelSelector] No se pudo descargar el modelo '%s' desde S3: %s. "
                "Usando fallback global.",
                model_route, exc,
            )
            return _fallback_model()
    else:
        logger.info(
            "[ModelSelector] Modelo en caché local: %s", local_model_dir
        )

    # ------------------------------------------------------------------ #
    # 3. Resolver time_priors.json
    # ------------------------------------------------------------------ #
    local_time_priors: Optional[str] = None

    if time_priors_route:
        # Ruta explícita en la tabla
        local_time_priors = _local_cache_path(time_priors_route, cache_root)
        if not os.path.exists(local_time_priors):
            # Intentar descargar el archivo individual
            try:
                tp_bucket, tp_key = _parse_s3_route(time_priors_route)
                os.makedirs(os.path.dirname(local_time_priors), exist_ok=True)
                s3 = dbcfg.generate_s3_client()
                s3.download_file(tp_bucket, tp_key, local_time_priors)
                logger.info(
                    "[ModelSelector] time_priors.json descargado: %s", local_time_priors
                )
            except Exception as exc:
                logger.warning(
                    "[ModelSelector] No se pudo descargar time_priors_route='%s': %s. "
                    "Buscando dentro del directorio del modelo.",
                    time_priors_route, exc,
                )
                local_time_priors = None

    if local_time_priors is None or not os.path.exists(local_time_priors):
        # Convención: buscar dentro del directorio del modelo
        candidate = os.path.join(local_model_dir, _TIME_PRIORS_FILENAME)
        if os.path.exists(candidate):
            local_time_priors = candidate
            logger.info(
                "[ModelSelector] time_priors.json encontrado dentro del modelo: %s",
                local_time_priors,
            )
        else:
            # Último recurso: variable de entorno
            env_tp = os.getenv("TIME_PRIORS_JSON")
            if env_tp and os.path.exists(env_tp):
                local_time_priors = env_tp
                logger.warning(
                    "[ModelSelector] time_priors.json no encontrado en el modelo personalizado. "
                    "Usando TIME_PRIORS_JSON del entorno: %s",
                    local_time_priors,
                )
            else:
                logger.warning(
                    "[ModelSelector] No se encontró time_priors.json en ninguna fuente. "
                    "El caller deberá manejar el None."
                )
                local_time_priors = None

    return local_model_dir, local_time_priors
