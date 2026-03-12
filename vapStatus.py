import os
import re
import json
import time
import datetime
import traceback
from datetime import timedelta

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

import database.dbConfig as dbcfg
from utils.VapFunctions import get_latest_directories
from database.dealerGen import parse_s3_url
from utils.campaignMetrics import count_s3_audio_files


# =========================
# === CONFIG / CONSTANTS ===
# =========================
RESET_WINDOW_HOURS = int(os.environ.get("RESET_WINDOW_HOURS", "10"))

# IMPORTANTE: el run_key debe ser el mismo para should_run_resets e insert_script_run
RUN_KEY = "vap_daily_run"


# =========================
# === QUERY HELPERS ===
# =========================
def fetch_campaigns_listed():
    """
    Campañas listadas = todas las filas de vap_status con:
      - file_count > 0
      - folder_date <= ayer (UTC_DATE()-1)
    Retorna lista de dicts con columnas:
      campaign_prefix, country, sponsor, campaign, campaign_id,
      folder_date, file_count, processed, assigned, machine
    """
    conn = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )
    try:
        cursor = conn.cursor()

        # Si folder_date siempre es YYYY-MM-DD, esto está bien.
        # Si a veces viene YYYYMMDD, dímelo y lo ajusto con CASE/REGEXP.
        sql = """
        SELECT
            campaign_prefix,
            country,
            sponsor,
            campaign,
            campaign_id,
            folder_date,
            file_count,
            processed,
            assigned,
            machine
        FROM vap_status
        WHERE file_count > 0
          AND STR_TO_DATE(folder_date, '%Y-%m-%d') <= DATE_SUB(UTC_DATE(), INTERVAL 1 DAY)
        ORDER BY sponsor, campaign;
        """

        cursor.execute(sql)
        rows = cursor.fetchall()
        cols = [c[0] for c in cursor.description]
        return [dict(zip(cols, r)) for r in rows]

    finally:
        cursor.close()
        conn.close()


def should_run_resets(window_hours: int = RESET_WINDOW_HOURS, run_key: str = RUN_KEY) -> bool:
    """
    Ejecuta reset si:
      - Nunca ha existido un run con did_reset=1 para este run_key, o
      - Ya pasaron >= window_hours desde el run_end_time del último run que sí hizo reset.

    OJO: Esto evita el bug de "ventana deslizante". Se ancla al último reset real.
    """
    conn = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT run_end_time
            FROM vap_run
            WHERE run_key = %s AND did_reset = 1
            ORDER BY run_end_time DESC
            LIMIT 1
        """, (run_key,))
        row = cursor.fetchone()

        # primera vez => resetea
        if row is None or row[0] is None:
            return True

        last_reset_end = row[0]

        # ahora en UTC desde MySQL para consistencia
        cursor.execute("SELECT UTC_TIMESTAMP()")
        now_utc = cursor.fetchone()[0]

        return (now_utc - last_reset_end) >= timedelta(hours=window_hours)

    finally:
        try:
            cursor.close()
        except Exception:
            pass
        conn.close()


def insert_script_run(
    run_key: str,
    run_started_utc,
    run_finished_utc,
    exec_seconds: float,
    campaigns_listed: list,
    warnings_exceptions: str | None,
    did_reset: int
):
    """
    Inserta un registro histórico del run en la tabla vap_run.

    Columnas esperadas en vap_run:
      - run_key
      - run_start_time
      - run_end_time
      - exec_time
      - warning_exceptions
      - campaigns
      - did_reset
    """
    conn = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )
    try:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO vap_run (
                run_key,
                run_start_time,
                run_end_time,
                exec_time,
                warning_exceptions,
                campaigns,
                did_reset
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            run_key,
            run_started_utc,
            run_finished_utc,
            exec_seconds,
            warnings_exceptions,
            json.dumps(campaigns_listed, ensure_ascii=False),
            int(did_reset)
        ))

        conn.commit()
    finally:
        cursor.close()
        conn.close()


# =========================
# === STATUS RESET FUNCS ===
# =========================
def reset_all_processed_to_zero():
    """Sets processed=0 for all rows in vap_status."""
    conn = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )
    cursor = conn.cursor()
    cursor.execute("UPDATE vap_status SET processed = 0")
    conn.commit()
    cursor.close()
    conn.close()
    print("All 'processed' values have been reset to 0.")


def reset_all_assigned_to_zero():
    """Sets assigned=0 for all rows in vap_status."""
    conn = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )
    cursor = conn.cursor()
    cursor.execute("UPDATE vap_status SET assigned = 0")
    conn.commit()
    cursor.close()
    conn.close()
    print("All 'assigned' values have been reset to 0.")


# =========================
# === S3 / DATE HELPERS ===
# =========================
def parse_folder_date(s3_path: str) -> str:
    """
    Given an S3 subfolder path ending with YYYYMMDD/ -> returns YYYY-MM-DD.
    If not matching, returns the last path token.
    """
    if not s3_path or not isinstance(s3_path, str):
        return None

    s3_path = s3_path.rstrip('/')
    date_candidate = s3_path.split('/')[-1]

    if re.match(r'^\d{8}$', date_candidate):
        year = date_candidate[:4]
        month = date_candidate[4:6]
        day = date_candidate[6:]
        return f"{year}-{month}-{day}"

    return date_candidate


def get_latest_date_folder(s3_url, s3_client):
    if not s3_url or not isinstance(s3_url, str):
        return None

    bucket, prefix = parse_s3_url(s3_url)
    directories = get_latest_directories(bucket, prefix, s3_client)

    if not directories:
        return None

    folder_path = directories[0].rstrip('/')
    folder_name = folder_path.split('/')[-1]
    return folder_name


# =========================
# === VAP STATUS UPSERT ===
# =========================
def upsert_folder_status(country, sponsor, campaign, campaign_id, campaign_prefix, folder_date, file_count):
    """
    Insert/update vap_status keyed by campaign_prefix.
    - If folder_date changes, processed resets to 0.
    """
    conn = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )
    cursor = conn.cursor()

    # OJO: tú dijiste que vap_status ya tiene assigned y machine.
    # Este CREATE TABLE solo aplica si alguien corre en BD vacía; lo dejo consistente.
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS vap_status (
        campaign_prefix VARCHAR(255) PRIMARY KEY,
        country VARCHAR(255),
        sponsor VARCHAR(255),
        campaign VARCHAR(255),
        campaign_id INT,
        folder_date VARCHAR(20),
        file_count INT,
        processed BOOLEAN,
        assigned BOOLEAN,
        machine INT
    );
    """
    cursor.execute(create_table_sql)

    upsert_sql = """
    INSERT INTO vap_status (
        country, sponsor, campaign, campaign_id,
        campaign_prefix, folder_date, file_count,
        processed, assigned, machine
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE, 0, NULL)
    ON DUPLICATE KEY UPDATE
        country = VALUES(country),
        sponsor = VALUES(sponsor),
        campaign = VALUES(campaign),
        campaign_id = VALUES(campaign_id),
        folder_date = VALUES(folder_date),
        file_count  = VALUES(file_count),
        processed   = CASE
            WHEN vap_status.folder_date <> VALUES(folder_date)
                 THEN 0
            ELSE vap_status.processed
        END
    """
    cursor.execute(upsert_sql, (
        country, sponsor, campaign, campaign_id,
        campaign_prefix, folder_date, file_count
    ))
    conn.commit()
    cursor.close()
    conn.close()


def fill_folder_processing_status():
    """
    1) Read marketing_campaigns
    2) Find newest date folder from S3
    3) Count audios per subprefix
    4) Upsert into vap_status
    """
    conn = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )
    cursor = conn.cursor()
    cursor.execute("SELECT country, sponsor, campaign, campaign_id, path, s3 FROM marketing_campaigns")
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    cursor.close()
    conn.close()

    s3_client = dbcfg.generate_s3_client()

    for _, row in df.iterrows():
        country = row["country"]
        sponsor = row["sponsor"]
        campaign = row["campaign"]
        campaign_id = row["campaign_id"]
        path = row["path"]
        s3_url = row["s3"]

        folder_date = get_latest_date_folder(s3_url, s3_client)
        if not folder_date:
            continue

        try:
            bucket, base_prefix = parse_s3_url(s3_url)
        except Exception:
            continue

        final_prefix = f"{base_prefix}{folder_date}/"
        subprefixes = [x.strip() for x in str(path).split(",") if x.strip()]

        total_audio_count = 0
        for subp in subprefixes:
            total_audio_count += count_s3_audio_files(bucket, subp, final_prefix, s3_client)

        upsert_folder_status(
            country, sponsor, campaign, campaign_id,
            path, folder_date, total_audio_count
        )

    return df


# =========================
# === ASSIGNMENT UTILITIES ===
# =========================
def reset_status_except_machines(keep_machines=None):
    """
    Resetea processed=0 y assigned=0 en vap_status para TODAS las filas,
    excepto aquellas con assigned=1 y machine en keep_machines.
    """
    def _normalize(arg):
        if arg is None:
            return []
        if isinstance(arg, int):
            return [arg]
        if isinstance(arg, str):
            parts = [p.strip() for p in re.split(r"[,\s]+", arg) if p.strip()]
            out = []
            for p in parts:
                try:
                    out.append(int(p))
                except ValueError:
                    pass
            return out
        try:
            return [int(x) for x in arg]
        except Exception:
            return []

    machines = _normalize(keep_machines)

    conn = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )
    try:
        cursor = conn.cursor()

        if not machines:
            cursor.execute("""
                UPDATE vap_status
                SET processed = 0,
                    assigned  = 0
            """)
        else:
            placeholders = ", ".join(["%s"] * len(machines))
            cursor.execute(f"""
                UPDATE vap_status
                SET processed = 0,
                    assigned  = 0
                WHERE NOT (assigned = 1 AND machine IN ({placeholders}))
            """, machines)

        conn.commit()
        print(f"Estado reseteado. Filas afectadas: {cursor.rowcount}")

    finally:
        try:
            cursor.close()
        except Exception:
            pass
        conn.close()


def mark_status_as_processed(campaign_prefixes):
    """Set assigned=1 for rows whose campaign_prefix matches provided patterns."""
    if not campaign_prefixes or not isinstance(campaign_prefixes, (list, tuple)):
        print("campaign_prefixes inválido, debe ser lista o tupla.")
        return

    conn = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )
    try:
        cursor = conn.cursor()

        sql = """
            UPDATE vap_status
            SET assigned = 1
            WHERE campaign_prefix LIKE %s
        """

        rows_updated = 0
        for prefix in campaign_prefixes:
            cursor.execute(sql, (f"%{prefix}%",))
            rows_updated += cursor.rowcount

        conn.commit()
        print(f"{rows_updated} filas actualizadas: assigned=1")

    finally:
        try:
            cursor.close()
        except Exception:
            pass
        conn.close()


def mark_status_as_unprocessed(campaign_prefixes):
    """Set assigned=0 for rows whose campaign_prefix matches provided patterns."""
    if not campaign_prefixes or not isinstance(campaign_prefixes, (list, tuple)):
        print("campaign_prefixes inválido, debe ser lista o tupla.")
        return

    conn = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )
    try:
        cursor = conn.cursor()

        sql = """
            UPDATE vap_status
            SET assigned = 0
            WHERE campaign_prefix LIKE %s
        """

        rows_updated = 0
        for prefix in campaign_prefixes:
            cursor.execute(sql, (f"%{prefix}%",))
            rows_updated += cursor.rowcount

        conn.commit()
        print(f"{rows_updated} filas actualizadas: assigned=0")

    finally:
        try:
            cursor.close()
        except Exception:
            pass
        conn.close()


def mark_unprocessed_unasigned():
    """Set assigned=0 where processed=0."""
    conn = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE vap_status
            SET assigned = 0
            WHERE processed = 0
        """)
        conn.commit()
        print("Filas con processed=0 actualizadas a assigned=0.")

    finally:
        try:
            cursor.close()
        except Exception:
            pass
        conn.close()


# =========================
# === MAIN ===
# =========================
def main():
    run_key = RUN_KEY
    start = datetime.datetime.utcnow()
    t0 = time.time()

    warnings = None
    campaigns = []
    did_reset = 0

    try:
        # Reset depende del último run_end_time que hizo reset (did_reset=1)
        if should_run_resets(run_key=run_key):
            reset_all_processed_to_zero()
            reset_all_assigned_to_zero()
            did_reset = 1

        print("Starting folder_date update process... ")
        updated_df = fill_folder_processing_status()

        campaigns = fetch_campaigns_listed()

        print("Successfully updated folder_date for marketing_campaigns!")
        print(updated_df.head(10))

    except Exception:
        warnings = traceback.format_exc()
        raise

    finally:
        end = datetime.datetime.utcnow()
        exec_seconds = round(time.time() - t0, 3)

        insert_script_run(
            run_key=run_key,
            run_started_utc=start,
            run_finished_utc=end,
            exec_seconds=exec_seconds,
            campaigns_listed=campaigns,
            warnings_exceptions=warnings,
            did_reset=did_reset
        )


if __name__ == "__main__":
    main()
