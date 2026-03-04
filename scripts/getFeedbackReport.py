from typing import List, Tuple
import json
from datetime import date, datetime, timedelta, timezone
import numpy as np
import pandas as pd
import mysql.connector


def conectar(host, port, database, username, password):
    return mysql.connector.connect(
        host=host,
        port=port,
        user=username,
        passwd=password,
        db=database
    )

def _read_sql(conexion: mysql.connector.MySQLConnection,
              sql: str,
              params: Tuple | List = ()) -> pd.DataFrame:
    return pd.read_sql(sql, conexion, params=params)


import pandas as pd
from datetime import datetime
from typing import Union
import mysql.connector

def format_datetime(dt_str):
    # Parse the input string into a datetime object
    dt = datetime.strptime(dt_str, "%Y%m%d-%H%M%S")
    # Format it into the desired string format
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def fetch_feedback_reports(con_main: mysql.connector.MySQLConnection) -> pd.DataFrame:
    """
    Descarga cada tabla por separado (filtradas por fecha) y hace los
    merge en Pandas.  
    """
    # ──────────────────────────────────────────────────────────────
    # 1. Fecha de corte (UTC; ajusta si tu server está en otra zona)
    # ──────────────────────────────────────────────────────────────
    cutoff = datetime.now(timezone.utc) - timedelta(days=42)
    cutoff_str_start = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    cutoff_str_start = "2025-04-01 00:00:00"
    cutoff_end = datetime.now(timezone.utc) - timedelta(days=40)
    cutoff_str_end = cutoff_end.strftime("%Y-%m-%d %H:%M:%S")
    cutoff_str_end = "2025-06-19 23:59:59"
    # ──────────────────────────────────────────────────────────────
    # 2. agent_audio_data  (prefijo aad_)
    # ──────────────────────────────────────────────────────────────
    df_cf = pd.read_sql(
        """
        SELECT
            id                          AS cf_id,
            user_id                     AS cf_user_id,
            text_call_feedback_id       AS cf_tcf_id,
            agent_audio_data_id         AS cf_aad_id,
            status                      AS cf_status,
            created_at                  AS cf_created_at,
            updated_at                  AS cf_updated_at
        FROM call_feedback
        WHERE `created_at` >= %s AND `created_at` <= %s;
        """,
        con_main,
        params=[cutoff_str_start, cutoff_str_end],
    )
    print(f"► agent_audio_data: {df_cf.shape[0]} filas entre {cutoff_str_start} y {cutoff_str_end}")
    if df_cf.empty:
        return df_cf

    # ──────────────────────────────────────────────────────────────
    # 3. aad_calls
    # ──────────────────────────────────────────────────────────────
    df_aad = pd.read_sql(
        """
        SELECT
            id                          AS aad_id,
            lead_id                     AS aad_lead_id,
            agent_id                    AS aad_agent_id,
            campaign_id                 AS aad_campaign_id,
            name                        AS aad_name,
            date                        AS aad_date,
            link_audio                  AS aad_link_audio,
            link_transcription_audio    AS aad_link_transcription
        FROM agent_audio_data
        WHERE `date` >= %s AND `date` <= %s;
        """,
        con_main,
        params=[cutoff_str_start, cutoff_str_end],
    )

    df_fr = pd.read_sql(
        """
        SELECT
            id                   AS fr_id,
            text_feedback        AS fr_text
        FROM text_call_feedback
        """,
        con_main,  
    )

    # ──────────────────────────────────────────────────────────────
    # 5.  MERGE en Pandas
    # ──────────────────────────────────────────────────────────────

    df_full = df_cf.merge(
        df_aad,
        how="left",
        left_on="cf_aad_id",
        right_on="aad_id",
        suffixes=("", "_dup"),
    ).drop(columns=[c for c in df_aad.columns if c.endswith("_dup")])
    df_full = df_full.drop_duplicates(subset="aad_lead_id")
    df_full["aad_lead_id"] = df_full["aad_lead_id"].astype(str)

    df_final = df_full.merge(
        df_fr,
        how="left",
        left_on="cf_tcf_id",
        right_on="fr_id",
    )

    df_final = df_final.drop_duplicates(subset="aad_lead_id")
    df_final.to_excel("merged_call_feedback.xlsx")
    return df_final

if __name__ == "__main__":
    HOST_vap, PORT_vap   = "vapdb.cjq4ek6ygqif.us-east-1.rds.amazonaws.com", 3306
    DB_vap,  USER_vap, PW_vap = "aihub_bd", "admindb", "VAPigs2024.*"
    con_vap  = conectar(HOST_vap,  PORT_vap,  DB_vap,  USER_vap,  PW_vap)
    try:
        print(f"►►► Descargando feedback de los últimos día(s)…")
        df_final = fetch_feedback_reports(con_vap)
        df_final.to_excel("df_fetched.xlsx")
    except Exception as e:
        print(f"⚠️  {e}")
    finally:
        con_vap.close()
        print("►►► Conexión cerrada.")
