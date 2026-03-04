from typing import List, Tuple
import json
import uuid
from datetime import date, datetime, timedelta, timezone
import numpy as np
import pandas as pd
import mysql.connector
import database.dbConfig as dbcfg
# ───────────────────────────────────────────────────────────────────────────────
#  Helper:  ejecutar consulta y devolver DataFrame
# ───────────────────────────────────────────────────────────────────────────────
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


def fetch_audios_last_day(con_main: mysql.connector.MySQLConnection,
                          con_vici: mysql.connector.MySQLConnection,
                          days_back: int = 40) -> pd.DataFrame:
    """
    Descarga cada tabla por separado (filtradas por fecha) y hace los
    merge en Pandas.  NO usa lista de placeholders.
    """
    # ──────────────────────────────────────────────────────────────
    # 1. Fecha de corte (UTC; ajusta si tu server está en otra zona)
    # ──────────────────────────────────────────────────────────────
    cutoff = datetime.now(timezone.utc) - timedelta(days=9)
    cutoff_str_start = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    #cutoff_str_start = "2025-05-10 00:00:00"
    cutoff_end = datetime.now(timezone.utc)
    cutoff_str_end = cutoff_end.strftime("%Y-%m-%d %H:%M:%S")
    #cutoff_str_end = "2025-05-18 23:59:59"
    # ──────────────────────────────────────────────────────────────
    # 2. agent_audio_data  (prefijo aad_)
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
    print(f"► agent_audio_data: {df_aad.shape[0]} filas entre {cutoff_str_start} y {cutoff_str_end}")
    if df_aad.empty:
        return df_aad

    # ──────────────────────────────────────────────────────────────
    # 3. call_affecteds  (prefijo ca_)
    # ──────────────────────────────────────────────────────────────
    df_ca = pd.read_sql(
        """
        SELECT
            id, lead_id, summary_rejection, updated_at
        FROM call_affecteds
        WHERE updated_at >= %s AND updated_at <= %s;
        """,
        con_vap,
        params=[cutoff_str_start, 
               cutoff_str_end]
    ).add_prefix("ca_")

    # ──────────────────────────────────────────────────────────────
    # 4. vapinfovicidial  (prefijo vici_)  
    # ──────────────────────────────────────────────────────────────
    df_vici = pd.read_sql(
        """
        SELECT
            lg_lead_id           AS lg_lead_id,
            lg_status            AS vici_status,
            rc_start_time        AS vici_last_call
        FROM vapinfovicidial
        WHERE rc_start_time >= %s  AND rc_start_time <= %s ;
        """,
        con_vici,
        params=[cutoff_str_start, 
                cutoff_str_end]     
    )

    if not df_vici.empty:
        df_vici = (
            df_vici.sort_values("vici_last_call", ascending=False)
                    .drop_duplicates(subset="lg_lead_id", keep="first")
                    .add_prefix("vici_")
        )

    # ──────────────────────────────────────────────────────────────
    # 5.  MERGE en Pandas
    # ──────────────────────────────────────────────────────────────
    # 5-a  agent_audio_data  ↔ call_affecteds
    if df_ca.empty:
        df_ac = df_aad.copy()
    else:
        df_ac = df_aad.merge(
            df_ca,
            how="left",
            left_on="aad_lead_id",
            right_on="ca_lead_id",
            suffixes=("", "_dup"),
        ).drop(columns=[c for c in df_aad.columns if c.endswith("_dup")])
        df_ac = df_ac.drop_duplicates(subset="aad_lead_id")
        df_ac["aad_lead_id"] = df_ac["aad_lead_id"].astype(str)

    if not df_vici.empty:
        df_vici["vici_lg_lead_id"]=df_vici["vici_lg_lead_id"].astype(str)
        df_final = df_ac.merge(
            df_vici,
            how="left",
            left_on="aad_lead_id",
            right_on="vici_lg_lead_id",
        )
    else:
        df_final = df_ac
    df_final = df_final.drop_duplicates(subset="aad_lead_id")
    df_final.to_excel("vici_vap_join.xlsx")
    return df_final


# ───────────────────────────────────────────────────────────────────────────────
#  1.  campaign_id -> selling_quality_id
# ───────────────────────────────────────────────────────────────────────────────

def _map_agent_pk(df: pd.DataFrame,
                           conexion,
                           igs_col="aad_agent_id",
                           pk_col="agent_pk") -> pd.DataFrame:
    agentes = pd.read_sql(
        "SELECT CAST(id_igs AS CHAR) AS id_igs, id AS agent_pk FROM agents;",
        conexion
    )
    agentes = agentes.rename(columns={"id_igs": igs_col})
    print(f"Mapeados {len(agentes)} agentes desde {igs_col} a {pk_col}.")
    return df.merge(agentes, how="left", on=igs_col)


def _map_selling_quality(df, conexion,
                         camp_col="aad_campaign_id"):
    """Añade selling_quality_id partiendo de aad_campaign_id"""
    campanas = df[camp_col].dropna().unique().tolist()
    if not campanas:
        df[camp_col] = pd.NA
        return df
    sql = f"""
      SELECT campaign_id AS {camp_col}, id AS selling_quality_id
      FROM selling_qualities
      WHERE campaign_id IN ({','.join(['%s']*len(campanas))});
    """
    mapa = pd.read_sql(sql, conexion, params=campanas)
    mapa.to_excel("mapa_selling_quality_id.xlsx")
    print(f"Mapeadas {len(mapa)} selling_quality_id desde {camp_col}.")
    return df.merge(mapa, how="left", on=camp_col)




# ───────────────────────────────────────────────────────────────────────────────
#  3.  (selling_quality_id, agent_pk) -> selling_quality_agent_id
# ───────────────────────────────────────────────────────────────────────────────
def _map_sqa_id(df: pd.DataFrame,
                conexion,
                sq_col: str = "selling_quality_id",
                agent_pk_col: str = "agent_pk",
                new_col: str = "sq_agent_pk") -> pd.DataFrame:
    pares = df[[sq_col, agent_pk_col]].dropna().drop_duplicates()
    if pares.empty:
        df[new_col] = pd.NA
        return df

    qs   = pares[sq_col].astype(int).unique().tolist()
    agts = pares[agent_pk_col].astype(int).unique().tolist()

    sql = f"""
        SELECT id                        AS sqa_id,
               selling_quality_id,
               agent_id
        FROM   selling_quality_agents
        WHERE  selling_quality_id IN ({','.join(['%s']*len(qs))})
          AND  agent_id            IN ({','.join(['%s']*len(agts))});
    """
    mapa = _read_sql(conexion, sql, qs + agts)
    df_out = df.merge(mapa,
                      how="left",
                      left_on=[sq_col, agent_pk_col],
                      right_on=["selling_quality_id", "agent_id"])
    df_out = df_out.rename(columns={"sqa_id": new_col})
    # re-añade la columna sq_col con nombre correcto
    df_out = df_out.rename(columns={sq_col: "selling_quality_id"})
    print(f"Mapeadas {len(mapa)} selling_quality_agent_id "
          f"desde {sq_col} y {agent_pk_col} a {new_col}.")
    return df_out

def insert_selling_quality_audio_agents(
        conexion: mysql.connector.MySQLConnection,
        df_raw: pd.DataFrame,
        sq_map:  dict[int, int],              # campaign_id ➜ selling_quality_id
        sqa_map: dict[tuple[int,int], int],   # (sq_id, agent_pk) ➜ sqa_id
        alert_const: str = "0",
        CHUNK: int = 500
) -> None:
    """
    Inserta los audios enlazándolos al selling_quality_agent_id correcto.
    """
    # 1) Copia para no tocar el DF original
    df = df_raw.copy()

    # 2) Añade selling_quality_id a partir del campaign_id
    df["selling_quality_id"] = df["aad_campaign_id"].map(sq_map)

    # 3) Resuelve agent_pk
    df = _map_agent_pk(df, conexion, igs_col="aad_agent_id")

    # 4) Construye la clave (sq_id, agent_pk) y obtén el sqa_id
    df["sqa_key"]   = list(zip(df["selling_quality_id"], df["agent_pk"]))
    df["sq_agent_pk"] = df["sqa_key"].map(sqa_map)

    # 5) Filtra filas que sí tengan FK completo
    df_valid = df.dropna(subset=["sq_agent_pk"])
    if df_valid.empty:
        print("⚠ No hay audios con selling_quality_agent_id resuelto.")
        return

    # 6) Preparar sentencia INSERT
    sql = """
        INSERT INTO selling_quality_audio_agents (
            uuid, selling_quality_agent_id, name,
            link_audio, link_transcription,
            alert, created_at, updated_at
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE                 -- ← evita choque si se repite
            link_audio       = VALUES(link_audio),
            link_transcription = VALUES(link_transcription),
            updated_at       = VALUES(updated_at)
    """

    # 7) Generar filas
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    df_valid["name_date"] = (
        df_valid["aad_name"]
        .str.split("_").str[1]
        .apply(format_datetime)
    )

    rows = [
        (
            str(uuid.uuid4()),
            int(r.sq_agent_pk),
            r.aad_name,
            r.aad_link_audio,
            r.aad_link_transcription,
            alert_const,
            r.name_date, r.name_date
        )
        for _, r in df_valid.iterrows()
    ]

    # 8) Ejecutar por lotes
    cur, total = conexion.cursor(), 0
    for i in range(0, len(rows), CHUNK):
        cur.executemany(sql, rows[i:i + CHUNK])
        conexion.commit()
        total += cur.rowcount
    cur.close()
    print(f"► Insertados/actualizados {total} audios vinculados.")


def _build_sq_rows(df: pd.DataFrame) -> list[tuple]:
    """
    Agrupa el DataFrame por `aad_campaign_id` y produce la lista de tuplas
    listas para el INSERT / UPSERT.
    """
    # 1. Normaliza nombres que el usuario ha usado (“vici_status” o “vici_vici_status”)
    if "vici_vici_status" in df.columns and "vici_status" not in df.columns:
        df = df.rename(columns={"vici_vici_status": "vici_status"})

    # 2. Métricas por campaña
    g = df.groupby("aad_campaign_id", dropna=True)

    rows = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for camp_id, grp in g:
        quantity_processed  = len(grp)

        # Filas auditadas (≠ "NO AUDITABLE", case-insensitive, nulos ≠ auditado)
        audited_mask        = ~grp["ca_summary_rejection"].fillna("").str.upper().eq("NO AUDITABLE")
        quantity_audited    = int(audited_mask.sum())

        percentage_audited  = (quantity_audited / quantity_processed * 100
                               if quantity_processed else 0)
        # Ventas  → vici_status que empieza con "V"
        sales_mask          = grp["vici_status"].fillna("").str.upper().str.startswith("V")
        quantity_sales      = int(sales_mask.sum())

        percentage_sales    = (quantity_sales / quantity_processed * 100
                               if quantity_audited else 0)

        # Rechazos → vici_status que empieza con "AUD"
        rejections_mask     = grp["vici_status"].fillna("").str.upper().str.startswith("AUD")
        quantity_rejections = int(rejections_mask.sum())

        # Estado
        if   percentage_sales > 95:
            status = "NORMAL"
        elif percentage_sales <= 95 and percentage_sales >= 90:
            status = "REVIEW"
        else:
            status = "CRITICAL"

        # uuid nuevo (se mantendrá el antiguo en el UPSERT si ya existe)
        uid = str(uuid.uuid4())

        rows.append((
            uid, camp_id,                              # uuid, campaign_id
            quantity_processed,
            quantity_audited,
            round(percentage_audited, 2),
            quantity_sales,
            round(percentage_sales, 2),
            quantity_rejections,
            status,
            now, now                                   # created_at, updated_at
        ))

    return rows


def insert_selling_qualities(
        conexion: mysql.connector.MySQLConnection,
        df: pd.DataFrame
) -> dict[int, int]:
    """
    Inserta un snapshot por campaña y devuelve
    {campaign_id: selling_quality_id (PK)} solo para esta corrida.
    """
    filas = _build_sq_rows(df)        # ⇢ lista de tuplas
    if not filas:
        return {}

    sql = ("""
        INSERT INTO selling_qualities (
            uuid, campaign_id, quantity_processed, quantity_audited,
            percentage_audited, quantity_sales, percentage_sales,
            quantity_rejections, status, created_at, updated_at
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """)
    cur, mapa = conexion.cursor(), {}
    for r in filas:
        cur.execute(sql, r)
        # r[1] es el campaign_id (posición fija en _build_sq_rows)
        mapa[r[1]] = cur.lastrowid
    conexion.commit()
    cur.close()
    print(f"► Insertadas {len(mapa)} filas en selling_qualities.")
    return mapa               # ← se usará como FK en el siguiente paso


# ────────────────────────────────────────────────────────────────
#  UTILIDADES DE MAPEO
# ────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────
#   NUEVO _build_sqa_rows  (agente → campaña)
# ────────────────────────────────────────────────────────────────
def _build_sqa_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame agregado por (aad_agent_id, aad_campaign_id)
    con las métricas necesarias para selling_quality_agents.
    """
    # —— normalizar nombres ————————————————————————————————
    if "vici_vici_status" in df.columns and "vici_status" not in df.columns:
        df = df.rename(columns={"vici_vici_status": "vici_status"})
    if "ca_summary_rejection" not in df.columns:
        df["ca_summary_rejection"] = ""

    def _startswith(series, prefix):
        return series.fillna("").str.upper().str.startswith(prefix)

    # —— agrupación agente → campaña ————————————————————————
    grouped = (
        df.groupby(["aad_agent_id", "aad_campaign_id"], dropna=True)
        .agg(
            quantity_call        = ("aad_agent_id", "size"),
            quantity_sales       = ("vici_status",
                                    lambda s: (s.fillna("")
                                                .str.upper()
                                                .str.startswith("V")).sum()),
            quantity_rejections  = ("vici_status",
                                    lambda s: (s.fillna("")
                                                .str.upper()
                                                .str.startswith("AUD")).sum()),
            quantity_macs        = ("ca_summary_rejection",
                                    lambda s: (~(s.fillna("")
                                                    .str.upper()
                                                    .str.contains("MAC"))).sum()),
            price                = ("ca_summary_rejection",
                                    lambda s: (~(s.fillna("")
                                                    .str.upper()
                                                    .str.contains("PRECIO"))).sum()),
            quantity_crit_errors = ("ca_summary_rejection",
                                    lambda s: s.fillna("").str.strip().ne("").sum()),
        )
        .reset_index()
    )

    # —— porcentajes y estado ————————————————————————————————
    grouped["percentage_sales"] = (
        grouped["quantity_sales"] / grouped["quantity_call"]
    ).fillna(0).round(2) * 100

    def _status(p):
        if p > 95:
            return "NORMAL"
        elif p <= 95 and p >= 90:
            return "REVIEW"
        else:
            return "CRITICAL"

    grouped["status"] = grouped["percentage_sales"].apply(_status)

    # —— extras constantes ————————————————————————————————
    grouped["quantity_inevitable"] = 0
    grouped["uuid"] = [str(uuid.uuid4()) for _ in range(len(grouped))]
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    grouped["created_at"] = grouped["updated_at"] = now
    grouped.to_excel("grouped_sqa_rows.xlsx", index=False)

    # —— renombrar para mapeos ————————————————————————————
    grouped = grouped.rename(
        columns={
            "aad_campaign_id": "camp_id",
            "aad_agent_id":    "agent_id",
            "quantity_crit_errors": "quantity_critical_errors",
        }
    )
    return grouped

def insert_selling_quality_agents(
                        df_source: pd.DataFrame,
                        conexion: mysql.connector.MySQLConnection,
                        sq_map: dict[int, int]        # campaign_id ➜ selling_quality_id
                                ) -> dict[tuple[int,int], int]:
    """
    Inserta una fila por (agent, campaign) y devuelve
    {(selling_quality_id, agent_pk): sqa_id}
    """
    df = _build_sqa_rows(df_source)
    df["selling_quality_id"] = df["camp_id"].map(sq_map)

    df = _map_agent_pk(df, conexion, igs_col="agent_id")
    df_valid = df.dropna(subset=["selling_quality_id", "agent_pk"])
    if df_valid.empty:
        print("⚠ Nada que insertar en selling_quality_agents.")
        return {}

    sql = """
    INSERT INTO selling_quality_agents (
        uuid, selling_quality_id, agent_id,
        quantity_call, quantity_sales, percentage_sales, status,
        quantity_rejections, quantity_macs, price,
        quantity_inevitable, quantity_critical_errors,
        created_at, updated_at
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON DUPLICATE KEY UPDATE
        id = LAST_INSERT_ID(id),
        quantity_call            = VALUES(quantity_call),
        quantity_sales           = VALUES(quantity_sales),
        percentage_sales         = VALUES(percentage_sales),
        status                   = VALUES(status),
        quantity_rejections      = VALUES(quantity_rejections),
        quantity_macs            = VALUES(quantity_macs),
        price                    = VALUES(price),
        quantity_inevitable      = VALUES(quantity_inevitable),
        quantity_critical_errors = VALUES(quantity_critical_errors),
        updated_at               = VALUES(updated_at);
    """
    cur, mapa = conexion.cursor(), {}
    for _, r in df_valid.iterrows():
        datos = (
            r.uuid,
            int(r.selling_quality_id),
            int(r.agent_pk),
            int(r.quantity_call),
            int(r.quantity_sales),
            float(r.percentage_sales),
            r.status,
            int(r.quantity_rejections),
            int(r.quantity_macs),
            r.price,
            int(r.quantity_inevitable),
            int(r.quantity_critical_errors),
            r.created_at,
            r.updated_at,
        )
        cur.execute(sql, datos)
        mapa[(int(r.selling_quality_id), int(r.agent_pk))] = cur.lastrowid
    conexion.commit()
    cur.close()
    print(f"► Insertadas {len(mapa)} filas en selling_quality_agents.")
    return mapa


# ───────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ───────────────────────────────────────────────────────────────




DAYS_BACK = 2           # <- rango de búsqueda

# ───────────────────────────────────────────────────────────────
#  MAIN
# ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings(
                                "ignore",
                                message="pandas only supports SQLAlchemy connectable",
                                category=UserWarning,
                            )
    print("⏳ Conectando a bases de datos…")
    con_vap = dbcfg.conectar(HOST=dbcfg.HOST_DB_VAP,  
                                  DATABASE=dbcfg.DB_NAME_VAP,  
                                  USERNAME=dbcfg.USER_DB_VAP,  
                                  PASSWORD=dbcfg.PASSWORD_DB_VAP)
    con_vici = dbcfg.conectar(HOST=dbcfg.HOST_DB_VICI,  
                              DATABASE=dbcfg.DB_NAME_VICI,  
                              USERNAME=dbcfg.USER_DB_VICI,  
                              PASSWORD=dbcfg.PASSWORD_DB_VICI)

    try:
        print(f"►►► Descargando audios de los últimos {DAYS_BACK} día(s)…")
        df_final = fetch_audios_last_day(con_vap, con_vici, DAYS_BACK)
        df_final.to_excel("df_fetched.xlsx")

        if df_final.empty:
            print("⚠ No se encontraron audios en el rango indicado.")
        else:
            # ─── 1) Upsert selling_qualities ─────────────────────
            print("►►► Actualizando selling_qualities…")
            sq_map = insert_selling_qualities(con_vap, df_final)
            
            # ─── 2) Upsert selling_quality_agents ───────────────
            print("►►► Actualizando selling_quality_agents…")
            sqa_map = insert_selling_quality_agents(df_final, con_vap, sq_map)

            # ─── 3) Insert selling_quality_audio_agents ─────────
            print("►►► Insertando selling_quality_audio_agents…")
            insert_selling_quality_audio_agents(con_vap, df_final, sq_map, sqa_map)
    finally:
        print("►►► Cerrando conexiones…")
        con_vici.close()
        con_vap.close()
        print("✅ Proceso completo.")


