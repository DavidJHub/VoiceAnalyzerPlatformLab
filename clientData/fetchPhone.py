import pandas as pd
from datetime import datetime, timedelta

import database.dbConfig as dbcfg


# ==========================
# CONEXIÓN A LA BD
# ==========================

def get_db_connection():
    """
    Obtiene la conexión a la BD usando la misma función dbcfg.conectar
    que ya usas en otros scripts.
    """
    conexion = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP,
    )
    return conexion


# ==========================
# PARSEO DEL TELÉFONO DESDE name
# ==========================

def extract_phone_from_name(name: str) -> str:
    """
    Extrae el teléfono del campo 'name'.

    Ejemplo:
    BANCOLBI_20240626-070451_127723_1719403491_52973157_3114528532-all.mp3
    -> split("_")[-1] = "3114528532-all.mp3"
    -> split("-")[0]  = "3114528532"
    """
    if not name:
        return ""

    try:
        last_segment = name.split("_")[-1]          # "3114528532-all.mp3"
        phone = last_segment.split("-")[0]          # "3114528532"
        return phone
    except Exception:
        # En caso de formato inesperado, devolvemos string vacío
        return ""


# ==========================
# CONSULTA A agent_audio_data (últimos N días)
# ==========================

def fetch_recent_agent_phones(days: int = 7) -> pd.DataFrame:
    """
    Consulta la tabla agent_audio_data filtrando por la columna `date`
    a los últimos `days` días y devuelve un DataFrame con columnas:
    - id  (id_agent_audio_data)
    - telefono

    La columna 'id' servirá luego como id_agent_audio_data en la inserción
    final sobre vap_clients.
    """
    conexion = get_db_connection()
    cutoff = datetime.now() - timedelta(days=days)

    query = """
        SELECT 
            id,
            name
        FROM agent_audio_data
        WHERE `date` >= %s
    """

    try:
        cursor = conexion.cursor()
        cursor.execute(query, (cutoff,))
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
    finally:
        cursor.close()
        conexion.close()

    # Creamos el DataFrame base
    df = pd.DataFrame(rows, columns=colnames)

    if df.empty:
        # Retornamos DataFrame vacío con las columnas esperadas
        return pd.DataFrame(columns=["id", "telefono"])

    # Extraer teléfono desde 'name'
    df["telefono"] = df["name"].apply(extract_phone_from_name)

    # Renombrar id_agent_audio_data -> id
    df = df.rename(columns={"id_agent_audio_data": "id"})

    # Dejar solo las columnas que necesitas
    df_result = df[["id", "telefono"]].copy()

    return df_result


# ==========================
# EJEMPLO DE USO
# ==========================

if __name__ == "__main__":
    df_phones = fetch_recent_agent_phones(days=7)
    print(df_phones.head())
    print(f"Total filas: {len(df_phones)}")
