from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import pandas as pd
import database.dbConfig as dbcfg



# ---------- CONFIGURACIÓN BD (RDS) ----------

SQL_QUERY = """
SELECT * FROM vap_report;
"""

# ---------- CONFIGURACIÓN GOOGLE SHEETS ----------
SERVICE_ACCOUNT_FILE = "gcplookerconfig.json"  # ruta al JSON de la cuenta de servicio
SPREADSHEET_ID = "1yeiu8RHYsr9xdv5xE2mA0dw8vn-CO0mTKXi1Gjj0sBo"   # ID del Google Sheet en Drive
SHEET_NAME = "Hoja 1"                  # nombre de la pestaña


def cast_and_clean_vap_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Castea y limpia el DataFrame con tipos razonables antes de enviarlo a Google Sheets.
    - Numéricos: NaN -> 0
    - Fechas: a string 'YYYY-MM-DD HH:MM:SS', vacías -> '0'
    - Strings: sin NaN y con espacios recortados
    - Al final: no queda ningún NaN (todo JSON-safe)
    """
    df = df.copy()

    # ======== 1. CASTEOS ESPECÍFICOS POR COLUMNA (ajusta nombres reales) ========
    for col in ["id", "call_id", "agent_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["duration_seconds", "ventas", "monto", "score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["is_sale", "is_contacted"]:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    datetime_candidate_cols = ["created_at", "updated_at", "call_datetime", "fecha"]
    for col in datetime_candidate_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ======== 2. RELLENAR NaN EN NUMÉRICOS CON 0 ========
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(0)

    # ======== 3. FORMATEAR FECHAS A STRING Y RELLENAR VACÍOS ========
    datetime_cols = df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns

    for col in datetime_cols:
        df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        df[col] = df[col].replace("NaT", "0").fillna("0")

    # ======== 4. STRINGS SIN NaN Y CON STRIP ========
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()  # 👈 aquí va .str.strip()

    # ======== 5. SANITY CHECK FINAL: NADA DE NaN EN NINGÚN LADO ========
    # Cualquier cosa suelta que quede como NaN -> 0 (cumple "castea los nan a cero")
    df = df.where(pd.notnull(df), 0)

    return df


def get_data_from_db():
    conn = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP,
    )
    try:
        df = pd.read_sql(SQL_QUERY, conn)
    finally:
        conn.close()

    df = cast_and_clean_vap_report(df)
    return df


def write_dataframe_to_google_sheet(df):
    """
    Reemplaza completamente el contenido de una hoja en Google Sheets
    con los datos de un DataFrame (incluyendo encabezados).
    """
    # df YA viene limpio y casteado desde get_data_from_db

    # Autenticación con cuenta de servicio
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=scopes
    )
    sheets_service = build("sheets", "v4", credentials=credentials)

    # Convertir DataFrame a lista de listas (primera fila = encabezados)
    values = [list(df.columns)] + df.values.tolist()

    # Limpiar toda la hoja
    range_all = SHEET_NAME
    sheets_service.spreadsheets().values().clear(
        spreadsheetId=SPREADSHEET_ID,
        range=range_all
    ).execute()

    # Escribir datos nuevos desde A1
    body = {"values": values}
    sheets_service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SHEET_NAME}!A1",
        valueInputOption="RAW",  # así Sheets puede inferir número/fecha
        body=body
    ).execute()

    print(f"Hoja '{SHEET_NAME}' actualizada con {len(df)} filas.")


def main():
    print("Obteniendo datos desde la base de datos (vap_report)...")
    df = get_data_from_db()
    print(f"Se obtuvieron {len(df)} filas. Subiendo a Google Sheets...")
    write_dataframe_to_google_sheet(df)
    print("Proceso completado con éxito.")


if __name__ == "__main__":
    main()
