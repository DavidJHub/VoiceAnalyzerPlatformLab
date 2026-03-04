from dotenv import load_dotenv

import database.dbConfig as dbcfg

import io
import datetime
import boto3
import mysql.connector
import pandas as pd

load_dotenv(".env")


def get_marketing_campaign_ids(db_conn, country=None, sponsor=None):
    """
    Retrieves campaign IDs from the 'marketing_campaigns' table, optionally filtered
    by country and/or sponsor.
    """
    base_query = "SELECT campaign_id FROM marketing_campaigns"
    filters = []
    params = []

    # Build the WHERE clause
    if country is not None:
        filters.append("country = %s")
        params.append(country)

    if sponsor is not None:
        filters.append("sponsor = %s")
        params.append(sponsor)

    if filters:
        base_query += " WHERE " + " AND ".join(filters)

    with db_conn.cursor() as cursor:
        cursor.execute(base_query, params)
        rows = cursor.fetchall()

    campaign_ids = [row[0] for row in rows]
    return campaign_ids


def upload_dataframe_to_s3(df, bucket_name, ruta_s3, start_date):
    """
    Uploads a DataFrame as CSV to an Amazon S3 path.
    The file name is derived from the start_date and end_date values.
    """
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    start = start_date.split(' ')[0]
    file_name = f"Reporte_{start}.csv"
    s3_key = f"{ruta_s3}/{file_name}"

    s3_client = dbcfg.generate_s3_client()
    

    s3_client.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=s3_key)
    print(f"DataFrame uploaded to s3://{bucket_name}/{s3_key}")


def build_agent_audio_query(campaign_ids, start_date, end_date):
    """
    Builds a parameterized SQL query that selects records from 'agent_audio_data'
    joined with 'call_affecteds', filtered by campaign_ids and date range.
    """
    if not campaign_ids:
        return "SELECT * FROM agent_audio_data WHERE 1=0", []

    placeholders = ', '.join(['%s'] * len(campaign_ids))
    query = f"""
    SELECT
        aad.name,
        aad.agent_id,
        aad.campaign_id,
        aad.date,
        aad.general_score,
        aad.tmo,
        aad.concatenated,
        ca.summary_rejection,
        ca.alert
    FROM
        agent_audio_data aad
    LEFT JOIN
        call_affecteds ca ON aad.id = ca.agent_audio_data_id
    WHERE
        aad.campaign_id IN ({placeholders})
        AND aad.date BETWEEN %s AND %s
    """
    params = list(campaign_ids) + [start_date, end_date]
    return query, params


def get_agent_audio_data_as_dataframe(db_conn, start_date, end_date, country=None, sponsor=None):
    """
    1) Retrieves campaign_ids from marketing_campaigns (optionally filtered).
    2) Builds the query to get data from agent_audio_data & call_affecteds.
    3) Executes the query and returns the results as a pandas DataFrame.
    """
    campaign_ids = get_marketing_campaign_ids(db_conn, country=country, sponsor=sponsor)
    query, params = build_agent_audio_query(campaign_ids, start_date, end_date)

    with db_conn.cursor() as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        if rows:
            column_names = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=column_names)
        else:
            # Return an empty DataFrame if no rows
            df = pd.DataFrame()
    return df


def get_distinct_countries(db_conn):
    """
    Retrieve the list of distinct countries from marketing_campaigns.
    """
    query = "SELECT DISTINCT country FROM marketing_campaigns WHERE country IS NOT NULL"
    with db_conn.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
    return [row[0] for row in rows if row[0] is not None]


def get_distinct_sponsors(db_conn, country):
    """
    Retrieve the list of distinct sponsors from marketing_campaigns for a given country.
    """
    query = "SELECT DISTINCT sponsor FROM marketing_campaigns WHERE sponsor IS NOT NULL AND country = %s"
    with db_conn.cursor() as cursor:
        cursor.execute(query, (country,))
        rows = cursor.fetchall()
    return [row[0] for row in rows if row[0] is not None]


def get_distinct_campaigns(db_conn, sponsor):
    """
    Retrieve the list of distinct sponsors from marketing_campaigns for a given country.
    """
    query = "SELECT DISTINCT campaign FROM marketing_campaigns WHERE campaign IS NOT NULL AND sponsor = %s"
    with db_conn.cursor() as cursor:
        cursor.execute(query, (sponsor,))
        rows = cursor.fetchall()
    return [row[0] for row in rows if row[0] is not None]

def _get_latest_country_report_key(s3_client, bucket_name, country, report_dir="reportesMensuales"):
    """
    Devuelve la KEY en S3 del último archivo CSV de reportesMensuales
    para un país dado, usando la fecha de última modificación en S3.
    """
    prefix = f"{country}/{report_dir}/"
    paginator = s3_client.get_paginator("list_objects_v2")

    latest_obj = None

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            # Nos aseguramos de quedarnos solo con CSVs
            if not obj["Key"].lower().endswith(".csv"):
                continue

            if latest_obj is None or obj["LastModified"] > latest_obj["LastModified"]:
                latest_obj = obj

    if latest_obj is None:
        return None

    return latest_obj["Key"]


def build_global_report_from_s3(bucket_name, countries, start_date, report_dir="reportesMensuales"):
    """
    Construye un reporte global concatenando el último reporte mensual de cada país
    desde S3 y sube el resultado a `s3://<bucket_name>/<report_dir>/`.

    Parámetros:
    - bucket_name: str, nombre del bucket S3 (ej: "documentos.aihub")
    - countries: list[str], lista de países (ej: ["Colombia", "Peru", ...])
    - start_date: str, fecha de inicio del rango en formato 'YYYY-MM-DD 00:00:00'
                  (se usa solo para el nombre del archivo de salida)
    - report_dir: str, subruta donde se guardan los reportes (por defecto "reportesMensuales")

    Retorna:
    - df_global: DataFrame concatenado con todos los países que tenían reporte.
    """
    s3_client = dbcfg.generate_s3_client()
    dataframes = []

    for country in countries:
        latest_key = _get_latest_country_report_key(
            s3_client=s3_client,
            bucket_name=bucket_name,
            country=country,
            report_dir=report_dir
        )

        if latest_key is None:
            print(f"[SKIP] No se encontró reporte en S3 para country='{country}'")
            continue

        print(f"[INFO] Usando último reporte de {country}: s3://{bucket_name}/{latest_key}")

        obj = s3_client.get_object(Bucket=bucket_name, Key=latest_key)
        body_bytes = obj["Body"].read()
        df_country = pd.read_csv(io.BytesIO(body_bytes))

        # Añadimos una columna opcional 'country' por trazabilidad
        if "country" not in df_country.columns:
            df_country["country"] = country

        dataframes.append(df_country)

    if not dataframes:
        print("[WARN] No se encontró ningún reporte para concatenar.")
        return pd.DataFrame()

    df_global = pd.concat(dataframes, ignore_index=True)

    # Reutilizamos la función existente para subir el resultado al path global
    global_s3_path = report_dir  # Ej: "reportesMensuales"
    upload_dataframe_to_s3(
        df=df_global,
        bucket_name=bucket_name,
        ruta_s3=global_s3_path,
        start_date=start_date,
    )

    print(f"[OK] Reporte global generado y subido a s3://{bucket_name}/{global_s3_path}/")
    return df_global



if __name__ == "__main__":
    # 1) Connect to your MySQL database
    conn = dbcfg.conectar(HOST=dbcfg.HOST_DB_VAP,  
                                DATABASE=dbcfg.DB_NAME_VAP,  
                                USERNAME=dbcfg.USER_DB_VAP,  
                                PASSWORD=dbcfg.PASSWORD_DB_VAP)

    # 2) Calculate a 15-day date range ending today
    now = datetime.datetime.now()
    DAY_OF_THE_MONTH = now.day
    LAST_FORTNIGHT = 15 if (DAY_OF_THE_MONTH > 15 or DAY_OF_THE_MONTH==1) else 1
    FORTNIGHT_MONTH = now.month - 1 if ( DAY_OF_THE_MONTH==1) else now.month
    FORTNIGHT_DATE = datetime.date(now.year, FORTNIGHT_MONTH, LAST_FORTNIGHT)

    start_dt = FORTNIGHT_DATE.strftime('%Y-%m-%d 00:00:00')
    end_dt = now.strftime('%Y-%m-%d 23:59:59')
    print("REPORTES DESDE :" + start_dt)
    print("REPORTES HASTA :" + end_dt)
    bucket_name = "documentos.aihub"

    # ---------------------------------------------------------------
    # PROCESS FOR ALL COUNTRIES AND ALL SPONSORS
    # ---------------------------------------------------------------
    countries = get_distinct_countries(conn)

    for country_filter in countries:
        # 1) Get all distinct sponsors for this country
        sponsors = get_distinct_sponsors(conn, country_filter)
        #sponsors = ["Davivienda",]

        # 2) For each sponsor in that country, run the script
        for sponsor_filter in sponsors:
            df_results = get_agent_audio_data_as_dataframe(
                db_conn=conn,
                start_date=start_dt,
                end_date=end_dt,
                country=country_filter,
                sponsor=sponsor_filter
            )
            if not df_results.empty:
                # Build the path: <country>/<sponsor>/reportesMensuales
                base_s3_path = f"{country_filter}/{sponsor_filter}/reportesMensuales"
                df_results.to_csv('output.csv', index=False)
                upload_dataframe_to_s3(
                    df=df_results,
                    bucket_name=bucket_name,
                    ruta_s3=base_s3_path,
                    start_date=start_dt,
                )
            else:
                print(f"[SKIP] No data for country='{country_filter}', sponsor='{sponsor_filter}'")

        # ---------------------------------------------------------------
        # PROCESS FOR THE COUNTRY BUT EXCLUDING SPONSOR (sponsor=None)
        # ---------------------------------------------------------------
        df_results_no_sponsor = get_agent_audio_data_as_dataframe(
            db_conn=conn,
            start_date=start_dt,
            end_date=end_dt,
            country=country_filter,
            sponsor=None
        )
        if not df_results_no_sponsor.empty:
            # Path: <country>/reportesMensuales
            base_s3_path_no_sponsor = f"{country_filter}/reportesMensuales"
            df_results_no_sponsor.to_csv('output.csv', index=False)
            upload_dataframe_to_s3(
                df=df_results_no_sponsor,
                bucket_name=bucket_name,
                ruta_s3=base_s3_path_no_sponsor,
                start_date=start_dt,
            )
        else:
            print(f"[SKIP] No data for country='{country_filter}' with sponsor=None")
    df_global = build_global_report_from_s3(
        bucket_name=bucket_name,
        countries=countries,
        start_date=start_dt,
        report_dir="reportesMensuales"
    )
    # 8) Close the connection
    conn.close()
