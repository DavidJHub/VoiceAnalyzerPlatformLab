
import mysql.connector
import ast
import json
from datetime import datetime, timedelta, date


import numpy as np
import pandas as pd

from mysql.connector import connect, Error
import database.dbConfig as dbcfg

def make_sql_safe(value):
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value



def obtener_id_sponsor(conexion, sponsor_name):
    """
    Obtiene el id del sponsor basado en el nombre, ignorando mayúsculas y minúsculas, y espacios en blanco.
    """
    try:
        cursor = conexion.cursor()

        cleaned_sponsor_name = ''.join(sponsor_name.split()).lower()

        consulta_sponsor_id = """
        SELECT id
        FROM sponsors
        WHERE LOWER(REPLACE(name, ' ', '')) = %s;
        """
        try:
            print("Trying with %s placeholders")
            cursor.execute(consulta_sponsor_id, (cleaned_sponsor_name,))
            sponsor_id = cursor.fetchone()

            if sponsor_id:
                return sponsor_id[0]

        except Exception as error:
            print(f"Error con %s placeholders: {error}")

            cursor.fetchall()

            consulta_sponsor_id = """
            SELECT id
            FROM sponsors
            WHERE LOWER(REPLACE(name, ' ', '')) = ?;
            """
            try:
                print("Trying with ? placeholders")
                cursor.execute(consulta_sponsor_id, (cleaned_sponsor_name,))
                sponsor_id = cursor.fetchone()

                if sponsor_id:
                    return sponsor_id[0]

            except Exception as error:
                print(f"Error con ? placeholders: {error}")
                return None

    except Exception as error:
        print(f"Error al obtener el id del sponsor: {error}")
        return None

from mysql.connector import connect, Error

def retrieve_agents_db(id_camp):
    query = f"""
        SELECT *
        FROM agents
        WHERE JSON_CONTAINS(campaigns, {id_camp}, '$')
    """
    try:
        conexion  = dbcfg.conectar(HOST=dbcfg.HOST_DB_VAP,  
                                    DATABASE=dbcfg.DB_NAME_VAP,  
                                    USERNAME=dbcfg.USER_DB_VAP,  
                                    PASSWORD=dbcfg.PASSWORD_DB_VAP)
        with conexion.cursor(buffered=True) as cursor:
            cursor.execute(query)
            df_agentes = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        conexion.close()
        df_agentes.to_csv("agentes.csv", index=False)
        return df_agentes
    except Error as e:
        print(f"Error al conectarse a la base de datos: {e}")
        return None


def obtener_o_generar_id_graf(conexion, campaign_id_param, sponsor_id_param):
    """
    Verifica si existe un elemento en campaign_graph_bases con el campaign_id y sponsor_id proporcionados.
    Si existe, retorna el id correspondiente. Si no existe, retorna el último id insertado más uno.
    """
    try:
        # Crear un cursor
        cursor = conexion.cursor()
        if campaign_id_param is None or sponsor_id_param is None:
            raise ValueError("Falta campaign_id o sponsor_id")

        print(f"campaign_id: {campaign_id_param}, sponsor_id: {sponsor_id_param}")
        campaign_id_param=str(campaign_id_param)
        sponsor_id_param=str(sponsor_id_param)
        consulta_campaña_id = """
        SELECT id
        FROM campaign_graph_bases
        WHERE campaign_id = %s AND sponsor_id = %s;
        """
        try:
            print("Trying with %s placeholders")
            cursor.execute(consulta_campaña_id, (campaign_id_param, sponsor_id_param))
            resultado = cursor.fetchone()

            if resultado:
                return resultado[0]

        except Exception as error:
            print(f"Error con %s placeholders: {error}")

            consulta_campaña_id = """
            SELECT id
            FROM campaign_graph_bases
            WHERE campaign_id = ? AND sponsor_id = ?;
            """
            try:
                print("Trying with ? placeholders")
                cursor.execute(consulta_campaña_id, (campaign_id_param, sponsor_id_param))
                resultado = cursor.fetchone()

                if resultado:
                    return resultado[0]

            except Exception as error:
                print(f"Error con ? placeholders: {error}")
                return None

        consulta_ultimo_id = """
        SELECT MAX(id)
        FROM campaign_graph_bases;
        """
        cursor.execute(consulta_ultimo_id)
        ultimo_id = cursor.fetchone()[0]

        nuevo_id = (ultimo_id + 1) if ultimo_id is not None else 1
        return nuevo_id

    except Exception as error:
        print(f"Error al obtener o generar id: {error}")
        return None



def config_agents(id_campania):
    df_agentes = retrieve_agents_db(id_campania)
    df_agentes['id_igs'] = df_agentes['id_igs'].fillna('-1')
    df_agentes['id_igs'] = df_agentes['id_igs'].apply(str)
    df_agentes['dni'] = df_agentes['dni'].fillna('-1')
    df_agentes['dni'] = df_agentes['dni'].apply(str)
    return df_agentes


def close_connection(conexion):
    """
    Cierra la conexión a la base de datos.
    """
    if conexion:
        conexion.close()
        print("Conexión cerrada.")




def make_json_serializable(dato):
    if isinstance(dato, (np.int64, np.int32)):
        return int(dato)
    elif isinstance(dato, (np.float64, np.float32)):
        return float(dato)
    elif isinstance(dato, np.ndarray):
        return json.dumps([float(x) if isinstance(x, (np.float64, np.float32)) else int(x) if isinstance(x, (np.int64, np.int32)) else x for x in dato.tolist()])  # Convertir numpy array a cadena JSON
    elif isinstance(dato, list):
        return json.dumps([make_json_serializable(x) for x in dato])
    elif isinstance(dato, (np.datetime64, datetime)):
        return dato.isoformat()
    elif dato is None:
        return None
    return dato

def insertar_datos_sponsor_kpis(conexion, datos):
    """
    Inserta datos en la tabla sponsor_kpis.
    """
    processed_datos = [make_json_serializable(dato) for dato in datos]

    consulta = (
        "INSERT INTO sponsor_kpis "
        "(sponsor_id,date,total_register,total_archives_process,total_minutes_process,total_calls_affected,"
        "total_buy_accepted,call_quality,unread_calls,deleted_at,created_at,updated_at,"
        "percentage_archives_process,chart_archives_process,percentage_calls_affected,chart_calls_affected,"
        "percentage_buy_accepted,chart_buy_accepted,percentage_call_quality,chart_call_quality,"
        "chart_total_minutes_process,percentage_total_minutes_process,tmo,percentage_tmo,chart_tmo,"
        "percentage_unread_calls,chart_unread_calls) "
        "VALUES(" + ",".join(["%s"] * 27) + ");"
    )

    cursor = conexion.cursor()
    print(processed_datos)
    cursor.execute(consulta, tuple(processed_datos))
    conexion.commit()
    cursor.close()

def insertar_datos_campanias_kpis(conexion, datos):
    """
    Inserta datos en la tabla campaign_kpis.
    """
    processed_datos = []
    for dato in datos:
        if isinstance(dato, list):
            cleaned_list = []
            for el in dato:
                if isinstance(el, np.integer):
                    cleaned_list.append(int(el))
                elif isinstance(el, np.floating):
                    cleaned_list.append(float(el))
                else:
                    cleaned_list.append(el)
            processed_dato = json.dumps(cleaned_list)
        else:
            processed_dato = dato
        processed_datos.append(processed_dato)
    processed_datos = [make_sql_safe(dato) for dato in datos]

    consulta = ("INSERT INTO campaign_kpis "
                "(status_vap, minutes_process, percentage_minutes_process, progress, "
                " total_audios, created_at, updated_at, total_archives_process, "
                " percentage_archives_process, chart_archives_process, "
                " total_calls_affected, percentage_calls_affected, chart_calls_affected, "
                " total_buy_accepted, percentage_buy_accepted, chart_buy_accepted, "
                " call_quality, percentage_call_quality, chart_call_quality, campaign_id, "
                " date, deleted_at, chart_total_minutes_process, tmo, percentage_tmo, chart_tmo, unread_calls)"
                "VALUES(" + ",".join(["%s"] * 27) + ");")

    cursor = conexion.cursor()
    cursor.execute(consulta, tuple(processed_datos))
    conexion.commit()
    cursor.close()

def insertar_datos_recomendaciones(conexion, datos):
    """
    Inserta datos en la tabla recommendations.
    """
    processed_datos = []
    for dato in datos:
        if isinstance(dato, list):
            processed_dato = json.dumps(dato)
        else:
            processed_dato = dato
        processed_datos.append(processed_dato)

    consulta = ("INSERT INTO recommendations "
                "(title,content,date,deleted_at,created_at,updated_at,campaign_id) "
                "VALUES(" + ",".join(["%s"] * 7) + ");")

    cursor = conexion.cursor()
    cursor.execute(consulta, tuple(processed_datos))
    conexion.commit()
    cursor.close()


def insertar_datos_graf(conexion, datos):
    processed_datos = []
    for dato in datos:
        if isinstance(dato, list):
            processed_dato = json.dumps(dato)
        else:
            processed_dato = dato
        processed_datos.append(processed_dato)
    consulta = ("INSERT INTO campaign_graph_bases(words_x_axis,time_y_axis,date,campaign_id,sponsor_id,score,evaluator,array_x_axis,array_y_axis,created_at,updated_at)"
                "VALUES(" + ",".join(["%s"] * 11) + ");")
    cursor = conexion.cursor()
    cursor.execute(consulta, tuple(processed_datos))
    conexion.commit()
    cursor.close()


def actualizar_datos_graf(conexion, datos, id):
    """
    Actualiza los datos en la tabla campaign_graph_bases para un registro específico identificado por id.
    """
    processed_datos = []
    for dato in datos:
        if isinstance(dato, list):
            processed_dato = json.dumps(dato)
        else:
            processed_dato = dato
        processed_datos.append(processed_dato)

    consulta = ("UPDATE campaign_graph_bases SET "
                "words_x_axis = %s, "
                "time_y_axis = %s, "
                "date = %s, "
                "campaign_id = %s, "
                "sponsor_id = %s, "
                "score = %s, "
                "evaluator = %s, "
                "array_x_axis = %s, "
                "array_y_axis = %s, "
                "created_at = %s, "
                "updated_at = %s "
                "WHERE id = %s;")

    cursor = conexion.cursor()
    cursor.execute(consulta, tuple(processed_datos) + (id,))
    conexion.commit()
    cursor.close()
    print(f'Datos actualizados para id={id}: {processed_datos}')


def insertar_datos_flags(conexion, datos):
    processed_datos = []
    for dato in datos:
        if isinstance(dato, list):
            processed_dato = json.dumps(dato)
        else:
            processed_dato = dato
        processed_datos.append(processed_dato)
    try:
        consulta = ("INSERT INTO campaign_graph_data(campaign_graph_base_id, campaign_id, sponsor_id, score_graph, flag_name, flag_hover, date) VALUES (%s, %s, %s, %s, %s, %s, %s);")
        cursor = conexion.cursor()
        cursor.execute(consulta, tuple(processed_datos))
        conexion.commit()
        cursor.close()
    except:
        consulta = ("INSERT INTO campaign_graph_data(campaign_graph_base_id, campaign_id, sponsor_id, score_graph, flag_name, flag_hover, date) VALUES (%s, %s, %s, %s, %s, %s, %s);")
        cursor = conexion.cursor()
        cursor.execute(consulta, tuple(processed_datos))
        conexion.commit()
        cursor.close()



def obtener_ultimo_id_graf(conexion):
    """
    Obtiene el último ID insertado en la tabla campaign_graph_bases.
    """
    try:
        cursor = conexion.cursor()

        consulta = """
        SELECT id
        FROM campaign_graph_bases
        ORDER BY id DESC
        LIMIT 1;
        """
        cursor.execute(consulta)
        resultado = cursor.fetchone()

        return resultado[0] if resultado else None

    except Exception as error:
        print(f"Error al obtener el último id de la campaña: {error}")
        return None

def convertir_a_array(cadena):
    """
    Convierte una cadena representando un array a un array de Python.
    """
    try:
        return ast.literal_eval(cadena)
    except (SyntaxError, ValueError):
        return [0] * 8

def force_array(string):
    try:
        new_array=string.replace('[','').replace(']','').split(',')
        _=new_array[2]
        return new_array
    except (SyntaxError, ValueError,IndexError,AttributeError):
        return ['0'] * 8

def obtener_charts_recientes_campania(conexion, campaign_id):
    """
    Obtiene el último registro creado en campaign_kpis para el campaign_id dado.
    """
    campaign_id=str(campaign_id)
    try:
        cursor = conexion.cursor()

        consulta_campaign_kpis = """
        SELECT
            chart_archives_process,
            chart_calls_affected,
            chart_buy_accepted,
            chart_call_quality,
            chart_total_minutes_process,
            chart_tmo
        FROM
            campaign_kpis
        WHERE
            campaign_id = %s
        ORDER BY
            date DESC
        LIMIT 1;
        """
        cursor.execute(consulta_campaign_kpis, (campaign_id,))
        campaign_kpis_data = cursor.fetchone()
        if not campaign_kpis_data:
            print(f"No se encontraron registros para el campaign_id {campaign_id}")
            datos_vacios = {
                'chart_archives_process': [[0] * 8],
                'chart_calls_affected': [[0] * 8],
                'chart_buy_accepted': [[0] * 8],
                'chart_call_quality': [[0] * 8],
                'chart_total_minutes_process': [[0] * 8],
                'chart_tmo': [[0] * 8],
            }
            return pd.DataFrame(datos_vacios)
        colnames = [desc[0] for desc in cursor.description]
        print(campaign_kpis_data)
        try:
            campaign_kpis_data = pd.DataFrame([campaign_kpis_data], columns=colnames)
            return campaign_kpis_data
        except (Exception) as error:
            print(f"Forzando conversión de datos")
            campaign_kpis_data = [force_array(valor) for valor in campaign_kpis_data]
            return campaign_kpis_data

    except (Exception) as error:
        print(f"Error al obtener datos de campañas: {error}")
        return pd.DataFrame()



def obtener_charts_recientes_sponsor(conexion, id_sponsor):
    """
    Obtiene el último registro creado en sponsor_kpis para el sponsor_id dado,
    filtrado por la fecha máxima de ayer.
    """
    try:
        cursor = conexion.cursor()

        fecha_maxima = (datetime.now()).strftime('%Y-%m-%d')

        consulta_sponsor_kpis = """
        SELECT
            chart_archives_process,
            chart_calls_affected,
            chart_buy_accepted,
            chart_call_quality,
            chart_total_minutes_process,
            chart_tmo,
            created_at,
            unread_calls
        FROM
            sponsor_kpis
        WHERE
            sponsor_id = %s
            AND created_at <= %s
        ORDER BY
            created_at DESC
        LIMIT 1;
        """
        cursor.execute(consulta_sponsor_kpis, (id_sponsor, fecha_maxima))
        sponsor_kpis_data = cursor.fetchone()

        if not sponsor_kpis_data:
            print(f"No se encontraron registros para el sponsor_id {id_sponsor} con fecha hasta {fecha_maxima}")
            datos_vacios = {
                'chart_archives_process': [[0] * 8],
                'chart_calls_affected': [[0] * 8],
                'chart_buy_accepted': [[0] * 8],
                'chart_call_quality': [[0] * 8],
                'chart_total_minutes_process': [[0] * 8],
                'chart_tmo': [[0] * 8],
                'created_at': [None],
                "unread_calls": [[0] * 8]
            }
            return pd.DataFrame(datos_vacios)

        colnames = [desc[0] for desc in cursor.description]

        try:
            df_sponsor_kpis = pd.DataFrame([sponsor_kpis_data], columns=colnames)
            return df_sponsor_kpis
        except Exception as error:
            print(f"Forzando conversión de datos debido a: {error}")
            df_sponsor_kpis = pd.DataFrame([
                [force_array(valor) for valor in sponsor_kpis_data]
            ], columns=colnames)
            return df_sponsor_kpis

    except Exception as error:
        print(f"Error al obtener datos de campañas: {error}")
        return pd.DataFrame()

def obtener_datos_ultima_campania(conexion, sponsor_id):
    """
    Obtiene el último registro creado en campaign_kpis para los campaign_id asociados a un sponsor_id,
    limitado a las campañas creadas el día actual.
    """
    try:
        cursor = conexion.cursor()

        consulta_campaigns = """
        SELECT id
        FROM campaigns
        WHERE sponsor_id = %s;
        """
        cursor.execute(consulta_campaigns, (sponsor_id,))
        campaign_ids = cursor.fetchall()

        if not campaign_ids:
            print(f"No se encontraron campañas para el sponsor_id {sponsor_id}")
            return pd.DataFrame()  # Devolver un DataFrame vacío

        campaign_ids = [id[0] for id in campaign_ids]

        campaign_ids_str = ','.join(map(str, campaign_ids))

        fecha_actual = datetime.now().strftime('%Y-%m-%d')

        consulta_campaign_kpis = f"""
        SELECT ck.*
        FROM campaign_kpis ck
        INNER JOIN (
            SELECT campaign_id, MAX(created_at) AS max_created_at
            FROM campaign_kpis
            WHERE campaign_id IN ({campaign_ids_str})
            AND DATE(created_at) = '{fecha_actual}'
            GROUP BY campaign_id
        ) AS latest
        ON ck.campaign_id = latest.campaign_id AND ck.created_at = latest.max_created_at;
        """
        cursor.execute(consulta_campaign_kpis)
        campaign_kpis_data = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]

        df_campaign_kpis = pd.DataFrame(campaign_kpis_data, columns=colnames)

        return df_campaign_kpis

    except Exception as e:
        print(f"Error al obtener los datos de la última campaña: {e}")
        return pd.DataFrame()


def _to_native(v):
    # None / NaN -> None
    if v is None:
        return None
    if pd is not None:
        # pandas NA/NaT
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        # pandas Timestamp -> datetime
        if isinstance(v, pd.Timestamp):
            return v.to_pydatetime()

    # NumPy escalares -> Python puros
    if isinstance(v, np.generic):
        return v.item()

    # Fechas nativas se pasan tal cual
    if isinstance(v, (datetime, date)):
        return v

    # Estructuras (listas/dicts) -> JSON
    if isinstance(v, (list, dict)):
        return json.dumps(v)

    # Cualquier otro tipo (str, int, float, Decimal, etc.)
    return v

def insertar_datos_agentes_kpi(conexion, datos):
    """
    Inserta datos en la tabla agent_scores.
    Columnas esperadas (13):
      score, inevitable_count, inevitable, not_allowed_count, not_allowed,
      mac, agent_id, campaign_id, date, deleted_at, created_at, updated_at, affected_audios
    """
    # Normaliza TODOS los valores a tipos nativos
    processed_datos = [_to_native(d) for d in datos]

    # (Opcional pero recomendado) valida longitud
    if len(processed_datos) != 13:
        raise ValueError(f"Se esperaban 13 valores, llegaron {len(processed_datos)}: {processed_datos}")

    consulta = (
        "INSERT INTO agent_scores "
        "(score, inevitable_count, inevitable, not_allowed_count, not_allowed, mac, "
        " agent_id, campaign_id, date, deleted_at, created_at, updated_at, affected_audios) "
        "VALUES (" + ",".join(["%s"] * 13) + ");"
    )

    cursor = conexion.cursor()
    cursor.execute(consulta, tuple(processed_datos))
    conexion.commit()
    cursor.close()
    print(f'Datos insertados en agent_scores: {processed_datos}')





def insertar_datos_statistics(conexion, datos):
    """
    Inserta datos en la tabla statistics.
    """
    processed_datos = []
    for dato in datos:
        if isinstance(dato, list):
            processed_dato = json.dumps(dato)
        else:
            processed_dato = dato
        processed_datos.append(processed_dato)

    consulta = ("INSERT INTO campaign_statistics "
                "(section,score_avg,evaluation_module,count_unmissable_found,unmissable_total,count_not_allowed_found,not_allowed_total,tmo,module_speed,module_volume,date,campaign_id,created_at,updated_at) "
                "VALUES (" + ",".join(["%s"] * 14) + ");")

    cursor = conexion.cursor()
    cursor.execute(consulta, tuple(processed_datos))
    conexion.commit()
    cursor.close()
    print(f'Datos insertados: {processed_datos}')

def insertar_datos_agents(conexion, datos):
    """
    Inserta datos en la tabla agent_audio_data.
    """
    processed_datos = []
    for dato in datos:
        if isinstance(dato, list):
            processed_dato = json.dumps(dato)
        else:
            processed_dato = dato
        processed_datos.append(processed_dato)

    consulta = (
        "INSERT INTO agent_audio_data "
        "(name, uuid, agent_id, campaign_id, date, average_score, purchase_acceptance, general_score, analyzed_audios, "
        "tmo, link_audio, unmissable, unmissable_percentage, not_allowed, not_allowed_percentage, sales_arguments, "
        "sales_arguments_percentage, sales_acceptance, sales_acceptance_percentage, created_at, updated_at, transcription, "
        "unmissable_not_found, link_tra"
        "nscription_audio, lead_id, summary_rejection, call_quality, user_feedback_id, "
        "text_feedback_id, concatenated, introduction, description, greeting_farewell, mac_price, empty_call, VC_Percent, concat_num) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
    )

    cursor = conexion.cursor()
    cursor.execute(consulta, tuple(processed_datos))
    conexion.commit()
    cursor.close()
    print(f'Datos insertados: {processed_datos}')



def insertar_datos_affected_calls(conn, datos):
    # Prepara los datos
    processed = []
    for d in datos:
        if isinstance(d, list):
            processed.append(json.dumps(d))   # ya sin fillna (es lista, no DataFrame)
        else:
            processed.append(d)

    query = (
        "INSERT INTO call_affecteds "
        "(agent_audio_data_id, lead_id, summary_rejection, call_quality, "
        " created_at, updated_at, warnings, alert, "
        " iconMute,iconVel,iconVol,iconOverlap,iconHangup,timeVol,timeVel,timeMute,timeHangup,timeOverlap,hoverVol,hoverVel,hoverHang,hoverOverlap) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
    )

    try:
        dbcfg.run_with_reconnect(conn, query, processed)
        print(f"[OK] Datos insertados: {processed}")
    except mysql.connector.Error as err:
        print(f"[ERROR] {err} – datos: {processed}")


def calls_this_campaign(id_campania, conexion):
    cursor = conexion.cursor()
    fecha_actual = datetime.now().date()
    query_id = f"""
        SELECT *
        FROM agent_audio_data
        WHERE campaign_id = {id_campania} 
        AND DATE(created_at) = '{fecha_actual}'
    """

    if cursor:
        try:
            cursor.execute(query_id)
            data = cursor.fetchall()
            if data:
                column_names = [i[0] for i in cursor.description]
                df_llamadas = pd.DataFrame(data, columns=column_names)
                return df_llamadas
            else:
                print("No se encontraron resultados para la consulta.")
            cursor.close()
            conexion.close()
        except mysql.connector.Error as err:
            print(f"Error al ejecutar la consulta: {err}")
    else:
        print("No se pudo establecer la conexión con la base de datos.")



def total_files_on_queue(conexion):
    cursor = conexion.cursor()
    hoy = datetime.now().date()
    ayer = hoy - timedelta(days=1)
    fechas_validas = [hoy.strftime('%Y%m%d'), ayer.strftime('%Y%m%d'),
                      hoy.strftime('%Y-%m-%d'), ayer.strftime('%Y-%m-%d')]

    # Preparamos las condiciones OR para los diferentes formatos
    condiciones_fecha = " OR ".join([f"folder_date = '{fecha}'" for fecha in fechas_validas])

    query = f"""
        SELECT file_count
        FROM vap_status
        WHERE {condiciones_fecha}
        AND processed = 0
    """

    if cursor:
        try:
            cursor.execute(query)
            data = cursor.fetchall()
            if data:
                total_files = sum([int(row[0]) for row in data if row[0] is not None])
                return total_files
            else:
                print("No se encontraron archivos para hoy o ayer.")
                return 0
        except mysql.connector.Error as err:
            print(f"Error al ejecutar la consulta: {err}")
            return 0
        finally:
            cursor.close()
            conexion.close()
    else:
        print("No se pudo establecer la conexión con la base de datos.")
        return 0




def merge_with_null_agent(agentes_db, mat_calls_campaign):
    """
    Realiza un merge entre dos DataFrames y agrega un agente "nulo" con id_igs=-1
    si no hay coincidencias entre las tablas.

    Parameters:
        agentes_db (pd.DataFrame): DataFrame con información de los agentes.
        mat_calls_campaign (pd.DataFrame): DataFrame con información de las llamadas.

    Returns:
        pd.DataFrame: Resultado del merge.
    """
    merged_df = pd.merge(
        mat_calls_campaign,
        agentes_db,
        left_on='AGENT_ID',
        right_on='id_igs',
        how='left'
    )

    merged_df['id_igs'] = merged_df['id_igs'].fillna(0)
    merged_df['name'] = merged_df['name'].fillna('Agente Nulo')
    return merged_df


def getAgentStats(LlamadasPorAgente: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa las llamadas por agente y calcula estadísticas agregadas.
    Parameters:
        LlamadasPorAgente (pd.DataFrame): DataFrame con las llamadas por agente.
    Returns:
        pd.DataFrame: DataFrame con estadísticas agregadas por agente.
    """
    df_concatenado_agentes_unicos = LlamadasPorAgente.groupby('id_igs').agg(
        CONFIDENCE=('confidence', 'mean'),
        TMO=('TMO', 'mean'),
        MUST_HAVE_COUNT=('count_must_have', 'sum'),
        MUST_HAVE_RATE=('must_have_rate', 'mean'),
        FORBIDDEN_COUNT=('count_forbidden', 'sum'),
        FORBIDDEN_RATE=('forbidden_rate', 'mean'),
        SCORE=('score', 'mean'),
        MAC_SCORE=('best_mac_likelihood_macs', 'mean'),
        PRICE_SCORE=('best_mac_likelihood_prices', 'mean'),
        words_p_m=('topic_words_p_m_macs', 'mean'),
        volume_rms=('volume_db_mac', 'mean'),
        volume_classification=('volume_classification_mac', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
        velocity_classification=('velocity_classification_macs', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
        DATE=('DATE_TIME', 'last'),
    ).reset_index().apply(lambda x: x.reset_index(drop=True)).reset_index(drop=True)
    df_concatenado_agentes_unicos = df_concatenado_agentes_unicos.drop_duplicates(subset='id_igs')
    return df_concatenado_agentes_unicos



def mark_campaign_processed(campaign_id: int):
    """
    Sets processed=1 for the given campaign_id in folder_processing_status.
    """
    conn = dbcfg.conectar(HOST=dbcfg.HOST_DB_VAP,  
                                    DATABASE=dbcfg.DB_NAME_VAP,  
                                    USERNAME=dbcfg.USER_DB_VAP,  
                                    PASSWORD=dbcfg.PASSWORD_DB_VAP)
    cursor = conn.cursor()

    update_sql = """
        UPDATE vap_status
        SET processed = 1
        WHERE campaign_id = %s
    """

    cursor.execute(update_sql, (campaign_id,))
    conn.commit()

    cursor.close()
    conn.close()

    print(f"Campaign '{campaign_id}' marked as processed!")





def insertar_datos_vap_report(conexion, datos):
    """
    Inserta datos en la tabla vap_report.
    Espera una lista con los siguientes campos en este orden:
    [campaign_prefix, country, sponsor, campaign, campaign_id,
     folder_date, file_count, process_warning, affected, unread, created_at]
    """
    processed_datos = []
    for dato in datos:
        # Si el dato es una lista (por ejemplo en process_warning), lo convierte a JSON
        if isinstance(dato, list):
            processed_dato = json.dumps(dato)
        else:
            processed_dato = dato
        processed_datos.append(processed_dato)

    consulta = (
        "INSERT INTO vap_report ("
        "campaign_prefix, country, sponsor, campaign, campaign_id, "
        "folder_date, file_count, affected, unread, created_at, "
        "rej_mac, rej_price, rej_noaudit, rej_emptycall, "
        "al_bpc, al_vel_mac, al_vol_mac, al_vel_price, al_vol_price, "
        "al_inex_mac, al_inex_price, al_lowscore, al_legal_na, al_low_tmo, "
        "al_mvd, al_hangup, al_overlap, al_mute, "
        "concat_count, recov_count"
        ") VALUES (" + ",".join(["%s"] * 30) + ");"
    )

    cursor = conexion.cursor()
    cursor.execute(consulta, tuple(processed_datos))
    conexion.commit()
    cursor.close()
    print(f'Datos insertados en vap_report: {processed_datos}')