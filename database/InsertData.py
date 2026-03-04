from datetime import datetime
import numpy as np
import time, json, math

import pandas as pd

from database.SQLDataManager import force_array, insertar_datos_campanias_kpis, \
    obtener_charts_recientes_sponsor, obtener_datos_ultima_campania, insertar_datos_sponsor_kpis, obtener_ultimo_id_graf, \
    actualizar_datos_graf, insertar_datos_graf, insertar_datos_flags, insertar_datos_agentes_kpi, \
    insertar_datos_statistics, insertar_datos_agents, insertar_datos_affected_calls, insertar_datos_vap_report
from Deprecated.ScoreEngine import smooth_array, process_directory_and_average
from database.dbConfig import generate_s3_client
from utils.campaignMetrics import count_local_files
from utils.VapUtils import get_data_types
from vapStatus import get_latest_date_folder

import numpy as np
import pandas as pd
import ast

def vols_to_numeric_array(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.array([], dtype=float)

    # Si ya es lista/np.array
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.asarray(v, dtype=float)

    # Si es string: intentar parsear
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return np.array([], dtype=float)

        # caso típico: "[1, 2, 3]"
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return np.asarray(parsed, dtype=float)
        except Exception:
            pass

        # fallback: "1,2,3" o "1 2 3"
        s = s.replace(",", " ")
        parts = [p for p in s.split() if p]
        try:
            return np.asarray([float(p) for p in parts], dtype=float)
        except Exception:
            return np.array([], dtype=float)

    # Cualquier otro tipo raro
    return np.array([], dtype=float)


def to_float_list(arr):
    out = []
    for x in arr:
        if isinstance(x, (int, float, np.number)):
            out.append(float(x))
        elif x is None:
            out.append(float('nan'))
        else:
            s = str(x).strip().replace('%','').replace(',','')
            try:
                out.append(float(s))
            except ValueError:
                out.append(float('nan'))
    return out

def str_to_array(s):
    if pd.isna(s):
        return np.array([], dtype=float)
    s = s.strip("[]")
    return np.fromstring(s, sep=" ")

def rel_change(arr, smooth=0.0):
    # (x_i - x_{i-1}) / (x_i + smooth), i >= 1
    a = np.asarray(to_float_list(arr), dtype=float)
    num = np.diff(a)
    den = a[1:] + smooth
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=~np.isnan(den) & (den != 0))
    return out.tolist()


def n0(x, as_int=False, rnd=None):
    """NaN/Inf/None -> 0.0; opcionalmente redondea y castea a int."""
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if math.isnan(v) or math.isinf(v):
        v = 0.0
    if rnd is not None:
        v = round(v, rnd)
    return int(v) if as_int else v


def insertar_campanias(conexion, directory_camp, id_campania_num, all_camps_same_spons, REJECTED, MAT_CALLS, MP3F, TMO, AUDIO_MINUTES,average_consciousness_score=0):

    def last_n_json(seq, n=8):
        """Toma últimos n, sanea NaN/Inf -> 0, devuelve JSON string."""
        arr = np.array(seq, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if len(arr) == 0:
            return json.dumps([0.0]*n)
        tail = arr[-n:].tolist()
        return json.dumps(tail)

    def safe_pct(series):
        """(x_t - x_{t-1}) / x_t si hay 2 puntos y x_t>0; si no, 0."""
        if series is None or len(series) < 2:
            return 0.0
        a = float(series[-1])
        b = float(series[-2])
        a = 0.0 if (math.isnan(a) or math.isinf(a)) else a
        b = 0.0 if (math.isnan(b) or math.isinf(b)) else b
        return 0.0 if a == 0 else (a - b) / a

    print(all_camps_same_spons)
    JSONS_TOT = count_local_files(directory_camp + '/transcript_sentences/', 'json')  # no se usa pero lo dejo
    KULLBACK_LEIBLER = 0.0
    AUDIO_QUALITY = 1.0
    AUDIO_MINUTES_PROCESSED = n0(AUDIO_MINUTES)

    # Fuerzo arrays y saneo NaN
    array_archives_process = force_array(all_camps_same_spons['chart_archives_process'][0])
    array_calls_affected   = force_array(all_camps_same_spons['chart_calls_affected'][0])
    array_buy_accepted     = force_array(all_camps_same_spons['chart_buy_accepted'][0])
    array_call_quality     = force_array(np.nan_to_num(all_camps_same_spons['chart_call_quality'][0], nan=0.0))
    array_total_minutes    = force_array(all_camps_same_spons['chart_total_minutes_process'][0])
    array_tmo              = force_array(all_camps_same_spons['chart_tmo'][0])

    AUDIOS_UNPROCESSED = n0(MP3F, as_int=True)
    REJECTED_LEN = n0(np.sum(REJECTED['reject']))
    print(REJECTED_LEN)
    TOTAL_AFFECTED = REJECTED_LEN
    TOTAL_SALES    = n0(np.mean(np.round(MAT_CALLS['best_mac_likelihood_macs'], 3) * 100.0))
    TMO_CAMP       = n0(TMO)

    # Append punto actual
    array_archives_process.append(AUDIOS_UNPROCESSED)
    array_calls_affected.append(TOTAL_AFFECTED)
    array_buy_accepted.append(TOTAL_SALES)
    array_call_quality.append(n0(AUDIO_QUALITY))
    array_total_minutes.append(AUDIO_MINUTES_PROCESSED)
    array_tmo.append(TMO_CAMP)

    # Cambios relativos
    archives_chg       = rel_change(np.nan_to_num(array_archives_process, nan=0.0, posinf=0.0, neginf=0.0).tolist(), smooth=1.0)
    calls_affected_chg = rel_change(np.nan_to_num(array_calls_affected,   nan=0.0, posinf=0.0, neginf=0.0).tolist(), smooth=1.0)
    buy_accepted_chg   = rel_change(np.nan_to_num(array_buy_accepted,     nan=0.0, posinf=0.0, neginf=0.0).tolist(), smooth=1.0)
    call_quality_chg   = rel_change(np.nan_to_num(array_call_quality,     nan=0.0, posinf=0.0, neginf=0.0).tolist(), smooth=1.0)
    total_minutes_chg  = rel_change(np.nan_to_num(array_total_minutes,    nan=0.0, posinf=0.0, neginf=0.0).tolist(), smooth=1.0)
    tmo_chg            = rel_change(np.nan_to_num(array_tmo,              nan=0.0, posinf=0.0, neginf=0.0).tolist(), smooth=1.0)

    array_archives_process = archives_chg
    array_calls_affected   = calls_affected_chg
    array_buy_accepted     = buy_accepted_chg
    array_call_quality     = call_quality_chg
    array_total_minutes    = total_minutes_chg
    array_tmo              = tmo_chg

    #unread_n = n0(count_local_files(directory_camp + "isolated/", 'mp3'), as_int=True)

    ###_________________________MODIFICACIONES PARA VENTAS CONSCIENTES_________________________###
    """
    Las únicas 2 modiciaciones son:
    1. Se recibe el parámetro average_consciousness_score en la función insertar_campanias, parámetro que se convertirá en 
    'average_consciousness_score_to_insert' y será el valor insertado en BD (todo lo demás se llama 'unread' pero ahora es venta consciente).

    2. Se elimina la línea que calculaba 'unread_n' (número de audios aislados) ya que ahora no se usa.
    """
    average_consciousness_score_to_insert = n0(average_consciousness_score, as_int=True)

    ###

    # Porcentajes seguros
    MINUTES_PROCESSED_PCT = safe_pct(array_total_minutes)
    FILES_PROCESSED_PCT   = safe_pct(array_archives_process)
    FILES_AFFECTED_PCT    = safe_pct(array_calls_affected)
    SALE_ACCEPTANCE_PCT   = safe_pct(array_buy_accepted)
    AUDIO_QUALITY_PCT     = safe_pct(array_call_quality)
    TMO_PCT               = safe_pct(array_tmo)

    now = time.strftime('%Y-%m-%d %H:%M:%S')

    # Charts a JSON (últimos 8)
    chart_archives  = last_n_json(array_archives_process, 8)
    chart_affected  = last_n_json(array_calls_affected,   8)
    chart_buy       = last_n_json(array_buy_accepted,     8)
    chart_quality   = last_n_json(array_call_quality,     8)
    chart_minutes   = last_n_json(array_total_minutes,    8)
    chart_tmo       = last_n_json(array_tmo,              8)

    # Armo tupla con tipos insertables
    datos_camp = (
        'ACTIVO',                                   # status_vap
        n0(AUDIO_MINUTES_PROCESSED, as_int=True),   # minutes_process
        n0(MINUTES_PROCESSED_PCT * 100.0, rnd=2),   # percentage_minutes_process
        n0(1.0 * 100.0, rnd=2),                     # progress
        n0(AUDIOS_UNPROCESSED, as_int=True),        # total_audios
        now,                                        # created_at
        now,                                        # updated_at
        n0(MP3F, as_int=True),                      # total_archives_process
        n0(FILES_PROCESSED_PCT * 100.0, rnd=2),     # percentage_archives_process
        chart_archives,                              # chart_archives_process (JSON)
        n0(TOTAL_AFFECTED, as_int=True),            # total_calls_affected
        n0(FILES_AFFECTED_PCT * 100.0, rnd=2),      # percentage_calls_affected
        chart_affected,                              # chart_calls_affected (JSON)
        n0(TOTAL_SALES, rnd=0),                     # total_buy_accepted
        n0(np.round(SALE_ACCEPTANCE_PCT,1) * 100.0, rnd=0),     # percentage_buy_accepted
        chart_buy,                                   # chart_buy_accepted (JSON)
        n0(AUDIO_QUALITY),                           # call_quality
        n0(AUDIO_QUALITY_PCT * 100.0, rnd=0),       # percentage_call_quality
        chart_quality,                                # chart_call_quality (JSON)
        str(id_campania_num),                       # campaign_id (si es FK string)
        now,                                        # date
        None,                                       # deleted_at
        chart_minutes,                               # chart_total_minutes_process (JSON)
        n0(TMO_CAMP, rnd=0),                        # tmo
        n0(TMO_PCT * 100.0, rnd=1),                 # percentage_tmo
        chart_tmo,                                   # chart_tmo (JSON)
        n0(average_consciousness_score_to_insert, as_int=True)# Antes era: unread_calls (asumo INT),(AHORA ES EL PROMEDIO DE VENTA CONSCIENTE)
    )

    print('inserting campaigns')
    insertar_datos_campanias_kpis(conexion, datos_camp)

def insertar_sponsors(conexion, id_sponsor):

    def last_n_json(seq, n=8):
        arr = np.array(seq, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if len(arr) == 0:
            return json.dumps([0.0]*n)
        return json.dumps(arr[-n:].tolist())

    def safe_pct(series):
        if series is None or len(series) < 2:
            return 0.0
        a = float(series[-1]); b = float(series[-2])
        if math.isnan(a) or math.isinf(a): a = 0.0
        if math.isnan(b) or math.isinf(b): b = 0.0
        return 0.0 if a == 0 else (a - b) / a

    df_sponsor_campañas = obtener_datos_ultima_campania(conexion, id_sponsor)

    TOTAL_SALES_spon             = n0(np.mean(df_sponsor_campañas['total_buy_accepted']))
    AUDIOS_UNPROCESSED_spon      = n0(np.sum(df_sponsor_campañas['total_audios']), as_int=True)
    AUDIOS_PROCESSED_spon        = n0(np.sum(df_sponsor_campañas['total_archives_process']), as_int=True)
    AUDIO_MINUTES_PROCESSED_spon = n0(np.sum(df_sponsor_campañas['minutes_process']), as_int=True)
    TMO_spon                     = n0(np.mean(df_sponsor_campañas['tmo']))
    TOTAL_AFFECTED_spon          = n0(np.sum(df_sponsor_campañas['total_calls_affected']), as_int=True)
    AUDIO_QUALITY_spon           = n0(np.mean(df_sponsor_campañas['call_quality']))
    unread_spon                  = n0(np.mean(df_sponsor_campañas['unread_calls']))#La variable 'unread_spon' ahora representa la suma de los promedios de venta consciente de las campañas del sponsor.

    array_spons = obtener_charts_recientes_sponsor(conexion, id_sponsor)

    array_archives_process_s = force_array(array_spons['chart_archives_process'][0])
    array_calls_affected_s   = force_array(array_spons['chart_calls_affected'][0])
    array_buy_accepted_s     = force_array(array_spons['chart_buy_accepted'][0])
    array_call_quality_s     = force_array(array_spons['chart_call_quality'][0])
    array_total_minutes_s    = force_array(array_spons['chart_total_minutes_process'][0])
    array_tmo_s              = force_array(array_spons['chart_tmo'][0])
    array_unread_spon        = force_array(array_spons['unread_calls'][0])

    # Append punto actual
    array_archives_process_s.append(AUDIOS_UNPROCESSED_spon)
    array_calls_affected_s.append(TOTAL_AFFECTED_spon)
    array_buy_accepted_s.append(TOTAL_SALES_spon)
    array_call_quality_s.append(AUDIO_QUALITY_spon)
    array_total_minutes_s.append(AUDIO_MINUTES_PROCESSED_spon)
    array_tmo_s.append(TMO_spon)
    array_unread_spon.append(unread_spon)

    # Cambios relativos sin suavizado (pediste smooth=0.0)
    archives_s_chg       = rel_change(np.nan_to_num(array_archives_process_s, nan=0.0, posinf=0.0, neginf=0.0).tolist(), smooth=0.0)
    calls_affected_s_chg = rel_change(np.nan_to_num(array_calls_affected_s,   nan=0.0, posinf=0.0, neginf=0.0).tolist(), smooth=0.0)
    buy_accepted_s_chg   = rel_change(np.nan_to_num(array_buy_accepted_s,     nan=0.0, posinf=0.0, neginf=0.0).tolist(), smooth=0.0)
    call_quality_s_chg   = rel_change(np.nan_to_num(array_call_quality_s,     nan=0.0, posinf=0.0, neginf=0.0).tolist(), smooth=0.0)
    total_minutes_s_chg  = rel_change(np.nan_to_num(array_total_minutes_s,    nan=0.0, posinf=0.0, neginf=0.0).tolist(), smooth=0.0)
    tmo_s_chg            = rel_change(np.nan_to_num(array_tmo_s,              nan=0.0, posinf=0.0, neginf=0.0).tolist(), smooth=0.0)
    unread_spon_chg      = rel_change(np.nan_to_num(array_unread_spon,        nan=0.0, posinf=0.0, neginf=0.0).tolist(), smooth=0.0)

    array_archives_process_s = archives_s_chg
    array_calls_affected_s   = calls_affected_s_chg
    array_buy_accepted_s     = buy_accepted_s_chg
    array_call_quality_s     = call_quality_s_chg
    array_total_minutes_s    = total_minutes_s_chg
    array_tmo_s              = tmo_s_chg
    array_unread_spon        = unread_spon_chg

    # Porcentajes seguros
    FILES_PROCESSED_PCT_spon  = safe_pct(array_archives_process_s)
    FILES_AFFECTED_PCT_spon   = safe_pct(array_calls_affected_s)
    SALE_ACCEPTANCE_PCT_spon  = safe_pct(array_buy_accepted_s)
    AUDIO_QUALITY_PCT_spon    = safe_pct(array_call_quality_s)
    MINUTES_PROCESSED_PCT_spon= safe_pct(array_total_minutes_s)
    TMO_PCT_spon              = safe_pct(array_tmo_s)
    unread_pct_spon           = safe_pct(array_unread_spon)

    now = time.strftime('%Y-%m-%d %H:%M:%S')

    # Charts a JSON (últimos 8)
    chart_archives  = last_n_json(array_archives_process_s, 8)
    chart_affected  = last_n_json(array_calls_affected_s,   8)
    chart_buy       = last_n_json(array_buy_accepted_s,     8)
    chart_quality   = last_n_json(array_call_quality_s,     8)
    chart_minutes   = last_n_json(array_total_minutes_s,    8)
    chart_tmo       = last_n_json(array_tmo_s,              8)
    chart_unread    = last_n_json(array_unread_spon,        8)

    datos_sponsor = (
        id_sponsor,
        now,
        n0(AUDIOS_UNPROCESSED_spon, as_int=True),
        n0(AUDIOS_PROCESSED_spon, as_int=True),
        n0(AUDIO_MINUTES_PROCESSED_spon, as_int=True),
        n0(TOTAL_AFFECTED_spon, as_int=True),
        n0(TOTAL_SALES_spon, rnd=0),
        n0(AUDIO_QUALITY_spon),
        n0(unread_spon),                                   # unread_calls actual
        None,
        now,
        now,
        n0(FILES_PROCESSED_PCT_spon * 100.0, rnd=2),
        chart_archives,                                     # chart_archives_process (JSON)
        n0(FILES_AFFECTED_PCT_spon * 100.0, rnd=2),
        chart_affected,                                     # chart_calls_affected (JSON)
        n0(SALE_ACCEPTANCE_PCT_spon * 100.0, rnd=2),
        chart_buy,                                          # chart_buy_accepted (JSON)
        n0(AUDIO_QUALITY_PCT_spon * 100.0, rnd=3),
        chart_quality,                                      # chart_call_quality (JSON)
        chart_minutes,                                      # chart_total_minutes_process (JSON)
        n0(MINUTES_PROCESSED_PCT_spon * 100.0, rnd=2),
        n0(TMO_spon, rnd=1),
        n0(TMO_PCT_spon * 100.0, rnd=2),
        chart_tmo,                                          # chart_tmo (JSON)
        n0(unread_pct_spon * 100.0, rnd=2),
        chart_unread                                        # chart_unread_calls (JSON)
    )

    print('inserting sponsors')
    insertar_datos_sponsor_kpis(conexion, datos_sponsor)



def insertar_grafica(conexion,id_campania_num,id_sponsor,id_graf,indexes,x_graph,y_graph,statistics,TMO,CALL_DATA):
    indice_saludo, indice_producto, indice_venta, indice_mac, indice_confirmacion, indice_despedida = indexes
    GRAF_MAIN_X = list(x_graph)
    GRAF_MAIN_Y = list(y_graph)
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    datos_graf_1 = (list(np.sort(list(np.round(np.array(
        [indice_saludo, indice_producto, indice_venta, indice_mac, indice_confirmacion, indice_despedida])/ (np.max(x_graph)-1) * TMO, 2)))),
                    ['SALUDO', 'PRODUCTO', 'MAC_DEF', 'PRECIO_DEF', 'CONFIRMACION DATOS', 'DESPEDIDA'],
                    now,
                    str(id_campania_num),
                    str(id_sponsor),
                    float(np.round(np.mean(CALL_DATA['score']), 2)),
                    float(np.round(np.mean(CALL_DATA['score']), 2)),
                    GRAF_MAIN_X,
                    GRAF_MAIN_Y,
                    now,
                    now)
    print('inserting graph data')

    if obtener_ultimo_id_graf(conexion):
        id_graf = obtener_ultimo_id_graf(conexion)
        actualizar_datos_graf(conexion, datos_graf_1, id_graf)
    else:
        print(get_data_types(datos_graf_1))
        insertar_datos_graf(conexion, datos_graf_1)

    heights = y_graph[indexes]

    t_1 = 'SALUDO'
    t_2 = 'PRODUCTO'
    t_3 = 'MAC'
    t_4 = 'PRECIO'
    t_5 = 'CONFIRMACION DATOS'
    t_6 = 'DESPEDIDA'

    saludo_score = statistics[statistics['final_label'] == t_1]['mean_conf'] * 100
    descripcion_score = statistics[statistics['final_label'] == t_2]['mean_conf'] * 100
    venta_score = statistics[statistics['final_label'] == t_3]['mean_conf'] * 100
    mac_score = statistics[statistics['final_label'] == t_4]['mean_conf'] * 100
    confirmacion_score = statistics[statistics['final_label'] == t_5]['mean_conf'] * 100
    despedida_score = statistics[statistics['final_label'] == t_6]['mean_conf'] * 100


    datos_graf_hov1 = (str(id_graf), str(id_campania_num), str(id_sponsor), str(heights[0]),
                       'Saludo', '[{"name": "Saludo", "value": ' + str(
        np.round(saludo_score.iloc[0], 1)) + '}, {"name": "Score", "value": ' + str(
        np.round(saludo_score.iloc[0], 1)) + '}]',
                       now)

    datos_graf_hov2 = (str(id_graf), str(id_campania_num), str(id_sponsor), str(heights[1]),
                       'PRODUCTO', '[{"name": "Description ", "value": ' + str(
        np.round(descripcion_score.iloc[0], 1)) + '}, {"name": "Score", "value": ' + str(
        np.round(descripcion_score.iloc[0], 1)) + '}]',
                       now)

    datos_graf_hov3 = (str(id_graf), str(id_campania_num), str(id_sponsor), str(heights[2]),
                       'MAC', '[{"name": "Terminos de venta ", "value": ' + str(
        np.round(venta_score, 1)) + '}, {"name": "Score", "value": ' + str(np.round(venta_score, 1)) + '}]',
                       now)
    datos_graf_hov4 = (str(id_graf), str(id_campania_num), str(id_sponsor), str(heights[3]),
                       'PRECIO', '[{"name": "MAC completado ", "value": ' + str(
        np.round(mac_score, 1)) + '}, {"name": "Score", "value": ' + str(np.round(mac_score, 1)) + '}]',
                       now)

    datos_graf_hov5 = (str(id_graf), str(id_campania_num), str(id_sponsor), str(heights[4]),
                       'CONFIRMACION DE DATOS', '[{"name": "Validacion de identidad ", "value": ' + str(
        np.round(confirmacion_score.iloc[0], 1)) + '}, {"name": "Score", "value": ' + str(
        np.round(confirmacion_score.iloc[0], 1)) + '}]',
                       now)

    datos_graf_hov6 = (str(id_graf), str(id_campania_num), str(id_sponsor), str(heights[5]),
                       'DESPEDIDA', '[ {"name": "Despedida", "value": ' + str(
        np.round(despedida_score.iloc[0], 1)) + '}, {"name": "Score", "value": ' + str(
        np.round(despedida_score.iloc[0], 1)) + '}]',
                       now)
    print('inserting hover 1')
    insertar_datos_flags(conexion, datos_graf_hov1)
    print('inserting hover 2')
    insertar_datos_flags(conexion, datos_graf_hov2)
    print('inserting hover 3')
    insertar_datos_flags(conexion, datos_graf_hov3)
    print('inserting hover 4')
    insertar_datos_flags(conexion, datos_graf_hov4)
    print('inserting hover 5')
    insertar_datos_flags(conexion, datos_graf_hov5)
    print('inserting hover 6')
    insertar_datos_flags(conexion, datos_graf_hov6)


def insertar_filas_dataframe_agente_score(conexion,id_campania_num, df,list_affected):
    """
    Inserta cada fila de un DataFrame en la base de datos.
    """
    df_alt= df.fillna(0)
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    for index, row in df_alt.iterrows():
        datos_agentes = (
            str(np.round(row['SCORE']*10, 1)),
            str(np.round(row['MUST_HAVE_COUNT'], 0)),
            str(np.round(row['MUST_HAVE_RATE'], 0)),
            str(np.round(row['FORBIDDEN_COUNT'], 0)),
            str(np.round(row['FORBIDDEN_RATE'], 0)),
            str(np.round(row['MAC_SCORE']*100, 2)),
            str(row['id_igs']),
            id_campania_num,
            now,
            None,
            now,
            now,
            list_affected[list_affected['id_igs']==row['id_igs']]['count'].iloc[0] if row['id_igs'] in list_affected['id_igs'].array else 0
        )
        insertar_datos_agentes_kpi(conexion, datos_agentes)



def insertar_filas_dataframe_estadisticas(conexion,id_campania_num, df,df_concatenado):
    """
    Inserta cada fila de un DataFrame en la base de datos.
    """
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    for index, row in df.iterrows():
        datos_secciones = (
            row['predicted_cluster'],
            str(np.round(row['confidence_score']*10, 1)),
            str(np.round(row['confidence_score']*10, 0)),
            str(np.round(row['confidence_score']*np.max(df_concatenado['count_must_have']), 0)),
            str(np.round(np.max(df_concatenado['count_must_have']), 0)),
            str(np.round(row['confidence_score']*np.max(df_concatenado['count_forbidden']), 0)),
            str(np.round(np.max(df_concatenado['count_forbidden']), 0)),
            str(np.round(row['relative_tmo'], 2)),
            row['velocity_classification'],
            row['velocity_classification'],
            now,
            id_campania_num,
            now,
            now)
        insertar_datos_statistics(conexion, datos_secciones)

import math
import numpy as np

def sql_float(x, default=None, ndigits=None):
    """
    Convierte a float seguro para MySQL:
    - np.nan / None / inf -> default (None o 0.0)
    - redondea si ndigits no es None
    """
    if x is None:
        return default
    try:
        v = float(x)
    except Exception:
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    if ndigits is not None:
        v = round(v, ndigits)
    return v


def insertar_filas_dataframe_agentes(conexion,campaign_directory,id_campania_num, statistics, MAT_DATAFRAME,route,route_transcripts,indexes):
    """
    Inserta cada fila de un DataFrame en la base de datos.
    """
    indice_saludo, indice_producto, indice_venta, indice_mac, indice_confirmacion, indice_despedida = indexes

    now = time.strftime('%Y-%m-%d %H:%M:%S')
    AUDIOS_PROCESSED = count_local_files(campaign_directory,'mp3')
    KULLBACK_LEIBLER = np.abs(0) ** (1 / 2)
    t_1 = 'SALUDO'
    t_2 = 'PRODUCTO'
    t_3 = 'MAC_DEF'
    t_4 = 'PRECIO_DEF'
    t_5 = 'CONFIRMACION DATOS'
    t_6 = 'DESPEDIDA'

    saludo_score = statistics[statistics['final_label'] == t_1]['mean_conf'] * 100
    descripcion_score = statistics[statistics['final_label'] == t_2]['mean_conf'] * 100
    venta_score = statistics[statistics['final_label'] == t_3]['mean_conf'] * 100
    mac_score = statistics[statistics['final_label'] == t_4]['mean_conf'] * 100
    confirmacion_score = statistics[statistics['final_label'] == t_5]['mean_conf'] * 100
    despedida_score = statistics[statistics['final_label'] == t_6]['mean_conf'] * 100
    
    for index, row in MAT_DATAFRAME.iterrows():
        average_score = sql_float(row.get("score"), default=None, ndigits=1)
        purchase_acceptance = sql_float(np.round(row.get("best_mac_likelihood_macs")*100, 0), default=None, ndigits=1)
        gen_score = sql_float(np.round(row.get("score")*100, 1), default=None, ndigits=2)
        sales_arguments = sql_float(np.round(row.get("best_mac_likelihood_macs")*100, 0), default=None, ndigits=1)
        sales_arguments_percentage = sql_float(np.round(row.get("best_mac_likelihood_macs")*100, 1), default=None, ndigits=2)
        VC_Percent = sql_float(np.round(row['porcentaje_venta_consciente'], 1)) if "porcentaje_venta_consciente" in MAT_DATAFRAME.columns else sql_float(np.round(0, 1))
        concat = 0
        sales_acceptance = sql_float(np.round(row.get("best_mac_likelihood_prices")*100, 0), default=None, ndigits=1)
        sales_acceptance_percentage = sql_float(np.round(row.get("best_mac_likelihood_prices")*100, 1), default=None, ndigits=2)
        tmo=row["TMO"]
        if row["TMO"] is None or pd.isna(row["TMO"]):
             tmo= 0
        if '-concat' in row['file_name']:
            concat = 1
        # Create the tuple matching the 30 fields in the SQL query
        datos_secciones = (
            row['file_name'],  # name
            None,  # uuid
            str(row['AGENT_ID']),  # agent_id
            str(id_campania_num),  # campaign_id
            now,  # date
            average_score,  # average_score
            purchase_acceptance,  # purchase_acceptance
            gen_score,  # general_score
            str(AUDIOS_PROCESSED),  # analyzed_audios
            tmo,  # tmo
            (route+row['file_name']).replace('.mp3', '.wav'),  # link_audio
            str(np.round(row['count_must_have'], 1)),  # unmissable
            str(np.round(row['must_have_rate'], 1)),  # unmissable_percentage
            str(np.round(row['count_forbidden'], 1)),  # not_allowed
            str(np.round(row['forbidden_rate'], 1)),  # not_allowed_percentage
            sales_arguments,  # sales_arguments
            sales_arguments_percentage,  # sales_arguments_percentage
            sales_acceptance,  # sales_acceptance
            sales_acceptance_percentage,  # sales_acceptance_percentage
            now,  # created_at
            now,  # updated_at
            None,  # transcription
            str(np.round(np.max(MAT_DATAFRAME['count_must_have']) - row['count_must_have'], 1)),
            # unmissable_not_found
            route_transcripts +  row['file_name'].replace('.mp3', '_transcript_paragraphs.json'),  # link_transcription_audio
            str(row['LEAD_ID']),  # lead_id (added)
            None,  # summary_rejection (added)
            str(np.round(KULLBACK_LEIBLER, 2)),  # call_quality (added)
            None,  # user_feedback_id (added)
            None,  # text_feedback_id (added)
            str(concat),  # concatenated
            str(float(np.round(10, 2))),  # introduction
            str(float(np.round(10, 2))),  # description
            str(float(np.round(10, 2))),  # greeting_farewell
            str(float(np.round((np.round(row['best_mac_likelihood_macs'] * 100, 1)
                                + np.round(row['best_mac_likelihood_prices'] * 100, 1)) / 2))), # mac_price
            0,
            VC_Percent,
            int(row['num_concat']) #concat_num
        )
        try:
            insertar_datos_agents(conexion, datos_secciones)
        except Exception as e:
            print(f"Error inserting row {index}: {e}")
            continue  # Skip to the next row on error

def insertar_filas_dataframe_afectadas(conexion, MAT_AFECTADAS):
    """
    Inserta cada fila de un DataFrame en la base de datos.
    """
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    #print(df)
    MAT_AFECTADAS['DATE_TIME'] = pd.to_datetime(MAT_AFECTADAS['DATE_TIME'], format='%Y%m%d-%H%M%S', errors='coerce')
    for index, row in MAT_AFECTADAS.iterrows():
        if isinstance(row['DATE_TIME'], str):
            # Convert string to datetime object
            row['DATE_TIME'] = datetime.strptime(row['DATE_TIME'], "%Y%m%d-%H%M%S")
        MAT_AFECTADAS.to_excel('afectadas_analysis.xlsx')
        #print("Number of NaT in DATE_TIME:", MAT_AFECTADAS['DATE_TIME'].isna().sum())
        val = row['DATE_TIME']

        if pd.isnull(val):
            print(f"Row {index} has a NaT or null: {val}")
        elif not isinstance(val, pd.Timestamp):
            print(f"Row {index} has a non-Timestamp type: {type(val)} => {val}")
        formatted_str = val.strftime("%Y-%m-%d %H:%M:%S")
        MAT_AFECTADAS['DATE_TIME'].fillna(pd.Timestamp(now), inplace=True)

        formatted_str = row['DATE_TIME'].strftime("%Y-%m-%d %H:%M:%S")
        vols_arr = vols_to_numeric_array(row["vols"])
        max_vol = np.round(np.max(100-vols_arr), 2) if vols_arr.size > 0 else 0
        print(max_vol)
        iconVel = 1 if (vols_arr.size > 0 and np.max(np.abs(vols_arr)) > 5) else 0
        if row["hangup_signaturetime"] is None or pd.isna(row["hangup_signaturetime"]):
            row["hangup_signaturetime"] = 0
        if row["time_overlap"] is None or pd.isna(row["time_overlap"]):
            row["time_overlap"] = 0

        timeHangup = float(np.round(row["hangup_signaturetime"], 2))
        timeOverlap = float(np.round(row["time_overlap"], 2))
        iconOverlap = int(1 if row["overlap_total_time"] else 0)
        iconHangup  = int(1 if row["hangup_n_detections"] > 0 else 0)
        try:
            datos_secciones = (
                str(row['agent_audio_data_id']), #agent_audio_data_id
                str(row['LEAD_ID']), #lead_id
                str(row['reject_reason']), #summary_rejection
                'rejected', #call_quality
                formatted_str, #created_at
                now, #updated_at
                '', #warnings
                str(row['warning_reason']), #alert 
                0, #iconMute
                iconVel, #iconVel
                iconVel, #iconVol
                iconOverlap, #iconOverlap
                iconHangup, #iconHangup
                str(max_vol), #timeVol
                str(max_vol), #timeVel
                0, #timeMute
                timeHangup, #timeHangup
                timeOverlap,#timeOverlap
                f"Volumen alto en: ",#hoverVol
                f"Velocidad alta en: ",#hoverVel
                f"hangup anticipado detectado: ",#hoverHangup
                f"overlapping detectado en:",#hoverOverlap
            )
            insertar_datos_affected_calls(conexion, datos_secciones)
        except:
            print("WARNING: Algunos agentes de planta no estan actualizados, insertando sin audio data id")
            datos_secciones = (
                str(row['agent_audio_data_id']),
                str(row['LEAD_ID']),
                str(row['reject_reason']),
                'rejected',
                formatted_str,
                now,
                '', #warnings
                str(row['warning_reason']), #alert 
                0, #iconMute
                iconVel, #iconVel
                iconVel, #iconVol
                iconOverlap, #iconOverlap
                iconHangup, #iconHangup
                str(max_vol), #timeVol
                str(max_vol), #timeVel
                0, #timeMute
                timeHangup, #timeHangup
                timeOverlap,#timeOverlap
                f"Volumen alto en: {max_vol}",#hoverVol
                f"Velocidad alta en: {max_vol}",#hoverVel
                f"hangup anticipado detectado en: ",#hoverHangup
                f"overlapping detectado en: ",#hoverOverlap
            )
            try:
                insertar_datos_affected_calls(conexion, datos_secciones)
            except Exception as e:
                print(f"Error inserting affected call with missing audio_data_id: {e}")
                continue  # Skip to the next row on error

#"(agent_audio_data_id, lead_id, summary_rejection, call_quality, "
#        " created_at, updated_at, warnings, alert, "
#        " iconMute,iconVel,iconVol,iconOverlap,iconHangup,timeVol,timeVel,timeMute,timeHangup) "

def _choose_agent_audio_date_col(conexion):
    """
    Returns the most suitable date/datetime column name in agent_audio_data,
    checking common candidates in priority order.
    """
    candidates = [
        'date', 'call_date', 'last_local_call_time',
        'call_datetime', 'created_at', 'inserted_at'
    ]
    sql = (
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() "
        "AND TABLE_NAME = 'agent_audio_data' "
        "AND COLUMN_NAME IN (" + ",".join(["%s"] * len(candidates)) + ")"
    )
    cur = conexion.cursor()
    cur.execute(sql, tuple(candidates))
    present = {row[0] for row in cur.fetchall()}
    cur.close()
    for c in candidates:
        if c in present:
            return c
    # Fallback: assume 'date' and let DB throw a clean error if absent
    return 'date'


def _has_column(conexion, table, col):
    sql = (
        "SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME=%s AND COLUMN_NAME=%s LIMIT 1"
    )
    cur = conexion.cursor()
    cur.execute(sql, (table, col))
    exists = cur.fetchone() is not None
    cur.close()
    return exists


def _fetch_concat_recov_counts(conexion, campaign_id, sponsor, folder_date):
    """
    Return (concat_count, recov_count) from agent_audio_data for the given
    campaign_id (+ optional sponsor) on folder_date (DATE).
    Counts rows where concatenated = 1 and recovered = 1 (TINYINT).
    """
    date_col    = _choose_agent_audio_date_col(conexion)  # e.g., 'date', 'last_local_call_time', etc.
    has_sponsor = _has_column(conexion, 'agent_audio_data', 'sponsor')
    has_concat  = _has_column(conexion, 'agent_audio_data', 'concatenated')
    has_recov   = _has_column(conexion, 'agent_audio_data', 'recovered')

    if not has_concat and not has_recov:
        return 0, 0

    concat_expr = "SUM(aad.concatenated = 1)" if has_concat else "0"
    recov_expr  = "SUM(aad.recovered   = 1)" if has_recov  else "0"

    # Use DATE(...) = %s to avoid any '%s' in format strings
    date_expr = f"DATE(aad.`{date_col}`)"
    where_clauses = [
        "aad.campaign_id = %s",
        f"{date_expr} = %s",
    ]
    params = [str(campaign_id), str(folder_date)]

    if has_sponsor:
        where_clauses.append("aad.sponsor = %s")
        params.append(str(sponsor))

    sql = (
        "SELECT "
        f"  {concat_expr} AS concat_count, "
        f"  {recov_expr}  AS recov_count "
        "FROM agent_audio_data aad "
        "WHERE " + " AND ".join(where_clauses)
    )

    cur = conexion.cursor()
    cur.execute(sql, tuple(params))
    row = cur.fetchone()
    cur.close()

    concat_count = int(row[0] or 0)
    recov_count  = int(row[1] or 0)
    return concat_count, recov_count




def insertar_fila_reporte(conexion, prefix, fcounter, campaign_parameters, REJECTED, unread):
    """
    Inserta fila de reporte en la base de datos y retorna (datos_reporte, bpc_count).

    - process_warning ahora es la cantidad de filas en df['warning'] que contienen 'bpc'
      (búsqueda case-insensitive).
    """
    s3_client = generate_s3_client()
    now = time.strftime('%Y-%m-%d %H:%M:%S')

    camp_data = campaign_parameters
    s3_url    = camp_data['s3']
    country   = camp_data['country']
    sponsor   = camp_data['sponsor']
    campaign  = camp_data['campaign']
    campaign_id = camp_data['campaign_id']

    file_count = fcounter
    affected   = int(np.sum(REJECTED['reject']))
    unread     = unread

    # ---- Nuevo: contar 'MAC INCORRECTO' en df['reject_reason'] ----
    # Tolerante a NaN.
    if REJECTED is not None and 'reject_reason' in REJECTED.columns:
        rej_mac_count       = int(REJECTED['reject_reason'].astype(str).str.contains('MAC INCORRECTO', case=False, na=False).sum())
        rej_price_count     = int(REJECTED['reject_reason'].astype(str).str.contains('MAC INCORRECTO', case=False, na=False).sum())
        rej_noaudit_count   = int(REJECTED['reject_reason'].astype(str).str.contains('NO AUDITABLE', case=False, na=False).sum())
        rej_emptycall_count = int(REJECTED['reject_reason'].astype(str).str.contains('LLAMADA VACÍA', case=False, na=False).sum())
    else:
        rej_mac_count       = 0
        rej_price_count     = 0
        rej_noaudit_count   = 0
        rej_emptycall_count = 0
    if REJECTED is not None and 'warning_reason' in REJECTED.columns:
        al_bpc_count        = int(REJECTED['warning_reason'].astype(str).str.contains('BPC', case=False, na=False).sum())
        al_vel_mac_count    = int(REJECTED['warning_reason'].astype(str).str.contains('MAC AFECTADO - VOZ ACELERADA', case=False, na=False).sum())
        al_vol_mac_count    = int(REJECTED['warning_reason'].astype(str).str.contains('MAC AFECTADO - VOLUMEN BAJO', case=False, na=False).sum())
        al_vel_price_count  = int(REJECTED['warning_reason'].astype(str).str.contains('PRECIO AFECTADO - VOZ ACELERADA', case=False, na=False).sum())
        al_vol_price_count  = int(REJECTED['warning_reason'].astype(str).str.contains('PRECIO AFECTADO - VOLUMEN BAJO', case=False, na=False).sum())
        al_inex_mac_count   = int(REJECTED['warning_reason'].astype(str).str.contains('MAC INEXACTO', case=False, na=False).sum())
        al_inex_price_count = int(REJECTED['warning_reason'].astype(str).str.contains('PRECIO INEXACTO', case=False, na=False).sum())
        al_lowscore_count   = int(REJECTED['warning_reason'].astype(str).str.contains('PUNTAJE BAJO', case=False, na=False).sum())
        al_legal_na_count   = int(REJECTED['warning_reason'].astype(str).str.contains('TERMINOS LEGALES NO ENCONTRADOS', case=False, na=False).sum())
        al_low_tmo_count    = int(REJECTED['warning_reason'].astype(str).str.contains('TMO BAJO', case=False, na=False).sum())
        al_mvd_count        = int(REJECTED['warning_reason'].astype(str).str.contains('MVD', case=False, na=False).sum())
        al_hangup_count     = int(REJECTED['warning_reason'].astype(str).str.contains('HANGUP', case=False, na=False).sum())
        al_overlap_count    = int(REJECTED['warning_reason'].astype(str).str.contains('OVERLAP', case=False, na=False).sum())
        al_mute_count       = int(REJECTED['warning_reason'].astype(str).str.contains('MUTE', case=False, na=False).sum())
    else:
        al_bpc_count        = 0
        al_vel_mac_count    = 0
        al_vol_mac_count    = 0
        al_vel_price_count  = 0
        al_vol_price_count  = 0
        al_inex_mac_count   = 0
        al_inex_price_count = 0
        al_lowscore_count   = 0
        al_legal_na_count   = 0
        al_low_tmo_count    = 0
        al_mvd_count        = 0
        al_hangup_count     = 0
        al_overlap_count    = 0
        al_mute_count       = 0

    
    folder_date = get_latest_date_folder(s3_url, s3_client)

    folder_date = get_latest_date_folder(s3_url, s3_client)

    # --- NEW: get concat / recovered from DB under (campaign_id, [sponsor], folder_date) ---
    concat_count, recov_count = _fetch_concat_recov_counts(
        conexion=conexion,
        campaign_id=campaign_id,
        sponsor=sponsor,
        folder_date=folder_date
    )

    datos_reporte = (
        str(camp_data['path']),   # campaign_prefix
        str(country),             # country
        str(sponsor),             # sponsor
        str(campaign),            # campaign
        str(campaign_id),         # campaign_id
        str(folder_date),         # folder_date (DATE(...) in the query above)
        str(file_count),          # file_count
        str(affected),            # affected
        str(unread),              # unread
        str(now),                 # created_at
        str(rej_mac_count      ),
        str(rej_price_count    ),
        str(rej_noaudit_count  ),
        str(rej_emptycall_count),
        str(al_bpc_count       ),
        str(al_vel_mac_count   ),
        str(al_vol_mac_count   ),
        str(al_vel_price_count ),
        str(al_vol_price_count ),
        str(al_inex_mac_count  ),
        str(al_inex_price_count),
        str(al_lowscore_count  ),
        str(al_legal_na_count  ),
        str(al_low_tmo_count   ),
        str(al_mvd_count       ),
        str(al_hangup_count    ),
        str(al_overlap_count   ),
        str(al_mute_count      ),
        str(concat_count       ),
        str(recov_count        ),
    )

    print('inserting campaigns')
    insertar_datos_vap_report(conexion, datos_reporte)