import glob
import json
import os
from difflib import SequenceMatcher

import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from segmentationModel.fittingDeep import fitCSVConversations

from lang.VapLangUtils import normalize_text, get_kws, word_count, \
    correctCommonTranscriptionMistakes, splitConversations

from setup.MatrixSetup import remove_connectors
from utils.VapFunctions import measure_speed_classification
from utils.VapUtils import jsonDecompose, get_data_from_name, jsonDecomposeSentencesHighlight, jsonTranscriptionToCsv, getTranscriptParagraphsJsonHighlights

import numpy as np

from segmentationModel.textPostprocessing import reconstruirDialogos, process_directory_mac_price_def
from vapScoreEngine.dfUtils import calculate_confidence_scores_per_topic, df_getWordRate, generateConvDataframe
from vapScoreEngine.schema import (
    CallRecord, DIMENSION_WEIGHTS, REQUIRED_COMPLIANCE_TOPICS,
    CANONICAL_SCRIPT_ORDER, SCORE_VERSION,
)



def smooth_array(data, sigma=2):
    """ 
    Smooths the data using a Gaussian filter.
    Args:
        data (np.array): The 1D array to be smoothed.
        sigma (float): The standard deviation for the Gaussian kernel.

    Returns:
        np.array: A smoothed version of the input data.
    """
    return gaussian_filter1d(data, sigma=sigma)

def calcular_actividad_norm(json_route):
    phrases_df = pd.DataFrame(jsonDecompose(json_route)[2])
    mm = np.max(phrases_df['end'])
    phrases_df['start_n'] = phrases_df['start'] / mm
    phrases_df['end_n'] = phrases_df['end'] / mm
    phrases_df['timedelta'] = phrases_df['end_n'] - phrases_df['start_n']
    phrases_df['speed_n'] = phrases_df['num_words'] / phrases_df['timedelta']
    phrases_df['speed_n'] = smooth_array(phrases_df['speed_n'])

    return phrases_df[['end_n', 'speed_n']]



def measureGlobalActivity(directory_path):
    json_files = glob.glob(os.path.join(directory_path, '*.json'))
    results = []
    for json_file in json_files:
        #print(json_file)
        try:
            result = calcular_actividad_norm(json_file)
            results.append(result)
        except:
            print(f'Audio vacío {json_file}')
            continue
    return results


def aggregate_and_average(results, num_windows=10):
    combined_df = pd.concat(results)
    combined_df['time_window'] = pd.cut(combined_df['end_n'], bins=num_windows, labels=False)
    averaged_results = combined_df.groupby('time_window')['speed_n'].mean().reset_index()

    return averaged_results

def process_directory_and_average(directory_path, num_windows=10):
    results = measureGlobalActivity(directory_path)
    averaged_results = aggregate_and_average(results, num_windows=num_windows)
    return averaged_results


def process_directory_conversations_with_memory(mainDir,rawDir,processedDir,rebuiltDir,keywords_good,keywords_bad):
    dataframes = []
    jsonTranscriptionToCsv(mainDir,rawDir)
    splitConversations(rawDir,rawDir,14)
    fitCSVConversations(rawDir,processedDir, 14, 6, 32)
    getTranscriptParagraphsJsonHighlights(mainDir,keywords_good, keywords_bad)
    jsonDecomposeSentencesHighlight(mainDir + '/transcript_sentences',mainDir + '/transcript_sentences',keywords_good)
    # Process main directory files
    reconstruirDialogos(rawDir, processedDir,rebuiltDir)
    df_res = process_directory_mac_price_def(rebuiltDir,rebuiltDir,topics_col='topics_sequence')
    files = [filename for filename in os.listdir(rebuiltDir) if filename.endswith('.csv')]
    for filename in tqdm(files, desc=f'Processing files in {rebuiltDir} directory'):
        filepath = os.path.join(rebuiltDir, filename)
        try:
            processed_df = generateConvDataframe(filepath)
            dataframes.append(processed_df)
        except Exception as e:
            print(f"Error al procesar: {e}")
    memory_dir = os.path.join(rebuiltDir, 'memory')
    if os.path.exists(memory_dir):
        memory_files = [filename for filename in os.listdir(memory_dir) if filename.endswith('.csv')]
        for filename in tqdm(memory_files, desc="Processing files in /memory/ directory"):
            filepath = os.path.join(memory_dir, filename)
            try:
                processed_df = generateConvDataframe(filepath)
                dataframes.append(processed_df)
            except:
                print("error with memory file: " + filename)
    return dataframes


def get_all_transcripts(directorio):
    print(directorio)
    archivos_json = [archivo for archivo in os.listdir(directorio) if archivo.endswith('.json')]
    print(archivos_json)
    resultados_totales = pd.DataFrame({'id': [], 'file_name': [], 'transcript': [], 'confidence': [],
                                       'conversation': [], 'speaker_order': [], 'TMO': [], 'agent_participation': []})
    counter = 0

    for archivo_json in archivos_json:
        ruta_completa = os.path.join(directorio, archivo_json)
        try:
            # Suponiendo que jsonDecompose devuelve transcript_df y sentences_df como DataFrames
            transcript_df, _, sentences_df = jsonDecompose(ruta_completa)
            transcript_df = transcript_df.copy()
            transcript_df['id'] = counter
            transcript_df['file_name'] = archivo_json.split('_transcript')[0] + '.mp3'
            transcript_df.loc[0, 'conversation'] = ' '.join(sentences_df['text'].astype(str).tolist())
            transcript_df.loc[0, 'speaker_order'] = ','.join(sentences_df['speaker'].astype(str).tolist())
            transcript_df.loc[0, 'TMO'] = np.max(sentences_df['end']) / 60 if len(sentences_df) > 0 else 0
            transcript_df.loc[0, 'agent_participation'] = np.max([
                np.sum(sentences_df[sentences_df['speaker'] == 1]['num_words']) / np.sum(sentences_df['num_words']),
                np.sum(sentences_df[sentences_df['speaker'] == 0]['num_words']) / np.sum(sentences_df['num_words'])
            ]) if 'num_words' in sentences_df.columns and np.sum(sentences_df['num_words']) > 0 else 0

            # Concatenando resultados
            resultados_totales = pd.concat([resultados_totales, transcript_df], ignore_index=True)
            counter += 1

        except Exception as e:
            print(f"Error al procesar '{archivo_json}': {e}")
            counter += 1
            continue

    return resultados_totales


def get_all_transcripts_memory(directorio):
    archivos_excel = [archivo for archivo in os.listdir(directorio) if archivo.endswith('.xlsx')]
    resultados_totales = pd.DataFrame({'id': [],'file_name': [],'transcript': [], 'confidence': [], 'conversation': [], 'speaker_order': [], 'TMO':[], 'agent_participation':[] })
    counter=0
    for archivo_xlsx in archivos_excel:
        ruta_completa = os.path.join(directorio, archivo_xlsx)
        #print(ruta_completa)
        try:
            transcript_df = pd.read_excel(ruta_completa)  # Usando el tercer DataFrame retornado
            array_transcript_df_file=archivo_xlsx.split('_transcript')[0]+('.mp3')
            #print(transcript_df)
            #resultados = contar_palabras_prohibidas(sentences_df, speaker_id, palabras_prohibidas)
            #print(sentences_df['text'].array)
            array_transcript_df_conv=transcript_df['text'].array
            array_transcript_df_speaker_order=transcript_df['speaker'].array
            array_transcript_df_TMO=np.max(transcript_df['end'])/60
            array_transcript_df_AGENT_PART=np.max([ np.sum( transcript_df[transcript_df['speaker']==1]['num_words'])/np.sum(transcript_df['num_words']),
                                                            np.sum( transcript_df[transcript_df['speaker']==0]['num_words'])/np.sum(transcript_df['num_words'])])
            #print(array_transcript_df_id,array_transcript_df_file,array_transcript_df_conv,
            #      array_transcript_df_speaker_order,array_transcript_df_TMO,array_transcript_df_AGENT_PART)
            new_df=pd.DataFrame({'id': [counter],'file_name': [array_transcript_df_file],'transcript': [' '.join(array_transcript_df_conv)],
                                 'confidence': [0.95], 'conversation': [array_transcript_df_conv], 'speaker_order': [array_transcript_df_speaker_order],
                                 'TMO': [array_transcript_df_TMO], 'agent_participation': [array_transcript_df_AGENT_PART] })
            resultados_totales=pd.concat([resultados_totales, new_df], ignore_index=True)
            resultados_totales['archivo'] = archivo_xlsx
            counter+=1
            #print(f"Procesado: {archivo_json}")
        except Exception as e:
            print(f"Error al procesar '{archivo_xlsx}': {e}")
            counter+=1
            continue
    return resultados_totales

def calculate_general_statistics(dfs,max_times):
    """
    Calculate the statistics for each topic across multiple DataFrames.

    Parameters:
    dfs (list of pd.DataFrame): List of DataFrames containing the centroid_start_time and TMO columns.
    tmo (float): The target TMO value to which the centroid_start_time should be rescaled.

    Returns:
    pd.DataFrame: DataFrame with the statistics for each topic.
    """
    all_rescaled_dfs = [pd.DataFrame({})]

    for i in range(len(dfs)):
        dfs[i]['time_centroid_pct']=dfs[i]['time_centroid']/max_times[i]
        all_rescaled_dfs.append(dfs[i])

    # Combine all rescaled DataFrames
    combined_df = pd.concat( all_rescaled_dfs , ignore_index=True)
    print(combined_df)
    # Calculate statistics for each topic
    #print(str(combined_df))
    statistics = combined_df.groupby('final_label').agg(
        mean_centroid=('time_centroid_pct', 'mean'),
        std_centroid =('time_centroid_pct', 'std'),
        min_centroid =('time_centroid_pct', 'min'),
        max_centroid=('time_centroid_pct', 'max'),
        mean_max_conf=('topic_max_conf', 'mean'),
        mean_conf=('topic_mean_conf', 'mean'),
        mean_words_p_m=('topic_words_p_m', 'mean'),
        mean_num_words=('topic_num_words', 'mean'),
        topic_frequency=('topic_occurrence', 'sum'),
    ).reset_index()

    return statistics



def list_y_n_words(df):
    df['permitida'] = df['cluster'].apply(
        lambda x: 0 if (x == 'no permitida' or x == 'nopermitida' or x=='no permitida' or x=='NP') else 1)
    #print(df)
    if 'KEYWORDS INFALTABLES' in df.columns:
        df['kws_array'] = df['KEYWORDS INFALTABLES'].apply(get_kws)
        df['kws_na'] = df.dropna(subset=['KEYWORDS NO PERMITIDAS'])[
            'KEYWORDS NO PERMITIDAS'].apply(get_kws)
        all_must_keywords_series  = df['kws_array'].explode()
        all_musnt_keywords_series = df['kws_na'].explode()
        all_must_keywords = list(set(all_must_keywords_series.dropna()))
        all_musnt_keywords = list(set(all_musnt_keywords_series.dropna()))
    else:
        all_must_keywords = df[df['permitida'] == 1]['name'].explode()
        all_musnt_keywords = df[df['permitida'] == 0]['name'].explode()
        all_must_keywords = list(set(all_must_keywords.dropna()))
        all_musnt_keywords = list(set(all_musnt_keywords.dropna()))
    df.to_excel('MATRIZ_CALIFICACION_HIPOT.xlsx')
    return all_must_keywords,all_musnt_keywords

def filter_dataframe_by_directory(dataframe, directory, file_column='file_name'):
    """
    Filters a DataFrame to include only rows where the filenames in the 'file_column'
    exist in the specified directory.

    Args:
    dataframe (pd.DataFrame): The DataFrame to filter.
    directory (str): The directory to check the files against.
    file_column (str): The column name in the DataFrame that contains the filenames. Default is 'file_name'.

    Returns:
    pd.DataFrame: A filtered DataFrame with rows where the 'file_column' value matches a file in the directory.
    """
    # Get the list of files in the directory
    directory_files = set(os.listdir(directory))
    #print(directory_files)
    # Filter the dataframe based on whether the filename in 'file_column' exists in the directory
    print(dataframe[file_column])
    filtered_df = dataframe[dataframe[file_column].apply(lambda x: x in directory_files)]

    return filtered_df




def calificarLikelyhoodConMatriz(df, input_keywords, stride=1):
    """
    Dado un DataFrame `df` que contiene en `text_col` una cadena de texto, esta función:
    1. Calcula el tamaño de ventana basándose en el promedio de longitud de
       los elementos en `input_keywords` filtrados por `cluster_label`.
    2. Recorre cada fila de `df` y construye ventanas de texto de ese tamaño en `text_col`.
    3. Compara cada ventana con todos los keywords MAC (usando similitud de difflib.SequenceMatcher)
       y se queda con la ventana que arroje la mayor similitud.
    4. Retorna el DataFrame original con dos columnas extra:
       - 'best_mac_window': la ventana de texto con mayor similitud.
       - 'best_mac_likelihood': la similitud (0 a 1) de dicha ventana.

    Params
    ------
    df : pd.DataFrame
        DataFrame principal que tiene la columna de texto.
    text_col : str
        Nombre de la columna que contiene el texto donde se buscarán las ventanas.
    input_keywords : pd.DataFrame
        DataFrame que contiene, al menos, las columnas 'cluster' y 'name'.
    cluster_label : str, opcional
        Nombre de la etiqueta usada en `input_keywords['cluster']` para filtrar. Por defecto 'MAC'.

    Returns
    -------
    pd.DataFrame
        Mismo DataFrame de entrada con dos columnas adicionales:
        'best_mac_window' y 'best_mac_likelihood'.
    """

    # 1) Filtrar los keywords para el cluster "
    ideal_keywords = input_keywords
    if not ideal_keywords:
        # En caso de que no haya keywords, retornamos el df sin cambios.
        df['best_mac_window'] = None
        df['best_mac_likelihood'] = 0.0
        return df

    # 2) Calcular el promedio de longitud de estos keywords
    avg_len = 6
    if avg_len < 1:
        avg_len = 1
    # Función de similitud basada en difflib
    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    best_windows = []
    best_scores = []

    for _, row in df.iterrows():
        text_value = row['text_string']
        text_value = correctCommonTranscriptionMistakes(text_value.lower())
        # Manejar casos donde no haya texto
        if not isinstance(text_value, str) or len(text_value.strip()) == 0:
            best_windows.append(None)
            best_scores.append(0.0)
            continue

        # Separar en palabras
        words = text_value.split()
        # Calcular el tamaño de la ventana en # de palabras
        window_size = min(avg_len, len(words))
        max_score = 0.0
        best_window_candidate = None

        # Generar ventanas avanzando de a 'stride' grupos de palabras
        for start in range(0, len(words) - window_size + 1, stride):
            window_words = words[start : start + window_size]
            window_text = " ".join(window_words)

            # Comparar contra cada keyword y tomar la similitud más alta
            local_max_score = max(similarity(str(window_text), str(kw)) for kw in input_keywords)
            if local_max_score > max_score:
                max_score = local_max_score
                best_window_candidate = window_text

        best_windows.append(best_window_candidate)
        best_scores.append(max_score)

    df['best_mac_window'] = best_windows
    df['best_mac_likelihood'] = best_scores

    return df


def _compute_mac_flow(row: pd.Series) -> dict:
    """
    Determina el resultado del flujo de Pregunta de Activación (MAC) para una
    llamada a partir de sus columnas agregadas.

    Reglas:
      - Si mac_times_said == 0                         → NO_MAC
      - Si best_mac_likelihood >= 0.88 y no hay mac_r  → VENTA_CONFIRMADA
      - Si best_mac_likelihood in [0.80, 0.88) o hay
        mac_r y mac_r_detected                         → VENTA_REFUERZO
      - En cualquier otro caso                         → NO_CONFIRMADA

    Retorna un dict con mac_flow_outcome, mac_r_triggered, mac_r_effective.
    """
    mac_count  = int(row.get('mac_times_said', 0) or 0)
    mac_like   = float(row.get('best_mac_likelihood', 0.0) or 0.0)
    mac_r_det  = bool(row.get('mac_r_detected', False) or row.get('mac_r', 0))

    if mac_count == 0:
        return {"mac_flow_outcome": "NO_MAC", "mac_r_triggered": False, "mac_r_effective": False}

    if mac_like >= 0.88 and not mac_r_det:
        return {"mac_flow_outcome": "VENTA_CONFIRMADA", "mac_r_triggered": False, "mac_r_effective": False}

    if mac_r_det:
        return {"mac_flow_outcome": "VENTA_REFUERZO", "mac_r_triggered": True, "mac_r_effective": True}

    if mac_like >= 0.80:
        # Respuesta ambigua y no se aplicó refuerzo → necesitaba refuerzo pero no se hizo
        return {"mac_flow_outcome": "NO_CONFIRMADA", "mac_r_triggered": True, "mac_r_effective": False}

    return {"mac_flow_outcome": "NO_CONFIRMADA", "mac_r_triggered": False, "mac_r_effective": False}


def _score_d1_mac_venta(row: pd.Series) -> float:
    """
    D1 — Pregunta de Activación + confirmación de venta (0–10).

    Componentes:
      - Presencia del MAC              (2.0 pts)
      - Calidad del fragmento MAC      (3.0 pts × best_mac_likelihood)
      - Resultado del flujo MAC        (3.0 pts: CONFIRMADA=3, REFUERZO=2, NO_CONF=0.5, NO_MAC=0)
      - Precio detectado y likelihood  (2.0 pts × best_price_likelihood)
    """
    mac_count  = int(row.get('mac_times_said', 0) or 0)
    mac_like   = float(row.get('best_mac_likelihood', 0.0) or 0.0)
    price_like = float(row.get('best_price_likelihood', 0.0) or 0.0)
    outcome    = str(row.get('mac_flow_outcome', 'NO_MAC'))

    score = 0.0
    if mac_count > 0:
        score += 2.0

    score += 3.0 * mac_like

    outcome_pts = {"VENTA_CONFIRMADA": 3.0, "VENTA_REFUERZO": 2.0,
                   "NO_CONFIRMADA": 0.5, "NO_MAC": 0.0}
    score += outcome_pts.get(outcome, 0.0)

    score += 2.0 * min(price_like, 1.0)

    return round(min(score, 10.0), 4)


def _score_d2_compliance(row: pd.Series) -> float:
    """
    D2 — Completitud de momentos legales y regulatorios (0–10).

    Cada momento de REQUIRED_COMPLIANCE_TOPICS vale igual.
    Bonus de 1 punto si el agente también mencionó confirmación de monitoreo.
    """
    flag_cols = {
        "TERMINOS LEGALES":       "terms_detected",
        "TRATAMIENTO DATOS":      "tratamiento_datos_detected",
        "LEY RETRACTO":           "ley_retracto_detected",
        "CONFIRMACION MONITOREO": "confirmacion_monitoreo_detected",
        "PRECIO":                 "precio_detected",
        "CONFIRMACION DATOS":     "confirmacion_datos_detected",
    }
    n_required = len(flag_cols)
    count = sum(1 for col in flag_cols.values() if bool(row.get(col, False)))
    base  = (count / n_required) * 9.0

    # Bonus: confirmación de monitoreo (ya incluida arriba, pero si excede la
    # base le damos un punto extra de calidad total hasta 10)
    bonus = 1.0 if bool(row.get('confirmacion_monitoreo_detected', False)) else 0.0
    return round(min(base + bonus, 10.0), 4)


def _score_d3_script(row: pd.Series) -> float:
    """
    D3 — Adherencia al guión y estructura conversacional (0–10).

    - script_completeness (fracción de tópicos cubiertos) → 6 pts
    - topic_order_score   (Kendall-tau vs orden canónico)  → 4 pts
    """
    completeness = float(row.get('script_completeness', 0.0) or 0.0)
    order_score  = float(row.get('topic_order_score',   0.0) or 0.0)
    return round(min(completeness * 6.0 + order_score * 4.0, 10.0), 4)


def _score_d4_engagement(row: pd.Series) -> float:
    """
    D4 — Calidad comunicativa del agente (0–10).

    - Participación óptima del agente: 55–70 % → 4 pts, se penaliza si > 0.85
    - WPM del agente: óptimo 120–160 → 3 pts
    - Turn count (interacción): más turnos = mejor → 2 pts (cap en 20 turnos)
    - Preguntas del cliente detectadas → 1 pt (cap en 5)
    """
    participation = float(row.get('agent_participation', 0.0) or 0.0)
    wpm           = float(row.get('agent_wpm', 140.0) or 140.0)
    turns         = int(row.get('turn_count', 0) or 0)
    client_q      = int(row.get('client_question_count', 0) or 0)

    # Participación: óptimo 0.55–0.70; penaliza si > 0.85 (agente habla demasiado)
    if 0.55 <= participation <= 0.70:
        part_pts = 4.0
    elif participation > 0.85:
        part_pts = max(0.0, 4.0 - (participation - 0.85) * 20)
    else:
        part_pts = 4.0 * (participation / 0.55)

    # WPM: óptimo 120–160; penaliza por velocidad excesiva (> 180) o muy lenta (< 80)
    if 120 <= wpm <= 160:
        wpm_pts = 3.0
    elif wpm > 160:
        wpm_pts = max(0.0, 3.0 - (wpm - 160) / 40)
    else:
        wpm_pts = max(0.0, 3.0 * (wpm / 120))

    turn_pts     = min(turns / 20, 1.0) * 2.0
    client_q_pts = min(client_q / 5, 1.0) * 1.0

    return round(min(part_pts + wpm_pts + turn_pts + client_q_pts, 10.0), 4)


def _score_d5_audio(row: pd.Series) -> float:
    """
    D5 — Calidad técnica del audio / transcripción ASR (0–10).

    Fuentes (todas pre-computadas por audioPrepDeep + audioOutputWpm):
      - confidence_score         → 5 pts  (confianza media ASR, 0–1)
      - silence_ratio            → 3 pts  (fracción de ventanas bajo umbral de silencio)
      - overlap_ratio_in_speech  → 2 pts  (fracción de habla con solapamiento de voces)
    """
    conf    = float(row.get('confidence_score', 0.0) or 0.0)
    silence = float(row.get('silence_ratio', 0.0) or 0.0)
    overlap = float(row.get('overlap_ratio_in_speech', 0.0) or 0.0)

    silence_pts = max(0.0, 3.0 * (1.0 - silence))
    overlap_pts = max(0.0, 2.0 * (1.0 - overlap))
    return round(min(conf * 5.0 + silence_pts + overlap_pts, 10.0), 4)


# Ventanas de 1 s cuyo volumen (dBFS) está por debajo de este umbral se
# consideran silencio. −50 dBFS ≈ señal de ruido de fondo sin voz activa.
_SILENCE_DB_THRESHOLD = -50.0


def _build_windows_index(df_windows: pd.DataFrame) -> dict:
    """
    Construye un índice {file_key: (times_array, wpm_array, vols_array)}
    desde df_windows para lookups O(log n) por timestamp.

    file_key = nombre de archivo sin extensión (match con MAT file_name).
    times_array = np.ndarray de enteros (fin de cada ventana de 1 s, en segundos).
    wpm_array   = np.ndarray de WPM por ventana (alineado con times_array).
    vols_array  = np.ndarray de dBFS por ventana (alineado con times_array).
    """
    import bisect

    def _key(fname: str) -> str:
        return os.path.splitext(os.path.basename(str(fname)))[0]

    index: dict = {}
    for _, row in df_windows.iterrows():
        t5  = row.get('times_5s') or row.get('times')
        w5  = row.get('wpm_5s')
        v5  = row.get('vols_5s') or row.get('vols')
        if t5 is None or w5 is None:
            continue
        t_arr = np.asarray(t5, dtype=float)
        w_arr = np.asarray(w5, dtype=float)
        v_arr = np.asarray(v5, dtype=float) if v5 is not None else np.full_like(t_arr, np.nan)
        if len(t_arr) == 0:
            continue
        index[_key(row['file_name'])] = (t_arr, w_arr, v_arr)

    return index


def _vol_at_time(index: dict, file_name: str, time_sec) -> float:
    """
    Devuelve el volumen (dBFS) de la ventana que contiene `time_sec`.
    Reutiliza el índice construido por _build_windows_index.
    Retorna NaN si el archivo no está en el índice o el tiempo es inválido.
    """
    if time_sec is None or (isinstance(time_sec, float) and np.isnan(time_sec)):
        return np.nan
    key = os.path.splitext(os.path.basename(str(file_name)))[0]
    if key not in index:
        return np.nan
    t_arr, _, v_arr = index[key]
    idx = int(np.searchsorted(t_arr, float(time_sec), side='left'))
    if idx >= len(v_arr):
        idx = len(v_arr) - 1
    val = float(v_arr[idx])
    return val if np.isfinite(val) else np.nan


def measure_volume_classification(db_value: float) -> str:
    """
    Clasifica el volumen en dBFS en tres categorías.

    Umbrales calibrados para audio de voz procesado a 8 kHz / −23 LUFS:
      high  : ≥ −20 dBFS  (voz alta o muy próxima al micrófono)
      mid   : −35 a −20   (voz normal, rango conversacional óptimo)
      low   : < −35 dBFS  (voz baja, posible problema de captación)
    """
    if db_value is None or (isinstance(db_value, float) and np.isnan(db_value)):
        return 'unknown'
    if db_value >= -20.0:
        return 'high'
    if db_value >= -35.0:
        return 'mid'
    return 'low'


def _wpm_at_time(index: dict, file_name: str, time_sec) -> float:
    """
    Devuelve el WPM de la ventana que contiene `time_sec` para `file_name`.
    Usa búsqueda binaria sobre el array de tiempos de fin de ventana.
    Retorna NaN si el archivo no está en el índice o el tiempo es inválido.
    """
    import bisect

    if time_sec is None or (isinstance(time_sec, float) and np.isnan(time_sec)):
        return np.nan
    key = os.path.splitext(os.path.basename(str(file_name)))[0]
    if key not in index:
        return np.nan
    t_arr, w_arr, _ = index[key]
    # times_5s son los fines de ventana → ventana i cubre (t[i-1], t[i]]
    # bisect_left da el primer índice donde t_arr[i] >= time_sec
    idx = int(np.searchsorted(t_arr, float(time_sec), side='left'))
    if idx >= len(w_arr):
        idx = len(w_arr) - 1
    return float(w_arr[idx])


def _enrich_with_topic_velocity(mat: pd.DataFrame,
                                 df_windows: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega a MAT la velocidad de habla (WPM) y su clasificación categórica
    en los momentos exactos donde se dijo el MAC y el PRECIO, usando las
    ventanas de tiempo pre-computadas por audioOutputWpm.

    Requiere en MAT las columnas:
      - file_name
      - time_centroid_macs   (tiempo central en segundos del mejor fragmento MAC)
      - time_centroid_prices (tiempo central en segundos del mejor fragmento PRECIO)

    Columnas que se agregan
    -----------------------
    wpm_at_mac                 : WPM en la ventana del instante MAC
    wpm_at_price               : WPM en la ventana del instante PRECIO
    velocity_classification_macs   : categoría de velocidad MAC  ("low"/"mid"/"high")
    velocity_classification_prices : categoría de velocidad PRECIO
    volume_db_mac              : volumen (dBFS) en la ventana del instante MAC
    volume_db_price            : volumen (dBFS) en la ventana del instante PRECIO
    volume_classification_mac  : categoría de volumen MAC  ("low"/"mid"/"high"/"unknown")
    volume_classification_price: categoría de volumen PRECIO
    """
    new_cols = (
        'wpm_at_mac', 'wpm_at_price',
        'velocity_classification_macs', 'velocity_classification_prices',
        'volume_db_mac', 'volume_db_price',
        'volume_classification_mac', 'volume_classification_price',
    )

    if df_windows is None or df_windows.empty:
        for col in new_cols:
            if col not in mat.columns:
                mat[col] = np.nan
        return mat

    index = _build_windows_index(df_windows)

    # ── WPM en el instante de MAC / PRECIO ───────────────────────────────────
    mat['wpm_at_mac'] = mat.apply(
        lambda r: _wpm_at_time(index, r['file_name'], r.get('time_centroid_macs')), axis=1
    )
    mat['wpm_at_price'] = mat.apply(
        lambda r: _wpm_at_time(index, r['file_name'], r.get('time_centroid_prices')), axis=1
    )
    mat['velocity_classification_macs'] = mat['wpm_at_mac'].apply(
        lambda x: measure_speed_classification(x) if pd.notna(x) else None
    )
    mat['velocity_classification_prices'] = mat['wpm_at_price'].apply(
        lambda x: measure_speed_classification(x) if pd.notna(x) else None
    )

    # ── Volumen (dBFS) en el instante de MAC / PRECIO ────────────────────────
    mat['volume_db_mac'] = mat.apply(
        lambda r: _vol_at_time(index, r['file_name'], r.get('time_centroid_macs')), axis=1
    )
    mat['volume_db_price'] = mat.apply(
        lambda r: _vol_at_time(index, r['file_name'], r.get('time_centroid_prices')), axis=1
    )
    mat['volume_classification_mac'] = mat['volume_db_mac'].apply(measure_volume_classification)
    mat['volume_classification_price'] = mat['volume_db_price'].apply(measure_volume_classification)

    return mat


def _enrich_with_audio_windows(mat: pd.DataFrame,
                                df_windows: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega a MAT estadísticas de audio derivadas de las ventanas de tiempo
    pre-computadas por audioPrepDeep.main_process_batch + audioOutputWpm.

    Parámetros
    ----------
    mat        : DataFrame principal de llamadas (una fila por llamada).
    df_windows : DataFrame producido por audioOutputWpm(), con columnas:
                   file_name, vols (np.ndarray de dBFS por ventana de 1 s),
                   times (np.ndarray de t_starts), overlap_ratio_in_speech.

    Columnas que se agregan / actualizan en MAT
    -------------------------------------------
    silence_ratio          : fracción de ventanas de 1 s bajo _SILENCE_DB_THRESHOLD
    overlap_ratio_in_speech: fracción de habla con solapamiento de voces
    vol_mean               : volumen medio (dBFS) de la llamada
    vol_std                : desviación estándar del volumen (dBFS)
    """
    audio_cols = ('silence_ratio', 'overlap_ratio_in_speech', 'vol_mean', 'vol_std')

    if df_windows is None or df_windows.empty:
        for col in audio_cols:
            if col not in mat.columns:
                mat[col] = np.nan
        return mat

    def _key(fname: str) -> str:
        """Normaliza nombre de archivo quitando extensión, para hacer el join."""
        return os.path.splitext(os.path.basename(str(fname)))[0]

    def _stats_from_row(row) -> dict:
        vols = row.get('vols')
        if vols is None or (hasattr(vols, '__len__') and len(vols) == 0):
            return {'silence_ratio': np.nan, 'vol_mean': np.nan, 'vol_std': np.nan}
        arr = np.asarray(vols, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return {'silence_ratio': np.nan, 'vol_mean': np.nan, 'vol_std': np.nan}
        return {
            'silence_ratio': float(np.mean(arr < _SILENCE_DB_THRESHOLD)),
            'vol_mean':      float(np.mean(arr)),
            'vol_std':       float(np.std(arr)),
        }

    stats = df_windows.apply(_stats_from_row, axis=1, result_type='expand')
    stats['file_key'] = df_windows['file_name'].apply(_key)

    if 'overlap_ratio_in_speech' in df_windows.columns:
        stats['overlap_ratio_in_speech'] = df_windows['overlap_ratio_in_speech'].values
    else:
        stats['overlap_ratio_in_speech'] = np.nan

    stats_map = stats.set_index('file_key')

    def _lookup(fname: str, col: str):
        key = _key(fname)
        if key in stats_map.index:
            val = stats_map.at[key, col]
            return float(val) if (val is not None and not (isinstance(val, float) and np.isnan(val))) else np.nan
        return np.nan

    for col in audio_cols:
        mat[col] = mat['file_name'].apply(lambda f, c=col: _lookup(f, c))

    return mat


def _enrich_with_compliance_flags(mat: pd.DataFrame,
                                   values_per_topic: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega a MAT los flags binarios de presencia por cada tópico SUBTAG,
    la completitud de compliance y los indicadores del flujo MAC.
    """
    topic_flags = {
        'saludo_detected':                  ['SALUDO'],
        'perfilamiento_detected':           ['PERFILAMIENTO'],
        'producto_detected':                ['PRODUCTO', 'OFERTA COMERCIAL'],
        'conformidad_detected':             ['CONFORMIDAD'],
        'confirmacion_monitoreo_detected':  ['CONFIRMACION MONITOREO'],
        'tratamiento_datos_detected':       ['TRATAMIENTO DATOS'],
        'ley_retracto_detected':            ['LEY RETRACTO'],
        'mac_detected':                     ['MAC', 'MAC_DEF'],
        'mac_r_detected':                   ['MAC REFUERZO'],
        'precio_detected':                  ['PRECIO', 'PRECIO_DEF'],
        'confirmacion_datos_detected':      ['CONFIRMACION DATOS'],
        'conformidad_atencion_detected':    ['ATENCION'],
        'despedida_detected':               ['DESPEDIDA'],
    }

    for flag_col, labels in topic_flags.items():
        present_files = values_per_topic[
            values_per_topic['final_label'].isin(labels)
        ]['file_name'].unique()
        mat[flag_col] = mat['file_name'].isin(present_files)

    # Aliases de compatibilidad legacy
    mat['terms_detected']  = mat['terms_detected'] if 'terms_detected' in mat.columns else mat['tratamiento_datos_detected']
    mat['mvd_detected']    = mat['confirmacion_datos_detected']

    # Compliance completeness
    req_flags = [
        'terms_detected', 'tratamiento_datos_detected', 'ley_retracto_detected',
        'confirmacion_monitoreo_detected', 'precio_detected', 'confirmacion_datos_detected',
    ]
    mat['compliance_moment_count'] = mat[req_flags].sum(axis=1)
    mat['compliance_completeness'] = mat['compliance_moment_count'] / len(REQUIRED_COMPLIANCE_TOPICS)

    return mat


def score_camp(campaign_directory, campaign_id, TMO, topics_combined_df,
               df_windows: pd.DataFrame = None):
    """
    Pipeline principal de calificación de llamadas — modelo MDCL v3.

    Parámetros
    ----------
    campaign_directory : str
        Directorio raíz de la campaña (con trailing slash).
    campaign_id : str
        Identificador de la campaña (se usa para construir subdirectorios).
    TMO : float
        Tiempo medio objetivo de la llamada en minutos (referencia de campaña).
    topics_combined_df : pd.DataFrame
        DataFrame de la matriz de calificación con columnas 'name' y 'cluster'.
    df_windows : pd.DataFrame, opcional
        Salida de audioOutputWpm() con columnas file_name, vols, times,
        overlap_ratio_in_speech, etc. Se usa para calcular silence_ratio y
        overlap en D5. Si es None, D5 sólo usa confidence_score de ASR.

    Retorna
    -------
    tuple(MAT_CALLS, statistics, MAT_complete_topics, NOPERM,
          topics_stats_convs_scores, values_per_topic_for_all_convs)
    """
    keywords_df = topics_combined_df
    PERM, NOPERM = list_y_n_words(keywords_df)

    print("Procesando conversaciones y generando caché...")
    routeRawCsvTranscripts = campaign_directory + campaign_id.replace('/', '') + '_RAW'
    routeRawCsvGraded      = campaign_directory + campaign_id.replace('/', '') + '_PROCESSED'
    routeRawCsvRebuilt     = campaign_directory + campaign_id.replace('/', '') + '_RECONS'

    scores = process_directory_conversations_with_memory(
        campaign_directory, routeRawCsvTranscripts,
        routeRawCsvGraded, routeRawCsvRebuilt, PERM, NOPERM,
    )
    print(f"TOTAL DE LLAMADAS A CALIFICAR: {len(scores)}")

    statistics = calculate_general_statistics(
        [scores[i][1] for i in range(len(scores))],
        [scores[i][2] for i in range(len(scores))],
    )
    statistics.to_excel(campaign_directory + 'misc/statistics_pre.xlsx')

    MAT_a = get_all_transcripts(campaign_directory)
    MAT_M = get_all_transcripts_memory(routeRawCsvRebuilt + '/memory/')
    MAT   = pd.concat([MAT_a, MAT_M], axis=0, ignore_index=True)
    MAT.to_excel(campaign_directory + "misc/MAT.xlsx")

    # ── Extraer keywords MAC y PRECIO de la matriz ───────────────────────────
    ALLOWED_MACS   = keywords_df[keywords_df['cluster'] == 'MAC']['name'].tolist()
    ALLOWED_PRICES = keywords_df[keywords_df['cluster'] == 'PRECIO']['name'].tolist()
    print(f"MACs permitidos: {ALLOWED_MACS}")
    print(f"Precios permitidos: {ALLOWED_PRICES}")

    MAT = df_getWordRate(MAT, keywords_df)

    # ── Metadatos del nombre de archivo ──────────────────────────────────────
    try:
        MAT[['DATE_TIME', 'LEAD_ID', 'EPOCH', 'AGENT_ID', 'CLIENT_ID']] = \
            MAT['file_name'].apply(lambda x: pd.Series(get_data_from_name(x)))
    except Exception:
        MAT[['DATE_TIME', 'LEAD_ID', 'AGENT_ID', 'CLIENT_ID']] = \
            MAT['file_name'].apply(lambda x: pd.Series(get_data_from_name(x)))

    MAT.to_excel(campaign_directory + "misc/MAT_BEFORE_MACS.xlsx")

    # ── Construir DataFrame de tópicos ───────────────────────────────────────
    values_per_topic_for_all_convs = pd.concat(
        [scores[i][1] for i in range(len(scores))], ignore_index=True,
    )
    topics_transcripts_convers = pd.concat(
        [scores[i][0] for i in range(len(scores))], ignore_index=False,
    )
    topics_transcripts_convers.to_excel(campaign_directory + "misc/topics_transcripts_convers.xlsx")

    values_per_topic_for_all_convs['velocity_classification'] = \
        values_per_topic_for_all_convs['topic_words_p_m'].apply(measure_speed_classification)
    values_per_topic_for_all_convs.to_excel(campaign_directory + "misc/values_per_topic_for_all_convs.xlsx")

    for conf_col in ('mean_transcript_confidence', 'mean_mac_transcript_confidence',
                     'mean_price_transcript_confidence'):
        if conf_col in values_per_topic_for_all_convs.columns:
            values_per_topic_for_all_convs[conf_col] = \
                values_per_topic_for_all_convs[conf_col].fillna(0)

    values_per_topic_for_all_convs['noauditable_transcript'] = \
        (values_per_topic_for_all_convs['mean_transcript_confidence'] <= 0.9).astype(int)

    # ── Filtrar fragmentos MAC y PRECIO ──────────────────────────────────────
    AllCallMacs   = values_per_topic_for_all_convs[
        values_per_topic_for_all_convs['final_label'].isin(['MAC_DEF', 'MAC'])
    ].copy()
    AllCallPrices = values_per_topic_for_all_convs[
        values_per_topic_for_all_convs['final_label'].isin(['PRECIO_DEF', 'PRECIO'])
    ].copy()
    AllCallMVDs     = values_per_topic_for_all_convs[values_per_topic_for_all_convs['mvd'] > 0]
    AllCallTERMS    = values_per_topic_for_all_convs[values_per_topic_for_all_convs['terms'] > 0]
    AllCallMacR     = values_per_topic_for_all_convs[values_per_topic_for_all_convs['mac_r'] > 0]
    AllCallIgsComps = values_per_topic_for_all_convs[values_per_topic_for_all_convs['igs_comp'] > 0]

    for label, df_sub in [('TRUE_MVDs', AllCallMVDs), ('TRUE_TERMS', AllCallTERMS),
                           ('TRUE_MAC_R', AllCallMacR), ('TRUE_IGS_COMPS', AllCallIgsComps),
                           ('TRUE_MACS', AllCallMacs), ('TRUE_PRICES', AllCallPrices)]:
        df_sub.to_excel(campaign_directory + f"misc/{label}.xlsx")

    print(f"MACs detectados: {len(AllCallMacs)} | Precios detectados: {len(AllCallPrices)}")

    # ── Likelihood de MAC y PRECIO ────────────────────────────────────────────
    AllCallMacs   = calificarLikelyhoodConMatriz(AllCallMacs,   ALLOWED_MACS,   1)
    AllCallPrices = calificarLikelyhoodConMatriz(AllCallPrices, ALLOWED_PRICES, 1)
    AllCallMacs.to_excel(campaign_directory + "misc/TRUE_MACS_graded.xlsx")
    AllCallPrices.to_excel(campaign_directory + "misc/TRUE_PRICES_graded.xlsx")

    # ── Confianza media por llamada ───────────────────────────────────────────
    topics_stats_convs_scores = (
        values_per_topic_for_all_convs
        .groupby('file_name', as_index=False)
        .agg(topic_mean_conf=('topic_mean_conf', 'mean'))
    )
    topics_stats_convs_scores.to_excel(campaign_directory + "misc/topics_stats_convs_scores.xlsx")

    conf_map = topics_stats_convs_scores.set_index('file_name')['topic_mean_conf']

    # ── Seleccionar el mejor fragmento MAC y PRECIO por llamada ──────────────
    AllCallMacs['times_said']   = AllCallMacs.groupby('file_name')['file_name'].transform('count')
    AllCallPrices['times_said'] = AllCallPrices.groupby('file_name')['file_name'].transform('count')

    AllCallMacsMarked   = AllCallMacs.loc[
        AllCallMacs.groupby('file_name')['best_mac_likelihood'].idxmax()
    ].copy()
    AllCallPricesMarked = AllCallPrices.loc[
        AllCallPrices.groupby('file_name')['best_mac_likelihood'].idxmax()
    ].copy()

    TOLERANCE = 0.85
    AllCallMacsMarked['mac_warn']   = AllCallMacsMarked['best_mac_likelihood'] < TOLERANCE
    AllCallPricesMarked['price_warn'] = AllCallPricesMarked['best_mac_likelihood'] < TOLERANCE

    # ── Merge de todos los datos en MAT ──────────────────────────────────────
    # time_centroid se incluye para el lookup de velocidad por ventana de audio.
    mac_merge_cols   = ['file_name', 'best_mac_likelihood', 'best_mac_window',
                        'times_said', 'mac_warn', 'time_centroid']
    price_merge_cols = ['file_name', 'best_mac_likelihood', 'times_said',
                        'price_warn', 'time_centroid']

    # Filtrar solo las columnas que realmente existen en cada DataFrame
    mac_merge_cols   = [c for c in mac_merge_cols   if c in AllCallMacsMarked.columns]
    price_merge_cols = [c for c in price_merge_cols if c in AllCallPricesMarked.columns]

    MAT_w = pd.merge(MAT, AllCallMacsMarked[mac_merge_cols],
                     on='file_name', how='left')
    MAT_w = pd.merge(MAT_w, AllCallPricesMarked[price_merge_cols],
                     on='file_name', how='left', suffixes=('_macs', '_prices'))
    MAT_w = pd.merge(MAT_w, AllCallMVDs[['file_name', 'mvd']],   on='file_name', how='left')
    MAT_w = pd.merge(MAT_w, AllCallTERMS[['file_name', 'terms']], on='file_name', how='left')
    MAT_w = pd.merge(MAT_w, AllCallMacR[['file_name', 'mac_r']],  on='file_name', how='left')
    MAT_w = pd.merge(MAT_w, AllCallIgsComps[['file_name', 'igs_comp']], on='file_name', how='left')
    MAT_w = MAT_w.drop_duplicates(subset=['file_name'], keep='first')
    MAT_w.to_excel(campaign_directory + "misc/MAT_WITH_EVERYTHING.xlsx")

    # ── Normalizar columnas de likelihood ────────────────────────────────────
    MAT_w[['mvd', 'terms', 'mac_r', 'igs_comp']] = \
        MAT_w[['mvd', 'terms', 'mac_r', 'igs_comp']].fillna(0)

    mac_like_col   = 'best_mac_likelihood_macs'
    price_like_col = 'best_mac_likelihood_prices'

    mac_max   = MAT_w[mac_like_col].max()
    price_max = MAT_w[price_like_col].max()
    MAT_w['best_mac_likelihood']   = MAT_w[mac_like_col].fillna(0.0) / (mac_max   if mac_max   > 0 else 1.0)
    MAT_w['best_price_likelihood'] = MAT_w[price_like_col].fillna(0.0) / (price_max if price_max > 0 else 1.0)
    MAT_w['mac_times_said']   = MAT_w.get('times_said_macs',   MAT_w.get('times_said', 0)).fillna(0)
    MAT_w['price_times_said'] = MAT_w.get('times_said_prices', 0)

    # ── Estadísticas de audio desde ventanas pre-computadas ──────────────────
    # Reemplaza measureDbAplitude (lectura de archivos de audio en disco).
    # df_windows proviene de audioOutputWpm() en main.py y ya contiene
    # silence_ratio, overlap_ratio_in_speech y vol_mean derivados de las
    # ventanas de 1 s calculadas por audioPrepDeep.main_process_batch.
    MAT_volumes = _enrich_with_audio_windows(MAT_w, df_windows)

    # ── Velocidad de habla en los momentos de MAC y PRECIO ───────────────────
    # time_centroid_macs / time_centroid_prices provienen del merge anterior.
    # La búsqueda en df_windows devuelve el WPM de la ventana de 1 s que
    # contiene ese instante; measure_speed_classification lo categoriza.
    MAT_volumes = _enrich_with_topic_velocity(MAT_volumes, df_windows)

    # ── Enriquecer con flags de compliance por tópico ────────────────────────
    MAT_volumes = _enrich_with_compliance_flags(MAT_volumes, values_per_topic_for_all_convs)

    # ── Determinar flujo MAC ─────────────────────────────────────────────────
    mac_flow_cols = MAT_volumes.apply(_compute_mac_flow, axis=1, result_type='expand')
    MAT_volumes = pd.concat([MAT_volumes, mac_flow_cols], axis=1)

    # ── Aplicar confianza media de transcripción ──────────────────────────────
    MAT_volumes['confidence_score'] = MAT_volumes['file_name'].map(conf_map).fillna(0.0)

    # ── Calcular las cinco dimensiones ───────────────────────────────────────
    MAT_volumes['score_d1_mac_venta']  = MAT_volumes.apply(_score_d1_mac_venta,  axis=1)
    MAT_volumes['score_d2_compliance'] = MAT_volumes.apply(_score_d2_compliance, axis=1)
    MAT_volumes['score_d3_script']     = MAT_volumes.apply(_score_d3_script,     axis=1)
    MAT_volumes['score_d4_engagement'] = MAT_volumes.apply(_score_d4_engagement, axis=1)
    MAT_volumes['score_d5_audio']      = MAT_volumes.apply(_score_d5_audio,      axis=1)

    # ── Score final ponderado (suma de dimensiones × pesos MDCL v3) ──────────
    MAT_volumes['score'] = (
        MAT_volumes['score_d1_mac_venta']  * DIMENSION_WEIGHTS['d1_mac_venta']
        + MAT_volumes['score_d2_compliance'] * DIMENSION_WEIGHTS['d2_compliance']
        + MAT_volumes['score_d3_script']     * DIMENSION_WEIGHTS['d3_script']
        + MAT_volumes['score_d4_engagement'] * DIMENSION_WEIGHTS['d4_engagement']
        + MAT_volumes['score_d5_audio']      * DIMENSION_WEIGHTS['d5_audio']
    )
    MAT_volumes['score'] = np.clip(MAT_volumes['score'], 0.0, 10.0)
    MAT_volumes['score_version'] = SCORE_VERSION

    MAT_volumes.to_excel(campaign_directory + "misc/MAT_VOLUMES.xlsx")

    # ── Post-proceso de IDs y estadísticas ───────────────────────────────────
    MAT_CALLS_THIS_CAMPAIGN = MAT_volumes.copy()
    print(f"SE TIENEN: {len(MAT_CALLS_THIS_CAMPAIGN)} LLAMADAS CALIFICADAS")

    statistics['mean_centroid'] = statistics['mean_centroid'] / np.max(statistics['mean_centroid'])

    for id_col in ('LEAD_ID', 'AGENT_ID'):
        MAT_CALLS_THIS_CAMPAIGN[id_col] = MAT_CALLS_THIS_CAMPAIGN[id_col].fillna(0).astype(str)

    MAT_complete_topics = pd.merge(
        topics_transcripts_convers,
        MAT_CALLS_THIS_CAMPAIGN[['id', 'file_name', 'transcript']],
        on='file_name', how='right',
    )

    return (
        MAT_CALLS_THIS_CAMPAIGN,
        statistics,
        MAT_complete_topics,
        NOPERM,
        topics_stats_convs_scores,
        values_per_topic_for_all_convs,
    )

