import glob
import json
import os
from difflib import SequenceMatcher
from sklearn.preprocessing import QuantileTransformer

import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from segmentationModel.fittingDeep import fitCSVConversations

from lang.VapLangUtils import normalize_text, get_kws, word_count, \
    correctCommonTranscriptionMistakes, splitConversations

from setup.MatrixSetup import remove_connectors
from utils.VapFunctions import measureDbAplitude, measure_speed_classification
from utils.VapUtils import jsonDecompose, get_data_from_name, jsonDecomposeSentencesHighlight, jsonTranscriptionToCsv, getTranscriptParagraphsJsonHighlights

import pandas as pd
import numpy as np

from segmentationModel.textPostprocessing import reconstruirDialogos, process_directory_mac_price_def
from vapScoreEngine.dfUtils import calculate_confidence_scores_per_topic, df_getWordRate, generateConvDataframe



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


def score_camp(campaign_directory,campaign_id,TMO,topics_combined_df):
    keywords_df = topics_combined_df
    PERM, NOPERM = list_y_n_words(keywords_df)
    print("Procesando conversaciones y generando caché...")
    routeRawCsvTranscripts = campaign_directory  + campaign_id.replace('/','') + '_RAW'
    routeRawCsvGraded  = campaign_directory  + campaign_id.replace('/','')  + '_PROCESSED'
    routeRawCsvRebuilt = campaign_directory + campaign_id.replace('/', '') + '_RECONS'
    scores = process_directory_conversations_with_memory(campaign_directory,
                                                         routeRawCsvTranscripts,
                                                         routeRawCsvGraded,
                                                         routeRawCsvRebuilt,
                                                         PERM, NOPERM)
    print("TOTAL DE LLAMADAS A CALIFICAR: "+ str(len(scores)))
    statistics = calculate_general_statistics([scores[i][1] for i in range(len(scores))],[scores[i][2] for i in range(len(scores))])
    statistics.to_excel(campaign_directory + 'misc/statistics_pre.xlsx')
    MAT_a = get_all_transcripts(campaign_directory)
    MAT_M = get_all_transcripts_memory(routeRawCsvRebuilt + '/memory/')

    MAT = pd.concat([MAT_a, MAT_M], axis=0)
    MAT.to_excel(campaign_directory + "misc/" + 'MAT.xlsx')
    

    ####### NEEDED MAC/PRICE ############

    ALLOWED_MACS = keywords_df[keywords_df['cluster']=='MAC']['name'].tolist()
    ALLOWED_PRICES = keywords_df[keywords_df['cluster'] == 'PRECIO']['name'].tolist()

    print('MACS PERMITIDOS: ' + str(ALLOWED_MACS))
    print('PRECIOS PERMITIDOS: ' + str(ALLOWED_PRICES))

    MAT=df_getWordRate(MAT,keywords_df)

    ## WORDS FACTOR
    MAT['score'] = np.sqrt(np.abs(
        (MAT['count_must_have'] - MAT['count_forbidden']) / (len(PERM) + len(NOPERM)))) * 10

    perfect_Score=np.sqrt(np.abs((100 + 0) / (200))) * 10
    ## TMO FACTOR
    MAT['score'] = np.maximum(0, MAT['score'] + MAT['score']*(MAT['TMO']  - np.mean(MAT['TMO']) ** (2)) / (
                np.mean(MAT['TMO']) ** (2)))
    perfect_Score=perfect_Score+1/2
    try:
        MAT[['DATE_TIME','LEAD_ID', 'EPOCH','AGENT_ID', 'CLIENT_ID']] = MAT['file_name'].apply(lambda x: pd.Series(get_data_from_name(x)))
    except:
        MAT[['DATE_TIME','LEAD_ID','AGENT_ID', 'CLIENT_ID']] = MAT['file_name'].apply(lambda x: pd.Series(get_data_from_name(x)))
    MAT.to_excel(campaign_directory + "misc/" + 'MAT_BEFORE_MACS.xlsx')
    print("TOTAL DE LLAMADAS CALIFICADAS: " + str(len(scores)))
    values_per_topic_for_all_convs = [scores[i][1] for i in range(len(scores))]
    values_per_topic_for_all_convs = pd.concat(values_per_topic_for_all_convs, ignore_index=True)
    topics_transcripts_convers = [scores[i][0] for i in range(len(scores))]
    topics_transcripts_convers = pd.concat(topics_transcripts_convers, ignore_index=False)
    topics_transcripts_convers.to_excel(campaign_directory + "misc/" + 'topics_transcripts_convers.xlsx')
    values_per_topic_for_all_convs['velocity_classification'] = values_per_topic_for_all_convs['topic_words_p_m'].apply(measure_speed_classification)
    values_per_topic_for_all_convs.to_excel(campaign_directory + "misc/" + 'values_per_topic_for_all_convs.xlsx')
    values_per_topic_for_all_convs['mean_transcript_confidence'].fillna(0)
    values_per_topic_for_all_convs['mean_mac_transcript_confidence'].fillna(0)
    values_per_topic_for_all_convs['mean_price_transcript_confidence'].fillna(0)
    values_per_topic_for_all_convs['noauditable_transcript']=values_per_topic_for_all_convs['mean_transcript_confidence'].apply(
        lambda x: 0 if x > 0.9 else 1
    )

    values_per_topic_for_all_convs['noauditable_transcript_mac']=values_per_topic_for_all_convs['mean_mac_transcript_confidence'].apply(
        lambda x: 0 if x > 0.9 else 1
    )

    values_per_topic_for_all_convs['noauditable_transcript_price']=values_per_topic_for_all_convs['mean_price_transcript_confidence'].apply(
        lambda x: 0 if x > 0.9 else 1
    )

    AllCallMacs = values_per_topic_for_all_convs[values_per_topic_for_all_convs['final_label'].isin(['MAC_DEF', 'MAC'])]
    AllCallPrices = values_per_topic_for_all_convs[values_per_topic_for_all_convs['final_label'].isin(['PRECIO_DEF', 'PRECIO'])]

    AllCallMVDs = values_per_topic_for_all_convs[values_per_topic_for_all_convs['mvd']>0]
    AllCallMVDs.to_excel(campaign_directory + "misc/" + 'TRUE_MVDs.xlsx')
    AllCallTERMS = values_per_topic_for_all_convs[values_per_topic_for_all_convs['terms']>0]
    AllCallMVDs.to_excel(campaign_directory + "misc/" + 'TRUE_TERMS.xlsx')
    AllCallMacR = values_per_topic_for_all_convs[values_per_topic_for_all_convs['mac_r']>0]
    AllCallMacR.to_excel(campaign_directory + "misc/" + 'TRUE_MAC_R.xlsx')
    AllCallIgsComps = values_per_topic_for_all_convs[values_per_topic_for_all_convs['igs_comp']>0]
    AllCallIgsComps.to_excel(campaign_directory + "misc/" + 'TRUE_IGS_COMPS.xlsx')

    print('Número de MACs detectados: ' + str(len(AllCallMacs)))
    print('Número de PRECIOS detectados: ' + str(len(AllCallPrices)))

    AllCallMacs.to_excel(campaign_directory + "misc/" + 'TRUE_MACS.xlsx')
    AllCallPrices.to_excel(campaign_directory + "misc/" + 'TRUE_PRICES.xlsx')

    AllCallMacs=calificarLikelyhoodConMatriz(AllCallMacs,ALLOWED_MACS, 1)
    AllCallPrices = calificarLikelyhoodConMatriz(AllCallPrices, ALLOWED_PRICES,  1)

    AllCallMacs.to_excel(campaign_directory + "misc/" + 'TRUE_MACS_graded.xlsx')
    AllCallPrices.to_excel(campaign_directory + "misc/" + 'TRUE_PRICES_graded.xlsx')

    topics_stats_convs_scores = values_per_topic_for_all_convs.groupby('file_name', as_index=False).agg({
        'topic_mean_conf': 'mean'
    }).reset_index()

    topics_stats_convs_scores.to_excel(campaign_directory + "misc/" + 'topics_stats_convs_scores.xlsx')

    # topics_stats_convs_scores      - > MEDIA DE CONFIDENCE DE TODA LA CONVERSACIÓN
    # values_per_topic_for_all_convs - > MEDIA DE CONFIDENCE DE TODA LA CONVERSACIÓN POR TEMA
    # values_per_topic_for_all_convs - > CONFIANZA DE TODOS LOS FRAGMENTOS DE CONVERSACIÓN DE TODAS LAS CONVERSACIONES

    # OPERAR LOS PUNTAJES CON LA MEDIA DE CONFIANZA DE LOS TEMAS (MAS ALTO = MAS CLARA LA CONVERSACIÓN)

    conf_map = topics_stats_convs_scores.set_index('file_name')['topic_mean_conf']
    MAT['score'] = MAT.apply(lambda row: row['score'] * conf_map[row['file_name']] if row['file_name'] in conf_map else row['score'], axis=1)
    MAT.to_excel(campaign_directory + "misc/" + 'MAT_complete.xlsx')
    perfect_Score = perfect_Score*0.9

    AllCallMacs['times_said'] = AllCallMacs.groupby('file_name')['file_name'].transform('count')
    AllCallPrices['times_said'] = AllCallPrices.groupby('file_name')['file_name'].transform('count')
    AllCallMacsMarked=AllCallMacs.loc[AllCallMacs.groupby('file_name')['best_mac_likelihood'].idxmax()]
    AllCallPricesMarked=AllCallPrices.loc[AllCallPrices.groupby('file_name')['best_mac_likelihood'].idxmax()]
    tolerance=0.85
    AllCallMacsMarked['warn'] = AllCallMacsMarked['best_mac_likelihood'].apply(lambda x: x < tolerance)
    AllCallPricesMarked['warn'] = AllCallPricesMarked['best_mac_likelihood'].apply(lambda x: x < tolerance)
    MAT_w_macs = pd.merge(MAT, AllCallMacsMarked, on='file_name',how='left')
    MAT_w_prices = pd.merge(MAT_w_macs, AllCallPricesMarked, on='file_name',how='left',suffixes=('_macs', '_prices'))
    MAT_w_prices.to_excel(campaign_directory + "misc/" + 'MAT_w_m_p.xlsx')

    MAT_w_mvd=pd.merge(MAT_w_prices, AllCallMVDs[['file_name','mvd']], on='file_name',how='left')
    MAT_w_terms=pd.merge(MAT_w_mvd, AllCallTERMS[['file_name','terms']], on='file_name',how='left')
    MAT_w_macr=pd.merge(MAT_w_terms, AllCallMacR[['file_name','mac_r']], on='file_name',how='left')
    MAT_w_igscomp=pd.merge(MAT_w_macr, AllCallIgsComps[['file_name','igs_comp']], on='file_name',how='left')
    MAT_w_igscomp=MAT_w_igscomp.drop_duplicates(subset=['file_name'], keep='first')
    MAT_w_igscomp.to_excel(campaign_directory + "misc/" + 'MAT_WITH_EVERYTHING.xlsx')

    MAT_volumes = measureDbAplitude(campaign_directory, measureDbAplitude(campaign_directory, MAT_w_igscomp,
                                    'time_centroid_macs','mac' ), 'time_centroid_prices', 'prices')
    MAT_volumes[['mvd','terms']] = MAT_volumes[['mvd','terms']].fillna(0)
    MAT_volumes['best_price_likelihood']=MAT_volumes['best_mac_likelihood_prices'].fillna(0.1)
    MAT_volumes['best_price_likelihood']=MAT_volumes['best_mac_likelihood_prices']/ np.max(MAT_volumes['best_mac_likelihood_prices'])
    MAT_volumes['best_mac_likelihood'] = MAT_volumes['best_mac_likelihood_macs'].fillna(0.1)
    MAT_volumes['best_mac_likelihood'] = MAT_volumes['best_mac_likelihood_macs'] / np.max(MAT_volumes['best_mac_likelihood_macs'])
    MAT_volumes['score']=MAT_volumes['score']*MAT_volumes['best_mac_likelihood_macs']*MAT_volumes['best_mac_likelihood_prices']

    MAT_volumes['score'] = MAT_volumes['score']*MAT_volumes['best_mac_likelihood_macs']*MAT_volumes['best_mac_likelihood_prices']
    MAT_volumes['pen_factor']=1+(0.9-MAT_volumes['agent_participation'])

    MAT_volumes['score'] = (MAT_volumes['score']*MAT_volumes['pen_factor']**(1/6))

    perfect_Score = perfect_Score * 0.95 * 0.95 * 0.9
    print(perfect_Score)
    print(MAT_volumes['score'])
    perfect_Score_final= np.max(MAT_volumes['score'])

    qt = QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=42)

    MAT_volumes['score']=MAT_volumes['score']/perfect_Score_final*10
    MAT_volumes['score'] = qt.fit_transform(MAT_volumes[['score']])
    MAT_volumes['score'] = MAT_volumes['score']*10
    print("score after quantile transform: " + str(MAT_volumes['score'].mean()))
    MAT_volumes['score']=np.clip(MAT_volumes['score'],0,10)

    MAT_volumes.to_excel(campaign_directory + "misc/" + 'MAT_VOLUMES.xlsx')
    #print('DROP: ' + str(MAT_complete))
    #print(MAT_complete['file_name'])
    #MAT_CALLS_THIS_CAMPAIGN=filter_dataframe_by_directory(MAT_w_prices,campaign_directory,'file_name')
    MAT_CALLS_THIS_CAMPAIGN=MAT_volumes
    print('SE TIENEN: ' + str(len(MAT_CALLS_THIS_CAMPAIGN)) + 'LLAMADAS CALIFICADAS')
    statistics['mean_centroid'] = statistics['mean_centroid'] / np.max(
        statistics['mean_centroid'])
    MAT_CALLS_THIS_CAMPAIGN['LEAD_ID'] = MAT_CALLS_THIS_CAMPAIGN['LEAD_ID'].fillna(0)
    MAT_CALLS_THIS_CAMPAIGN['LEAD_ID'] = MAT_CALLS_THIS_CAMPAIGN['LEAD_ID'].astype(str)
    MAT_CALLS_THIS_CAMPAIGN['AGENT_ID'] = MAT_CALLS_THIS_CAMPAIGN['AGENT_ID'].fillna(0)
    MAT_CALLS_THIS_CAMPAIGN['AGENT_ID'] = MAT_CALLS_THIS_CAMPAIGN['AGENT_ID'].astype(str)
    MAT_complete_topics = pd.merge(topics_transcripts_convers, MAT_CALLS_THIS_CAMPAIGN[['id','file_name','transcript']], left_on='file_name',
                                   right_on='file_name', how='right')

    return MAT_CALLS_THIS_CAMPAIGN,statistics,MAT_complete_topics,NOPERM,topics_stats_convs_scores,values_per_topic_for_all_convs

