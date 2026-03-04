import ast
import json
import os
import numpy as np
import pandas as pd
from lang.VapLangUtils import get_kws, normalize_text, word_count
from setup.MatrixSetup import remove_connectors


def calculate_confidence_scores_per_topic(df):
    """
    Calculates average confidence scores for:
    1. The entire DataFrame
    2. Rows with 'final_label' in ['MAC', 'MAC_DEF']
    3. Rows with 'final_label' in ['PRECIO', 'PRECIO_DEF']

    Parameters:
        df (pd.DataFrame): DataFrame containing columns 'avg_confidence' and 'final_label'

    Returns:
        tuple: (total_avg_confidence, mac_avg_confidence, precio_avg_confidence)
    """

    total_avg_conf = df['avg_confidence'].mean()

    mac_mask = df['final_label'].isin(['MAC', 'MAC_DEF'])
    mac_avg_conf = df.loc[mac_mask, 'avg_confidence'].mean()

    precio_mask = df['final_label'].isin(['PRECIO', 'PRECIO_DEF'])
    precio_avg_conf = df.loc[precio_mask, 'avg_confidence'].mean()

    return total_avg_conf, mac_avg_conf, precio_avg_conf

def getKeywords(df):
    df['permitida'] = df['cluster'].apply(
        lambda x: 'No' if (x == 'no permitida' or x == 'nopermitida' or x=='no permitida') else 'Sí')
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
        all_must_keywords = df[df['permitida'] == 'Sí']['name'].explode()
        all_musnt_keywords = df[df['permitida'] == 'No']['name'].explode()
        all_must_keywords = list(set(all_must_keywords.dropna()))
        all_musnt_keywords = list(set(all_musnt_keywords.dropna()))
    df.to_excel('MATRIZ_CALIFICACION_HIPOT.xlsx')
    return all_must_keywords,all_musnt_keywords


def count_keywords_in_text(keywords, target_text):
    normalized_text = normalize_text(target_text)
    normalized_text = remove_connectors(normalized_text)

    count = 0
    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)
        normalized_keyword = remove_connectors(normalized_keyword)
        if normalized_keyword in normalized_text:
            count += 1
    return count


def confidenceMetrics(
    df: pd.DataFrame,
    col: str = "topics_sequence",
    mean_col: str = "mean_conf",
    min_col: str = "min_conf",
    max_col: str = "max_conf",
    std_col: str = "std_conf",
    n_col: str = "n_conf",
    default: float = 0.0,
    ddof: int = 1,  # 1 = desviación estándar muestral (como pandas)
) -> pd.DataFrame:
    """
    Calcula para cada fila las métricas sobre 'confidence' dentro de `col`:
      - promedio (mean_col)
      - mínimo (min_col)
      - máximo (max_col)
      - desviación estándar (std_col) con ddof configurable (1=muestral)
      - conteo de valores válidos (n_col)

    La columna `col` puede venir como string JSON o lista de dicts.
    Si no hay 'confidence' válidos, usa `default` y std=default (0.0 por defecto).

    Retorna el mismo DataFrame con 5 columnas nuevas.
    """

    means, mins, maxs, stds, counts = [], [], [], [], []

    serie_vals = df[col] if col in df.columns else pd.Series([None] * len(df), index=df.index)

    for val in serie_vals:
        # Normaliza a lista de dicts
        topics_list = []
        if isinstance(val, list):
            topics_list = val
        elif isinstance(val, str):
            try:
                topics_list = json.loads(val)
            except Exception:
                try:
                    parsed = ast.literal_eval(val)
                    topics_list = parsed if isinstance(parsed, list) else []
                except Exception:
                    topics_list = []
        else:
            topics_list = []

        # Extrae confidences válidos
        confidences = []
        for item in topics_list:
            if isinstance(item, dict) and "confidence" in item:
                try:
                    confidences.append(float(item["confidence"]))
                except (TypeError, ValueError):
                    pass

        n = len(confidences)
        if n > 0:
            c_arr = np.array(confidences, dtype=float)
            means.append(float(c_arr.mean()))
            mins.append(float(c_arr.min()))
            maxs.append(float(c_arr.max()))
            if n > ddof:  # evita NaN por grados de libertad <= 0
                stds.append(float(np.std(c_arr, ddof=ddof)))
            else:
                stds.append(default)
            counts.append(int(n))
        else:
            means.append(default)
            mins.append(default)
            maxs.append(default)
            stds.append(default)
            counts.append(0)

    df[mean_col] = means
    df[min_col]  = mins
    df[max_col]  = maxs
    df[std_col]  = stds
    df[n_col]    = counts
    return df

def df_getWordRate(convDataframe,wordMatrix):
    PERM,NOPERM=getKeywords(wordMatrix)
    convDataframe['normalized']      = convDataframe['transcript'].apply(normalize_text)
    convDataframe['normalized']      = convDataframe['normalized'].apply(remove_connectors)
    convDataframe['count_must_have'] = convDataframe['normalized'].apply(lambda x: count_keywords_in_text(PERM, x))
    convDataframe['count_forbidden'] = convDataframe['normalized'].apply(lambda x: count_keywords_in_text(NOPERM, x))
    convDataframe['must_have_rate']  = convDataframe['count_must_have'] / len(PERM) * 100
    if len(NOPERM) != 0:
        convDataframe['forbidden_rate'] = convDataframe['count_forbidden'] / (len(NOPERM)) * 100
    else:
        convDataframe['forbidden_rate'] = 0
    return convDataframe



def generateConvDataframe(filepath):
    classifiedConv = pd.read_csv(filepath)
    classifiedConv = confidenceMetrics(classifiedConv)
    classifiedConv['num_words'] = classifiedConv['text'].apply(word_count)
    classifiedConv['words_p_m'] = classifiedConv['num_words'] / np.abs(classifiedConv['end'] - classifiedConv['start']) * 60
    classifiedConv['file_name'] = os.path.basename(filepath).split('_transcript')[0] + '.mp3'
    classifiedConv[['total_avg_conf_transcript',
                    'mac_avg_conf_transcript',
                    'precio_avg_conf_transcript']]=calculate_confidence_scores_per_topic(classifiedConv)
    classifiedConv['MVD'] = classifiedConv['final_label'].apply(lambda x: 1 if x == 'CONFIRMACION DATOS' else 0)
    classifiedConv['TERMS'] = classifiedConv['final_label'].apply(lambda x: 1 if x == 'TERMINOS LEGALES' else 0)
    classifiedConv['MAC_R'] = classifiedConv['final_label'].apply(lambda x: 1 if x == 'MAC REFUERZO' else 0)
    classifiedConv['IGS_COMP'] = classifiedConv.apply(lambda x: 1 if (("integral" in x["text"].lower())) else 0 ,axis=1)
    aggregated = classifiedConv.groupby('final_label').agg(
        topic_start=('start', 'mean'),
        topic_end=('end', 'mean'),
        topic_max_conf=('max_conf', 'mean'),
        topic_mean_conf=('mean_conf', 'mean'),
        topic_words_p_m=('words_p_m', 'mean'),
        topic_num_words=('num_words', 'mean'),
        topic_occurrence=('final_label', 'size'),
        text_string=('text', lambda x: ' '.join(x)),
        mean_transcript_confidence = ('total_avg_conf_transcript','mean'),
        mean_mac_transcript_confidence = ('mac_avg_conf_transcript', 'mean'),
        mean_price_transcript_confidence=('precio_avg_conf_transcript', 'mean'),
        mvd = ('MVD', 'max'),
        terms = ('TERMS', 'max'),
        mac_r = ('MAC_R', 'max'),
        igs_comp = ('IGS_COMP', 'max')
    ).reset_index()
    aggregated['file_name'] = os.path.basename(filepath).split('_transcript')[0] + '.mp3'
    aggregated['time_centroid'] = (aggregated['topic_start'] + aggregated['topic_end']) / 2
    return [classifiedConv, aggregated, np.max(classifiedConv['end'])]