import ast
import os
import re
import string
from collections import Counter
import Levenshtein
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tika import parser

from setup.CampaignSetup import obtener_inventario
from database.S3Loader import download_grading_matrix, download_guion
from lang.VapLangUtils import preprocess_text
from utils.VapFunctions import clean_column_blank_regex, get_campaign_parameters

stop_words = set(stopwords.words('spanish'))

common_spanish_stopwords = list({
    'de', 'la', 'en', 'con', 'el', 'y', 'a', 'que', 'los', 'del',
    'se', 'por', 'las', 'un', 'para', 'o', 'es', 'una', 'al','si'
})+list(stop_words)
#spanish_model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')

stemmer = SnowballStemmer("spanish")

mac_sign='MAC'
precio_sign='PRECIO'

mapping = pd.read_excel('TOPIC_MAPPING.xlsx')




def replace_cluster_values(df, mapping_df,clust_name):
    """
    Reemplaza los valores de la columna 'cluster' en df según el mapeo en mapping_df.

    :param df: DataFrame que contiene la columna 'cluster'.
    :param mapping_df: DataFrame con dos columnas, donde la primera es el valor original y la segunda el valor mapeado.
    :return: DataFrame con los valores de 'cluster' reemplazados.
    """
    mapping_dict = dict(zip(mapping_df.iloc[:, 0], mapping_df.iloc[:, 1]))
    df[clust_name] = df[clust_name].map(mapping_dict).fillna(df['cluster'])
    return df



def GetMatrix(df):
    # Define possible topics for the "cluster" column
    # Initialize empty columns
    cluster_col = None
    keywords_col = None

    # Initialize a list to store other keyword columns
    other_keyword_cols = []

    # Simple checks for exact matches or contains "keyword"
    for column in df.columns:
        if column.lower() == 'cluster':
            cluster_col = column
        if 'name' or 'keyword' in column.lower():
            if keywords_col is None:
                keywords_col = column
            else:
                # If there are multiple keyword columns, take the one with the longest average content
                avg_length_current = df[keywords_col].dropna().apply(lambda x: len(str(x))).mean()
                avg_length_new = df[column].dropna().apply(lambda x: len(str(x))).mean()
                if avg_length_new > avg_length_current:
                    other_keyword_cols.append(keywords_col)
                    keywords_col = column
                else:
                    other_keyword_cols.append(column)
    if keywords_col is None:
        avg_lengths = {column: df[column].dropna().apply(lambda x: len(str(x))).mean() for column in df.columns}
        keywords_col = max(avg_lengths, key=avg_lengths.get)

    return cluster_col, keywords_col, other_keyword_cols

def list_files(directory, extension):
    all_files = os.listdir(directory)
    files = [file for file in all_files if file.endswith(extension)]
    return files


def match_matrix(campaign_name, files):
    # Initialize the best match and minimum distance
    best_match = None
    min_distance = float('inf')
    csv_files = [file for file in files if ("planta" not in file.lower()) and (("matriz") in file.lower() or ("palabras") in file.lower() or ("guion") in file.lower())] 
    # Calculate the Levenshtein distance for each file name
    for file_name in csv_files:
        # Remove the file extension for comparison
        base_name = os.path.splitext(file_name)[0]

        # Calculate the distance
        distance = Levenshtein.distance(str.lower(campaign_name), str.lower(" ".join(base_name.split("_"))))
        print(f"Distance between '{campaign_name}' and '{base_name}': {distance}")

        # Update the best match if this distance is the smallest
        if distance < min_distance:
            min_distance = distance
            best_match = file_name

    return best_match


def preprocess_and_extract_keywords(sentences, stop_words,use_stemming=True):
    # Join all sentences into one string
    text = ' '.join(sentences)
    # Lowercase
    text = text.lower()
    # Remove accents manually
    text = re.sub(r'[áàäâ]', 'a', text)
    text = re.sub(r'[éèëê]', 'e', text)
    text = re.sub(r'[íìïî]', 'i', text)
    text = re.sub(r'[óòöô]', 'o', text)
    text = re.sub(r'[úùüû]', 'u', text)
    text = re.sub(r'[ñ]', 'n', text)
    # Remove punctuation
    text = re.sub(r'[.]', '', text)
    text = re.sub(r'\W+', ' ', text)
    # Split into tokens
    tokens = text.split()
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    if use_stemming:
        tokens = [stemmer.stem(word) for word in tokens]
    return tokens


def remove_connectors(text):
    # Define a list of connectors (can be expanded)
    connectors = stop_words
    words = text.split()
    filtered_words = [word for word in words if word not in connectors]
    return ' '.join(filtered_words)


def matrix_setup(path_campaña,campaign_directory,inventario):
    """
    Realiza el setup de la campaña y retorna los parámetros de la campaña.
    """
    df_inventory = obtener_inventario()
    if df_inventory.empty:
        print("No se pudo obtener el inventario.")
        return None
    else:
        mapping_camps = clean_column_blank_regex(df_inventory, 'path')
        mapping_camps_expanded = mapping_camps.assign(path=mapping_camps['path'].str.split(',')).explode('path')
        campaign_parameters = get_campaign_parameters(path_campaña, mapping_camps_expanded)
        print(campaign_parameters)
        if campaign_parameters is not None:
            #print(f"Parámetros de la campaña {path_campaña}:{campaign_parameters}")
            MAT=campaign_parameters['document']
            camp_data_campaign = campaign_parameters['campaign']
            download_grading_matrix('matrices.aihub',MAT, campaign_directory)
            download_guion('matrices.aihub',MAT, campaign_directory)
            matrix_path = campaign_directory + 'misc/'
            csv_files = list_files(matrix_path, '.csv')
            xlsx_files = list_files(matrix_path, '.xlsx')
            pdf_files = list_files(matrix_path, '.pdf')
            all_files = csv_files + xlsx_files
            camp_matrix_path = match_matrix(camp_data_campaign, all_files)
            GUION_PATH = ""
            try:
                camp_guion_path = match_matrix(camp_data_campaign, pdf_files)
            except:
                camp_guion_path = None
            if camp_matrix_path:
                MATRIX_PATH = matrix_path + camp_matrix_path
            if camp_guion_path:
                GUION_PATH = matrix_path + camp_guion_path
            print("LA MATRIZ ASIGNADA ES: " + MATRIX_PATH)
            print("EL GUION ASIGNADO ES: " + GUION_PATH)
            try:
                if MATRIX_PATH.split('.')[-1] == 'xlsx':
                    evaluation_matrix_df = pd.read_excel(MATRIX_PATH)
                elif MATRIX_PATH.split('.')[-1] == 'csv':
                    try:
                        evaluation_matrix_df = pd.read_csv(MATRIX_PATH,
                                                           on_bad_lines='skip')
                    except Exception as e:
                        print("Codec error: ")
                        print(e)
                        try:
                            evaluation_matrix_df = pd.read_csv(MATRIX_PATH, encoding='utf-8',
                                                               on_bad_lines='skip')
                        except Exception as e:
                            print("Codec error: ")
                            print(e)
                            evaluation_matrix_df = pd.read_csv(MATRIX_PATH, encoding='latin-1',
                                                               on_bad_lines='skip')
            except Exception as e:
                raise Exception(f"No existe la matriz de calificación: {e}.")
            raw_guion_text = " "
            try:
                if GUION_PATH.split('.')[-1] == 'pdf':
                    try:
                        guion = parser.from_file(GUION_PATH)
                        raw_guion_text = guion['content']
                        print(raw_guion_text)
                    except Exception as e:
                        print("Codec error: ")
                        print(e)
            except Exception as e:
                raise Exception(f"No existe el guion de campaña: {e}.")
            try:
                clust_name, kw_name, other_kws = GetMatrix(evaluation_matrix_df)
            except:
                evaluation_matrix_df = pd.read_csv(MATRIX_PATH,encoder='latin1')
                clust_name, kw_name, other_kws = GetMatrix(evaluation_matrix_df)
            clust_name = clust_name.lower() if clust_name else 'cluster'
            kw_name = kw_name.lower() if kw_name else 'name'
            evaluation_matrix_df.columns = evaluation_matrix_df.columns.str.lower()
            evaluation_matrix_df[clust_name] = evaluation_matrix_df[clust_name]
            if 'MODULO' in evaluation_matrix_df.columns:
                evaluation_matrix_df['MODULO'] = evaluation_matrix_df['MODULO'].ffill()
            evaluation_matrix_df =  replace_cluster_values(evaluation_matrix_df,mapping,clust_name)
            for i in other_kws:
                if ('NO' or 'no' in i):
                    evaluation_matrix_df[clust_name] = evaluation_matrix_df[clust_name].replace(i, 'nopermitida')
            evaluation_matrix_df['permitida'] = evaluation_matrix_df[clust_name].apply(
                lambda x: 'No' if (x == 'no permitida' or x == 'nopermitida') else 'Sí')
            permitidas = evaluation_matrix_df[evaluation_matrix_df['permitida'] == 'Sí'][kw_name].tolist()
            no_permitidas = evaluation_matrix_df[evaluation_matrix_df['permitida'] == 'No'][kw_name].tolist()
            topics_df_grouped = evaluation_matrix_df.groupby('cluster', as_index=False).agg({
                'name': lambda x: list(x)
            }).reset_index()
            topics_keywords = topics_df_grouped[['cluster', 'name']].copy()
            topics_keywords = topics_keywords[topics_keywords['cluster'] != 'no permitida']
            topics_df = evaluation_matrix_df[evaluation_matrix_df['cluster'] != 'no permitida']
            try:
                topics_keywords['name'] = topics_keywords['name'].apply(
                    ast.literal_eval)  # Convert string representation of list to list
            except:
                topics_keywords['name'] = topics_keywords['name']

            print(topics_df_grouped)
            return topics_df,topics_df_grouped,permitidas, no_permitidas, raw_guion_text
        else:
            return None,None,None,None,None


