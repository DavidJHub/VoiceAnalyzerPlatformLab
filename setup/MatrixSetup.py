import ast
import logging
import os
import re
import string
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tika import parser

from setup.CampaignSetup import obtener_inventario
from database.S3Loader import download_grading_matrix, download_guion
from lang.VapLangUtils import preprocess_text
from utils.VapFunctions import clean_column_blank_regex, get_campaign_parameters

logger = logging.getLogger(__name__)

stop_words = set(stopwords.words('spanish'))

common_spanish_stopwords = list({
    'de', 'la', 'en', 'con', 'el', 'y', 'a', 'que', 'los', 'del',
    'se', 'por', 'las', 'un', 'para', 'o', 'es', 'una', 'al','si'
})+list(stop_words)

stemmer = SnowballStemmer("spanish")

mac_sign='MAC'
precio_sign='PRECIO'

# ---------------------------------------------------------------------------
# Canonical topics — single source of truth
# ---------------------------------------------------------------------------
CANONICAL_TOPICS = [
    "SALUDO",
    "PERFILAMIENTO",
    "PRODUCTO",
    "CONFIRMACION MONITOREO",
    "LEY RETRACTO",
    "TERMINOS LEGALES",
    "TRATAMIENTO DATOS",
    "MAC",
    "MAC REFUERZO",
    "PRECIO",
    "CONFIRMACION DATOS",
    "CONFORMIDAD",
    "ATENCION",
    "DESPEDIDA",
]

# Non-canonical topic names that should map to canonical ones
_TOPIC_NORMALIZATION = {
    "CONFIRMACION DE DATOS": "CONFIRMACION DATOS",
    "CONFIRMACIÓN DE DATOS": "CONFIRMACION DATOS",
    "CONFIRMACION DE MONITOREO": "CONFIRMACION MONITOREO",
    "PREGUNTA DE REFUERZO": "MAC REFUERZO",
    "PDC": "CONFORMIDAD",
    "NP": "NP",  # keep as-is, filtered later as 'no permitida'
}

mapping = pd.read_excel('TOPIC_MAPPING.xlsx')
mapping['cluster'] = mapping['cluster'].astype(str).str.lower()


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def _normalize_text(text):
    """Lowercase, strip accents, and remove non-alphanumeric characters."""
    text = str(text).lower().strip()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def _tokenize(text):
    """Normalize and split into a set of meaningful tokens."""
    normalized = _normalize_text(text)
    tokens = set(normalized.split())
    tokens -= {'de', 'la', 'el', 'los', 'las', 'del', 'en', 'con', 'a',
               'y', 'o', 'un', 'una', 'por', 'para', 'al', 'que', 'se',
               'es', 'si', 'matriz', 'palabras', 'guion', 'clave'}
    return tokens


# ---------------------------------------------------------------------------
# match_matrix — token-based Jaccard + overlap matching
# ---------------------------------------------------------------------------

def match_matrix(campaign_name, files):
    """
    Match a campaign name to the best file using token-based similarity.

    Uses a combined score of Jaccard similarity (intersection/union of word
    tokens) and overlap coefficient (intersection/min-set-size).  This is
    robust to word reordering, extra/missing words, and name length
    differences — problems that plagued the old Levenshtein approach.
    """
    best_match = None
    best_score = -1.0

    candidate_files = [
        f for f in files
        if "planta" not in f.lower()
        and ("matriz" in f.lower() or "palabras" in f.lower() or "guion" in f.lower())
    ]

    campaign_tokens = _tokenize(campaign_name)

    if not campaign_tokens:
        logger.warning(f"Campaign name '{campaign_name}' produced no tokens for matching.")
        return candidate_files[0] if candidate_files else None

    for file_name in candidate_files:
        base_name = os.path.splitext(file_name)[0]
        file_tokens = _tokenize(" ".join(base_name.split("_")))

        if not file_tokens:
            continue

        intersection = campaign_tokens & file_tokens
        union = campaign_tokens | file_tokens

        jaccard = len(intersection) / len(union) if union else 0.0
        overlap = len(intersection) / min(len(campaign_tokens), len(file_tokens))

        # Combined score: weighted towards overlap to handle length differences
        score = 0.4 * jaccard + 0.6 * overlap

        logger.info(
            f"match_matrix: '{campaign_name}' vs '{base_name}' -> "
            f"jaccard={jaccard:.3f}, overlap={overlap:.3f}, score={score:.3f}"
        )

        if score > best_score:
            best_score = score
            best_match = file_name

    logger.info(f"match_matrix: best match for '{campaign_name}' -> '{best_match}' (score={best_score:.3f})")
    return best_match


# ---------------------------------------------------------------------------
# replace_cluster_values — map matrix clusters to general topics
# ---------------------------------------------------------------------------

def replace_cluster_values(df, mapping_df, clust_name):
    """
    Replace cluster values in *df* using the TOPIC_MAPPING lookup.

    Returns the DataFrame with the cluster column updated.
    """
    mapping_dict = dict(zip(
        mapping_df.iloc[:, 0].apply(lambda x: str(x).lower()),
        mapping_df.iloc[:, 1]
    ))
    df[clust_name] = df[clust_name].map(mapping_dict).fillna(df[clust_name])
    return df


# ---------------------------------------------------------------------------
# normalize_to_canonical — ensure final topics are canonical uppercase
# ---------------------------------------------------------------------------

def normalize_to_canonical(topic):
    """
    Normalize a topic string to its canonical uppercase form.

    1. Strip, uppercase.
    2. Apply known normalization aliases.
    3. Return as-is if already canonical or unrecognized.
    """
    topic = str(topic).strip().upper()
    if topic in _TOPIC_NORMALIZATION:
        topic = _TOPIC_NORMALIZATION[topic]
    return topic


# ---------------------------------------------------------------------------
# map_unmapped_topics — fuzzy-match topics not found in TOPIC_MAPPING
# ---------------------------------------------------------------------------

def map_unmapped_topics(df, clust_name, campaign_directory):
    """
    Attempt to map topics that are not in the canonical set.

    Strategy (in priority order):
      1. Direct normalization via _TOPIC_NORMALIZATION aliases.
      2. Substring containment — if any canonical topic is fully contained
         in the unmapped name (or vice versa), use that.
      3. Token overlap — pick the canonical topic with the highest overlap
         coefficient against the unmapped name's tokens.

    Writes a report of unmapped topics and their resolutions to
    ``<campaign_directory>/misc/unmapped_topics_report.csv``.

    Returns the DataFrame with the cluster column updated.
    """
    canonical_set = set(CANONICAL_TOPICS)
    unique_topics = df[clust_name].dropna().unique()
    unmapped_report = []

    for topic in unique_topics:
        normalized = normalize_to_canonical(topic)

        if normalized in canonical_set:
            # Already canonical after normalization
            if normalized != str(topic).strip().upper():
                df.loc[df[clust_name] == topic, clust_name] = normalized
                unmapped_report.append({
                    'original_topic': topic,
                    'mapped_to': normalized,
                    'method': 'alias_normalization'
                })
            continue

        # Skip known non-topic categories
        if normalized in ('NP', 'NO PERMITIDA', 'NOPERMITIDA'):
            continue

        # Strategy 2: Substring containment
        topic_norm = _normalize_text(topic)
        best_canonical = None
        best_method = None

        for canonical in CANONICAL_TOPICS:
            canonical_norm = _normalize_text(canonical)
            if canonical_norm in topic_norm or topic_norm in canonical_norm:
                best_canonical = canonical
                best_method = 'substring_containment'
                break

        # Strategy 3: Token overlap
        if best_canonical is None:
            topic_tokens = _tokenize(topic)
            if topic_tokens:
                best_overlap = 0.0
                for canonical in CANONICAL_TOPICS:
                    canonical_tokens = _tokenize(canonical)
                    if not canonical_tokens:
                        continue
                    intersection = topic_tokens & canonical_tokens
                    overlap = len(intersection) / min(len(topic_tokens), len(canonical_tokens))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_canonical = canonical
                        best_method = f'token_overlap({best_overlap:.2f})'
                # Only accept if overlap is meaningful
                if best_overlap < 0.5:
                    best_canonical = None
                    best_method = None

        if best_canonical:
            df.loc[df[clust_name] == topic, clust_name] = best_canonical
            unmapped_report.append({
                'original_topic': topic,
                'mapped_to': best_canonical,
                'method': best_method
            })
            logger.info(f"Unmapped topic '{topic}' -> '{best_canonical}' via {best_method}")
        else:
            unmapped_report.append({
                'original_topic': topic,
                'mapped_to': 'UNMAPPED',
                'method': 'no_match_found'
            })
            logger.warning(f"Topic '{topic}' could not be mapped to any canonical topic.")

    # Write report
    if unmapped_report:
        report_df = pd.DataFrame(unmapped_report)
        report_path = os.path.join(campaign_directory, 'misc', 'unmapped_topics_report.csv')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report_df.to_csv(report_path, index=False, encoding='utf-8')
        logger.info(f"Unmapped topics report saved to {report_path}")

        mapped_count = sum(1 for r in unmapped_report if r['mapped_to'] != 'UNMAPPED')
        still_unmapped = [r['original_topic'] for r in unmapped_report if r['mapped_to'] == 'UNMAPPED']
        logger.info(f"Topic mapping summary: {mapped_count} resolved, {len(still_unmapped)} still unmapped.")
        if still_unmapped:
            logger.warning(f"Still unmapped topics: {still_unmapped}")

    return df


# ---------------------------------------------------------------------------
# GetMatrix — detect cluster and keyword columns in the matrix DataFrame
# ---------------------------------------------------------------------------

def GetMatrix(df):
    """Detect the cluster column and keyword/name columns in the matrix."""
    cluster_col = None
    keywords_col = None
    other_keyword_cols = []

    for column in df.columns:
        col_lower = column.lower()
        if col_lower == 'cluster':
            cluster_col = column
        elif 'name' in col_lower or 'keyword' in col_lower:
            if keywords_col is None:
                keywords_col = column
            else:
                avg_length_current = df[keywords_col].dropna().apply(lambda x: len(str(x))).mean()
                avg_length_new = df[column].dropna().apply(lambda x: len(str(x))).mean()
                if avg_length_new > avg_length_current:
                    other_keyword_cols.append(keywords_col)
                    keywords_col = column
                else:
                    other_keyword_cols.append(column)

    if keywords_col is None:
        avg_lengths = {
            column: df[column].dropna().apply(lambda x: len(str(x))).mean()
            for column in df.columns
        }
        keywords_col = max(avg_lengths, key=avg_lengths.get)

    return cluster_col, keywords_col, other_keyword_cols


def list_files(directory, extension):
    all_files = os.listdir(directory)
    files = [file for file in all_files if file.endswith(extension)]
    return files


def preprocess_and_extract_keywords(sentences, stop_words, use_stemming=True):
    text = ' '.join(sentences)
    text = text.lower()
    text = re.sub(r'[áàäâ]', 'a', text)
    text = re.sub(r'[éèëê]', 'e', text)
    text = re.sub(r'[íìïî]', 'i', text)
    text = re.sub(r'[óòöô]', 'o', text)
    text = re.sub(r'[úùüû]', 'u', text)
    text = re.sub(r'[ñ]', 'n', text)
    text = re.sub(r'[.]', '', text)
    text = re.sub(r'\W+', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    if use_stemming:
        tokens = [stemmer.stem(word) for word in tokens]
    return tokens


def remove_connectors(text):
    connectors = stop_words
    words = text.split()
    filtered_words = [word for word in words if word not in connectors]
    return ' '.join(filtered_words)


def matrix_setup(path_campaña, campaign_directory, inventario):
    """
    Realiza el setup de la campaña y retorna los parámetros de la campaña.
    """
    df_inventory = obtener_inventario()
    if df_inventory.empty:
        logger.error("No se pudo obtener el inventario.")
        return None
    else:
        mapping_camps = clean_column_blank_regex(df_inventory, 'path')
        mapping_camps_expanded = mapping_camps.assign(path=mapping_camps['path'].str.split(',')).explode('path')
        campaign_parameters = get_campaign_parameters(path_campaña, mapping_camps_expanded)
        logger.info(f"Campaign parameters: {campaign_parameters}")
        if campaign_parameters is not None:
            MAT = campaign_parameters['document']
            camp_data_campaign = campaign_parameters['campaign']
            download_grading_matrix('matrices.aihub', MAT, campaign_directory)
            download_guion('matrices.aihub', MAT, campaign_directory)
            matrix_path = campaign_directory + 'misc/'
            csv_files = list_files(matrix_path, '.csv')
            xlsx_files = list_files(matrix_path, '.xlsx')
            pdf_files = list_files(matrix_path, '.pdf')
            all_files = csv_files + xlsx_files
            camp_matrix_path = match_matrix(camp_data_campaign, all_files)
            GUION_PATH = ""
            try:
                camp_guion_path = match_matrix(camp_data_campaign, pdf_files)
            except Exception:
                camp_guion_path = None
            if camp_matrix_path:
                MATRIX_PATH = matrix_path + camp_matrix_path
            if camp_guion_path:
                GUION_PATH = matrix_path + camp_guion_path
            logger.info("LA MATRIZ ASIGNADA ES: " + MATRIX_PATH)
            logger.info("EL GUION ASIGNADO ES: " + GUION_PATH)
            try:
                if MATRIX_PATH.split('.')[-1] == 'xlsx':
                    evaluation_matrix_df = pd.read_excel(MATRIX_PATH)
                elif MATRIX_PATH.split('.')[-1] == 'csv':
                    try:
                        evaluation_matrix_df = pd.read_csv(MATRIX_PATH,
                                                           on_bad_lines='skip')
                    except Exception as e:
                        logger.warning(f"Codec error: {e}")
                        try:
                            evaluation_matrix_df = pd.read_csv(MATRIX_PATH, encoding='utf-8',
                                                               on_bad_lines='skip')
                        except Exception as e:
                            logger.warning(f"Codec error: {e}")
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
                        logger.info(f"Guion loaded, length={len(raw_guion_text)}")
                    except Exception as e:
                        logger.warning(f"Codec error reading guion: {e}")
            except Exception as e:
                raise Exception(f"No existe el guion de campaña: {e}.")
            try:
                clust_name, kw_name, other_kws = GetMatrix(evaluation_matrix_df)
            except Exception:
                evaluation_matrix_df = pd.read_csv(MATRIX_PATH, encoding='latin1')
                clust_name, kw_name, other_kws = GetMatrix(evaluation_matrix_df)
            clust_name = clust_name.lower() if clust_name else 'cluster'
            kw_name = kw_name.lower() if kw_name else 'name'
            evaluation_matrix_df[clust_name] = evaluation_matrix_df[clust_name].astype(str).str.lower()
            evaluation_matrix_df.columns = evaluation_matrix_df.columns.str.lower()
            if 'modulo' in evaluation_matrix_df.columns:
                evaluation_matrix_df['modulo'] = evaluation_matrix_df['modulo'].ffill()

            # Step 1: Apply TOPIC_MAPPING.xlsx lookups
            evaluation_matrix_df = replace_cluster_values(evaluation_matrix_df, mapping, clust_name)

            # Step 2: Normalize all topics to canonical uppercase form
            evaluation_matrix_df[clust_name] = evaluation_matrix_df[clust_name].apply(normalize_to_canonical)

            # Step 3: Map any remaining unmapped topics via fuzzy matching
            evaluation_matrix_df = map_unmapped_topics(evaluation_matrix_df, clust_name, campaign_directory)

            # Mark 'no permitida' rows
            for i in other_kws:
                if 'no' in str(i).lower():
                    evaluation_matrix_df[clust_name] = evaluation_matrix_df[clust_name].replace(i, 'NP')
            evaluation_matrix_df['permitida'] = evaluation_matrix_df[clust_name].apply(
                lambda x: 'No' if x.upper() in ('NO PERMITIDA', 'NOPERMITIDA', 'NP') else 'Sí')
            permitidas = evaluation_matrix_df[evaluation_matrix_df['permitida'] == 'Sí'][kw_name].tolist()
            no_permitidas = evaluation_matrix_df[evaluation_matrix_df['permitida'] == 'No'][kw_name].tolist()

            # Filter out 'no permitida' for topic groupings
            topics_mask = ~evaluation_matrix_df[clust_name].str.upper().isin(('NO PERMITIDA', 'NOPERMITIDA', 'NP'))

            topics_df_grouped = evaluation_matrix_df[topics_mask].groupby(clust_name, as_index=False).agg({
                kw_name: lambda x: list(x)
            }).reset_index()

            # Rename columns to standard names for downstream compatibility
            topics_df_grouped.columns = ['index', 'cluster', 'name'] if len(topics_df_grouped.columns) == 3 else topics_df_grouped.columns
            if 'index' in topics_df_grouped.columns:
                topics_df_grouped = topics_df_grouped.drop(columns=['index'])

            topics_df = evaluation_matrix_df[topics_mask]

            try:
                topics_df_grouped['name'] = topics_df_grouped['name'].apply(ast.literal_eval)
            except Exception:
                pass

            logger.info(f"Topics found: {topics_df_grouped['cluster'].tolist() if 'cluster' in topics_df_grouped.columns else topics_df_grouped.iloc[:, 0].tolist()}")
            return topics_df, topics_df_grouped, permitidas, no_permitidas, raw_guion_text
        else:
            return None, None, None, None, None
