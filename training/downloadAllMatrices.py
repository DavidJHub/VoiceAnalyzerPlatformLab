import os
import warnings
import boto3
import pandas as pd

from database.dbConfig import generate_s3_client

import unicodedata
import re

def normalize_spanish_text(x: str, lowercase: bool=True) -> str:
    """
    Normaliza texto para matching consistente:
      - Preserva 'ñ/Ñ'
      - Elimina diacríticos (á→a, ü→u, etc.)
      - Normaliza espacios
      - (Opcional) lowercase → True por defecto
    """
    if x is None:
        return x
    s = str(x)

    # Preservar ñ
    s = s.replace('ñ', '<__ENYE__>').replace('Ñ', '<__ENYE_UP__>')

    # Quitar diacríticos
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')

    # Restaurar ñ
    s = s.replace('<__ENYE__>', 'ñ').replace('<__ENYE_UP__>', 'Ñ')

    # Normalizar espacios
    s = re.sub(r'\s+', ' ', s).strip()

    # Lowercase
    if lowercase:
        s = s.lower()

    return s

def preprocess_text_df(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    df = df.copy()
    if cols is None:
        cols = [c for c in df.columns if df[c].dtype == 'object']
    else:
        cols = [c for c in cols if c in df.columns]
    for c in cols:
        df[c] = df[c].map(normalize_spanish_text)
    return df

def preprocess_text_df(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """
    Aplica normalize_spanish_text a columnas de texto (por defecto 'name' y 'cluster'
    si existen). Si 'cols' es None, se aplica a TODAS las columnas tipo 'object'.
    """
    df = df.copy()
    if cols is None:
        cols = [c for c in df.columns if df[c].dtype == 'object']
    else:
        cols = [c for c in cols if c in df.columns]

    for c in cols:
        df[c] = df[c].map(normalize_spanish_text)
    return df

###############################################################################
# ------------------------ DESCARGA DESDE S3 -------------------------------- #
###############################################################################
def download_data_files(bucket_name,
                        local_output_folder):
    """
    Descarga .csv (salvo general_dictionary) y .xlsx (salvo los que contengan “planta”)
    preservando la estructura del bucket.
    """
    s3 = generate_s3_client()

    if not os.path.exists(local_output_folder):
        os.makedirs(local_output_folder)

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name)

    n = 0
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            key_l = key.lower()

            # ---- FILTRO GLOBAL -------------------------------------------
            if key_l.endswith('.csv') and 'REENTRENO' in key_l:
                if 'general_dictionary' in key_l:
                    continue
            elif key_l.endswith('.xlsx'):
                if 'planta' in key_l:
                    continue
            else:
                continue  # ni csv ni xlsx

            # ---- DESCARGA ------------------------------------------------
            local_path = os.path.join(local_output_folder, key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            #s3.download_file(bucket_name, key, local_path)
            #print("Descargado:", local_path)
            n += 1

    print(f"\nTotal de archivos descargados: {n}")


def validar_unicidad_mapping(df_concat, mapping_df, col='cluster'):
    # normalizados (como en tu pipeline)
    map_df = mapping_df.iloc[:, :2].copy()
    map_df.columns = ['orig', 'gen']
    map_df = preprocess_text_df(map_df, cols=['orig', 'gen'])
    dict_map = dict(zip(map_df['orig'], map_df['gen']))

    s_orig = df_concat[col].map(normalize_spanish_text)
    s_gen  = s_orig.map(dict_map)

    tabla = pd.DataFrame({'orig': s_orig, 'gen': s_gen})
    # para detectar originales que apuntan a múltiples generales
    amb = (tabla.dropna()
                 .drop_duplicates()
                 .groupby('orig', as_index=False)['gen'].nunique()
                 .query('gen > 1'))
    return amb  # filas con conflictivos (orig → >1 gen)

###############################################################################
# ------------- CONCATENAR CSV + XLSX Y ENRIQUECER COLUMNAS ----------------- #
###############################################################################

def generar_reportes_mapeo(df_origen, mapping_df, columna='cluster', outdir='ALL_LANG_DATA', sample_n=10):
    os.makedirs(outdir, exist_ok=True)

    dict_map = build_mapping_dict(mapping_df)

    # Serie original normalizada
    s_orig_norm = df_origen[columna].map(normalize_spanish_text)
    s_mapped    = s_orig_norm.map(dict_map)

    base = df_origen.copy()
    base['cluster_original_norm'] = s_orig_norm
    base['cluster_general']       = s_mapped

    # 1) Pares mapeados usados
    pairs = (base.dropna(subset=['cluster_general'])
                .groupby(['cluster_original_norm','cluster_general'], as_index=False)
                .size()
                .rename(columns={'size':'conteo'})
                .sort_values(['cluster_general','conteo'], ascending=[True,False]))
    pairs.to_csv(os.path.join(outdir,'MAPPING_PAIRS.csv'), index=False)

    # 2) NO mapeados (resumen)
    unmapped_base = base[base['cluster_general'].isna()].copy()
    unmapped = (unmapped_base
                .groupby('cluster_original_norm', as_index=False)
                .size()
                .rename(columns={'size':'conteo'})
                .sort_values('conteo', ascending=False))
    unmapped.rename(columns={'cluster_original_norm':'cluster_original'}, inplace=True)
    unmapped.to_csv(os.path.join(outdir,'UNMAPPED_VALUES.csv'), index=False)

    # 2b) NO mapeados con ORIGEN
    cols_origen = [c for c in ['pais','sponsor','s3_route'] if c in base.columns]
    if cols_origen:
        unmapped_by_origin = (unmapped_base
            .groupby(['cluster_original_norm']+cols_origen, as_index=False)
            .size()
            .rename(columns={'size':'conteo'})
            .sort_values('conteo', ascending=False))
        unmapped_by_origin.rename(columns={'cluster_original_norm':'cluster_original'}, inplace=True)
        unmapped_by_origin.to_csv(os.path.join(outdir,'UNMAPPED_VALUES_BY_ORIGIN.csv'), index=False)

    # 3) Claves del mapping no usadas (comparar contra únicos NORMALIZADOS del dataset)
    usados = set(pairs['cluster_original_norm'].unique())
    definidos = set(build_mapping_dict(mapping_df).keys())  # ya normalizado/colapsado
    no_usados = sorted(definidos - usados)
    pd.DataFrame({'clave_mapping_no_usada': no_usados}).to_csv(
        os.path.join(outdir,'MAPPING_KEYS_UNUSED.csv'), index=False
    )

    # 4) Ambigüedad del mapping (orig → múltiples generales)
    amb = (base.dropna(subset=['cluster_general'])
               .drop_duplicates(subset=['cluster_original_norm','cluster_general'])
               .groupby('cluster_original_norm', as_index=False)['cluster_general']
               .nunique()
               .rename(columns={'cluster_general':'n_generales'})
               .query('n_generales > 1'))
    amb.rename(columns={'cluster_original_norm':'cluster_original'}, inplace=True)
    amb.to_csv(os.path.join(outdir,'MAPPING_AMBIGUO.csv'), index=False)

    # 5) Distribución por clase (sobre lo mapeado)
    dist = (base.dropna(subset=['cluster_general'])['cluster_general']
              .value_counts()
              .rename_axis('cluster_general')
              .reset_index(name='conteo'))
    total = dist['conteo'].sum()
    dist['porcentaje'] = (dist['conteo'] / max(total,1))*100
    dist.sort_values('conteo', ascending=False, inplace=True)
    dist.to_csv(os.path.join(outdir,'CLASS_DISTRIBUTION.csv'), index=False)

    # 6) Muestra estratificada (≥10 por clase)
    cols_show = [c for c in ['name', columna, 'cluster_general','s3_route','pais','sponsor'] if c in base.columns]
    df_show = base.dropna(subset=['cluster_general'])
    frames = []
    for k,g in df_show.groupby('cluster_general'):
        take = min(len(g), max(10, sample_n))
        frames.append(g.sample(n=take, random_state=7)[cols_show])
    sample = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols_show)
    sample.to_csv(os.path.join(outdir,'CLASS_STRATIFIED_SAMPLE.csv'), index=False)

    # 7) Auditoría de normalización del mapping (para cazar causas de "no usadas")
    #    Muestra el "raw" vs "normalizado" y si colapsó por duplicados.
    md = mapping_df.iloc[:, :2].copy()
    md.columns = ['orig_raw','gen_raw']
    md['orig_norm'] = md['orig_raw'].map(normalize_spanish_text)
    md['gen_norm']  = md['gen_raw'].map(normalize_spanish_text)
    md['duplicado_por_norm'] = md.duplicated(subset=['orig_norm'], keep='first')
    md.to_csv(os.path.join(outdir,'MAPPING_NORMALIZATION_AUDIT.csv'), index=False)

    print(f"[OK] Reportes en {outdir}:")
    print(" - MAPPING_PAIRS.csv")
    print(" - UNMAPPED_VALUES.csv")
    print(" - UNMAPPED_VALUES_BY_ORIGIN.csv")
    print(" - MAPPING_KEYS_UNUSED.csv")
    print(" - MAPPING_AMBIGUO.csv")
    print(" - CLASS_DISTRIBUTION.csv")
    print(" - CLASS_STRATIFIED_SAMPLE.csv")
    print(" - MAPPING_NORMALIZATION_AUDIT.csv")


import os, warnings, csv, pandas as pd
from tqdm import tqdm

EXPECTED = ['name', 'cluster']

def _read_csv_two_cols(path, encodings=('utf-8-sig','utf-8','cp1252','latin-1')):
    import csv
    EXPECTED = ['name','cluster']
    for enc in encodings:
        try:
            rows = []
            with open(path, encoding=enc, errors='replace', newline='') as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if not row: 
                        continue
                    joined = ','.join(row)
                    if ',' in joined:
                        name, cluster = joined.rsplit(',', 1)
                    else:
                        name, cluster = joined, ''
                    rows.append([name.strip(), cluster.strip()])
            df = pd.DataFrame(rows, columns=EXPECTED)
            # NORMALIZAR SIEMPRE
            df = preprocess_text_df(df, cols=EXPECTED)
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            warnings.warn(f"⚠️ {path} — salto por: {e}")
            return pd.DataFrame(columns=EXPECTED)
    warnings.warn(f"⚠️ {path} — no pudo abrirse con {encodings}")
    return pd.DataFrame(columns=EXPECTED)

def build_mapping_dict(mapping_df: pd.DataFrame) -> dict:
    """
    - Toma 2 primeras columnas (orig→gen)
    - Normaliza ambas
    - Elimina filas vacías
    - Colapsa duplicados por 'orig' (si hay conflicto, prioriza la 1ª aparición)
    """
    map_df = mapping_df.iloc[:, :2].copy()
    map_df.columns = ['orig', 'gen']

    # Normalizar SIEMPRE
    map_df = preprocess_text_df(map_df, cols=['orig','gen'])

    # Quitar vacíos
    map_df = map_df[(map_df['orig'].notna()) & (map_df['orig'].str.len()>0)]
    map_df = map_df[(map_df['gen'].notna()) & (map_df['gen'].str.len()>0)]

    # Colapsar duplicados por 'orig' conservando primera ocurrencia
    map_df = map_df.drop_duplicates(subset=['orig'], keep='first')

    return dict(zip(map_df['orig'], map_df['gen']))

def concatenate_data_files(folder_path, bucket_name):
    frames = []

    for root, _, files in os.walk(folder_path):
        for fname in files:
            low_path = os.path.join(root, fname).lower()
            full_path = os.path.join(root, fname)

            # ---------- CSV ----------
            if low_path.endswith('.csv'):
                if 'general_dictionary' in low_path or 'planta' in low_path:
                    continue
                df = _read_csv_two_cols(full_path)

            # ---------- XLSX ----------
            elif low_path.endswith('.xlsx'):
                if 'planta' in low_path:
                    continue
                try:
                    df = pd.read_excel(full_path, engine='openpyxl', dtype=str)
                except Exception as e:
                    warnings.warn(f"⚠️  {fname} — {e}")
                    continue

                if df.shape[1] < 2:
                    warnings.warn(f"⚠️  {fname} — solo {df.shape[1]} columna; se omite.")
                    continue

                df = df.iloc[:, :2]
                df.columns = EXPECTED
                df = preprocess_text_df(df, cols=EXPECTED)

            else:
                continue  # ni csv ni xlsx

            if df.empty:
                continue

            """rel_key  = os.path.relpath(full_path, folder_path).replace('\\', '/')
            s3_route = f"{bucket_name}/{rel_key}"
            parts    = s3_route.split('/')

            df['s3_route'] = s3_route
            df['pais']     = parts[1] if len(parts) > 1 else ''
            df['sponsor']  = parts[2] if len(parts) > 2 else '' """

            frames.append(df)

    cols_fin = EXPECTED  # + ['s3_route', 'pais', 'sponsor']
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols_fin)



     #####################################################################
# -------------------------- MAPEAR ETIQUETAS ------------------------------- #
     #####################################################################
def aplicar_mapping(df, mapping, columna='cluster'):
    """
    Reemplaza valores de `columna` según mapping (dict o DataFrame).
    - Normaliza tildes tanto en df[columna] como en el mapping.
    - Si mapping es DataFrame: 1a col = original, 2a col = general.
    """
    tmp = df.copy()

    if isinstance(mapping, pd.DataFrame):
        map_df = mapping.iloc[:, :2].copy()
        map_df.columns = ['orig', 'gen']
        map_df = preprocess_text_df(map_df, cols=['orig', 'gen'])
        mapping_dict = dict(zip(map_df['orig'], map_df['gen']))
    else:
        # Si es dict, normalizamos claves y valores
        mapping_dict = {
            normalize_spanish_text(k): normalize_spanish_text(v)
            for k, v in mapping.items()
        }

    # Normalizar la columna objetivo del DF antes del map
    if columna in tmp.columns:
        tmp[columna] = tmp[columna].map(normalize_spanish_text)
        tmp['__mapped__'] = tmp[columna].map(mapping_dict)
        tmp = tmp.dropna(subset=['__mapped__'])
        tmp[columna] = tmp['__mapped__']
        tmp = tmp.drop(columns='__mapped__')
    else:
        warnings.warn(f"⚠️  Columna '{columna}' no existe en df.")

    return tmp


def muestra_estratificada(df_mapped, col='cluster', n=50, random_state=7):
    frames = []
    for k, g in df_mapped.groupby(col):
        frames.append(g.sample(n=min(len(g), n), random_state=random_state))
    return pd.concat(frames, ignore_index=True)

def distribucion_por_clase(df_mapped, col='cluster'):
    dist = (df_mapped[col]
            .value_counts(dropna=False)
            .rename_axis('cluster_general')
            .reset_index(name='conteo'))
    dist['porcentaje'] = dist['conteo'] / dist['conteo'].sum() * 100
    return dist.sort_values('conteo', ascending=False)


###############################################################################
# -------------------------------- MAIN ------------------------------------- #
###############################################################################
if __name__ == "__main__":
    BUCKET = "matrices.aihub"
    OUTDIR = "ALL_LANG_DATA"

    download_data_files(
        bucket_name=BUCKET,
        local_output_folder=OUTDIR,
    )

    df_concat = concatenate_data_files(OUTDIR, BUCKET)
    df_concat.to_excel(os.path.join(OUTDIR, "TRAIN_MATRIX.xlsx"), index=False)
    print("Concatenación completada.")

    mapping_df = pd.read_excel('TOPIC_MAPPING.xlsx', dtype=str)
    mapping_df = preprocess_text_df(mapping_df, cols=list(mapping_df.columns[:2]))
    df_mapped = aplicar_mapping(df_concat, mapping_df, columna='cluster')
    df_mapped["cluster"] = df_mapped["cluster"].apply(lambda x: x.upper() if isinstance(x, str) else x)
    df_mapped.to_csv(os.path.join(OUTDIR, "MAPPED_TRAIN_MATRIX.csv"), index=False)
    print("Mapping aplicado y archivo guardado.")

    generar_reportes_mapeo(df_concat, mapping_df, columna='cluster', outdir=OUTDIR)


