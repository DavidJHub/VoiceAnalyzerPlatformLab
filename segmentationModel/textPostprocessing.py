import re
import numpy as np
import os
import pandas as pd
import json
from lang.mvdModule import palabras_a_digitos


import numpy as np
import pandas as pd

# 1) Robustly infer which columns are per-class probabilities
def infer_prob_cols(df: pd.DataFrame):
    """
    Return columns that look like per-class probabilities:
    - float dtype (or numeric coercible)
    - not known metadata columns
    """
    exclude = {
        "text","start","end","turn_idx",
        "predicted_cluster","predicted_cluster_smooth",
        "conversation_id","call_id","time","time_sec",
        "call_duration_sec","rel_time","time_tag","time_bin"
    }
    # start with floats
    prob_like = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    if prob_like:
        return prob_like

    # fallback: try coercing to numeric and keep those with at least some numeric values
    prob_like = []
    for c in df.columns:
        if c in exclude: 
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            prob_like.append(c)
    return prob_like

# 2) Safely get the second-best label for row idx
def seleccionarSegundaMejor(df: pd.DataFrame, idx: int, prob_cols=None):
    if prob_cols is None:
        prob_cols = infer_prob_cols(df)

    # take only prob columns, force floats, drop NaNs
    row_probs = pd.to_numeric(df.loc[idx, prob_cols], errors="coerce").dropna()

    if row_probs.empty:
        # fallback: keep current prediction
        return df.at[idx, "predicted_cluster"]

    # drop the current predicted label first, then take the best among the rest
    curr = df.at[idx, "predicted_cluster"]
    if curr in row_probs.index and len(row_probs) >= 2:
        row_probs = row_probs.drop(index=curr, errors="ignore")

    if row_probs.empty:
        return df.at[idx, "predicted_cluster"]

    return row_probs.idxmax()


def ajustarSaludoDespedida(df: pd.DataFrame,
                           saludo_max_turn: int = 6,
                           despedida_min_turn: int = 20) -> pd.DataFrame:
    """
    Heuristics:
      - If model predicts SALUDO but it's too late in the call, replace with 2nd-best.
      - If model predicts DESPEDIDA but it's too early, replace with 2nd-best.
    Tune thresholds as you wish.
    """
    df = df.copy()
    prob_cols = infer_prob_cols(df)

    if "turn_idx" not in df.columns:
        df["turn_idx"] = np.arange(len(df))

    # SALUDO too late -> pick 2nd best
    m_saludo = (df["predicted_cluster"].str.lower() == "saludo") & (df["turn_idx"] > saludo_max_turn)
    for idx in df.index[m_saludo]:
        df.at[idx, "predicted_cluster"] = seleccionarSegundaMejor(df, idx, prob_cols)

    # DESPEDIDA too early -> pick 2nd best
    m_desp = (df["predicted_cluster"].str.lower() == "despedida") & (df["turn_idx"] < despedida_min_turn)
    for idx in df.index[m_desp]:
        df.at[idx, "predicted_cluster"] = seleccionarSegundaMejor(df, idx, prob_cols)

    return df

# -----------------------------------------------------------------------------

def reconstruirDialogos(directorio_original: str,
                        directorio_procesado: str,
                        directorio_salida: str):
    """
    Reconstruye la estructura de los diálogos originales y agrega 'MVD'
    cuando el tema es CONFIRMACION DATOS.
    - 'MVD' = lista de números (strings) extraídos del texto del fragmento
      tras convertir palabras numéricas a dígitos con `palabras_a_digitos()`.
    *Se asume que 'text' existe en df_original y que `palabras_a_digitos()`
     ya está definida en tu entorno.*
    """

    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)

    archivos_originales = {
        os.path.splitext(f)[0]: os.path.join(directorio_original, f)
        for f in os.listdir(directorio_original)
        if f.endswith(".csv")
    }

    archivos_procesados = {
        os.path.splitext(f)[0]: os.path.join(directorio_procesado, f)
        for f in os.listdir(directorio_procesado)
        if f.endswith(".csv")
    }

    for nombre_base, ruta_origen in archivos_originales.items():
        if nombre_base not in archivos_procesados:
            print(f"No existe archivo procesado para: {nombre_base}")
            continue

        ruta_procesado = archivos_procesados[nombre_base]

        # 1) Cargar ambos DataFrames
        df_original = pd.read_csv(ruta_origen)
        df_procesado = pd.read_csv(ruta_procesado)

        # 2) Ajustar SALUDO / DESPEDIDA en df_procesado
        df_procesado.sort_values("start", inplace=True)
        df_procesado = ajustarSaludoDespedida(df_procesado)  # ya existente

        # 3) Reconstruir con la nueva versión de df_procesado
        df_original.sort_values("start", inplace=True)

        resultados = []

        for _, row_orig in df_original.iterrows():
            fragment_start = row_orig["start"]
            fragment_end = row_orig["end"]
            frag_text = row_orig.get("text", "")  # <=== usamos SOLO 'text'

            # Ventanas solapadas
            solapadas = df_procesado[
                (df_procesado["start"] < fragment_end) &
                (df_procesado["end"] > fragment_start)
            ].copy()
            solapadas.sort_values("start", inplace=True)

            # Construir secuencia de temas
            temas_en_orden = []
            if len(solapadas) == 0:
                temas_en_orden.append({"topic": "SIN_INFO", "confidence": 0.0})
            else:
                tema_actual = None
                suma_conf = 0.0
                contador = 0

                def _cerrar_e_insertar(tema, suma, cnt):
                    prom_conf = suma / cnt if cnt else 0.0
                    item = {"topic": tema, "confidence": prom_conf}
                    temas_en_orden.append(item)

                for _, row_vent in solapadas.iterrows():
                    tema_vent = row_vent["predicted_cluster"]

                    # MAC_DEF / PRECIO_DEF
                    if tema_vent == "MAC_DEF":
                        conf_vent = row_vent.get('MAC', 0.0)
                    elif tema_vent == "PRECIO_DEF":
                        conf_vent = row_vent.get('PRECIO', 0.0)
                    else:
                        if (tema_vent in df_procesado.columns and
                            tema_vent not in ('MAC_DEF', 'PRECIO_DEF')):
                            conf_vent = row_vent.get(tema_vent, 0.0)
                        else:
                            conf_vent = 0.0

                    if tema_actual is None:
                        tema_actual = tema_vent
                        suma_conf = conf_vent
                        contador = 1
                    else:
                        if tema_vent == tema_actual:
                            suma_conf += conf_vent
                            contador += 1
                        else:
                            _cerrar_e_insertar(tema_actual, suma_conf, contador)
                            tema_actual = tema_vent
                            suma_conf = conf_vent
                            contador = 1

                if tema_actual is not None and contador > 0:
                    _cerrar_e_insertar(tema_actual, suma_conf, contador)

            # Guardar fila resultante
            nueva_fila = {col: row_orig[col] for col in df_original.columns}
            nueva_fila["topics_sequence"] = json.dumps(temas_en_orden, ensure_ascii=False)
            resultados.append(nueva_fila)

        df_resultado = pd.DataFrame(resultados)
        nombre_salida = f"{nombre_base}_reconstruido.csv"
        ruta_salida = os.path.join(directorio_salida, nombre_salida)
        df_resultado.to_csv(ruta_salida, index=False)
        print(f"Reconstruido: {ruta_salida}")


import json


def marcar_mac_price_def(df, topics_col='topics_sequence'):
    """
    Lee la columna topics_col (que contiene JSON con [{"topic":..., "confidence":...}]),
    agrupa filas consecutivas con MAC y PRECIO, busca la de mayor suma de confianza
    y marca esas filas como MAC_DEF o PRICE_DEF.
    Si no están en ese grupo ganador, escoge el tema con mayor confianza de cada fila.
    se toma el mínimo confidence de MAC y PRECIO por JSON, y la suma final se mantiene con esos valores.

    Param:
      df : DataFrame con la columna 'topics_col'
      topics_col : nombre de la columna con el JSON (por defecto 'topics_sequence')
    Salida:
      DataFrame con una nueva columna 'final_label'
    """

    # 1) Parsear el JSON en cada fila
    parsed_topics = []
    for idx, row in df.iterrows():
        contenido_json = row[topics_col]
        try:
            lista_temas = json.loads(contenido_json)
        except:
            lista_temas = []
        parsed_topics.append(lista_temas)

    df['parsed_topics'] = parsed_topics

    # 2) Calcular el mínimo confidence de MAC y PRECIO por fila
    mac_conf = []
    price_conf = []
    for lista_temas in parsed_topics:
        # Obtener el mínimo confidence de MAC y PRECIO si existen
        mac_vals = [t['confidence'] for t in lista_temas if t['topic'] == "MAC"]
        price_vals = [t['confidence'] for t in lista_temas if t['topic'] in ["PRECIO", "PRICE"]]

        min_mac = min(mac_vals) if mac_vals else 0.0
        min_price = min(price_vals) if price_vals else 0.0

        mac_conf.append(min_mac)
        price_conf.append(min_price)

    df['mac_conf'] = mac_conf
    df['price_conf'] = price_conf

    # 3) Hallar el grupo consecutivo con la mayor suma de MAC
    df['mac_group_id'] = agrupar_consecutivos(df['mac_conf'])
    grupo_mac_sum = df.groupby('mac_group_id')['mac_conf'].sum()
    grupo_mac_sum = grupo_mac_sum.drop(labels=0, errors='ignore')
    best_mac_group = grupo_mac_sum.idxmax() if len(grupo_mac_sum) > 0 else None

    # 4) Hallar el grupo consecutivo con la mayor suma de PRECIO
    df['price_group_id'] = agrupar_consecutivos(df['price_conf'])
    grupo_price_sum = df.groupby('price_group_id')['price_conf'].sum()
    grupo_price_sum = grupo_price_sum.drop(labels=0, errors='ignore')
    best_price_group = grupo_price_sum.idxmax() if len(grupo_price_sum) > 0 else None

    # 5) Asignar la etiqueta final
    final_labels = []
    for idx, row in df.iterrows():
        mg = row['mac_group_id']
        pg = row['price_group_id']
        if mg == best_mac_group and mg != 0:
            final_labels.append("MAC_DEF")
        elif pg == best_price_group and pg != 0:
            final_labels.append("PRECIO_DEF")
        else:
            temas = row['parsed_topics']
            if not temas:
                final_labels.append("SIN_INFO")
            else:
                best_tema = max(temas, key=lambda x: x['confidence'])
                final_labels.append(best_tema['topic'])

    df['final_label'] = final_labels

    # Limpiar columnas auxiliares
    df.drop(columns=['parsed_topics'], inplace=True)

    return df

def agrupar_consecutivos(conf_series):
    """
    Asigna un "group_id" positivo para cada fila que tenga conf_series>0
    y sea consecutiva con la anterior.
    0 => no pertenece a un grupo (conf=0)
    1,2,3,... => ID del grupo consecutivo.
    """
    group_ids = []
    current_group = 0
    previous_nonzero = False

    for val in conf_series:
        if val > 0:
            if not previous_nonzero:
                # Abrimos un nuevo grupo
                current_group += 1
            group_ids.append(current_group)
            previous_nonzero = True
        else:
            # val=0 => no forma parte de un grupo
            group_ids.append(0)
            previous_nonzero = False

    return group_ids



def process_directory_mac_price_def(input_dir: str,
                                    output_dir: str,
                                    topics_col: str = 'topics_sequence'):
    """
    Lee todos los archivos CSV de 'input_dir', aplica la función 'marcar_mac_price_def'
    para marcar MAC_DEF y PRECIO_DEF, y guarda el resultado en 'output_dir'.

    Parámetros:
    -----------
    input_dir : str
        Directorio donde se encuentran los CSV originales.
    output_dir : str
        Directorio donde se guardarán los CSV procesados.
    topics_col : str
        Nombre de la columna que contiene el JSON con los topics (por defecto 'topics_sequence').
    """

    # Asegurarse de que el directorio de salida exista
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Listar todos los CSV en 'input_dir'
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"No se encontraron archivos CSV en {input_dir}")
        return

    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(csv_file)[0]}.csv")
        try:
            df = pd.read_csv(input_path)
            # Llamamos a la función que marca MAC_DEF / PRECIO_DEF
            df_processed = marcar_mac_price_def(df, topics_col=topics_col)
            # Guardar el DataFrame resultante
            df_processed.to_csv(output_path, index=False)
            print(f"Procesado y guardado: {output_path}")

        except Exception as e:
            print(f"Error procesando {input_path}: {e}")




if __name__ == "__main__":
    dir_original = "BANCOLSE"
    dir_procesado = "BANCOLSE_PREDICTED"
    dir_salida = "BANCOLSE_RECONSTRUIDO"

    # Reconstruir el diálogo agrupando y promediando ventanas
    reconstruirDialogos(dir_original, dir_procesado, dir_salida)
    df_res = process_directory_mac_price_def(dir_salida,dir_salida,topics_col='topics_sequence')