import pandas as pd
import os, re

def calcular_promedio_venta_consciente(df: pd.DataFrame) -> float:
    """
    Recibe un DataFrame, busca la columna 'porcentaje_venta_consciente',
    calcula y retorna su promedio.

    Args:
        df (pd.DataFrame): DataFrame con la columna 'porcentaje_venta_consciente'

    Returns:
        float: Promedio de la columna 'porcentaje_venta_consciente'

    Raises:
        ValueError: Si la columna no existe en el DataFrame.
    """

    columna = "porcentaje_venta_consciente"

    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    # Convertir a numérico por seguridad (maneja strings o valores raros)
    serie_numerica = pd.to_numeric(df[columna], errors="coerce")

    # Calcular promedio (ignorando NaN)
    promedio = serie_numerica.mean()

    return promedio


def merge_consent_into_campaign(
    MAT_CALLS_CAMPAIGN: pd.DataFrame,
    consent_evaluation_df: pd.DataFrame,
    campaign_key: str = "file_name",
    consent_key: str = "Nombre llamada",
    drop_consent_key_after_merge: bool = True,
    handle_consent_duplicates: str = "first",  # "first" | "error"
) -> pd.DataFrame:
    """
    Retorna MAT_CALLS_CAMPAIGN con las columnas de consent_evaluation_df agregadas,
    mapeando consent_key -> campaign_key, incluso si difieren en sufijos/extensiones:
      - MAT:  ... .wav / .mp3 / ...
      - CONS: ... _transcript.json

    - Mantiene todas las filas de MAT_CALLS_CAMPAIGN (left join).
    - Si una llamada no existe en consent_evaluation_df, las nuevas columnas quedan NaN.
    - Controla duplicados en consent_evaluation_df por la key normalizada (opcional).
    """

    def build_merge_key_from_campaign_filename(x: str) -> str:
        """
        Ej:
        '..._169.wav' -> '..._169'
        '...-all.mp3' -> '...-all'
        """
        name=x.split("-all")[0] +"-all.mp3"  # quita extensión
        return name


    mat = MAT_CALLS_CAMPAIGN.copy()
    cons = consent_evaluation_df.copy()

    # Validaciones básicas
    if campaign_key not in mat.columns:
        raise KeyError(f"'{campaign_key}' no existe en MAT_CALLS_CAMPAIGN. Columnas: {list(mat.columns)}")
    if consent_key not in cons.columns:
        raise KeyError(f"'{consent_key}' no existe en consent_evaluation_df. Columnas: {list(cons.columns)}")

    # Normaliza a string (strip)
    mat[campaign_key] = mat[campaign_key].astype(str).str.strip()
    cons[consent_key] = cons[consent_key].astype(str).str.strip()

    # Construye llave normalizada para hacer match real
    _mat_merge_key = "__merge_key__"
    _cons_merge_key = "__merge_key__"
    mat[_mat_merge_key] = mat[campaign_key].map(build_merge_key_from_campaign_filename)
    cons[_cons_merge_key] = cons[consent_key].map(build_merge_key_from_campaign_filename)

    # Manejo de duplicados en consent por la llave normalizada
    dup_mask = cons[_cons_merge_key].duplicated(keep=False)
    if dup_mask.any():
        dups = cons.loc[dup_mask, _cons_merge_key].value_counts()
        if handle_consent_duplicates == "error":
            raise ValueError(
                "Hay duplicados en consent_evaluation_df para la key normalizada "
                f"'{_cons_merge_key}'. Ejemplos:\n{dups.head(10)}"
            )
        elif handle_consent_duplicates == "first":
            cons = cons.drop_duplicates(subset=[_cons_merge_key], keep="first")

    # Trae todas las columnas del consent excepto la key original (más limpio),
    # pero conservando la merge key normalizada
    consent_cols_to_add = [c for c in cons.columns if c not in (consent_key, _cons_merge_key)]
    cons_reduced = cons[[_cons_merge_key] + consent_cols_to_add]

    merged = mat.merge(
        cons_reduced,
        how="left",
        left_on=_mat_merge_key,
        right_on=_cons_merge_key,
    )

    # Limpieza
    merged = merged.drop(columns=[_mat_merge_key, _cons_merge_key], errors="ignore")
    if drop_consent_key_after_merge and consent_key in merged.columns:
        merged = merged.drop(columns=[consent_key])

    return merged
