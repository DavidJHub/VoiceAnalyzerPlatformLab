import math
import pandas as pd
from typing import Iterable, List, Optional, Union

NumberLike = Union[int, float, str]

def _to_ratio(x: NumberLike) -> Optional[float]:
    """
    Convierte x a 'ratio' en [0,1] cuando sea posible.
    Acepta: 0.74, 74, '74%', '0.74', etc.
    Retorna None si no se puede convertir.
    """
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None

    # Strings como "74%" o " 0.74 "
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        if s.endswith("%"):
            s = s[:-1].strip()
            try:
                return float(s) / 100.0
            except ValueError:
                return None
        try:
            x = float(s)
        except ValueError:
            return None

    # Numéricos
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None

    # Heurística: si viene como 0-1 => ratio; si viene como 0-100 => porcentaje
    if v > 1.0:
        return v / 100.0
    return v


def filter_low_likelihood_rows(
    df: pd.DataFrame,
    mac_col: str = "best_mac_likelihood_macs",
    price_col: str = "best_mac_likelihood_prices",
    mac_threshold_pct: float = 35.0,
    price_threshold_pct: float = 30.0,
    include_na_as_fail: bool = False,
) -> pd.DataFrame:
    """
    Retorna las filas donde:
      - best_mac_likelihood_macs < 65%  OR
      - best_mac_likelihood_prices < 60%

    mac_threshold_pct y price_threshold_pct van en % (65, 60).
    """
    if mac_col not in df.columns:
        raise KeyError(f"Falta columna '{mac_col}' en el DataFrame.")
    if price_col not in df.columns:
        raise KeyError(f"Falta columna '{price_col}' en el DataFrame.")

    mac_thr = mac_threshold_pct / 100.0
    price_thr = price_threshold_pct / 100.0

    mac_ratio = df[mac_col].map(_to_ratio)
    price_ratio = df[price_col].map(_to_ratio)

    if include_na_as_fail:
        mac_fail = mac_ratio.isna() | (mac_ratio < mac_thr)
        price_fail = price_ratio.isna() | (price_ratio < price_thr)
    else:
        mac_fail = mac_ratio.notna() & (mac_ratio < mac_thr)
        price_fail = price_ratio.notna() & (price_ratio < price_thr)

    return df[mac_fail | price_fail].copy()


def clear_summary_rejection_for_agent_audio_ids(
    conexion,
    agent_audio_ids: Iterable[Union[int, float, str]],
    chunk_size: int = 800,
    set_to_null: bool = False,
) -> int:
    """
    Dada una conexión ya establecida, hace JOIN:
      agent_audio_data.id = call_affecteds.agent_audio_data_id
    y para esos IDs, vacía call_affecteds.summary_rejection ('' o NULL).

    Retorna: número de filas afectadas (según cursor.rowcount acumulado).
    """
    # Normaliza ids a int (ignorando nulos/no convertibles)
    ids: List[int] = []
    for x in agent_audio_ids:
        if x is None:
            continue
        try:
            if isinstance(x, float) and math.isnan(x):
                continue
        except Exception:
            pass
        try:
            ids.append(int(float(x)))
        except Exception:
            continue

    if not ids:
        return 0

    value_expr = "NULL" if set_to_null else "''"

    total_updated = 0
    with conexion.cursor() as cur:
        # Actualiza por chunks para evitar límites de placeholders
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i : i + chunk_size]
            placeholders = ",".join(["%s"] * len(chunk))

            sql = f"""
            UPDATE call_affecteds ca
            INNER JOIN agent_audio_data aad
                ON aad.id = ca.agent_audio_data_id
            SET ca.summary_rejection = {value_expr}
            WHERE aad.lead_id IN ({placeholders})
            """

            cur.execute(sql, chunk)
            total_updated += cur.rowcount

    conexion.commit()
    return total_updated

if __name__ == "__main__":
    import pandas as pd
    import database.dbConfig as dbcfg

    df = pd.read_excel("result_rejected_general.xlsx")

    # 1) Filtrar filas
    df_low = filter_low_likelihood_rows(df)
    df_low.to_excel("test.xlsx")
    # 2) Conexión (ya la tienes así)
    conexion = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )

    # 3) Vaciar summary_rejection en call_affecteds para los ids filtrados
    updated = clear_summary_rejection_for_agent_audio_ids(conexion, df_low["LEAD_ID"])
    print("Filas call_affecteds actualizadas:", updated)