import math
import pandas as pd
from typing import Optional, Union, Dict, List

NumberLike = Union[int, float, str]

def _to_ratio(x: NumberLike) -> Optional[float]:
    """Convierte a ratio [0,1] desde 0.74, 74, '74%', etc."""
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None

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

    try:
        v = float(x)
    except (TypeError, ValueError):
        return None

    return v / 100.0 if v > 1.0 else v


def build_rejection_map_by_lead_id(
    df: pd.DataFrame,
    lead_col: str = "lead_id",
    mac_col: str = "best_mac_likelihood_macs",
    price_col: str = "best_mac_likelihood_prices",
    mac_threshold_pct: float = 65.0,
    price_threshold_pct: float = 60.0,
    include_na_as_fail: bool = False,
    both_separator: str = ", ",
) -> Dict[int, str]:
    """
    Devuelve dict {lead_id: "MAC INCORRECTO" / "PRECIO INCORRECTO" / "MAC INCORRECTO | PRECIO INCORRECTO"}
    Si un lead aparece múltiples veces, agrega flags (OR lógico): si alguna fila falla MAC, el lead falla MAC.
    """
    for c in (lead_col, mac_col, price_col):
        if c not in df.columns:
            raise KeyError(f"Falta columna '{c}' en el DataFrame.")

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

    # Nos quedamos solo con filas que fallan algo
    df_fail = df[mac_fail | price_fail].copy()

    # Acumular por lead_id (OR)
    acc: Dict[int, Dict[str, bool]] = {}
    for idx, row in df_fail.iterrows():
        lead = row.get(lead_col)
        try:
            if lead is None or (isinstance(lead, float) and math.isnan(lead)):
                continue
            lead_int = int(float(lead))
        except Exception:
            continue

        if lead_int not in acc:
            acc[lead_int] = {"mac": False, "price": False}

        if bool(mac_fail.loc[idx]):
            acc[lead_int]["mac"] = True
        if bool(price_fail.loc[idx]):
            acc[lead_int]["price"] = True

    # Convertir flags a string final
    out: Dict[int, str] = {}
    for lead_int, flags in acc.items():
        parts: List[str] = []
        if flags["mac"]:
            parts.append("MAC INCORRECTO")
        if flags["price"]:
            parts.append("PRECIO INCORRECTO")
        if parts:
            out[lead_int] = both_separator.join(parts)

    return out

def upsert_summary_rejection_by_lead_id(
    conexion,
    rejection_map: Dict[int, str],
    aad_lead_col: str = "LEAD_ID",
    chunk_size: int = 300,
) -> int:
    """
    UPDATE call_affecteds usando JOIN:
      aad.id = ca.agent_audio_data_id
    pero filtrando por lead_id:
      aad.<aad_lead_col> IN (...)

    Escribe ca.summary_rejection según lead_id (CASE WHEN).
    Retorna total de filas afectadas (rowcount acumulado).
    """
    if not rejection_map:
        return 0

    lead_ids = list(rejection_map.keys())
    total_updated = 0

    with conexion.cursor() as cur:
        for i in range(0, len(lead_ids), chunk_size):
            chunk = lead_ids[i:i+chunk_size]

            case_clauses = []
            params: List[Union[int, str]] = []
            for lid in chunk:
                case_clauses.append(f"WHEN aad.{aad_lead_col} = %s THEN %s")
                params.extend([lid, rejection_map[lid]])

            case_sql = "CASE " + " ".join(case_clauses) + " ELSE ca.summary_rejection END"
            placeholders = ",".join(["%s"] * len(chunk))
            params.extend(chunk)

            sql = f"""
            UPDATE call_affecteds ca
            INNER JOIN agent_audio_data aad
                ON aad.id = ca.agent_audio_data_id
            SET ca.summary_rejection = {case_sql}
            WHERE aad.{aad_lead_col} IN ({placeholders})
            """

            cur.execute(sql, params)
            total_updated += cur.rowcount

    conexion.commit()
    return total_updated


if __name__ == "__main__":
    import pandas as pd
    import database.dbConfig as dbcfg
    df = pd.read_excel("result_rejected_general.xlsx")
    conexion = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )

    rejection_map = build_rejection_map_by_lead_id(
        df,
        lead_col="LEAD_ID",
        mac_threshold_pct=25,
        price_threshold_pct=30,
        include_na_as_fail=True
    )

    updated = upsert_summary_rejection_by_lead_id(
        conexion,
        rejection_map,
        aad_lead_col="lead_id"
    )

    print("Filas call_affecteds actualizadas:", updated)