from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Tuple, Optional

def calibrate_thresholds(
    df: pd.DataFrame,
    percent: int = 20
) -> Tuple[float, float, Optional[float]]:
    """
    Retorna:
      - price_lh_threshold
      - mac_lh_threshold
      - general_threshold (solo se usa si la única solución exacta fuerza thresholds "iguales")
        Si no se usa, retorna None.

    Regla de conteo:
      selected = (VACIA == 1) OR (best_price_likelihood < price_th) OR (best_mac_likelihood < mac_th)
    """
    required_cols = {"VACIA", "best_price_likelihood", "best_mac_likelihood"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    if not isinstance(percent, int) or not (0 <= percent <= 100):
        raise ValueError("percent debe ser un entero entre 0 y 100.")

    total_calls_number = int(len(df))
    calls_amount_target = int(math.ceil(total_calls_number * (percent / 100.0)))

    print(f"Calibrando thresholds para seleccionar {calls_amount_target} llamadas de {total_calls_number}, equivalente al ({percent}%)")

    vacia = df["VACIA"].fillna(0)
    # Si viene como texto, lo intentamos interpretar
    try:
        vacia = vacia.astype(int)
    except Exception:
        vacia = pd.to_numeric(vacia, errors="coerce").fillna(0).astype(int)

    vacia_mask = (vacia == 1)
    empty_calls_number = int(vacia_mask.sum())

    # Caso trivial: target 0
    if calls_amount_target == 0:
        return 0.0, np.nextafter(0.0, 1.0), None

    # Si VACIA ya cumple o excede el target, no necesitamos "abrir" thresholds
    if empty_calls_number >= calls_amount_target:
        # thresholds mínimos para no sumar casi nada extra
        price_th = 0.0
        mac_th = np.nextafter(0.0, 1.0)  # distinto
        return float(price_th), float(mac_th), None

    # Likelihoods como float (NaN si no parsea)
    price = pd.to_numeric(df["best_price_likelihood"], errors="coerce")
    mac = pd.to_numeric(df["best_mac_likelihood"], errors="coerce")

    # Solo filas NO VACIA para calibrar el "faltante"
    non_empty_mask = ~vacia_mask

    # Construimos un "min_like" por fila para generar candidatos de frontera (boundary values)
    # Si ambos NaN => +inf (nunca contará por threshold)
    min_like = np.minimum(
        price.where(price.notna(), np.inf).to_numpy(dtype=float),
        mac.where(mac.notna(), np.inf).to_numpy(dtype=float),
    )

    min_like_nonempty = min_like[non_empty_mask.to_numpy()]
    finite_vals = min_like_nonempty[np.isfinite(min_like_nonempty)]
    if finite_vals.size == 0:
        # No hay manera de sumar llamadas por thresholds (todo NaN), así que no se puede alcanzar target
        # Retornamos thresholds que no agregan nada y general_threshold None
        return 0.0, np.nextafter(0.0, 1.0), None

    # Candidatos de frontera: valores únicos ordenados del min_like
    boundary_vals = np.unique(finite_vals)

    # Helper para contar seleccionadas dado thresholds
    non_empty_mask_np = non_empty_mask.to_numpy()
    price_np = price.to_numpy(dtype=float)
    mac_np = mac.to_numpy(dtype=float)

    def count_selected(price_th: float, mac_th: float) -> int:
        # Nota: NaN < threshold => False (por eso chequeamos isfinite)
        sel_nonempty = (
            (np.isfinite(price_np) & (price_np < price_th)) |
            (np.isfinite(mac_np) & (mac_np < mac_th))
        ) & non_empty_mask_np
        return int(empty_calls_number + sel_nonempty.sum())

    best_exact_distinct = None  # (price_th, mac_th)
    best_exact_equal = None     # (th, th)
    best_approx = None          # (abs_diff, price_th, mac_th)

    for v in boundary_vals:
        # v como frontera: threshold==v incluye "< v", y threshold==nextafter(v,1) incluye "<= v" efectivamente
        v_up = float(np.nextafter(v, 1.0))

        candidates = [
            (float(v), float(v)),         # ninguno incluye ==v
            (float(v_up), float(v)),      # incluye ==v por price
            (float(v), float(v_up)),      # incluye ==v por mac
            (float(v_up), float(v_up)),   # incluye ==v por ambos
        ]

        for p_th, m_th in candidates:
            total_sel = count_selected(p_th, m_th)
            diff = abs(total_sel - calls_amount_target)

            # guardar aproximación por si no existe exacta
            if best_approx is None or diff < best_approx[0]:
                best_approx = (diff, p_th, m_th)

            if total_sel == calls_amount_target:
                if p_th != m_th:
                    best_exact_distinct = (p_th, m_th)
                    break
                else:
                    best_exact_equal = (p_th, m_th)
        if best_exact_distinct is not None:
            break

    if best_exact_distinct is not None:
        price_th, mac_th = best_exact_distinct
        return float(price_th), float(mac_th), None

    # Si solo existe solución exacta con thresholds "iguales"
    if best_exact_equal is not None:
        th = float(best_exact_equal[0])

        # Intentamos “forzar” diferencia mínima sin cambiar el conteo (si es posible)
        # Si th < 1, subir un hair a uno de ellos normalmente NO cambia nada
        # salvo que existan valores en (th, nextafter(th,1)) (casi imposible con floats típicos).
        if th < 1.0:
            price_th = th
            mac_th = float(np.nextafter(th, 1.0))
        else:
            # th == 1.0: uno hacia abajo
            price_th = float(np.nextafter(th, 0.0))
            mac_th = th

        # general_threshold reporta el "escenario igual"
        general_threshold = th
        return float(price_th), float(mac_th), float(general_threshold)

    # Si no hay forma de llegar exacto (limitación matemática por empates/NaNs),
    # retornamos la mejor aproximación
    _, p_th, m_th = best_approx
    # Asegura que sean distintos
    if p_th == m_th:
        m_th = float(np.nextafter(m_th, 1.0)) if m_th < 1.0 else float(np.nextafter(m_th, 0.0))
    return float(p_th), float(m_th), None
