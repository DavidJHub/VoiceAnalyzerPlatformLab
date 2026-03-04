import numpy as np
import pandas as pd

def _smooth_wpm(
    t,
    wpm,
    window_n: int = 11,     # impar recomendado: 11, 21, 31...
    sigma: float | None = None,
    interpolate_nans: bool = True,
):
    """
    Grafica una curva suave aplicando Gaussian blur 1D.

    - Si wpm tiene NaN (bins vacíos), opcionalmente interpola.
    - Gaussian blur con kernel normalizado y padding reflect para bordes.

    Retorna:
      t (np.ndarray), wpm_smooth (np.ndarray)
    """

    t = np.asarray(t, dtype=float)
    y = np.asarray(wpm, dtype=float)

    if len(t) != len(y):
        raise ValueError(f"t y wpm deben tener el mismo tamaño. Got {len(t)} vs {len(y)}")

    # 1) Interpolar NaNs (opcional)
    if interpolate_nans:
        mask = np.isfinite(y)
        if mask.sum() >= 2:
            y_filled = y.copy()
            y_filled[~mask] = np.interp(t[~mask], t[mask], y[mask])
            y = y_filled
        else:
            # Muy pocos puntos válidos: no hay qué suavizar
            y = np.nan_to_num(y, nan=0.0)

    # 2) Gaussian blur 1D
    if window_n < 3:
        y_smooth = y
    else:
        if window_n % 2 == 0:
            window_n += 1  # asegurar impar

        if sigma is None:
            sigma = window_n / 6.0  # regla común: 99% masa ~ +/-3sigma ~ ventana completa

        half = window_n // 2
        k = np.arange(-half, half + 1, dtype=float)
        kernel = np.exp(-(k**2) / (2 * sigma**2))
        kernel /= kernel.sum()

        ypad = np.pad(y, (half, half), mode="reflect")
        y_smooth = np.convolve(ypad, kernel, mode="valid")

    return t, y_smooth


def global_mean_wpm_vs_normalized_time(
    df: pd.DataFrame,
    times_col: str = "times_5s",
    wpm_col: str = "wpm_5s",
    n_bins: int = 50,              # puntos en [0,1], ej 101 -> paso 0.01
    agg: str = "mean",              # "mean" (recomendado) o "median"
    min_points_per_row: int = 2,    # ignora filas con menos puntos
):
    """
    Produce una curva global promedio WPM vs tiempo normalizado.

    Pasos:
      1) Por fila: normaliza times_5s a [0,1] dividiendo por el último tiempo (T).
      2) Binning a grilla común en [0,1].
      3) Promedia WPM por bin sobre todas las filas.

    Retorna:
      t_grid: (n_bins,) tiempos normalizados
      wpm_global: (n_bins,) promedio global por bin (NaN si bin vacío)
      n_per_bin: (n_bins,) cantidad de aportes por bin
    """
    t_grid = np.linspace(0.0, 1.0, n_bins)
    sum_wpm = np.zeros(n_bins, dtype=float)
    sum_wpm2 = np.zeros(n_bins, dtype=float)  # por si luego quieres std
    n_per_bin = np.zeros(n_bins, dtype=int)

    for _, row in df.iterrows():
        times = row.get(times_col)
        wpm = row.get(wpm_col)

        if times is None or wpm is None:
            continue

        times = np.asarray(times, dtype=float)
        wpm = np.asarray(wpm, dtype=float)

        if len(times) < min_points_per_row or len(wpm) < min_points_per_row:
            continue

        L = min(len(times), len(wpm))
        times = times[:L]
        wpm = wpm[:L]

        T = times[-1]
        if not np.isfinite(T) or T <= 0:
            continue

        t_norm = times / T
        m = np.isfinite(t_norm) & np.isfinite(wpm)
        t_norm = t_norm[m]
        wpm = wpm[m]
        if len(t_norm) == 0:
            continue

        # Asignación a bins (0..n_bins-1) por redondeo a grilla uniforme
        # (equivalente a "binarizar" en una grilla común)
        idx = np.rint(t_norm * (n_bins - 1)).astype(int)
        idx = np.clip(idx, 0, n_bins - 1)

        if agg == "median":
            # Para mediana global: acumular listas por bin (más costoso)
            # -> aquí dejamos mean como estándar; si necesitas mediana real te la hago.
            raise NotImplementedError("agg='median' requiere acumulación por bin; te lo implemento si lo necesitas.")
        else:
            np.add.at(sum_wpm, idx, wpm)
            np.add.at(sum_wpm2, idx, wpm * wpm)
            np.add.at(n_per_bin, idx, 1)

    wpm_global = np.full(n_bins, np.nan, dtype=float)
    mask = n_per_bin > 0
    wpm_global[mask] = sum_wpm[mask] / n_per_bin[mask]

    return t_grid, wpm_global, n_per_bin



def buildGraphData(audioDataFrame: pd.DataFrame, n_bins: int = 50):
    t_norm, wpm_mean, _ = global_mean_wpm_vs_normalized_time(audioDataFrame,n_bins=n_bins)
    x,y=_smooth_wpm(
            t_norm,
            wpm_mean,
            7,     
            )
    return x,y