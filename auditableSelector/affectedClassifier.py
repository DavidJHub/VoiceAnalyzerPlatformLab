import pandas as pd
from utils.VapUtils import _sec_to_min_sec

def combinedRejectionReason(df: pd.DataFrame,
                            all_musnt_keywords: list,
                            CUT: float,
                            MAC_CUT: float,
                            PRICE_CUT: float) -> pd.DataFrame:
    """
    Aplica un filtro global de rechazo y genera columnas 'reject_reason', 
    'warning_reason' y 'reject' con base en las condiciones combinadas.
    """

    # Convertir fechas
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='%Y%m%d-%H%M%S', errors='coerce')
    df['mac_r'] = df['mac_r'].fillna(0)
    df['terms'] = df['terms'].fillna(0)
    df['igs_comp'] = df['igs_comp'].fillna(0)
    # Medias precalculadas
    score_mean = df['score'].mean()
    TMO_mean = df['TMO'].mean()
    forbidden_rate_mean = df['forbidden_rate'].mean()
    df['hangup_signaturetime']=df['hangup_signaturetime'].fillna(df['TMO'])
    # Filtro global de rechazo (OR de todas las condiciones)
    def evaluate_reasons(row):
        rej_reason = ''
        warning = ''
        rej = False
        # MAC modulado
        if (row['velocity_classification_macs'] == 'high'):
            warning += 'MAC AFECTADO - VOZ ACELERADA, '

        if (row['volume_classification_mac'] == 'low'):
            warning += 'MAC AFECTADO - VOLUMEN BAJO, '

        if (row['velocity_classification_prices'] == 'high'):
            warning += 'PRECIO AFECTADO - VOZ ACELERADA, '

        if (row['volume_classification_prices'] == 'low'):
            warning += 'PRECIO AFECTADO - VOLUMEN BAJO, '

        # hangup
        if pd.notna(row['hangup_signaturetime']):
            if (_sec_to_min_sec(row['hangup_signaturetime']) < _sec_to_min_sec(row['TMO']-15) 
                and _sec_to_min_sec(row['hangup_signaturetime']) > 20):
                    warning += 'HANGUP ANTICIPADO, '
        # overlapping
        if (row['overlap_total_time'] > 5):
                warning += 'OVERLAPPING, '
        # MAC INCORRECTO
        if (row['best_mac_likelihood'] < MAC_CUT *0.95):
            rej_reason += 'MAC INCORRECTO, '
            rej = True
        # Precio INCORRECTO
        if (row['best_price_likelihood'] < PRICE_CUT * 0.8):
            rej_reason += 'PRECIO INCORRECTO, '
            rej = True
        # MAC inexacto
        if row['best_mac_likelihood'] < MAC_CUT:
            warning += 'MAC INEXACTO, '
        
        if row['best_price_likelihood'] < MAC_CUT:
            warning += 'PRECIO INEXACTO, '

        # Palabras prohibidas
        if row['forbidden_rate'] < (forbidden_rate_mean * 0.9):
            warning += 'PALABRAS PROHIBIDAS, '

        # TMO bajo
        if row['TMO'] < (TMO_mean * 0.8):
            warning += 'TMO BAJO, '

        # Puntaje bajo
        if row['score'] < (score_mean * 0.8):
            warning += 'PUNTAJE BAJO, '
        
        # Modulo validación de documento
        if row['mvd'] < 1:
            warning += 'MVD NO DETECTADO, '

        # Modulo términos legales
        
        if row['terms'] < 1:
            warning += 'TERMINOS NO DETECTADOS, '

                # Modulo validación de documento
        if row['mac_r'] < 1:
            warning += 'PDR NO DETECTADA, '

        # BPC
        if row['agent_participation'] > 0.9:
            warning += 'BPC, '

        # No auditable
        if row['auditable_strikes'] > 3 or (
            pd.isna(row['final_label_macs']) & pd.isna(row['final_label_prices'])
        ):
            rej_reason = 'NO AUDITABLE'
            warning = ''

        # Llamada vacía
        if row['VACIA'] == 1:
            rej_reason = 'LLAMADA VACÍA'
            warning = ''

        # MAC FORZADO
        if row["times_said_macs"] > 2:
            rej_reason += 'MAC FORZADO, '
            rej = True
        if row["igs_comp"] == 0:
            warning += "IGS INCOMPLETO, "
        return pd.Series([rej_reason.strip(', '), warning.strip(', '), rej])

    # Evaluar razones
    df[['reject_reason', 'warning_reason', 'reject']] = df.apply(evaluate_reasons, axis=1)

    # Aplicar el filtro global de rechazo
    return df
