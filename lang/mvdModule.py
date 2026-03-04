import re
from text_to_num import alpha2digit

def palabras_a_digitos(texto: str, convertir_ordinals: bool = False) -> str:
    """
    Reemplaza números escritos en palabras por dígitos en español.
    - convertir_ordinals=False: solo cardinales/decimales (por defecto)
    - convertir_ordinals=True: también ordinales (p.ej., 'tercera' -> '3ª')
    """
    # En 3.x el parámetro 'threshold' controla qué tan agresivo es con
    # cardinals y ordinales. Con 0 convierte incluso ordinales simples.
    threshold = 0 if convertir_ordinals else 1
    return alpha2digit(texto, "es", threshold=threshold)

def getNums(texto: str) -> list:
    """
    Reemplaza números escritos en palabras por dígitos en español.
    - convertir_ordinals=False: solo cardinales/decimales (por defecto)
    - convertir_ordinals=True: también ordinales (p.ej., 'tercera' -> '3ª')
    """
    # En 3.x el parámetro 'threshold' controla qué tan agresivo es con
    # cardinals y ordinales. Con 0 convierte incluso ordinales simples.
    _num_pat = re.compile(r"\d+(?:[.,]\d+)?")
    return _num_pat.findall(texto or "")