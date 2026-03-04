import json
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd

import database.dbConfig as dbcfg


# ==========================
# CONFIG / CLIENTES
# ==========================

S3_BUCKET_NAME = "documentos.aihub"


def get_db_connection():
    """Devuelve conexión a la BD VAP."""
    conexion = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP,
    )
    return conexion


def get_s3_client():
    return dbcfg.generate_s3_client()


# ==========================
# UTILES BD
# ==========================

def fetch_agent_audio_rows_last_days(days: int = 2) -> pd.DataFrame:
    """
    Trae de agent_audio_data los registros de los últimos `days` días,
    con columnas:
      - id
      - name
      - link_transcription_audio
    """
    conexion = get_db_connection()
    cutoff = datetime.now() - timedelta(hours=days)

    query = """
        SELECT
            id,
            name,
            link_transcription_audio
        FROM agent_audio_data
        WHERE `date` >= %s
          AND link_transcription_audio IS NOT NULL
          AND link_transcription_audio <> ''
    """

    try:
        cursor = conexion.cursor()
        cursor.execute(query, (cutoff,))
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
    finally:
        cursor.close()
        conexion.close()

    df = pd.DataFrame(rows, columns=colnames)
    return df


# ==========================
# S3: LECTURA DEL TRANSCRITO
# ==========================

def load_transcript_json_from_s3(s3_client, key: str) -> Optional[Dict[str, Any]]:
    """
    Lee un JSON de S3 dado su 'key'.
    Devuelve el dict o None si falla.
    """
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        content = response["Body"].read()
        data = json.loads(content)
        return data
    except Exception as e:
        print(f"[!] Error leyendo S3 key={key}: {e}")
        return None


# ==========================
# TRANSCRITO → UTTERANCES Y TEXTO
# ==========================

def transcript_to_utterances(transcript_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convierte el JSON en una lista de 'utterances':
    [
      {"speaker": 0, "text": "..."},
      {"speaker": 1, "text": "..."},
      ...
    ]
    speaker 0 suele ser agente, 1 cliente (según el pipeline).
    """
    utterances: List[Dict[str, Any]] = []

    if not transcript_json:
        return utterances

    try:
        channels = transcript_json["results"]["channels"]
    except (KeyError, TypeError):
        return utterances

    for ch in channels:
        alternatives = ch.get("alternatives", [])
        for alt in alternatives:
            paragraphs = alt.get("paragraphs", [])
            for p in paragraphs:
                speaker = p.get("speaker")
                sentences = p.get("sentences", [])
                for s in sentences:
                    t = s.get("text")
                    if t:
                        utterances.append(
                            {
                                "speaker": speaker,
                                "text": t.strip()
                            }
                        )

    return utterances


def transcript_to_text(transcript_json: Dict[str, Any]) -> str:
    """
    Concatena todos los textos en un string grande.
    """
    utterances = transcript_to_utterances(transcript_json)
    return " ".join(u["text"] for u in utterances if u.get("text"))


# ==========================
# REGLAS PARA CEDULA / EMAIL / NOMBRE COMPLETO
# ==========================

# Candidatos de DNI: secuencias de 7–15 dígitos
DIGITS_REGEX = re.compile(r"\b\d{7,15}\b")

# Palabras clave alrededor de documento
DNI_KEYWORDS = [
    "dni", "documento", "cédula", "cedula", "identificación", "identificacion",
    "id", "identidad"
]

# Patrones típicos de agente presentando o preguntando nombre
NAME_TRIGGER_REGEX = re.compile(
    r"(?i)(tengo el gusto de hablar con|hablo con|estoy hablando con|"
    r"con la se[nñ]ora|con el se[nñ]or|con la se[nñ]orita)"
)

# Nombre después de "hablar con ..." en la misma frase (nombre completo)
NAME_AFTER_TRIGGER_REGEX = re.compile(
    r"(?i)(?:tengo el gusto de hablar con|hablo con|estoy hablando con|"
    r"con la se[nñ]ora|con el se[nñ]or|con la se[nñ]orita)\s+"
    r"([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑñü]+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑñü]+){0,3})"
)

# Respuesta típica del cliente: "Sí, habla Nancy Corona" / "Habla Nancy Corona"
CLIENT_NAME_RESPONSE_REGEX = re.compile(
    r"(?i)\b(habla|soy|sí,? con|si,? con)\s+"
    r"([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑñü]+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑñü]+){0,3})"
)


def score_dni_candidate(context: str, number: str) -> int:
    """
    Da un score a un candidato de DNI según si hay palabras clave cerca.
    Mientras más palabras clave, más score.
    """
    c_lower = context.lower()
    score = 0
    for kw in DNI_KEYWORDS:
        if kw in c_lower:
            score += 1
    # Pequeña preferencia por longitudes típicas (8-10)
    if 8 <= len(number) <= 10:
        score += 1
    return score


def extract_dni_from_utterances(utterances: List[Dict[str, Any]]) -> Optional[str]:
    """
    Busca posibles DNIs en todas las intervenciones y se queda con
    el candidato mejor puntuado por contexto.
    """
    best_number = None
    best_score = 0

    for u in utterances:
        text = u.get("text", "")
        for m in DIGITS_REGEX.finditer(text):
            number = m.group(0)
            start = max(m.start() - 40, 0)
            end = m.end() + 40
            context = text[start:end]
            score = score_dni_candidate(context, number)
            if score > best_score:
                best_score = score
                best_number = number

    return best_number


def extract_full_name_from_agent_utterances(utterances: List[Dict[str, Any]]) -> Optional[str]:
    """
    Primero intentamos extraer el nombre completo desde frases del agente
    tipo 'tengo el gusto de hablar con la señorita X Y'.
    """
    for u in utterances:
        if u.get("speaker") != 0:
            continue  # 0 = agente (asumiendo tu pipeline)
        text = u.get("text", "")
        if not NAME_TRIGGER_REGEX.search(text):
            continue
        m = NAME_AFTER_TRIGGER_REGEX.search(text)
        if m:
            full_name = m.group(1).strip()
            return full_name
    return None


def extract_full_name_from_client_utterances(utterances: List[Dict[str, Any]]) -> Optional[str]:
    """
    Si no se encontró nombre en el agente, probamos con frases del cliente
    tipo 'habla Nancy Corona', 'soy Nancy Corona', etc.
    """
    for u in utterances:
        if u.get("speaker") != 1:
            continue  # 1 = cliente
        text = u.get("text", "")
        m = CLIENT_NAME_RESPONSE_REGEX.search(text)
        if m:
            full_name = m.group(2).strip()
            return full_name

    # Fallback muy laxo: primera intervención del cliente con 2-4 palabras capitalizadas
    for u in utterances:
        if u.get("speaker") != 1:
            continue
        text = u.get("text", "")
        candidates = re.findall(
            r"\b([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑñü]+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑñü]+){1,3})\b",
            text
        )
        if candidates:
            return candidates[0].strip()

    return None


def extract_email_fragment(text: str) -> Optional[str]:
    """
    Busca un token que contenga:
      - '@'
      - 'arroba'
      - '.com'
      - 'punto' seguido de 'com'

    y devuelve un fragmento de texto alrededor (unas palabras antes y después).
    """
    tokens = text.split()
    n = len(tokens)

    for i, tok in enumerate(tokens):
        low = tok.lower()

        hit = False
        if "@" in tok:
            hit = True
        elif low == "arroba":
            hit = True
        elif ".com" in low:
            hit = True
        elif low == "com" and i > 0 and tokens[i-1].lower() == "punto":
            hit = True

        if hit:
            start = max(0, i - 3)
            end = min(n, i + 3)
            frag = " ".join(tokens[start:end])
            return frag

    return None


# ==========================
# EXTRAER CAMPOS DEL TRANSCRITO
# ==========================

def extract_fields_from_transcript(transcript_json: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    A partir del JSON de transcripción, devuelve un dict con:
      - cedula
      - nombre (nombre completo)
      - email (fragmento alrededor de arroba / punto com)
    """
    utterances = transcript_to_utterances(transcript_json)
    full_text = " ".join(u["text"] for u in utterances if u.get("text"))

    # Email: fragmento alrededor de @ / arroba / .com / punto com
    email_fragment = extract_email_fragment(full_text)

    # DNI: candidato mejor puntuado
    dni = extract_dni_from_utterances(utterances)

    # Nombre completo: agente primero, cliente después
    nombre_completo = extract_full_name_from_agent_utterances(utterances)
    if not nombre_completo:
        nombre_completo = extract_full_name_from_client_utterances(utterances)

    return {
        "cedula": dni,
        "nombre": nombre_completo,
        "email": email_fragment,
    }

# ==========================
# PIPELINE PRINCIPAL
# ==========================

def scrape_client_data_from_transcripts_last_days(days: int = 2) -> pd.DataFrame:
    """
    1. Consulta agent_audio_data de los últimos `days` días.
    2. Usa link_transcription_audio como key en S3.
    3. Lee el JSON de S3.
    4. Extrae cedula, nombre completo, email vía reglas.
    5. Devuelve un DataFrame con:
         id, cedula, nombre, email
    """
    df_aad = fetch_agent_audio_rows_last_days(days=days)
    if df_aad.empty:
        return pd.DataFrame(columns=["id", "cedula", "nombre", "email"])

    s3_client = get_s3_client()

    registros = []

    for _, row in df_aad.iterrows():
        id_aad = row["id"]
        key = row["link_transcription_audio"]
        if not key:
            print(f"[!] Sin key de transcript para id={id_aad}")
            continue

        transcript_json = load_transcript_json_from_s3(s3_client, key)
        if not transcript_json:
            print(f"[!] No se pudo cargar transcript para id={id_aad}")
            continue

        fields = extract_fields_from_transcript(transcript_json)

        registros.append({
            "id": id_aad,
            "cedula": fields.get("cedula"),
            "nombre": fields.get("nombre"),
            "email": fields.get("email"),
        })

    return pd.DataFrame(registros)


# ==========================
# EJEMPLO DE USO
# ==========================

if __name__ == "__main__":
    df_result = scrape_client_data_from_transcripts_last_days(days=2)
    print(df_result.head())
    df_result.to_excel("client_data_extracted.xlsx", index=False)
    print(f"Total registros extraídos: {len(df_result)}")
