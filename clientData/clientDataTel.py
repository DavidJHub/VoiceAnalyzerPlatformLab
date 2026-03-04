import uuid
from typing import Iterable, Any, Dict, Optional, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import pandas as pd

import database.dbConfig as dbcfg

# 🔇 (Opcional) desactivar warnings por verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================
# CONFIG API
# ==========================

BASE_URL = "https://asignacionautomatica.igroupsolution.com/Api/index.php"
TOKEN_URL = f"{BASE_URL}?r=token"

CLIENT_ID = "igs-client"
CLIENT_SECRET = "TEST"   # cambia por el real si aplica

# ==========================
# SESIÓN HTTP CON REINTENTOS
# ==========================

def create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=2,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Igual que en tu Lambda: SIN verificación de certificado
    session.verify = False
    return session

SESSION = create_session()

# ==========================
# TOKEN
# ==========================

def get_token() -> Optional[str]:
    """Obtiene el access_token de la API (mismo flujo que tu Lambda)."""
    try:
        resp = SESSION.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
            },
            timeout=(5, 5),
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token")
        print(f"[✓] Token obtenido: {token}")
        return token
    except requests.RequestException as e:
        print(f"[✗] Error al obtener token: {e}")
        return None

# ==========================
# ENDPOINT AFILIADOS POR TELÉFONO
# ==========================

def build_afiliado_url_por_telefono(telefono: str) -> str:
    """
    URL correcta para consultar por teléfono.
    Ajusta el nombre del parámetro si tu API usa otro (ej. 'tel' en vez de 'telefono').
    """
    return f"{BASE_URL}?r=afiliados&telefono={telefono}"


def get_afiliado_by_telefono(
    telefono: str,
    token: str,
    timeout: int = 10,
) -> Optional[Dict]:
    """Llama al endpoint afiliados consultando por teléfono."""
    url = build_afiliado_url_por_telefono(str(telefono).strip())

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        resp = SESSION.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data
    except requests.HTTPError as e:
        print(f"[HTTPError] Tel {telefono} -> {e} | body: {getattr(resp, 'text', '')}")
    except requests.RequestException as e:
        print(f"[RequestException] Tel {telefono} -> {e}")
    except ValueError as e:
        print(f"[JSONError] Tel {telefono} -> {e} | body: {getattr(resp, 'text', '')}")
    return None

# ==========================
# PARSEAR RESPUESTA -> CAMPOS vap_clients
# ==========================

def parse_afiliado_to_client_row(
    raw_data: Any,
    telefono: str,
    id_agent_audio_data: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convierte la respuesta de la API en un dict listo para insertar en vap_clients.

    Estructura esperada de raw_data:
    {
        "count": N,
        "items": [
            {
                "dni": "...",
                "nombre": "...",
                "apellido": "...",
                "telefono1": "...",
                "email": "...",
                "cuenta": "...",
                "plan": "...",
                "estado": "...",
                "uuid": "..." (opcional)
            },
            ...
        ]
    }
    """

    if raw_data is None:
        return None

    # Esperamos un dict con "items"
    if isinstance(raw_data, dict) and "items" in raw_data:
        items = raw_data.get("items") or []
        if not items:
            return None
        data = items[0]  # Tomamos el primer afiliado
    elif isinstance(raw_data, list):
        # Fallback por si la API en algún caso devuelve lista directa
        if not raw_data:
            return None
        data = raw_data[0]
    else:
        # Estructura no esperada
        return None

    # Mapeo según la estructura nueva
    cedula = data.get("dni", "") or ""
    nombre = data.get("nombre", "") or ""
    apellido = data.get("apellido", "") or ""
    email = data.get("email", "") or ""
    cuenta = data.get("cuenta", "") or ""
    plan = data.get("plan", "") or ""
    estado = data.get("estado", "") or ""
    telefono_api = data.get("telefono1", "") or str(telefono).strip()
    api_uuid = data.get("uuid")  # por si en algún momento la API lo incluye

    # uuid sigue siendo un identificador propio
    if not api_uuid:
        api_uuid = str(uuid.uuid4())

    # Dict con las columnas de vap_clients (excepto id, que es autoincremental)
    row = {
        "uuid": api_uuid,
        "id_agent_audio_data": id_agent_audio_data,
        "cedula": cedula,
        "telefono": telefono_api,
        "nombre": nombre,
        "apellido": apellido,
        "email": email,
        "cuenta": cuenta,
        "plan": plan,
        "estado": estado,
    }
    return row

# ==========================
# BD: INSERCIÓN EN vap_clients
# ==========================

def get_db_connection():
    """
    Obtiene la conexión a la BD usando tu dbcfg.
    """
    conexion = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP,
    )
    return conexion


SQL_INSERT_CLIENT = """
INSERT INTO vap_clients
    (uuid, id_agent_audio_data, cedula, telefono, nombre, apellido, email, cuenta, plan, estado)
VALUES
    (%(uuid)s, %(id_agent_audio_data)s, %(cedula)s, %(telefono)s, %(nombre)s, %(apellido)s, %(email)s, %(cuenta)s, %(plan)s, %(estado)s)
ON DUPLICATE KEY UPDATE
    id_agent_audio_data = VALUES(id_agent_audio_data),
    cedula   = VALUES(cedula),
    telefono = VALUES(telefono),
    nombre   = VALUES(nombre),
    apellido = VALUES(apellido),
    email    = VALUES(email),
    cuenta   = VALUES(cuenta),
    plan     = VALUES(plan),
    estado   = VALUES(estado);
"""

def save_client_to_db(conexion, client_row: Dict[str, Any]) -> None:
    """
    Inserta o actualiza un registro en vap_clients.
    Usa uuid como clave única (UNIQUE) para ON DUPLICATE KEY.
    """
    with conexion.cursor() as cursor:
        cursor.execute(SQL_INSERT_CLIENT, client_row)
    conexion.commit()

# ==========================
# FLUJO PRINCIPAL: TELEFONOS -> API -> BD (sin IDs)
# ==========================

def process_telefonos_and_save(
    telefonos: Iterable[Any],
) -> None:
    """
    Versión simple para cuando solo tienes teléfonos sueltos.
    El campo id_agent_audio_data quedará NULL.
    """
    token = get_token()
    if not token:
        raise RuntimeError("No se pudo obtener token. Abortando.")

    conexion = get_db_connection()

    try:
        for tel in telefonos:
            tel_str = str(tel).strip()
            if not tel_str:
                continue

            raw_data = get_afiliado_by_telefono(tel_str, token=token)
            client_row = parse_afiliado_to_client_row(
                raw_data,
                telefono=tel_str,
                id_agent_audio_data=None,
            )

            if client_row is None:
                print(f"[!] Sin datos útiles para teléfono {tel_str}")
                continue

            save_client_to_db(conexion, client_row)
            print(f"[✓] Guardado/actualizado teléfono {tel_str} (uuid={client_row['uuid']})")
    finally:
        conexion.close()

# ==========================
# VERSIÓN DESDE DATAFRAME (con id_agent_audio_data)
# ==========================

def process_telefonos_from_dataframe(
    df: pd.DataFrame,
    telefono_column: str,
    id_column: str = "id",
) -> None:
    """
    Procesa un DataFrame con:
      - df[id_column]  = id_agent_audio_data
      - df[telefono_column] = teléfono

    Por cada fila:
      - Llama a la API.
      - Inserta/actualiza en vap_clients
        usando uuid (nuevo o de la API) y
        llenando id_agent_audio_data con el id de agent_audio_data.
    """
    token = get_token()
    if not token:
        raise RuntimeError("No se pudo obtener token. Abortando.")

    conexion = get_db_connection()

    try:
        for _, row in df.iterrows():
            tel_str = str(row[telefono_column]).strip()
            id_agent_audio_data = row[id_column]

            if not tel_str:
                continue

            raw_data = get_afiliado_by_telefono(tel_str, token=token)

            client_row = parse_afiliado_to_client_row(
                raw_data,
                telefono=tel_str,
                id_agent_audio_data=id_agent_audio_data,
            )

            if client_row is None:
                print(f"[!] Sin datos útiles para teléfono {tel_str} (id_agent_audio_data={id_agent_audio_data})")
                continue

            save_client_to_db(conexion, client_row)
            print(
                f"[✓] Guardado/actualizado teléfono {tel_str} "
                f"(id_agent_audio_data={id_agent_audio_data}, uuid={client_row['uuid']})"
            )
    finally:
        conexion.close()

# ==========================
# EJEMPLO DE USO
# ==========================

if __name__ == "__main__":
    # Ejemplo 1: lista manual de teléfonos (sin id_agent_audio_data)
    telefonos = ["3001234567", "3109876543"]
    process_telefonos_and_save(telefonos)

    # Ejemplo 2: desde un DataFrame con id (agent_audio_data) y teléfono
    # df = pd.DataFrame({
    #     "id": [101, 102],
    #     "telefono": ["3001234567", "3109876543"],
    # })
    # process_telefonos_from_dataframe(df, telefono_column="telefono", id_column="id")
