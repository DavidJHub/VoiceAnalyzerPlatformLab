import logging
import os
import time
import random
import boto3
from dotenv import load_dotenv
from mysql.connector import connect, Error as MySQLError

load_dotenv()

def _require(var: str) -> str:
    value = os.getenv(var)
    if not value:
        raise RuntimeError(f"Variable de entorno «{var}» no definida")
    return value




HOST_DB_VAP     = _require("HOST_DB_VAP")
PORT_DB_VAP     = _require("PORT_DB_VAP")
DB_NAME_VAP     = _require("DB_NAME_VAP")
USER_DB_VAP     = _require("USER_DB_VAP")
PASSWORD_DB_VAP = _require("PASSWORD_DB_VAP")

HOST_DB_VICI     = _require("HOST_DB_VICI")
PORT_DB_VICI     = _require("PORT_DB_VICI")
DB_NAME_VICI     = _require("DB_NAME_VICI")
USER_DB_VICI     = _require("USER_DB_VICI")
PASSWORD_DB_VICI = _require("PASSWORD_DB_VICI")


#### AWS S3 CREDENTIALS AND CONFIGURATION ####

AWS_ACCESS_KEY_ID = _require("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = _require("AWS_SECRET_ACCESS_KEY")
AWS_REGION = _require("AWS_REGION")

#### HUGGING FACE TOKEN MODELS ####
HF_TOKEN     = _require("HF_TOKEN")


def conectar(HOST,USERNAME,PASSWORD,DATABASE,max_retries=10, base_delay=2, cap_delay=60,jitter=0.3):
    """
    Devuelve una conexión MySQL.  
    * max_retries = None  → intenta para siempre  
    * base_delay  = 2 s   → primer back-off  
    * cap_delay   = 60 s  → tope para no esperar demasiado
    """
    intentos = 0
    while True:
        try:
            cnx = connect(
                host=HOST,
                database=DATABASE,
                user=USERNAME,
                password=PASSWORD,
                autocommit=True,          # evita transacciones abiertas
                connection_timeout=20,    # falla rápido si no hay respuesta
            )
            logging.info("✔ Conectado a RDS en el intento %s", intentos + 1)
            return cnx

        except MySQLError as e:
            intentos += 1
            wait = min(base_delay * 2 ** (intentos - 1), cap_delay)
            wait *= 1 + jitter * random.random()   # pequeño azar
            logging.warning("✖ %s – reintento %s en %.1f s", e.__class__.__name__, intentos, wait)

            if max_retries is not None and intentos >= max_retries:
                logging.error("Se alcanzó el máximo de %s reintentos; abortando.", max_retries)
                raise

            time.sleep(wait)


def ensure_connected(conn, attempts=3, delay=2):
    if conn.is_connected():
        return True
    for _ in range(attempts):
        try:
            conn.reconnect(attempts=1, delay=0)  # reconecta el mismo objeto
            if conn.is_connected():
                return True
        except Exception as e:
            time.sleep(delay)
    return False

def run_with_reconnect(conn, query, params):
    if not ensure_connected(conn):
        raise RuntimeError("No fue posible reconectar a MySQL")
    with conn.cursor() as cur:
        cur.execute(query, params)
    conn.commit()


def generate_s3_client():
    return boto3.client('s3',
                             aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
                             region_name = AWS_REGION)

def generate_s3_resource():
    return boto3.resource('s3',
                             aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
                             region_name = AWS_REGION)