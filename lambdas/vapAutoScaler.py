import os
import json
import time
import math
import base64
import datetime
import traceback
import socket
from typing import Optional, List, Dict

import boto3
from botocore.config import Config

try:
    import pymysql
except Exception:
    pymysql = None

try:
    import paramiko
except Exception:
    paramiko = None


# =========================================================
# ENV / CONFIG
# =========================================================
REGION = os.environ.get("AWS_REGION", "us-east-1")
EC2_REGION = os.environ.get("EC2_REGION", REGION)

# ---- DB ----
DB_HOST = os.environ.get("DB_HOST", "")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_USER = os.environ.get("DB_USER", "")
DB_PASS = os.environ.get("DB_PASS", "")
DB_NAME = os.environ.get("DB_NAME", "aihub_bd")

TABLE_VAP_STATUS = os.environ.get("DB_TABLE_VAP_STATUS", "vap_status")
TABLE_VAP_RUN = os.environ.get("DB_TABLE_VAP_RUN", "vap_run")

RUN_KEY = os.environ.get("RUN_KEY", "vap_daily_run")

# ---- Guardrails ----
MIN_REMAINING_MS = int(os.environ.get("MIN_REMAINING_MS", "15000"))

# ---- EC2 Orchestration ----
EC2_PRIMARY_ID = os.environ.get("EC2_PRIMARY_ID", "").strip()
EC2_POOL_IDS = [x.strip() for x in os.environ.get("EC2_POOL_IDS", "").split(",") if x.strip()]

EC2_SSH_HOST_PRIMARY = os.environ.get("EC2_SSH_HOST_PRIMARY", "").strip()  # private ip/dns (recomendado)
EC2_SSH_USER = os.environ.get("EC2_SSH_USER", "ubuntu").strip()
EC2_SSH_PORT = int(os.environ.get("EC2_SSH_PORT", "22"))
SSH_KEY_B64 = os.environ.get("SSH_KEY_B64", "").strip()

# Script a ejecutar en EC2 (ajustable por env si quieres)
REMOTE_WORKDIR = os.environ.get("REMOTE_WORKDIR", "~/VAP_RELEASE").strip()
REMOTE_SCRIPT = os.environ.get("REMOTE_SCRIPT", "vap_status_run.sh").strip()
REMOTE_LOG = os.environ.get("REMOTE_LOG", "~/vap_status_run.log").strip()

# Espera fija del pipeline
WAIT_AFTER_TRIGGER_SECONDS = int(os.environ.get("WAIT_AFTER_TRIGGER_SECONDS", "120"))

# boto3 reliability
BOTO_CONFIG = Config(
    region_name=REGION,
    retries={"max_attempts": 6, "mode": "standard"},
    connect_timeout=5,
    read_timeout=20,
)
EC2_BOTO_CONFIG = Config(
    region_name=EC2_REGION,
    retries={"max_attempts": 6, "mode": "standard"},
    connect_timeout=5,
    read_timeout=20,
)

ec2 = boto3.client("ec2", config=EC2_BOTO_CONFIG)


# =========================================================
# Logging helpers
# =========================================================
def log_step(label: str, t_last: float) -> float:
    t = time.time()
    print(f"[STEP] {label} dt={t - t_last:.3f}s")
    return t

def remaining_ms(ctx) -> Optional[int]:
    try:
        if ctx is None:
            return None
        return int(ctx.get_remaining_time_in_millis())
    except Exception:
        return None

def should_abort_for_time(ctx) -> bool:
    rm = remaining_ms(ctx)
    if rm is None:
        return False
    return rm <= MIN_REMAINING_MS

def safe_sleep(total_seconds: int, ctx=None, tick: float = 2.0):
    end = time.time() + float(total_seconds)
    while time.time() < end:
        if ctx and should_abort_for_time(ctx):
            print("[ABORT] Near timeout during sleep.")
            return
        time.sleep(min(tick, max(0.0, end - time.time())))


# =========================================================
# DB UTILITIES (PyMySQL)
# =========================================================
def db_connect():
    if not pymysql:
        raise RuntimeError("pymysql no disponible (layer faltante).")
    if not (DB_HOST and DB_USER and DB_PASS and DB_NAME):
        raise RuntimeError("Credenciales DB incompletas (DB_HOST/DB_USER/DB_PASS/DB_NAME).")

    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=8,
        read_timeout=30,
        write_timeout=30,
        autocommit=False,
    )

def insert_script_run(
    conn,
    run_key: str,
    run_started_utc,
    run_finished_utc,
    exec_seconds: float,
    total_files_target: int,
    target_date_str: str,
    desired_machines: int,
    started_instance_ids: List[str],
    ssh_out: str,
    warnings_exceptions: Optional[str],
):
    """
    Log simple en vap_run (si tu tabla tiene estas columnas diferentes, ajusta aquí).
    Si tu vap_run NO tiene estas columnas, borra esta función o mapea a tus campos reales.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO `{TABLE_VAP_RUN}` (
                    run_key,
                    run_start_time,
                    run_end_time,
                    exec_time,
                    warning_exceptions,
                    campaigns,
                    did_reset
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_key,
                    run_started_utc,
                    run_finished_utc,
                    exec_seconds,
                    warnings_exceptions,
                    json.dumps(
                        {
                            "target_date": target_date_str,
                            "total_files_target": int(total_files_target),
                            "desired_machines": int(desired_machines),
                            "started_instance_ids": started_instance_ids,
                            "ssh_out_tail": (ssh_out or "")[-2000:],
                        },
                        ensure_ascii=False,
                    ),
                    0,
                ),
            )
    except Exception as e:
        print(f"[WARN] insert_script_run failed (schema mismatch?): {e}")

def fetch_total_files_for_target_day_sql(conn) -> Dict:
    """
    Implementa la lógica del filter_subfolders_within_one_day SIN pandas, en SQL:
    - Parsea folder_date aceptando YYYYMMDD o YYYY-MM-DD
    - most_recent = MAX(parsed_date)
    - si most_recent == hoy(UTC_DATE) => target=hoy, else target=ayer
    - suma file_count donde parsed_date == target
    """
    parsed_expr = """
    (
      CASE
        WHEN folder_date REGEXP '^[0-9]{8}$'
          THEN STR_TO_DATE(folder_date, '%Y%m%d')
        WHEN folder_date REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
          THEN STR_TO_DATE(folder_date, '%Y-%m-%d')
        ELSE NULL
      END
    )
    """

    with conn.cursor() as cur:
        cur.execute("SELECT UTC_DATE() AS today_utc")
        today_utc = (cur.fetchone() or {}).get("today_utc")
        if not today_utc:
            today_utc = datetime.datetime.utcnow().date()

        cur.execute(
            f"""
            SELECT MAX({parsed_expr}) AS most_recent
            FROM `{TABLE_VAP_STATUS}`
            """
        )
        most_recent = (cur.fetchone() or {}).get("most_recent")

        if most_recent and hasattr(most_recent, "strftime"):
            most_recent_date = most_recent
        else:
            most_recent_date = None

        if most_recent_date == today_utc:
            target_date = today_utc
        else:
            target_date = today_utc - datetime.timedelta(days=1)

        cur.execute(
            f"""
            SELECT
              COALESCE(SUM(file_count), 0) AS total_files,
              COUNT(*) AS row_count
            FROM `{TABLE_VAP_STATUS}`
            WHERE {parsed_expr} = %s
            """,
            (target_date,),
        )
        row = cur.fetchone() or {}
        total_files = int(row.get("total_files") or 0)
        row_count = int(row.get("row_count") or 0)

        # Para debug: trae algunas folder_date crudas que entraron al filtro
        cur.execute(
            f"""
            SELECT folder_date
            FROM `{TABLE_VAP_STATUS}`
            WHERE {parsed_expr} = %s
            GROUP BY folder_date
            ORDER BY folder_date DESC
            LIMIT 25
            """,
            (target_date,),
        )
        raw_dates = [r.get("folder_date") for r in (cur.fetchall() or []) if r.get("folder_date")]

        return {
            "today_utc": str(today_utc),
            "most_recent": str(most_recent_date) if most_recent_date else None,
            "target_date": str(target_date),
            "total_files": total_files,
            "row_count": row_count,
            "raw_folder_dates": raw_dates,
        }


# =========================================================
# EC2 HELPERS
# =========================================================
def describe_instance_states(instance_ids: List[str]) -> Dict[str, str]:
    """
    Return dict {instance_id: state_name}
    """
    if not instance_ids:
        return {}

    resp = ec2.describe_instances(InstanceIds=instance_ids)
    out = {}
    for r in resp.get("Reservations", []):
        for inst in r.get("Instances", []):
            iid = inst.get("InstanceId")
            st = (inst.get("State") or {}).get("Name")
            if iid and st:
                out[iid] = st
    return out

def start_instances_if_needed(instance_ids: List[str]) -> List[str]:
    """
    Starts instances only if state is stopped/stopping.
    Returns the list of instance IDs for which a start was requested.
    """
    if not instance_ids:
        return []

    states = describe_instance_states(instance_ids)
    to_start = []
    for iid in instance_ids:
        st = states.get(iid)
        # si no lo encontramos, igual intentamos
        if st in (None, "stopped", "stopping"):
            to_start.append(iid)

    if to_start:
        print(f"[EC2] start_instances -> {to_start}")
        ec2.start_instances(InstanceIds=to_start)
    else:
        print(f"[EC2] no start needed. states={states}")

    return to_start

def wait_instance_running(instance_id: str, max_wait_s: int = 240) -> bool:
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        st = describe_instance_states([instance_id]).get(instance_id)
        print(f"[EC2] {instance_id} state={st}")
        if st == "running":
            return True
        time.sleep(5)
    return False

def wait_tcp_open(host: str, port: int, timeout_s: int = 120, tick: float = 3.0) -> bool:
    """
    Espera a que el puerto SSH esté abierto (best-effort).
    """
    if not host:
        return False
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=5):
                return True
        except Exception:
            time.sleep(tick)
    return False

def ssh_trigger_vap_status_run(host: str, user: str, port: int, key_b64: str) -> Dict:
    """
    Ejecuta REMOTE_WORKDIR/REMOTE_SCRIPT en background:
      cd ~/VAP_RELEASE && nohup bash vap_status_run.sh > ~/vap_status_run.log 2>&1 & disown
    """
    if not paramiko:
        raise RuntimeError("paramiko no disponible (layer faltante).")
    if not (host and user and key_b64):
        raise RuntimeError("Faltan envs SSH: EC2_SSH_HOST_PRIMARY / EC2_SSH_USER / SSH_KEY_B64")

    key_path = "/tmp/id_rsa"
    with open(key_path, "wb") as f:
        f.write(base64.b64decode(key_b64))
    os.chmod(key_path, 0o600)

    # intenta RSA; si usas ed25519, cae a Ed25519Key
    try:
        pkey = paramiko.RSAKey.from_private_key_file(key_path)
    except Exception:
        pkey = paramiko.Ed25519Key.from_private_key_file(key_path)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client.connect(
        hostname=host,
        port=port,
        username=user,
        pkey=pkey,
        timeout=12,
        banner_timeout=12,
        auth_timeout=12,
    )

    cmd = (
        "bash -lc '"
        f"cd {REMOTE_WORKDIR} && "
        f"nohup bash {REMOTE_SCRIPT} > {REMOTE_LOG} 2>&1 & disown; "
        "echo STARTED; "
        f"tail -n 25 {REMOTE_LOG} 2>/dev/null || true"
        "'"
    )

    stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
    out = stdout.read().decode("utf-8", "ignore")
    err = stderr.read().decode("utf-8", "ignore")
    client.close()

    return {"ssh_out": out[-3000:], "ssh_err": err[-3000:], "cmd": cmd}


# =========================================================
# Demand rules
# =========================================================
def desired_machines_from_rules(total_files: int) -> int:
    """
    Reglas EXACTAS pedidas:
      <600 => 1
      600-1199 => 2
      1200-1799 => 3
      >=1800 => todas (len(pool))
    """
    if total_files < 600:
        return 1
    if total_files < 1200:
        return 2
    if total_files < 1800:
        return 3
    # "todas": lo resolvemos al máximo posible después (según pool)
    return 10**9

def build_ordered_pool(primary_id: str, pool_ids: List[str]) -> List[str]:
    pool = [x for x in pool_ids if x]
    if primary_id:
        pool = [x for x in pool if x != primary_id]
        pool = [primary_id] + pool
    return pool

def choose_instances_to_start(primary_id: str, pool_ids: List[str], desired_n: int) -> List[str]:
    ordered = build_ordered_pool(primary_id, pool_ids)
    if desired_n >= len(ordered):
        return ordered
    return ordered[:max(1, desired_n)]


# =========================================================
# LAMBDA HANDLER
# =========================================================
print("### MODULE IMPORTED ###")

def lambda_handler(event, ctx):
    t0 = time.time()
    started_utc = datetime.datetime.utcnow()
    t_last = time.time()

    conn = None
    warnings = None

    # ---- Validaciones mínimas ----
    if not EC2_PRIMARY_ID:
        return {"ok": False, "error": "Falta env EC2_PRIMARY_ID"}
    if not EC2_POOL_IDS:
        return {"ok": False, "error": "Falta env EC2_POOL_IDS (lista de instancias)"}

    # 1) Encender la máquina principal inmediatamente (si ya está running/pending, no hace nada)
    try:
        print("[START] event_keys=", list(event.keys()) if isinstance(event, dict) else str(type(event)))
        rm = remaining_ms(ctx)
        if rm is not None:
            print(f"[CTX] remaining_ms={rm} MIN_REMAINING_MS={MIN_REMAINING_MS}")

        start_instances_if_needed([EC2_PRIMARY_ID])
        t_last = log_step("ec2_start_primary_if_needed", t_last)

        ok_running = wait_instance_running(EC2_PRIMARY_ID, max_wait_s=240)
        t_last = log_step(f"ec2_wait_running_primary={ok_running}", t_last)

        # Best-effort: esperar a que el puerto SSH abra
        if EC2_SSH_HOST_PRIMARY:
            ok_ssh = wait_tcp_open(EC2_SSH_HOST_PRIMARY, EC2_SSH_PORT, timeout_s=120, tick=3.0)
            print(f"[EC2] ssh_port_open={ok_ssh} host={EC2_SSH_HOST_PRIMARY}:{EC2_SSH_PORT}")
            t_last = log_step("wait_ssh_port_open", t_last)

        # 2) Ejecutar el script VAP_RELEASE/vap_status_run.sh
        ssh_result = {}
        if EC2_SSH_HOST_PRIMARY and SSH_KEY_B64:
            ssh_result = ssh_trigger_vap_status_run(
                host=EC2_SSH_HOST_PRIMARY,
                user=EC2_SSH_USER,
                port=EC2_SSH_PORT,
                key_b64=SSH_KEY_B64,
            )
            t_last = log_step("ssh_trigger_vap_status_run", t_last)
        else:
            print("[WARN] SSH envs missing; cannot run remote script. (EC2 started only).")

        # 3) Esperar 2 minutos
        print(f"[WAIT] sleeping {WAIT_AFTER_TRIGGER_SECONDS}s ...")
        safe_sleep(WAIT_AFTER_TRIGGER_SECONDS, ctx=ctx, tick=2.0)
        t_last = log_step("wait_2_minutes", t_last)

        # 4) Leer vap_status y sumar file_count del día objetivo (según tu lógica)
        if not pymysql:
            raise RuntimeError("pymysql no disponible (layer faltante).")
        if not (DB_HOST and DB_USER and DB_PASS and DB_NAME):
            raise RuntimeError("Credenciales DB incompletas (DB_HOST/DB_USER/DB_PASS/DB_NAME).")

        conn = db_connect()
        t_last = log_step("db_connect", t_last)

        target_info = fetch_total_files_for_target_day_sql(conn)
        total_files_target = int(target_info["total_files"])
        target_date = target_info["target_date"]
        t_last = log_step(f"db_total_files_target={total_files_target} target_date={target_date}", t_last)

        # 5) Encender máquinas según demanda (reglas)
        desired_n = desired_machines_from_rules(total_files_target)
        target_ids = choose_instances_to_start(EC2_PRIMARY_ID, EC2_POOL_IDS, desired_n)

        # Enciende solo las que lo necesiten
        started_now = start_instances_if_needed(target_ids)
        t_last = log_step(f"ec2_start_targets started_now={len(started_now)} target_n={len(target_ids)}", t_last)

        finished_utc = datetime.datetime.utcnow()
        exec_seconds = round(time.time() - t0, 3)

        # Log en vap_run (best-effort)
        insert_script_run(
            conn,
            run_key=RUN_KEY,
            run_started_utc=started_utc,
            run_finished_utc=finished_utc,
            exec_seconds=exec_seconds,
            total_files_target=total_files_target,
            target_date_str=target_date,
            desired_machines=min(desired_n, len(build_ordered_pool(EC2_PRIMARY_ID, EC2_POOL_IDS))),
            started_instance_ids=target_ids,
            ssh_out=(ssh_result or {}).get("ssh_out", ""),
            warnings_exceptions=None,
        )
        t_last = log_step("insert_script_run", t_last)

        conn.commit()
        t_last = log_step("db_commit", t_last)

        return {
            "ok": True,
            "run_key": RUN_KEY,
            "primary_id": EC2_PRIMARY_ID,
            "ssh_trigger": ssh_result,
            "wait_seconds": WAIT_AFTER_TRIGGER_SECONDS,
            "target_day_info": target_info,  # incluye today_utc, most_recent, target_date, raw_folder_dates
            "total_files_target": total_files_target,
            "desired_machines_rule": desired_n,
            "started_instance_ids_target": target_ids,
            "started_instance_ids_now": started_now,
            "exec_seconds": exec_seconds,
            "remaining_ms_end": remaining_ms(ctx),
        }

    except Exception:
        warnings = traceback.format_exc()
        print("[ERROR] exception:\n", warnings)

        # Intentar loguear error (best-effort)
        try:
            if conn:
                finished_utc = datetime.datetime.utcnow()
                exec_seconds = round(time.time() - t0, 3)
                insert_script_run(
                    conn,
                    run_key=RUN_KEY,
                    run_started_utc=started_utc,
                    run_finished_utc=finished_utc,
                    exec_seconds=exec_seconds,
                    total_files_target=-1,
                    target_date_str="",
                    desired_machines=0,
                    started_instance_ids=[],
                    ssh_out="",
                    warnings_exceptions=warnings[-12000:],
                )
                conn.commit()
        except Exception as e:
            print(f"[WARN] failed to log vap_run on error: {e}")

        return {
            "ok": False,
            "error": "vapStatus-orchestrator-lambda failed",
            "trace": warnings[-3500:] if warnings else "",
            "remaining_ms_end": remaining_ms(ctx),
        }

    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass
