import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import pandas as pd
import database.dbConfig as dbcfg


# =========================
# === CONFIG ===
# =========================
MACHINE = int(os.environ.get("MACHINE", "1"))           # máquina actual
NUM_MACHINES = int(os.environ.get("NUM_MACHINES", "4")) # total de máquinas
PARAMS_DIR = os.environ.get("PARAMS_DIR", ".")          # dónde guardar params_machine{n}.txt
MODE = os.environ.get("MODE", "ASSIGN").upper()            # ASSIGN | PARAMS | ALL
# ALL = asigna (si hay campañas nuevas) + genera params para esta máquina


# =========================
# === DATE PARSING ===
# =========================
def _parse_folder_date_to_ts(val) -> pd.Timestamp:
    """
    Acepta 'YYYY-MM-DD' o 'YYYYMMDD'. Retorna Timestamp normalizado (00:00) o NaT.
    """
    if val is None:
        return pd.NaT
    s = str(val).strip()
    if not s:
        return pd.NaT

    # Intento 1: YYYY-MM-DD
    t1 = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if pd.notna(t1):
        return t1.normalize()

    # Intento 2: YYYYMMDD
    if re.match(r"^\d{8}$", s):
        t2 = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        if pd.notna(t2):
            return t2.normalize()

    return pd.NaT


def get_latest_load_date_from_status(df_status: pd.DataFrame) -> Optional[pd.Timestamp]:
    """
    "Última carga" = máximo folder_date parseable en vap_status.
    """
    if df_status.empty:
        return None

    parsed = df_status["folder_date"].apply(_parse_folder_date_to_ts)
    parsed_valid = parsed.dropna()
    if parsed_valid.empty:
        return None

    return parsed_valid.max().normalize()


# =========================
# === DB READS ===
# =========================
def get_vap_status_df() -> pd.DataFrame:
    """
    Lee vap_status con las columnas reales que dijiste que existen.
    """
    conn = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )
    try:
        query = """
            SELECT
                campaign_prefix,
                country,
                sponsor,
                campaign,
                campaign_id,
                folder_date,
                file_count,
                processed,
                assigned,
                machine
            FROM vap_status
        """
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()


def fetch_eligible_campaigns_for_latest_load() -> Tuple[pd.Timestamp, pd.DataFrame]:
    """
    Elegibles para asignación:
      - Última carga (max folder_date)
      - file_count > 0
      - assigned = 0
      - processed = 0
    """
    df = get_vap_status_df()
    latest_ts = get_latest_load_date_from_status(df)

    if latest_ts is None:
        return None, df.iloc[0:0].copy()

    df = df.copy()
    df["parsed_ts"] = df["folder_date"].apply(_parse_folder_date_to_ts)

    eligible = df[
        (df["parsed_ts"] == latest_ts) &
        (df["file_count"].fillna(0).astype(int) > 0) &
        (df["assigned"].fillna(0).astype(int) == 0) &
        (df["processed"].fillna(0).astype(int) == 0)
    ].copy()

    # normaliza tipos
    eligible["file_count"] = eligible["file_count"].fillna(0).astype(int)

    return latest_ts, eligible


def fetch_campaigns_for_machine_latest_load(machine: int) -> Tuple[pd.Timestamp, pd.DataFrame]:
    """
    Para generar params: trae campañas asignadas a machine y pertenecientes a la última carga.
    (No depende del archivo local; la fuente es DB.)
    """
    df = get_vap_status_df()
    latest_ts = get_latest_load_date_from_status(df)
    if latest_ts is None:
        return None, df.iloc[0:0].copy()

    df = df.copy()
    df["parsed_ts"] = df["folder_date"].apply(_parse_folder_date_to_ts)

    dfm = df[
        (df["parsed_ts"] == latest_ts) &
        (df["machine"].fillna(-1).astype(int) == int(machine)) &
        (df["file_count"].fillna(0).astype(int) > 0) &
        (df["processed"].fillna(0).astype(int) == 0) &
        (df["assigned"].fillna(0).astype(int) == 1)
    ].copy()
    dfm["file_count"] = dfm["file_count"].fillna(0).astype(int)
    return latest_ts, dfm


# =========================
# === BALANCED DISTRIBUTION ===
# =========================
@dataclass
class AssignmentResult:
    machine_to_prefixes: Dict[int, List[str]]
    machine_loads: Dict[int, int]
    total_files: int


def distribute_balanced_by_file_count(df: pd.DataFrame, num_machines: int) -> AssignmentResult:
    """
    Greedy load balancing por file_count (desc). Minimiza desequilibrio en práctica.
    """
    if df.empty:
        return AssignmentResult(
            machine_to_prefixes={i: [] for i in range(1, num_machines + 1)},
            machine_loads={i: 0 for i in range(1, num_machines + 1)},
            total_files=0
        )

    df_sorted = df.sort_values(by="file_count", ascending=False).copy()

    machine_loads = {i: 0 for i in range(1, num_machines + 1)}
    machine_to_prefixes = {i: [] for i in range(1, num_machines + 1)}

    for _, row in df_sorted.iterrows():
        # máquina con menor carga
        m = min(machine_loads, key=machine_loads.get)
        machine_to_prefixes[m].append(row["campaign_prefix"])
        machine_loads[m] += int(row["file_count"])

    total_files = int(df_sorted["file_count"].sum())
    return AssignmentResult(machine_to_prefixes, machine_loads, total_files)


# =========================
# === DB WRITES ===
# =========================
def apply_machine_assignment_to_db(machine_to_prefixes: Dict[int, List[str]]):
    """
    Escribe en vap_status:
      - assigned = 1
      - machine = <n>
    Para los campaign_prefix asignados.
    """
    # Flatten
    updates: List[Tuple[int, str]] = []
    for m, prefixes in machine_to_prefixes.items():
        for p in prefixes:
            updates.append((int(m), str(p)))

    if not updates:
        print("[ASSIGN] No hay campañas para actualizar en DB.")
        return

    conn = dbcfg.conectar(
        HOST=dbcfg.HOST_DB_VAP,
        DATABASE=dbcfg.DB_NAME_VAP,
        USERNAME=dbcfg.USER_DB_VAP,
        PASSWORD=dbcfg.PASSWORD_DB_VAP
    )
    try:
        cursor = conn.cursor()

        # Actualiza fila a fila (simple, seguro y suficiente para cientos/miles)
        # Si quieres mega-optimizar, se puede hacer con CASE WHEN.
        cursor.executemany("""
            UPDATE vap_status
            SET assigned = 1,
                machine = %s
            WHERE campaign_prefix = %s
        """, updates)

        conn.commit()
        print(f"[ASSIGN] Filas actualizadas: {cursor.rowcount}")

    finally:
        try:
            cursor.close()
        except Exception:
            pass
        conn.close()


# =========================
# === PARAMS FILE (QUEUE APPEND) ===
# =========================
def _read_existing_params_prefixes(filepath: str) -> set:
    """
    Lee params_machineN.txt y retorna conjunto de prefixes ya presentes.
    Formato esperado por línea: "<campaign_prefix> 0"
    """
    if not os.path.exists(filepath):
        return set()

    existing = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # toma primer token como prefix
            prefix = line.split()[0].strip()
            if prefix:
                existing.add(prefix)
    return existing


def append_new_campaigns_to_params(filepath: str, prefixes_in_order: List[str]):
    """
    No re-genera; hace append solo de los que no están ya en el archivo.
    """
    existing = _read_existing_params_prefixes(filepath)
    new_items = [p for p in prefixes_in_order if p and p not in existing]

    if not new_items:
        print(f"[PARAMS] No hay nuevas campañas para encolar en {filepath}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        for p in new_items:
            f.write(f"{p} 0\n")

    print(f"[PARAMS] Encoladas {len(new_items)} campañas nuevas en {filepath}")


def build_params_for_machine(machine: int):
    """
    Genera/actualiza params_machine{machine}.txt usando la columna machine en DB.
    Si ya existe el archivo, NO duplica; hace append de nuevas campañas.
    """
    latest_ts, dfm = fetch_campaigns_for_machine_latest_load(machine)

    if latest_ts is None:
        print("[PARAMS] No se pudo determinar última carga (sin folder_date válido).")
        return

    if dfm.empty:
        print(f"[PARAMS] No hay campañas pendientes para machine={machine} en última carga ({latest_ts.date()}).")
        return

    # Orden sugerido: por file_count desc (primero lo pesado)
    dfm_sorted = dfm.sort_values(by="file_count", ascending=False).copy()
    prefixes = dfm_sorted["campaign_prefix"].apply(lambda x: x.split(",")[0]).astype(str).tolist()
    print(prefixes)

    filename = os.path.join(PARAMS_DIR, f"params_machine{machine}.txt")
    append_new_campaigns_to_params(filename, prefixes)

    total_files = int(dfm_sorted["file_count"].sum())
    print(f"[PARAMS] machine={machine} campañas={len(prefixes)} total_file_count={total_files} última_carga={latest_ts.date()}")


# =========================
# === OPTIONAL REPORT ===
# =========================
def generate_report_excel(filepath: str = "VAP_REPORT.xlsx"):
    """
    Reporte simple basado en DB para la última carga.
    Si quieres unir con KPIs como antes, me dices y lo integro acá.
    """
    df = get_vap_status_df()
    latest_ts = get_latest_load_date_from_status(df)
    if latest_ts is None:
        print("[REPORT] No hay folder_date válido.")
        return

    df = df.copy()
    df["parsed_ts"] = df["folder_date"].apply(_parse_folder_date_to_ts)
    dfl = df[(df["parsed_ts"] == latest_ts) & (df["file_count"].fillna(0).astype(int) > 0)].copy()
    dfl["file_count"] = dfl["file_count"].fillna(0).astype(int)

    dfl.sort_values(["machine", "file_count"], ascending=[True, False], inplace=True)
    dfl.to_excel(filepath, index=False)
    print(f"[REPORT] Guardado: {filepath} filas={len(dfl)} última_carga={latest_ts.date()}")


# =========================
# === MAIN ===
# =========================
def run_assignment_if_needed():
    """
    Asigna SOLO campañas nuevas (assigned=0 & processed=0) de la última carga.
    """
    latest_ts, eligible = fetch_eligible_campaigns_for_latest_load()

    if latest_ts is None:
        print("[ASSIGN] No se pudo determinar última carga.")
        return

    if eligible.empty:
        print(f"[ASSIGN] No hay campañas nuevas para asignar en última carga ({latest_ts.date()}).")
        return

    # Distribución balanceada
    result = distribute_balanced_by_file_count(eligible, NUM_MACHINES)

    print(f"[ASSIGN] última_carga={latest_ts.date()} campañas_nuevas={len(eligible)} total_file_count={result.total_files}")
    for m in range(1, NUM_MACHINES + 1):
        print(f"  - machine {m}: campañas={len(result.machine_to_prefixes[m])} load={result.machine_loads[m]}")

    # Aplicar a DB
    apply_machine_assignment_to_db(result.machine_to_prefixes)


def main():
    if MODE not in {"ASSIGN", "PARAMS", "ALL"}:
        raise ValueError("MODE debe ser ASSIGN | PARAMS | ALL")

    if MODE in {"ASSIGN", "ALL"}:
        run_assignment_if_needed()

    if MODE in {"PARAMS", "ALL"}:
        build_params_for_machine(MACHINE)


if __name__ == "__main__":
    main()
