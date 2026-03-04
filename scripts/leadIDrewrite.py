import json
from datetime import datetime

from setup.CampaignSetup import connection_mysql
from mysql.connector import connect, Error

nombrebd = "aihub_bd"
usuario = "admindb"
contraseña = "VAPigs2024.*"
host = "vapdb.cjq4ek6ygqif.us-east-1.rds.amazonaws.com"

conexion, cursor = connection_mysql(nombrebd, usuario, contraseña, host)


def actualizar_agent_id_en_audio_data(conexion, batch_size=500, ts_col="date"):

    sel = f"""
        SELECT id, name
        FROM   agent_audio_data
        WHERE  agent_audio_data.{ts_col} >= (NOW() - INTERVAL 100 DAY)
    """
    cur_sel = conexion.cursor(dictionary=True, buffered=True)
    cur_upd = conexion.cursor()

    cur_sel.execute(sel)

    pendientes, tot = [], 0
    for fila in cur_sel:
        try:
            partes = fila["name"].split("_")
            if len(partes) > 2 and partes[-2]:
                pendientes.append((partes[-2], fila["id"]))
        except:
            continue

        if len(pendientes) >= batch_size:
            cur_upd.executemany(
                "UPDATE agent_audio_data SET agent_id = %s WHERE id = %s",
                pendientes
            )
            print(f"Actualizando {len(pendientes)} filas... con {pendientes}")
            conexion.commit()
            tot += cur_upd.rowcount
            pendientes.clear()

    if pendientes:
        cur_upd.executemany(
            "UPDATE agent_audio_data SET agent_id = %s WHERE id = %s",
            pendientes
        )
        conexion.commit()
        tot += cur_upd.rowcount

    cur_sel.close()
    cur_upd.close()
    #print(f"Filas actualizadas (últimos 7 días): {tot}")



import mysql.connector

def actualizar_lead_id_en_call_affecteds(
        conexion,
        batch_size: int = 500,
        ts_col: str = "date",
        idx_lead: int = 2  # posición del fragmento con el lead_id en el string name
    ):
    """
    Actualiza call_affecteds.lead_id tomando el valor del campo 'name'
    de agent_audio_data.  Se filtran únicamente los registros creados en
    los últimos 3 días (o según la columna ts_col indicada).

    Parámetros
    ----------
    conexion   : conexión abierta de mysql.connector
    batch_size : tamaño del lote para ejecutar UPDATE por lotes
    ts_col     : nombre de la columna timestamp en call_affecteds
    idx_lead   : índice del fragmento en 'name' que contiene el lead_id
    """

    # 1) Seleccionar IDs de call_affecteds + el 'name' relacionado
    sel = f"""
        SELECT ca.id                 AS ca_id,
               aad.name              AS audio_name
        FROM   call_affecteds  AS ca
        JOIN   agent_audio_data AS aad
               ON ca.agent_audio_data_id = aad.id
        WHERE  aad.{ts_col} >= (NOW() - INTERVAL 100 DAY) 
    """ # AND aad.{ts_col} <= (NOW() - INTERVAL 12 DAY)

    cur_sel = conexion.cursor(dictionary=True, buffered=True)
    cur_upd = conexion.cursor()
    cur_sel.execute(sel)

    pendientes, tot = [], 0
    for fila in cur_sel:
        partes = (fila["audio_name"] or "").split("_")
        if len(partes) < 4:
            continue                      # formato malo

        lead_id = partes[2]
        try:
            cur_upd.execute(
                "UPDATE call_affecteds SET lead_id = %s WHERE id = %s",
                (lead_id, fila["ca_id"])
            )
            conexion.commit()          
        except mysql.connector.Error as e:
            print(f"⚠️  Saltando {fila['ca_id']} – {e}")
            conexion.rollback()        
        # 2) Ejecutar UPDATE por lotes
        if len(pendientes) >= batch_size:
            cur_upd.executemany(
                "UPDATE call_affecteds SET lead_id = %s WHERE id = %s",
                pendientes
            )
            conexion.commit()
            tot += cur_upd.rowcount
            print(f"Actualizadas {cur_upd.rowcount} filas…")
            pendientes.clear()

    # 3) Procesar lo que quede pendiente
    if pendientes:
        cur_upd.executemany(
            "UPDATE call_affecteds SET lead_id = %s WHERE id = %s",
            pendientes
        )
        conexion.commit()
        tot += cur_upd.rowcount

    cur_sel.close()
    cur_upd.close()
    print(f"Filas actualizadas (últimos 3 días): {tot}")



def actualizar_lead_id_en_agent_audio_data(
        conexion,
        batch_size: int = 500,
        ts_col: str = "date",
        idx_lead: int = 2  # posición del fragmento con el lead_id en el string name
    ):
    """
    Actualiza call_affecteds.lead_id tomando el valor del campo 'name'
    de agent_audio_data.  Se filtran únicamente los registros creados en
    los últimos 3 días (o según la columna ts_col indicada).

    Parámetros
    ----------
    conexion   : conexión abierta de mysql.connector
    batch_size : tamaño del lote para ejecutar UPDATE por lotes
    ts_col     : nombre de la columna timestamp en call_affecteds
    idx_lead   : índice del fragmento en 'name' que contiene el lead_id
    """

    # 1) Seleccionar IDs de call_affecteds + el 'name' relacionado
    sel = f"""
        SELECT aad.id                 AS ca_id,
               aad.name              AS audio_name,
               aad.lead_id
        FROM   agent_audio_data  AS aad
        WHERE  aad.{ts_col} >= (NOW() - INTERVAL 100 DAY) 
    """ # AND aad.{ts_col} <= (NOW() - INTERVAL 12 DAY)

    cur_sel = conexion.cursor(dictionary=True, buffered=True)
    cur_upd = conexion.cursor()
    cur_sel.execute(sel)

    pendientes, tot = [], 0
    for fila in cur_sel:
        partes = (fila["audio_name"] or "").split("_")
        if len(partes) < 4:
            continue                      # formato malo

        lead_id = partes[2]
        try:
            cur_upd.execute(
                "UPDATE agent_audio_data SET lead_id = %s WHERE id = %s",
                (lead_id, fila["ca_id"])
            )
            conexion.commit()          
        except mysql.connector.Error as e:
            print(f"⚠️  Saltando {fila['ca_id']} – {e}")
            conexion.rollback()        
        # 2) Ejecutar UPDATE por lotes
        if len(pendientes) >= batch_size:
            cur_upd.executemany(
                "UPDATE agent_audio_data SET lead_id = %s WHERE id = %s",
                pendientes
            )
            conexion.commit()
            tot += cur_upd.rowcount
            print(f"Actualizadas {cur_upd.rowcount} filas…")
            pendientes.clear()

    # 3) Procesar lo que quede pendiente
    if pendientes:
        cur_upd.executemany(
            "UPDATE agent_audio_data SET lead_id = %s WHERE id = %s",
            pendientes
        )
        conexion.commit()
        tot += cur_upd.rowcount

    cur_sel.close()
    cur_upd.close()
    print(f"Filas actualizadas (últimos 3 días): {tot}")



if __name__ == "__main__":
    actualizar_lead_id_en_agent_audio_data(conexion)
    actualizar_lead_id_en_call_affecteds(conexion)
    actualizar_agent_id_en_audio_data(conexion)
