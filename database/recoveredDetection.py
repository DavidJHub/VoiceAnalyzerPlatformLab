from mysql.connector import connect, Error
import database.dbConfig as dbcfg


def actualizar_recovered(reset=False):
    """
    Actualiza la columna 'recovered' en agent_audio_data:
    - 0 si el lead_id es único en los últimos 30 días
    - 1 si el lead_id está repetido en los últimos 30 días
    """
    conn = dbcfg.conectar(HOST=dbcfg.HOST_DB_VAP,  
                                    DATABASE=dbcfg.DB_NAME_VAP,  
                                    USERNAME=dbcfg.USER_DB_VAP,  
                                    PASSWORD=dbcfg.PASSWORD_DB_VAP)
    cursor = conn.cursor()

    if reset ==True:
        consulta_reset = """
                                UPDATE agent_audio_data
                                SET recovered = 0
                                WHERE date >= CURRENT_DATE - INTERVAL 30 DAY;
                            """
        cursor.execute(consulta_reset)

    consulta_repetidos = """
        UPDATE agent_audio_data
        SET recovered = 1
        WHERE date >= CURRENT_DATE - INTERVAL 30 DAY
          AND (lead_id, campaign_id) IN (
                SELECT lead_id, campaign_id
                FROM (
                    SELECT lead_id, campaign_id
                    FROM agent_audio_data
                    WHERE date >= CURRENT_DATE - INTERVAL 30 DAY
                    GROUP BY lead_id, campaign_id
                    HAVING COUNT(*) > 1
                ) AS repetidos
            );
    """
    cursor.execute(consulta_repetidos)

    conn.commit()
    cursor.close()
    print("Columna 'recovered' actualizada correctamente para los últimos 30 días.")

if __name__ == "__main__":
    actualizar_recovered(reset=True)