from clientData.fetchPhone import fetch_recent_agent_phones
from clientData.clientDataTel import process_telefonos_from_dataframe

# 1. Traer id + teléfono desde agent_audio_data (últimos 7 días)
df_phones = fetch_recent_agent_phones(days=2)  # columnas: id, telefono

# 2. Procesar y guardar en vap_clients
process_telefonos_from_dataframe(
    df_phones,
    telefono_column="telefono",
    id_column="id",          # este es id_agent_audio_data
)