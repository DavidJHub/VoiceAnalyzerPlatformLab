import json
import os
from datetime import datetime
import sys

import torch

from auditableSelector.main import classify_audio_quality
from auditableSelector.confidenceSelection import calculate_confidence_scores_per_topic


script_dir = os.path.dirname(os.path.abspath(__file__))



class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, message):
        for file in self.files:
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            file.write(safe_message)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()




import numpy as np
import pandas as pd

from database.InsertData import insertar_campanias, insertar_sponsors, insertar_grafica, insertar_filas_dataframe_agente_score, \
    insertar_filas_dataframe_estadisticas, insertar_filas_dataframe_agentes, insertar_filas_dataframe_afectadas
from setup.MatrixSetup import matrix_setup
from database.S3Loader import cargar_archivos_json_a_s3
from database.SQLDataManager import config_agents, obtener_id_sponsor, conexion, obtener_o_generar_id_graf, \
    obtener_charts_recientes_campania, conectar, verificar_id_graf, calls_this_campaign, merge_with_null_agent, \
    mark_campaign_processed
from vapScoreEngine.ScoreEngineBeta import score_camp, process_directory_and_average, genRejectionReason
from utils.VapFunctions import measure_speed_classification
from utils.campaignMetrics import count_local_files
from transcript.VapTranscript import transcribe_with_retry, retry_attempts
from utils.VapUtils import get_data_from_name, jsonDecompose, calculate_total_audio_minutes, insertTopicTagsJson, \
    list_files_to_dataframe, getTranscriptParagraphsJsonHighlights
from setup.CampaignSetup import default_json_structure


def main(PREFIX,route):
    from setup.CampaignSetup_noInsert import campaign_setup

    ###########################################################
    #################### PARAMETROS ###########################
    ###########################################################

    campaign_id_route = PREFIX
    log_file_name = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{PREFIX}.txt"
    script_dir_logs = os.path.join(script_dir, 'logs')
    log_file_path = os.path.join(script_dir_logs, log_file_name)
    log_file = open(log_file_path, "w")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)
    OPTION_TRANSCRIPT_ENGINE="DEEPGRAM"
    campaign_directory = 'process/'+campaign_id_route +'/'
    campaign_parameters,route = campaign_setup(campaign_id_route,route)
    print("RUTA DE LOS AUDIOS EN S3")
    print(route)
    camp_id = str(campaign_parameters['campaign_id'])
    id_campania_num = str(camp_id)
    conexion = conectar()
    id_sponsor = str(obtener_id_sponsor(conexion, campaign_parameters['sponsor']))
    id_graf = obtener_o_generar_id_graf(conexion, id_campania_num, id_sponsor)
    print(f"ID DE CAMPAÑA: {camp_id}")
    processed_output_directory = campaign_directory + "processed/"
    if not os.path.exists(processed_output_directory):
        os.makedirs(processed_output_directory, exist_ok=True)
    audio_files = [f for f in os.listdir(campaign_directory) if f.endswith('.mp3')]
    for filename in audio_files :
        input_path_filename = os.path.join(campaign_directory, filename)
        output_path = os.path.join(processed_output_directory, filename)
        if filename.endswith(".mp3") and filename not in os.listdir(processed_output_directory):
            #procesar_audio(input_path_filename, processed_output_directory)
    for filename in audio_files:
        ruta_completa = os.path.join(campaign_directory, filename)
        nombre_archivo_sin_extension = os.path.splitext(ruta_completa)[0]
        if OPTION_TRANSCRIPT_ENGINE=="DEEPGRAM":
            output_file = os.path.join(f"{nombre_archivo_sin_extension}_transcript.json")
            if not os.path.exists(output_file):
                transcribe_with_retry(ruta_completa, output_file, retry_attempts)
            if os.path.exists(output_file):
                if os.path.getsize(output_file) < 4000:
                    transcribe_with_retry(ruta_completa, output_file, retry_attempts)
            if os.path.exists(output_file):
                if os.path.getsize(output_file) > 4000:
                    print(f"Transcript for {filename} already exists. Skipping transcription.")
            print(f'Verifying transcript {output_file}')
            try:
                jsonDecompose(output_file)
            except (Exception,) as e:
                print(f"{output_file} unsolvable, using default empty file")
                data = default_json_structure
                file = open(output_file, "w")
                json.dump(data, file, indent=4)
                file.close()
    print("CONFIGURANDO MATRIZ DE CALIFICACIÓN...")
    (topics_df,topics_df_grouped,permitidas,no_perm,clust_name,
     kw_names)=matrix_setup(campaign_id_route)
    print("CALIFICANDO LLAMADAS EN: " + str(campaign_directory))
    
    try:
        MP3F = count_local_files(campaign_directory,'.mp3') + count_local_files(campaign_directory + '/isolated' ,extension='.mp3')
        TAM = calculate_total_audio_minutes(campaign_directory) + calculate_total_audio_minutes(campaign_directory + '/isolated')
    except:
        MP3F = count_local_files(campaign_directory,'.mp3')
        TAM = calculate_total_audio_minutes(campaign_directory)
    TMO = TAM / MP3F
    if campaign_id_route == 'BR_ESP_':
        campaign_id_route = 'BRESP_'
    if not os.path.exists(campaign_directory+campaign_id_route+'_RAW'):
        os.makedirs(campaign_directory+campaign_id_route+'_RAW', exist_ok=True)
    if not os.path.exists(campaign_directory+campaign_id_route+'_PROCESSED'):
        os.makedirs(campaign_directory+campaign_id_route+'_PROCESSED', exist_ok=True)
    if not os.path.exists(campaign_directory+campaign_id_route+'_RECONS'):
        os.makedirs(campaign_directory+campaign_id_route+'_RECONS', exist_ok=True)
    (MAT_CALLS_CAMPAIGN,STATISTICS,MAT_COMPLETE_TOPICS,all_musnt_keywords,
    topics_stats_convs_scores,topics_stats_convs)=score_camp(campaign_directory, campaign_id_route, TMO,topics_df)

    AGENTES_DB=config_agents(camp_id)
    AGENTES_DB.to_excel(campaign_directory+"misc/AGENTES_DB.xlsx")

    LlamadasPorAgente= merge_with_null_agent(AGENTES_DB, MAT_CALLS_CAMPAIGN)
    LlamadasPorAgente.to_excel(campaign_directory+"misc/calls_with_agents.xlsx")

    final_json_route = campaign_directory + 'transcript_sentences/'
    if not os.path.exists(final_json_route):
        os.makedirs(final_json_route)

    getTranscriptParagraphsJsonHighlights(final_json_route, permitidas, no_perm)


    MAT_CALLS_CAMPAIGN['velocity_classification_macs']=MAT_CALLS_CAMPAIGN['topic_words_p_m_macs'].apply(lambda x: measure_speed_classification(x))
    MAT_CALLS_CAMPAIGN['velocity_classification_prices']=MAT_CALLS_CAMPAIGN['topic_words_p_m_prices'].apply(lambda x: measure_speed_classification(x))

    insertTopicTagsJson(MAT_CALLS_CAMPAIGN, final_json_route)
    MAT_CALLS_CAMPAIGN.to_excel(campaign_directory+"misc/all_graded_full.xlsx")

    '''    df_concatenado_llamadas_unicas = df_concatenado_all.groupby('file_name').apply(
        lambda x: x.reset_index(drop=True)).reset_index(drop=True)

    df_concatenado_llamadas_unicas.to_excel(campaign_directory + "misc/" + 'df_concatenado_llamadas_unicas.xlsx')'''
    df_concatenado_agentes_unicos = LlamadasPorAgente.groupby('id_igs').agg(
        CONFIDENCE=('confidence', 'mean'),
        TMO=('TMO', 'mean'),
        MUST_HAVE_COUNT=('count_must_have', 'sum'),
        MUST_HAVE_RATE=('must_have_rate', 'mean'),
        FORBIDDEN_COUNT=('count_forbidden', 'sum'),
        FORBIDDEN_RATE=('forbidden_rate', 'mean'),
        SCORE=('score', 'mean'),
        MAC_SCORE=('best_mac_likelihood_macs', 'mean'),
        PRICE_SCORE=('best_mac_likelihood_prices', 'mean'),
        words_p_m=('topic_words_p_m_macs', 'mean'),
        volume_rms=('volume_db_mac', 'mean'),
        volume_classification=('volume_classification_mac', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
        velocity_classification=('velocity_classification_macs', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
        DATE=('DATE_TIME', 'last'),
    ).reset_index().apply(lambda x: x.reset_index(drop=True)).reset_index(drop=True)

    df_concatenado_agentes_unicos = df_concatenado_agentes_unicos.drop_duplicates(subset='id_igs')
    df_concatenado_agentes_unicos.to_excel(campaign_directory + "misc/df_concatenado_agentes_unicos.xlsx")
    CUT=0.8
    MAC_CUT=0.8

    MAT_CALLS_CAMPAIGN['audio_quality_auditable']=MAT_CALLS_CAMPAIGN['file_name'].apply(classify_audio_quality)
    MAT_CALLS_CAMPAIGN['auditable_strikes']=(MAT_CALLS_CAMPAIGN['audio_quality_auditable']+MAT_CALLS_CAMPAIGN['noauditable_transcript_macs']
    +MAT_CALLS_CAMPAIGN['noauditable_transcript_mac_macs']+MAT_CALLS_CAMPAIGN['noauditable_transcript_price_macs'])

    REJECTED = MAT_CALLS_CAMPAIGN[(MAT_CALLS_CAMPAIGN['count_forbidden'] > len(all_musnt_keywords)*0.2)
                                       | (MAT_CALLS_CAMPAIGN['score'] < np.mean(MAT_CALLS_CAMPAIGN['score']) * CUT)
                                       | (MAT_CALLS_CAMPAIGN['best_mac_likelihood_prices'] < MAC_CUT)
                                       | (MAT_CALLS_CAMPAIGN['best_mac_likelihood_macs'] < MAC_CUT)
                                       | ((MAT_CALLS_CAMPAIGN['volume_classification_mac']=='low') & (MAT_CALLS_CAMPAIGN['velocity_classification_macs'] == 'high'))
                                       | ((MAT_CALLS_CAMPAIGN['volume_classification_prices']=='low') & (MAT_CALLS_CAMPAIGN['velocity_classification_prices'] == 'high'))
                                       | (MAT_CALLS_CAMPAIGN['TMO'] < np.mean(MAT_CALLS_CAMPAIGN['TMO'])*MAC_CUT)
                                       | (MAT_CALLS_CAMPAIGN['auditable_strikes'] > 3)]

    print("PUNTAJES RECHAZADAS: " + str(REJECTED))

    agents_with_reject = pd.merge( REJECTED,AGENTES_DB, left_on='AGENT_ID', right_on='id_igs', how='left')
    agents_with_reject.to_excel(campaign_directory + "misc/" + 'agents_with_reject.xlsx')

    list_affected = agents_with_reject['id_igs'].value_counts().reset_index()
    list_affected.columns = ['id_igs', 'count']

    averaged_results_p = process_directory_and_average(campaign_directory, num_windows=400)
    averaged_results_p['speed_n'] = averaged_results_p['speed_n'] / 400
    averaged_results_p.to_excel(campaign_directory + "misc/" + 'averaged_results_p.xlsx')
    STATISTICS.to_excel(campaign_directory + "misc/" + 'STATISTICS.xlsx')

    try:
        indice_saludo = STATISTICS[STATISTICS['final_label']   == 'SALUDO']['mean_centroid'].reset_index(drop=True)[0]
    except Exception as e:
        print(f"NO PROFILING DETECTED! USING DEFAULT INDEX, {e}")
        indice_saludo = 0.1
    try:
        print("NO OFFERINGS DETECTED! USING DEFAULT INDEX")
        if len(STATISTICS[STATISTICS['final_label'] == 'OFRECIMIENTO'])>0:
            indice_producto = STATISTICS[STATISTICS['final_label'] == 'OFRECIMIENTO']['mean_centroid'].reset_index(drop=True)[0]
        else:
            indice_producto = STATISTICS[STATISTICS['final_label'] == 'PERFILAMIENTO']['mean_centroid'].reset_index(drop=True)[0]
    except Exception as e:
        print(f"NO PROFILING DETECTED! USING DEFAULT INDEX {e}")
        indice_producto = 0.3

    try:
        indice_mac = STATISTICS[STATISTICS['final_label'] == 'MAC_DEF']['mean_centroid'].reset_index(drop=True)[0]
    except Exception as e:
        print(f"NO MACS DETECTED! USING DEFAULT INDEX {e}")
        indice_mac = 0.5

    try:
        indice_precio = STATISTICS[STATISTICS['final_label'] == 'PRECIO_DEF']['mean_centroid'].reset_index(drop=True)[0]
    except Exception as e:
        print(f"NO PRICES DETECTED! USING DEFAULT INDEX {e}")
        indice_precio = 0.6
    try:
        if len(STATISTICS[STATISTICS['final_label'] == 'CONFIRMACION DE DATOS'])>0:
            indice_confirmacion = STATISTICS[STATISTICS['final_label'] == 'CONFIRMACION DE DATOS']['mean_centroid'].reset_index(drop=True)[0]
        else:
            indice_confirmacion = STATISTICS[STATISTICS['final_label'] == 'TERMINOS LEGALES']['mean_centroid'].reset_index(drop=True)[0]
    except Exception as e:
        print(f"NO DATA CONFIRMATION DETECTED! USING DEFAULT INDEX {e}")
        indice_confirmacion = 0.7
    try:
        indice_despedida = STATISTICS[STATISTICS['final_label'] == 'DESPEDIDA']['mean_centroid'].reset_index(drop=True)[0]
    except Exception as e:
        print(f"NO FAREWELL DETECTED! USING DEFAULT INDEX {e}")
        indice_despedida = 0.9

    indices = len(averaged_results_p) * np.array([indice_saludo, indice_producto,
                              indice_precio, indice_mac,
                              indice_confirmacion, indice_despedida])

    indices = np.sort(np.rint(indices).astype(int))
    conexion=conectar()
    array_camps = obtener_charts_recientes_campania(conexion, id_campania_num)

    topics_stats_convs.to_excel(campaign_directory + "misc/" + 'topics_stats_convs.xlsx')
    topics_stats_convs['velocity_classification'] = topics_stats_convs['topic_words_p_m'].apply(measure_speed_classification)
    topics_stats_convs_concat = topics_stats_convs.groupby('final_label').agg({
        'topic_mean_conf': 'mean',
        'topic_words_p_m': 'mean',
        'time_centroid': 'mean',
        'velocity_classification': lambda x: x.mode().iloc[0],
    }).reset_index()

    topics_stats_convs_concat_all = pd.merge(topics_stats_convs_concat, STATISTICS, left_on='final_label',
                                             right_on='final_label', how='inner')

    topics_stats_convs_concat_all.to_excel(campaign_directory + "misc/" + 'topics_stats_convs_concat_all.xlsx')
    topics_stats_convs_concat_all['section_score']=topics_stats_convs_concat_all['topic_mean_conf']*10/np.max(topics_stats_convs_concat_all['topic_mean_conf'])

    df_files = list_files_to_dataframe(campaign_directory + '/isolated/')


    df_files['DATE_TIME'] = df_files['file_name'].apply(lambda x: datetime.strptime(x.split('_')[1], "%Y%m%d-%H%M%S"))
    df_files['LEAD_ID'] = df_files['file_name'].apply(lambda x: x.split('_')[2])
    df_files['AGENT_ID'] = df_files['file_name'].apply(lambda x: x.split('_')[-2])
    df_files['VACIA'] = 1
    REJECTED['VACIA'] = 0
    MAT_CALLS_CAMPAIGN['VACIA'] = 0
    #REJECTED['date-time'] = REJECTED['file_name'].apply(lambda x: datetime.strptime( x.split('_')[1], "%Y%m%d-%H%M%S") )

    #res = cct.reindex(columns=REJECTED.columns, fill_value=None)
    REJECTED = pd.concat([REJECTED, df_files], axis=0, ignore_index=True, sort=False)
    REJECTED=genRejectionReason(REJECTED)
    MAT_CALLS_CAMPAIGN = pd.concat([MAT_CALLS_CAMPAIGN, df_files], axis=0, ignore_index=True, sort=False)
    REJECTED.to_excel(campaign_directory + "misc/" + 'REJECTED.xlsx')
    MAT_CALLS_CAMPAIGN=MAT_CALLS_CAMPAIGN.fillna(0)
    MAT_CALLS_CAMPAIGN.to_excel(campaign_directory + "misc/" + 'MAT_WITH_COLADAS.xlsx')
    REJECTED.to_excel(campaign_directory + 'misc/result_rejected.xlsx')
    conexion = conectar()
    print(int(id_campania_num))
    print("Generando reportes...")
    def transform_datetime(date_string, date_format='%Y%m%d-%H%M%S'):
        return datetime.strptime(date_string, date_format)
    route_transcripts = (campaign_parameters['country'].replace(' ', '') +
                         '/' + campaign_parameters['sponsor'].replace(' ', '') + '/transcript_sentences/' +
                         route.split('/')[-2] + '/')
    topics_stats_convs.to_excel(campaign_directory + "misc/" + 'topics_stats_convs.xlsx')
    topics_stats_convs['LEAD_ID'] = topics_stats_convs['file_name'].apply(lambda x: x.split('_')[-2])

    bring_llamadas = calls_this_campaign(int(id_campania_num), conexion)
    bring_llamadas['sales_arguments_percentage'] = bring_llamadas['sales_arguments_percentage'].apply(
        lambda x: np.clip(x, 0, 100))
    
    bring_llamadas[['DATE_TIME','LEAD_ID', 'EPOCH','AGENT_ID', 'PHONE']] = bring_llamadas['name'].apply(lambda x: pd.Series(get_data_from_name(x)))

    bring_llamadas=bring_llamadas[['id','name','LEAD_ID', 'EPOCH','AGENT_ID']]
    bring_llamadas.rename(columns={'id': 'agent_audio_data_id'}, inplace=True)
    
    df_concatenado_afectadas = pd.merge(REJECTED, bring_llamadas, left_on='LEAD_ID', right_on='LEAD_ID', how='left')
    df_concatenado_afectadas.to_excel(campaign_directory + "misc/" + 'df_concatenado_afectadas.xlsx')
    mark_campaign_processed(int(id_campania_num))

if __name__ == '__main__':
    # Comprobamos que vengan exactamente 2 parámetros extra además del nombre del script
    if len(sys.argv) != 3:
        print("Uso correcto: python main.py 'PREFIX', días atras")
        sys.exit(1)
    param1 = str(sys.argv[1])
    param2 = int(sys.argv[2])
    print("INICIANDO PROCESO PARA "+ param1 +" CON DELAY "+ str(param2))
    print("CUDA disponible:", torch.cuda.is_available())
    print("Dispositivo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    main(param1, param2)
    print("FINALIZADO PROCESO PARA " + param1 + " CON DELAY " + str(param2) + " EXITOSAMENTE.")



# Cerrar el archivo de log
#log_file.close()


