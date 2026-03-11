import os
from datetime import datetime
import sys

from dotenv import load_dotenv

from audio.audioPrepDeep import main_process_batch,audioOutputWpm
from audio.audioShazam import merge_peaks, ncc_fft
from auditableSelector.affectedClassifier import combinedRejectionReason
import shutil

load_dotenv()

import torch
from auditableSelector.main import classify_audio_quality
from database.ModelSelector import resolve_model_for_sponsor
script_dir = os.path.dirname(os.path.abspath(__file__))

from config.logger import logger
import numpy as np
import pandas as pd

import database.dbConfig as dbcfg
from database.InsertData import insertar_campanias, insertar_fila_reporte, insertar_sponsors, insertar_grafica, insertar_filas_dataframe_agente_score, \
    insertar_filas_dataframe_estadisticas, insertar_filas_dataframe_agentes, insertar_filas_dataframe_afectadas
from setup.MatrixSetup import matrix_setup
from database.S3Loader import cargar_archivos_json_a_s3, cargar_audios_concat_a_s3, cargar_excel_a_s3
from database.SQLDataManager import config_agents, getAgentStats, obtener_id_sponsor, obtener_o_generar_id_graf, \
    obtener_charts_recientes_campania, obtener_ultimo_id_graf, calls_this_campaign, merge_with_null_agent, \
    mark_campaign_processed
from vapScoreEngine.ScoreEngineBeta import score_camp
from utils.VapFunctions import measure_speed_classification,get_campaign_parameters
from transcript.batchTranscript import batchTranscript
from utils.VapUtils import get_data_from_name, calculate_total_audio_minutes, insertTopicTagsJson, \
    list_files_to_dataframe, getTranscriptParagraphsJsonHighlights
from setup.CampaignSetup import obtener_inventario,campaign_setup,campaign_setup_manual_route, tagCleanning
from utils.campaignMetrics import count_local_files
from dataVis.generalGraph import buildGraphData
retry_attempts = 3


hangup_signature = "data/hangout/BANCOLBI_20250910-180407_2863174_1757545447_1032414142_3204579981-all_fragment_signature1.wav"  

def chunk_list(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]



def preprocess_in_batches(
    campaign_directory,
    processed_output_directory,
    hangup_signature,
    template_resample_to=8000,
    batch_size=100
):
    """
    Llama a main_process_batch en uno o varios batches, dependiendo de cuántos audios haya.
    Devuelve (df_audio, df_windows): DataFrames con métricas por archivo y ventanas de volumen.
    """

    # Asegurar carpeta misc
    misc_dir = os.path.join(campaign_directory, "misc")
    os.makedirs(misc_dir, exist_ok=True)

    # 1. Encontrar todos los audios que se deben preprocesar
    audio_files = []
    for root, dirs, files in os.walk(campaign_directory):
        # Saltar la carpeta de procesados para no re-preprocesar
        if os.path.abspath(root).startswith(os.path.abspath(processed_output_directory)):
            continue

        for f in files:
            if f.lower().endswith((".wav", ".mp3")):
                audio_files.append(os.path.join(root, f))

    print(f"TOTAL AUDIO FILES FOUND FOR PREPROCESSING: {len(audio_files)}")

    if len(audio_files) == 0:
        print("No se encontraron audios para preprocesar.")
        return pd.DataFrame(), pd.DataFrame()

    # 2. Si no excede el batch_size, se comporta como antes: un solo llamado
    if len(audio_files) <= batch_size:
        print("Número de archivos menor o igual al batch_size, usando main_process_batch normal.")
        df_audio, df_windows = main_process_batch(
            input_dir=campaign_directory,
            output_dir=processed_output_directory,
            template_path=hangup_signature,
            template_resample_to=template_resample_to,
            preserve_subdirs=True,
            force_wav_out=True,
            verbose=True,
        )
        return df_audio, df_windows

    # 3. Si excede, partimos en batches
    print(f"Procesando en batches de tamaño {batch_size}...")

    tmp_input_dir = os.path.join(campaign_directory, "_tmp_batch_input")
    os.makedirs(tmp_input_dir, exist_ok=True)

    all_audio_batches = []
    all_window_batches = []

    for batch_idx, files_batch in enumerate(chunk_list(audio_files, batch_size), start=1):
        print(f"--- PREPROCESSING BATCH {batch_idx} ({len(files_batch)} files) ---")

        # Limpiar carpeta temporal
        for item in os.listdir(tmp_input_dir):
            item_path = os.path.join(tmp_input_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

        # Copiar o hacer hardlink de los archivos al dir temporal
        for src in files_batch:
            dst = os.path.join(tmp_input_dir, os.path.basename(src))
            try:
                os.link(src, dst)  # hardlink (rápido, poco disco)
            except OSError:
                shutil.copy2(src, dst)  # si no se puede hardlink, copiar

        df_audio_batch, df_windows_batch = main_process_batch(
            input_dir=tmp_input_dir,
            output_dir=processed_output_directory,
            template_path=hangup_signature,
            template_resample_to=template_resample_to,
            preserve_subdirs=True,
            force_wav_out=True,
            verbose=True,
        )

        all_audio_batches.append(df_audio_batch)
        all_window_batches.append(df_windows_batch)

    # 4. Unir todos los resultados
    if len(all_audio_batches) == 0:
        return pd.DataFrame(), pd.DataFrame()
    df_audio = pd.concat(all_audio_batches, ignore_index=True)
    df_windows = pd.concat(all_window_batches, ignore_index=True)

    return df_audio, df_windows


def main(PREFIX,days_ago,mode,oparam1=None):
    #    write_auto, write_manual, dry_run
    #    route = "Colombia/Bancolombia/Abril/"    POR SI ACASO SE NECESITA PROCESAR MANUAL
    if not os.path.exists("process/"):
        os.makedirs("process/", exist_ok=True)
    if not os.path.exists("logs/"):
        os.makedirs("logs/", exist_ok=True)
    log_file_name = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{PREFIX}.txt"
    script_dir_logs = os.path.join(script_dir, 'logs')
    log_file_path = os.path.join(script_dir_logs, log_file_name)
    log_file = open(log_file_path, "w") 

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = logger(original_stdout, log_file)
    sys.stderr = logger(original_stderr, log_file)
    # ---------------------------------------------------------------
    # INVENTARIO Y PARÁMETROS DE CAMPAÑA (CONSULTA ÚNICA A BD)
    # ---------------------------------------------------------------
    df_inventory = obtener_inventario()
    mapping_camps_expanded = df_inventory.assign(path=df_inventory['path'].str.split(',')).explode('path') #ALL PREFIX
    campaign_parameters = get_campaign_parameters(PREFIX, mapping_camps_expanded)
    
    OPTION_TRANSCRIPT_ENGINE="DEEPGRAM"

    campaign_directory = 'process/'+ PREFIX +'/'
    if mode == "write_auto" or mode == "dry_run":
        campaign_parameters,route,audioList = campaign_setup(PREFIX,mapping_camps_expanded,campaign_parameters,days_ago,oparam1)
    if mode == "write_manual":
        campaign_parameters,route,audioList = campaign_setup_manual_route(PREFIX,route,oparam1)
    print("RUTA DE LOS AUDIOS EN S3")
    print(route)

    # Tomar ids e identificadores de campaña por config

    camp_id = str(campaign_parameters['campaign_id'])
    id_campania_num = str(camp_id)

    # Conectarse a la base de datos

    conexion = dbcfg.conectar(HOST=dbcfg.HOST_DB_VAP,  
                                  DATABASE=dbcfg.DB_NAME_VAP,  
                                  USERNAME=dbcfg.USER_DB_VAP,  
                                  PASSWORD=dbcfg.PASSWORD_DB_VAP)
    
    # Consultar ids adicionales

    id_sponsor = str(obtener_id_sponsor(conexion, campaign_parameters['sponsor']))
    id_graf = obtener_o_generar_id_graf(conexion, id_campania_num, id_sponsor)
    print(f"ID DE CAMPAÑA: {camp_id}")

    # ---------------------------------------------------------------
    # RESOLUCIÓN DEL MODELO DE SEGMENTACIÓN POR SPONSOR
    # Se consulta vap_models para obtener el modelo personalizado más
    # reciente (y validado) del sponsor. Si no existe, se usa el modelo
    # global definido en TEXT_MODEL_DIR / TIME_PRIORS_JSON.
    # ---------------------------------------------------------------
    print(f"RESOLVIENDO MODELO DE SEGMENTACIÓN PARA SPONSOR {id_sponsor}...")
    sponsor_model_dir, sponsor_time_priors = resolve_model_for_sponsor(
        conexion, int(id_sponsor)
    )
    if sponsor_model_dir:
        print(f"MODELO SELECCIONADO: {sponsor_model_dir}")
    else:
        print("USANDO MODELO GLOBAL POR DEFECTO (TEXT_MODEL_DIR)")

    # Generar directorio local

    processed_output_directory = campaign_directory + "processed/"
    if not os.path.exists(processed_output_directory):
        os.makedirs(processed_output_directory, exist_ok=True)

    # ---------------------------------------------------------------
    # PREPROCESAMIENTO DE AUDIOS
    # ---------------------------------------------------------------

    
    # Asegurar carpeta misc (por si acaso)
    if not os.path.exists(os.path.join(campaign_directory, "misc")):
        os.makedirs(os.path.join(campaign_directory, "misc"), exist_ok=True)

    # ---------------------------------------------------------------
    # PREPROCESAMIENTO DE AUDIOS 
    # ---------------------------------------------------------------
    BATCH_SIZE = 200

    audioData, df_windows = preprocess_in_batches(
        campaign_directory=campaign_directory,
        processed_output_directory=processed_output_directory,
        hangup_signature=hangup_signature,
        template_resample_to=8000,
        batch_size=BATCH_SIZE,
    )
    audioData = audioData.merge(audioList, on="file_name", how="left")
    audio_files = [f for f in os.listdir(processed_output_directory) if f.endswith('.mp3') or f.endswith('.wav')]
    if len(audio_files) == 0:
        print("No se encontraron archivos de audio en la campaña.")
        print("Asegúrate de que la ruta de la campaña sea correcta y contenga archivos de audio.")
            
    # ---------------------------------------------------------------
    # TRANSCRIBIR AUDIOS
    # ---------------------------------------------------------------

    batchTranscript(audio_files,processed_output_directory,campaign_directory, retry_attempts, OPTION_TRANSCRIPT_ENGINE)

    # ---------------------------------------------------------------
    # MEDICION DE VELOCIDAD Y VOLUMEN EN VENTANAS
    # ---------------------------------------------------------------
    df_windows_wpm, audioData_vel = audioOutputWpm(
                                    audio_outputs=audioData,
                                    df_windows=df_windows,
                                    transcripts_dir=campaign_directory,
                                    window_sec=1,
                                    hop_sec=1,
                                    vol_agg="mean"
                                )
    summary_excel_path = os.path.join(campaign_directory, "misc/audio_outputs_test.xlsx")
    df_windows_wpm.to_excel(summary_excel_path, index=False)
    audioData_vel.to_excel(os.path.join(campaign_directory, "misc/audio_windows_test.xlsx"), index=False)

    # ---------------------------------------------------------------
    # SETUP DE CALIFICACIÓN (EN MODO GPU INSTANCE ESTA PARTE SE HACE EXTERNA A LA EC2 Y LOCAL)
    # ---------------------------------------------------------------

    print("CONFIGURANDO MATRIZ DE CALIFICACIÓN...")
    (topics_df,topics_df_grouped,permitidas,no_perm,guion)=matrix_setup(PREFIX,campaign_directory,df_inventory)
    print("CALIFICANDO LLAMADAS EN: " + str(campaign_directory))
    
    try:
        MP3F = (count_local_files(campaign_directory,'.mp3') +count_local_files(campaign_directory,'.wav')+ 
                count_local_files(campaign_directory + '/isolated' ,extension='.mp3') +
                count_local_files(campaign_directory + '/isolated' ,extension='.wav'))
        TAM = calculate_total_audio_minutes(campaign_directory) + calculate_total_audio_minutes(campaign_directory + '/isolated')
    except:
        MP3F = count_local_files(campaign_directory,'.mp3') +count_local_files(campaign_directory,'.wav')
        TAM = calculate_total_audio_minutes(campaign_directory)
    print("TOTAL LLAMADAS VACÍAS: " + str(count_local_files(campaign_directory + '/isolated' ,extension='.mp3') +count_local_files(campaign_directory + '/isolated' ,extension='.wav')))
    print("TOTAL LLAMADAS NO VACÍAS: " + str(count_local_files(campaign_directory,'.mp3') + count_local_files(campaign_directory,'.wav') ) )
    try:
        TMO = TAM / MP3F
    except ZeroDivisionError:
        TMO = 0
        raise Exception("Campaña vacía: No se encontraron archivos de audio válidos para la fecha.")
    print("TOTAL WAV : " + str(count_local_files(campaign_directory,'.wav')))
    if not os.path.exists(campaign_directory+PREFIX+'_RAW'):
        os.makedirs(campaign_directory+PREFIX+'_RAW', exist_ok=True)
    if not os.path.exists(campaign_directory+PREFIX+'_PROCESSED'):
        os.makedirs(campaign_directory+PREFIX+'_PROCESSED', exist_ok=True)
    if not os.path.exists(campaign_directory+PREFIX+'_RECONS'):
        os.makedirs(campaign_directory+PREFIX+'_RECONS', exist_ok=True)
    (MAT_CALLS_CAMPAIGN,STATISTICS,MAT_COMPLETE_TOPICS,all_musnt_keywords,
    topics_stats_convs_scores,topics_stats_convs)=score_camp(campaign_directory, PREFIX, TMO,topics_df,
                                                             df_windows=audioData_vel,
                                                             model_dir=sponsor_model_dir,
                                                             time_priors_json=sponsor_time_priors)

    AGENTES_DB=config_agents(camp_id)
    AGENTES_DB.to_excel(campaign_directory+"misc/AGENTES_DB.xlsx")

    # ---------------------------------------------------------------
    # TRAER Y MERGEAR CON AGENTES (FUTURA VERSIÓN #TODO NUEVO MÓDULO)
    # ---------------------------------------------------------------

    LlamadasPorAgente= merge_with_null_agent(AGENTES_DB, MAT_CALLS_CAMPAIGN)

    final_json_route = campaign_directory + 'transcript_sentences/'
    if not os.path.exists(final_json_route):
        os.makedirs(final_json_route)

    # ---------------------------------------------------------------
    # MEDIR PALABRAS PERMITIDAS Y NO PERMITIDAS
    # ---------------------------------------------------------------
    print(" PALABRAS NO PERMITIDAS :" + str(no_perm+all_musnt_keywords))
    getTranscriptParagraphsJsonHighlights(final_json_route, permitidas, no_perm)

    # ---------------------------------------------------------------
    # EVALUACIÓN DE CONCIENCIA DE VENTA
    # Se realiza luego de determinar la ruta de transcritos 'final_json_route'
    # y antes de convertir MAT_CALLS_CAMPAIGN a Excel
    # ---------------------------------------------------------------


    def consent_evaluation(MAT_CALLS_CAMPAIGN,do_it: bool = False):
        global consent_evaluation_security_variable, promedio_vc
        promedio_vc = 0.0
        if do_it:
            from consent_evaluator.consent_evaluator import process_transcripts_consent_evaluator
            from consent_evaluator.consent_evaluator_helpers import merge_consent_into_campaign, calcular_promedio_venta_consciente
            try:
                consent_evaluation_df = process_transcripts_consent_evaluator(transcripts_path = final_json_route, script = guion,mode='sample',only_n=2)#Dataframe con evaluación de consentimiento de todo el batch
                print(consent_evaluation_df)
                promedio_vc = calcular_promedio_venta_consciente(consent_evaluation_df)#Promedio de venta consciente en el batch, para posting en Campaña y sponsor.
                MAT_CALLS_CAMPAIGN = merge_consent_into_campaign(MAT_CALLS_CAMPAIGN,consent_evaluation_df)#Merge de evaluación de consentimiento en DataFrame principal de campaña.
            except Exception as e:
                print(f"[ERROR] Ocurrió un error en consent_evaluation: {e}, promedio de venta consciente seteado en 0.0")
        return MAT_CALLS_CAMPAIGN, promedio_vc
    print(f"campaign_parameters['has_vc'] = {campaign_parameters['has_vc']}")
    promedio_vc=0.0
    if campaign_parameters["has_vc"]:
        MAT_CALLS_CAMPAIGN, promedio_vc = consent_evaluation(MAT_CALLS_CAMPAIGN, do_it=False)

    # ---------------------------------------------------------------
    # AÑADIR ETIQUETAS ADICIONALES AL JSON DE DISPLAY
    # ---------------------------------------------------------------
    insertTopicTagsJson(MAT_CALLS_CAMPAIGN, final_json_route)
    MAT_CALLS_CAMPAIGN.to_excel(campaign_directory+"misc/all_graded_full.xlsx")

    # ---------------------------------------------------------------
    # JUNTAR MÉTRICAS POR AGENTE
    # ---------------------------------------------------------------
    LlamadasPorAgente.to_excel(campaign_directory+"misc/llamadas_por_agente.xlsx")
    df_concatenado_agentes_unicos = getAgentStats(LlamadasPorAgente)

    CUT=0.7
    MAC_CUT=0.7
    MAT_CALLS_CAMPAIGN['audio_quality_auditable']=MAT_CALLS_CAMPAIGN['file_name'].apply(classify_audio_quality)
    MAT_CALLS_CAMPAIGN['auditable_strikes']=(MAT_CALLS_CAMPAIGN['audio_quality_auditable']+MAT_CALLS_CAMPAIGN['noauditable_transcript_macs']
    +MAT_CALLS_CAMPAIGN['noauditable_transcript_mac_macs']+MAT_CALLS_CAMPAIGN['noauditable_transcript_price_macs'])
    
    df_files = list_files_to_dataframe(campaign_directory + '/isolated/')


    df_files['DATE_TIME'] = df_files['file_name'].apply(lambda x: datetime.strptime(x.split('_')[1], "%Y%m%d-%H%M%S"))
    df_files['LEAD_ID'] = df_files['file_name'].apply(lambda x: x.split('_')[2])
    df_files['AGENT_ID'] = df_files['file_name'].apply(lambda x: x.split('_')[-2])

    df_files['VACIA'] = 1
    MAT_CALLS_CAMPAIGN['VACIA'] = 0
    MAT_CALLS_CAMPAIGN = pd.concat([MAT_CALLS_CAMPAIGN, df_files], axis=0, ignore_index=True, sort=False)
    MAT_CALLS_CAMPAIGN=MAT_CALLS_CAMPAIGN.merge(audioData_vel, on='file_name', how='left')

    REJECTED = combinedRejectionReason(MAT_CALLS_CAMPAIGN, all_musnt_keywords, CUT, MAC_CUT,MAC_CUT)
    REJECTED['VACIA'] = 0
    print("PUNTAJES RECHAZADAS: " + str(REJECTED))

    agents_with_reject = pd.merge(REJECTED,AGENTES_DB, left_on='AGENT_ID', right_on='id_igs', how='left')
    agents_with_reject.to_excel(campaign_directory + "misc/" + 'agents_with_reject.xlsx')

    list_affected = agents_with_reject['id_igs'].value_counts().reset_index()
    list_affected.columns = ['id_igs', 'count']

    x_graph,y_graph = buildGraphData(audioData)

    STATISTICS.to_excel(campaign_directory + "misc/" + 'STATISTICS.xlsx')

    try:
        indice_saludo = STATISTICS[STATISTICS['final_label']  == 'SALUDO']['mean_centroid'].reset_index(drop=True)[0]
    except Exception as e:
        print(f"NO PROFILING DETECTED! USING DEFAULT INDEX, {e}")
        indice_saludo = 0.1
    try:
        if len(STATISTICS[STATISTICS['final_label'] == 'PRODUCTO'])>0:
            indice_producto = STATISTICS[STATISTICS['final_label'] == 'PRODUCTO']['mean_centroid'].reset_index(drop=True)[0]
        else:
            indice_producto = STATISTICS[STATISTICS['final_label'] == 'PERFILAMIENTO']['mean_centroid'].reset_index(drop=True)[0]
    except Exception as e:
        print(f"NO PROFILING DETECTED! USING DEFAULT INDEX {e}")
        indice_producto = 0.3

    try:
        indice_mac = STATISTICS[STATISTICS['final_label'] == 'MAC_DEF']['mean_centroid'].reset_index(drop=True)[0]
    except:
        try:
            indice_mac = STATISTICS[STATISTICS['final_label'] == 'MAC']['mean_centroid'].reset_index(drop=True)[0]
        except Exception as e:
            print(f"NO MACS DETECTED! USING DEFAULT INDEX {e}")
            indice_mac = 0.5

    try:
        indice_precio = STATISTICS[STATISTICS['final_label'] == 'PRECIO_DEF']['mean_centroid'].reset_index(drop=True)[0]
    except:
        try:
            indice_precio = STATISTICS[STATISTICS['final_label'] == 'PRECIO']['mean_centroid'].reset_index(drop=True)[0]
        except Exception as e:
            print(f"NO PRECIOS DETECTED! USING DEFAULT INDEX {e}")
            indice_precio = 0.6

    try:
        if len(STATISTICS[STATISTICS['final_label'] == 'CONFIRMACION DATOS'])>0:
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

    indices = len(x_graph) * np.array([indice_saludo, indice_producto,
                              indice_precio, indice_mac,
                              indice_confirmacion, indice_despedida])

    indices = np.sort(np.rint(indices).astype(int))
    conexion = dbcfg.conectar(HOST=dbcfg.HOST_DB_VAP,  
                                  DATABASE=dbcfg.DB_NAME_VAP,  
                                  USERNAME=dbcfg.USER_DB_VAP,  
                                  PASSWORD=dbcfg.PASSWORD_DB_VAP)
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
    MAT_CALLS_CAMPAIGN['num_concat'] = MAT_CALLS_CAMPAIGN['num_concat'].fillna(1)
    MAT_CALLS_CAMPAIGN.to_excel(campaign_directory + "misc/" + 'MAT_WITH_COLADAS.xlsx')

    conexion = dbcfg.conectar(HOST=dbcfg.HOST_DB_VAP,  
                                  DATABASE=dbcfg.DB_NAME_VAP,  
                                  USERNAME=dbcfg.USER_DB_VAP,  
                                  PASSWORD=dbcfg.PASSWORD_DB_VAP)
    print(int(id_campania_num))


    REJECTED.to_excel(campaign_directory + 'misc/result_rejected_'+ PREFIX +'.xlsx')
    if mode == "dry_run":
        print("Modo dry_run activado. No se realizarán inserciones en la base de datos.")
        print("REJECTED:")
        print(REJECTED.head())
        print("MAT_CALLS_CAMPAIGN:")
        print(MAT_CALLS_CAMPAIGN.head())
        print("Averaged Results P:")
        print("Indices:")
        print(indices)
        REJECTED.to_excel(campaign_directory + 'misc/result_rejected_'+ PREFIX +'.xlsx')
        MAT_CALLS_CAMPAIGN.to_excel(campaign_directory + 'misc/all_calls.xlsx')
        sys.exit(0)
    if mode == "write_auto" or mode == "write_manual":
        print("INSERTANDO KPIS DE CAMPAÑA:")
        insertar_campanias(conexion,campaign_directory,id_campania_num, array_camps, REJECTED, MAT_CALLS_CAMPAIGN, MP3F, TMO,TAM,average_consciousness_score=promedio_vc)
        print("INSERTANDO KPIS DE SPONSOR:")
        insertar_sponsors(conexion,id_sponsor)
        print("DATOS DE GRAFICA EXISTEN: "+ str(obtener_ultimo_id_graf(conexion)))
        print("INSERTANDO GRAFICA:")
        insertar_grafica(conexion,id_campania_num, id_sponsor, id_graf,indices,x_graph,y_graph, STATISTICS, TMO, MAT_CALLS_CAMPAIGN)
        print("INSERTANDO ESTADISTICAS DE AGENTES:")
        insertar_filas_dataframe_agente_score(conexion, id_campania_num,df_concatenado_agentes_unicos,list_affected)
        print("INSERTANDO ESTADISTICAS DE CAMPAÑA:")
        #TODO insertar_filas_dataframe_estadisticas(conexion,id_campania_num, topics_stats_convs_concat_all,df_concatenado_all)
        print("INSERTANDO LLAMADAS:")
        route_transcripts = (campaign_parameters['country'].replace(' ', '') +
                         '/' + campaign_parameters['sponsor'].replace(' ', '') + '/transcript_sentences/' +
                         route.split('/')[-2] + '/')
        route_audios = (campaign_parameters['country'].replace(' ', '') +
                         '/' + campaign_parameters['sponsor'].replace(' ', '') + '/' +
                         route.split('/')[-2] + '/')
        insertar_filas_dataframe_agentes(conexion, campaign_directory,id_campania_num, STATISTICS, MAT_CALLS_CAMPAIGN,route,route_transcripts,indices)
        bring_llamadas = calls_this_campaign(int(id_campania_num), conexion)
        bring_llamadas['sales_arguments_percentage'] = bring_llamadas['sales_arguments_percentage'].apply(
            lambda x: np.clip(x, 0, 100))
        try:
            bring_llamadas[['DATE_TIME','LEAD_ID', 'EPOCH','AGENT_ID', 'PHONE']] = bring_llamadas['name'].apply(lambda x: pd.Series(get_data_from_name(x)))
        except Exception as e:
            bring_llamadas[['DATE_TIME','LEAD_ID', 'AGENT_ID', 'CLIENT_ID']] = bring_llamadas['name'].apply(lambda x: pd.Series(get_data_from_name(x)))

        bring_llamadas=bring_llamadas[['id','name','LEAD_ID' ,'AGENT_ID']]
        bring_llamadas.rename(columns={'id': 'agent_audio_data_id'}, inplace=True)
        
        df_concatenado_afectadas = pd.merge(REJECTED, bring_llamadas, left_on='LEAD_ID', right_on='LEAD_ID', how='left')
        df_concatenado_afectadas.to_excel(campaign_directory + "misc/" + 'df_concatenado_afectadas.xlsx')

        print("RUTA A SUBIR LOS TRANSCRIPTOS")

        print(route_transcripts)
        cargar_archivos_json_a_s3(campaign_directory + '/transcript_sentences/', 'documentos.aihub', route_transcripts)
        cargar_audios_concat_a_s3(campaign_directory + '/processed/', 's3iahub.igs', route_audios)

        ulpoad_result_rejected = True #Seguridad para no subir excel si no se ha analizado todo @david y @ Yop

        if ulpoad_result_rejected:
            try:
                cargar_excel_a_s3(campaign_directory + 'misc/result_rejected_'+ PREFIX +'.xlsx', 'documentos.aihub', route_transcripts + 'Analytics/')#UPLOAD EXCELS (en modificación aún).
            except Exception as e:
                print(f"[ERROR] No se pudo cargar result_rejected.xlsx a S3: {e}")


        print("RUTA A SUBIR LOS AUDIOS")

        print(route_audios)
        insertar_filas_dataframe_afectadas(conexion, df_concatenado_afectadas)
        mark_campaign_processed(int(id_campania_num))
        insertar_fila_reporte(conexion, 
                            PREFIX,
                            MP3F,
                            campaign_parameters,
                            REJECTED,
                            str(count_local_files(campaign_directory + '/isolated' ,extension='.mp3')))


if __name__ == '__main__':
    param1 = str(sys.argv[1])
    param2 = int(sys.argv[2])
    mode   = "write_auto"  # write_auto, write_manual, dry_run
    oparam1 = param1
    param1 = tagCleanning(param1)
    print("INICIANDO PROCESO PARA "+ oparam1 +" CON DELAY "+ str(param2))
    print("CUDA disponible:", torch.cuda.is_available())
    print("Dispositivo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    main(param1, param2, mode, oparam1)
    print("FINALIZADO PROCESO PARA " + param1 + " CON DELAY " + str(param2) + " EXITOSAMENTE.")
