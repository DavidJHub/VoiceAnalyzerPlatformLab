
from database.InsertData import insertar_campanias, insertar_sponsors, insertar_grafica, insertar_filas_dataframe_agente_score, \
    insertar_filas_dataframe_estadisticas, insertar_filas_dataframe_agentes, insertar_filas_dataframe_afectadas
from setup.MatrixSetup import matrix_setup
from database.S3Loader import cargar_archivos_json_a_s3
from database.SQLDataManager import config_agents, obtener_id_sponsor, conexion, obtener_o_generar_id_graf, \
    obtener_charts_recientes_campania, conectar, verificar_id_graf, calls_this_campaign, merge_with_null_agent, \
    mark_campaign_processed
from Deprecated.ScoreEngineRegrade import score_camp, process_directory_and_average, smooth_array, filter_dataframe_by_directory
from utils.VapFunctions import calculate_folder_kld, measure_speed_classification
from utils.campaignMetrics import count_local_files
from transcript.VapTranscript import transcribe_with_retry, retry_attempts
from utils.VapUtils import jsonDecompose, calculate_total_audio_minutes, insertTopicTagsJson, merge_keep_right, sort_by_variance, \
    list_files_to_dataframe, getTranscriptParagraphsJsonHighlights
from setup.CampaignSetup import default_json_structure

import json
import os
from datetime import datetime
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))



import numpy as np
import pandas as pd



def main(PREFIX,days_ago):
    from Deprecated.CampaignSetupRegrade import campaign_setup
    from audio.audioPreprocessing import procesar_audio

    ###########################################################
    #################### PARAMETROS ###########################
    ###########################################################

    campaign_id_route = PREFIX  # Definir campaña a procesar
                                # #definir dias previos al último registro en s3
    log_file_name = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{PREFIX}.txt"
    log_file_path = os.path.join(script_dir, log_file_name)
    # Abrir el archivo de log para escribir


    OPTION_TRANSCRIPT_ENGINE="DEEPGRAM"


    campaign_directory = campaign_id_route +'/'
    campaign_parameters,route = campaign_setup(campaign_id_route,days_ago)
    print("RUTA DE LOS AUDIOS EN S3")
    print(route)
    camp_id = str(campaign_parameters['campaign_id'])
    id_campania_num = str(camp_id)
    conexion = conectar()
    id_sponsor = str(obtener_id_sponsor(conexion, campaign_parameters['sponsor']))
    id_graf = obtener_o_generar_id_graf(conexion, id_campania_num, id_sponsor)
    print(f"ID DE CAMPAÑA: {camp_id}")
    #processed_output_directory = campaign_directory + "processed/"
    #unread_directory= campaign_directory+ "isolated/"
    #if not os.path.exists(processed_output_directory):
    #    os.makedirs(processed_output_directory, exist_ok=True)
    audio_files = [f for f in os.listdir(campaign_directory) if f.endswith('.mp3')]
    for filename in audio_files :
        input_path_filename = os.path.join(campaign_directory, filename)
        #output_path = os.path.join(processed_output_directory, filename)
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
    (topics_df,topics_df_grouped,permitidas,no_perm,clust_name,
     kw_names,knn_preprocessed_alternative,vectorizer_ngram)=matrix_setup(campaign_id_route)
    print("CALIFICANDO LLAMADAS EN: " + str(campaign_directory))
    TAM = calculate_total_audio_minutes(campaign_directory)
    MP3F = count_local_files(campaign_directory,'.mp3')
    #print(campaign_directory)
    TMO = TAM / MP3F
    (MAT_CALLS_CAMPAIGN,STATISTICS,
     MAT_COMPLETE_TOPICS,all_musnt_keywords,
     topics_stats_convs_scores,topics_stats_convs)=score_camp(campaign_id_route, campaign_directory, TMO,
                                                                                 topics_df,vectorizer_ngram,
                                                                                 knn_preprocessed_alternative)
    AGENTES_DB=config_agents(camp_id)

    AGENTES_DB.to_excel(campaign_directory+"misc/AGENTES_DB.xlsx")
    df_concatenado= merge_with_null_agent(AGENTES_DB, MAT_CALLS_CAMPAIGN)
    df_concatenado.to_excel(campaign_directory+"misc/concat_agents.xlsx")
    #print("LLAMADAS CON AGENTES DE LA CAMPAÑA")
    #print(df_concatenado)
    print("EXTRAYENDO MACS Y PRECIOS")

    MACS = MAT_COMPLETE_TOPICS[MAT_COMPLETE_TOPICS['predicted_cluster']=='confirmacion']
    MACS['Topic scores'] = np.mean(topics_stats_convs['confidence_score'])
    PRICES = MAT_COMPLETE_TOPICS[MAT_COMPLETE_TOPICS['predicted_cluster']=='precio']
    PRICES['Topic scores'] = np.mean(topics_stats_convs['confidence_score'])


    #MACS.to_excel(campaign_directory+'macs.xlsx')'DATE-TIME','CALL_ID','CLIENT_ID','LEAD_ID','PHONE'

    #PRICES.to_excel(campaign_directory+'prices.xlsx')

    MACS['CALL_ID'] = MACS['file_name'].apply(lambda x: x.split('_')[2])
    PRICES['CALL_ID'] = PRICES['file_name'].apply(lambda x: x.split('_')[2])
    MACS['date-time'] = MACS['file_name'].apply(lambda x: x.split('_')[1])
    PRICES['date-time'] = PRICES['file_name'].apply(lambda x: x.split('_')[1])
    MACS['CALL_ID'] = MACS['CALL_ID'].fillna(-1)
    PRICES['CALL_ID'] = PRICES['CALL_ID'].fillna(-1)
    MACS['CALL_ID'] = MACS['CALL_ID'].astype(str)
    PRICES['CALL_ID'] = PRICES['CALL_ID'].astype(str)
    #print('MACS: ')
    #print(MACS)
    #print('PRICES: ')
    #print(PRICES)
    MACS_unique = (
        MACS
        .sort_values(by='confidence_score', ascending=False)
        .drop_duplicates(subset='file_name', keep='first')
    )
    PRICES_unique = (
        PRICES
        .sort_values(by='confidence_score', ascending=False)
        .drop_duplicates(subset='file_name', keep='first')
    )
    final_json_route = campaign_directory + 'transcript_sentences/'
    if not os.path.exists(final_json_route):
        os.makedirs(final_json_route)

    getTranscriptParagraphsJsonHighlights(final_json_route, permitidas, no_perm)

    insertTopicTagsJson(MACS_unique, PRICES_unique, final_json_route)

    MACS.to_excel(campaign_directory+"misc/"+'macs.xlsx')
    MACS_unique.to_excel(campaign_directory + "misc/"+ 'MACS_unique.xlsx')
    PRICES.to_excel(campaign_directory + "misc/" +'prices.xlsx')
    PRICES_unique.to_excel(campaign_directory + "misc/" +'PRICES_unique.xlsx')
    STATISTICS.to_excel(campaign_directory + "misc/" + 'STATISTICS.xlsx')

    df_concatenado_macs = merge_keep_right(df_concatenado, MACS_unique, on='file', how='inner')
    df_concatenado_macs_empty = merge_keep_right(df_concatenado, MACS_unique, on='file', how='left')
    macs_this_date = filter_dataframe_by_directory(df_concatenado_macs, campaign_directory)
    macs_volume=measure_volume_for_dataframe(campaign_directory,macs_this_date)[['file_name','volume_rms', 'volume_classification']]
    df_concatenado_macs =pd.merge(df_concatenado_macs,macs_volume,how='left',left_on='file_name',right_on='file_name')
    df_concatenado_macs['velocity_classification']=df_concatenado_macs['words_p_m'].apply(lambda x: measure_speed_classification(x))
    df_concatenado_macs.to_excel(campaign_directory + "misc/" + 'df_concatenado_macs.xlsx')


    df_concatenado_prices = merge_keep_right(df_concatenado, PRICES_unique, on='file', how='inner')
    df_concatenado_prices_empty = merge_keep_right(df_concatenado, PRICES_unique, on='file', how='left')
    prices_this_date = filter_dataframe_by_directory(df_concatenado_macs, campaign_directory)
    prices_volume=measure_volume_for_dataframe(campaign_directory,prices_this_date)[['file_name','volume_rms', 'volume_classification']]
    df_concatenado_prices =pd.merge(df_concatenado_prices,prices_volume,how='left',left_on='file_name',right_on='file_name')
    df_concatenado_prices['velocity_classification']=df_concatenado_prices['words_p_m'].apply(lambda x: measure_speed_classification(x))
    df_concatenado_prices.to_excel(campaign_directory + "misc/" + 'df_concatenado_prices.xlsx')

    insertTopicTagsJson(df_concatenado_macs, df_concatenado_prices, final_json_route)

    df_concatenado_all=pd.merge(df_concatenado_macs,df_concatenado_prices[['volume_rms','volume_classification','velocity_classification']])

    df_concatenado_llamadas_unicas = df_concatenado_all.groupby('file').apply(
        lambda x: x.reset_index(drop=True)).reset_index(drop=True)

    #df_concatenado_llamadas_unicas.to_excel(campaign_directory + "misc/" + 'df_concatenado_llamadas_unicas.xlsx')
    df_concatenado_agentes_unicos = df_concatenado_all.groupby('LEAD_ID').agg(
        CONFIDENCE=('confidence', 'mean'),
        TMO=('TMO', 'mean'),
        MUST_HAVE_COUNT=('count_must_have', 'sum'),
        MUST_HAVE_RATE=('must_have_rate', 'mean'),
        FORBIDDEN_COUNT=('count_forbidden', 'sum'),
        FORBIDDEN_RATE=('forbidden_rate', 'mean'),
        SCORE=('score', 'mean'),
        MAC_SCORE=('mac_score_norm', 'mean'),
        PRICE_SCORE=('price_score_norm', 'last'),
        words_p_m=('words_p_m', 'mean'),
        volume_rms=('volume_rms', 'mean'),
        volume_classification=(
        'volume_classification', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
        DATE=('date-time', 'last'),
    ).reset_index().apply(lambda x: x.reset_index(drop=True)).reset_index(drop=True)
    #print("AGENTES UNICOS" +str(df_concatenado_agentes_unicos))
    df_concatenado_agentes_unicos = df_concatenado_agentes_unicos.drop_duplicates(subset='LEAD_ID')
    CUT=0.85

    MAT_CALLS_CAMPAIGN.to_excel(campaign_directory + "misc/" + 'call_analysis.xlsx')
    #df_concatenado_all.to_excel(campaign_directory + "misc/" + 'df_concatenado_all.xlsx')

    print("MAC CAMPAÑA: " + str(np.mean(df_concatenado_all['price_score_norm']) * CUT))
    print("SCORE CAMPAÑA: " + str(np.mean(df_concatenado_all['score']) * CUT))
    print("PRICE CAMPAÑA: " + str(np.mean(df_concatenado_all['mac_score_norm']) * CUT))

    REJECTED = df_concatenado_all[(df_concatenado_all['count_forbidden'] > len(all_musnt_keywords)*0.2)
                                       | (df_concatenado_all['score'] < np.mean(df_concatenado_all['score']) * CUT)
                                       | (df_concatenado_all['price_score_norm'] < np.mean(
        PRICES_unique['confidence_score']) * CUT)
                                       | (df_concatenado_all['mac_score_norm'] < np.mean(
        MACS_unique['confidence_score']) * CUT)
                                       | (df_concatenado_all['volume_classification']=='low')
                                       | (df_concatenado_all['volume_classification']=='low')]

    print("PUNTAJES RECHAZADAS: " + str(REJECTED))
    MAT_COMPLETE_TOPICS.to_excel(campaign_directory + "misc/" + 'MAT_COMPLETE_TOPICS.xlsx')
    REJECTED.to_excel(campaign_directory + "misc/" + 'REJECTED.xlsx')

    REJECTED_LEN = len(REJECTED)

    agents_with_reject = pd.merge(df_concatenado, REJECTED, left_on='file', right_on='file', how='right')

    list_affected = agents_with_reject['LEAD_ID_x'].value_counts().reset_index()
    list_affected.columns = ['LEAD_ID_x', 'count']


    call_ids = list_affected['LEAD_ID_x']
    averaged_results_p = process_directory_and_average(campaign_directory, num_windows=400)
    averaged_results_p['speed_n'] = averaged_results_p['speed_n'] / 400
    tems = sort_by_variance(STATISTICS, 'mean_rescaled_centroid')

    tems = tems[(tems['predicted_cluster'] != 'confirmacion') & (
                tems['predicted_cluster'] != 'precio')].reset_index()



    indice_saludo = STATISTICS[STATISTICS['predicted_cluster'] == tems['predicted_cluster'][0]][
        'mean_rescaled_2_centroid'].reset_index(drop=True)[0]
    indice_producto = STATISTICS[STATISTICS['predicted_cluster'] == tems['predicted_cluster'][1]][
        'mean_rescaled_2_centroid'].reset_index(drop=True)[0]
    indice_venta = \
    STATISTICS[STATISTICS['predicted_cluster'] == 'precio']['mean_rescaled_2_centroid'].reset_index(
        drop=True)[0]
    indice_mac = \
    STATISTICS[STATISTICS['predicted_cluster'] == 'confirmacion']['mean_rescaled_2_centroid'].reset_index(drop=True)[0]
    STATISTICS['relative_tmo'] = (STATISTICS['max_rescaled_centroid'] - STATISTICS['min_rescaled_centroid']) / \
                                 STATISTICS['mean_rescaled_centroid']

    indice_confirmacion = STATISTICS[STATISTICS['predicted_cluster'] == tems['predicted_cluster'][2]][
        'mean_rescaled_2_centroid'].reset_index(drop=True)[0]
    indice_despedida = STATISTICS[STATISTICS['predicted_cluster'] == tems['predicted_cluster'][3]][
        'mean_rescaled_2_centroid'].reset_index(drop=True)[0]

    indices = len(smooth_array((averaged_results_p['speed_n']), 2)) * np.array([indice_saludo, indice_producto,
                                                                                indice_venta, indice_mac,
                                                                                indice_confirmacion, indice_despedida])

    indices = np.sort(np.rint(indices).astype(int))
    conexion=conectar()
    array_camps = obtener_charts_recientes_campania(conexion, id_campania_num)
    topics_stats_convs['velocity_classification'] = topics_stats_convs['words_p_m'].apply(measure_speed_classification)
    topics_stats_convs_concat = topics_stats_convs.groupby('predicted_cluster').agg({
        'confidence_score': 'mean',
        'words_p_m': 'mean',
        'centroid_start_time': 'mean',
        'rescaled_centroid_start_time': 'mean',
        'velocity_classification': lambda x: x.mode().iloc[0],
    }).reset_index()

    topics_stats_convs_concat_all = pd.merge(topics_stats_convs_concat, STATISTICS, left_on='predicted_cluster',
                                             right_on='predicted_cluster', how='inner')

    topics_stats_convs_concat_all['confidence_score']=topics_stats_convs_concat_all['section_score']*10/np.max(topics_stats_convs_concat_all['confidence_score'])

    df_files = list_files_to_dataframe(campaign_id_route + '/isolated/')


    df_files['DATE_TIME'] = df_files['file_name'].apply(lambda x: datetime.strptime(x.split('_')[1], "%Y%m%d-%H%M%S"))
    df_files['LEAD_ID'] = df_files['file_name'].apply(lambda x: x.split('_')[-2])
    df_files['call_id'] = df_files['file_name'].apply(lambda x: x.split('_')[-1].split('-')[0])
    df_files['file'] = df_files['file_name']
    df_files['CALL_ID'] = df_files['file_name'].apply(lambda x: x.split('_')[2])

    #REJECTED['date-time'] = REJECTED['file_name'].apply(lambda x: datetime.strptime( x.split('_')[1], "%Y%m%d-%H%M%S") )
    common_columns = ['file_name', 'LEAD_ID', 'CALL_ID', 'DATE_TIME', 'CALL_ID']
    cct = pd.concat([REJECTED[common_columns], df_files[common_columns]], axis=0).drop_duplicates().reset_index(
        drop=True)
    #res = cct.reindex(columns=REJECTED.columns, fill_value=None)
    result_rejected = pd.concat([REJECTED, df_files], axis=0, ignore_index=True, sort=False)
    REJECTED_info = pd.merge(result_rejected, df_concatenado_llamadas_unicas, left_on='CALL_ID', right_on='CALL_ID', how='inner')

    conexion = conectar()
    print(int(id_campania_num))
    MAT_COMPLETE_TOPICS.to_excel(campaign_directory + "misc/" + 'MAT_COMPLETE_TOPICS.xlsx')


    route_transcripts = (campaign_parameters['country'].replace(' ', '') +
                         '/' + campaign_parameters['sponsor'].replace(' ', '') + '/transcript_sentences/' +
                         route.split('/')[-2] + '/')
    #insertar_filas_dataframe_agentes(conexion, campaign_id_route,id_campania_num, STATISTICS, df_concatenado, df_concatenado_prices, df_concatenado_macs, df_concatenado_llamadas_unicas,route,route_transcripts)
    topics_stats_convs.to_excel(campaign_directory + "misc/" + 'topics_stats_convs.xlsx')
    topics_stats_convs['LEAD_ID'] = topics_stats_convs['file_name'].apply(lambda x: x.split('_')[-2])

    bring_llamadas = calls_this_campaign(int(id_campania_num), conexion)
    bring_llamadas['sales_arguments_percentage'] = bring_llamadas['sales_arguments_percentage'].apply(
        lambda x: np.clip(x, 0, 100))
    bring_llamadas['name'] = bring_llamadas['name'].apply(lambda x: x.replace('BH_TEST', 'BHTEST'))
    bring_llamadas['date-time'] = bring_llamadas['name'].apply(lambda x: datetime.strptime(x.split('_')[1],  "%Y%m%d-%H%M%S"))
    bring_llamadas['CALL_ID'] = bring_llamadas['name'].apply(lambda x: x.split('_')[2])
    result_rejected.to_excel(campaign_directory + 'misc/result_rejected.xlsx')
    result_rejected.drop(['id_x', 'id_y'], axis=1, inplace=True)
    df_concatenado_afectadas = pd.merge(result_rejected, bring_llamadas, left_on='CALL_ID', right_on='CALL_ID', how='left')
    df_concatenado_afectadas = df_concatenado_afectadas.drop_duplicates(subset=['file_name'])
    df_concatenado_afectadas = df_concatenado_afectadas.groupby('CALL_ID').agg(
        agent_audio_data_id=('id_y', 'first'),
        LEAD_ID=('LEAD_ID', 'first'),
        DATE=('date-time_x', 'last'),
        SCORE=('score', 'first'),
        F_COUNT=('count_forbidden', 'first'),
        S_ACC=('sales_acceptance', 'first')
    ).reset_index().apply(lambda x: x.reset_index(drop=True)).reset_index(drop=True)

    df_concatenado_macs_info = topics_stats_convs[
        ['LEAD_ID', 'confidence_score', 'velocity_classification']]
    result_general = pd.merge(result_rejected, df_concatenado_macs_info, left_on='LEAD_ID', right_on='LEAD_ID', how='left')
    result_general = result_general.drop_duplicates(subset='CALL_ID', keep='first').reset_index(drop=True)
    result_general_ids = pd.merge(result_general,
                                  df_concatenado_afectadas[['CALL_ID', 'LEAD_ID', 'agent_audio_data_id', 'DATE']],
                                  left_on='CALL_ID', right_on='CALL_ID', how='inner')
    result_general_ids.to_excel(campaign_directory + 'misc/result_general.xlsx')




if __name__ == '__main__':
    # Comprobamos que vengan exactamente 2 parámetros extra además del nombre del script
    if len(sys.argv) != 3:
        print("Uso correcto: python main.py 'PREFIX', días atras")
        sys.exit(1)

    param1 = str(sys.argv[1])
    param2 = int(sys.argv[2])
    print("INICIANDO PROCESO PARA "+ param1 +" CON DELAY "+ str(param2))
    main(param1, param2)
    print("FINALIZADO PROCESO PARA " + param1 + " CON DELAY " + str(param2) + " EXITOSAMENTE.")
