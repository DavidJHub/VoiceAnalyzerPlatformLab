from lang.VapLangUtils import splitConversations
from segmentationModel.fitting import fitCSVConversations
from segmentationModel.textPostprocessing import process_directory_mac_price_def, reconstruirDialogos
from utils.VapUtils import getTranscriptParagraphsJson, jsonDecomposeSentencesHighlight, jsonTranscriptionToCsv


def process_directory_conversations_with_memory(mainDir,rawDir,processedDir,rebuiltDir,kws):
    dataframes = []
    jsonTranscriptionToCsv(mainDir,rawDir)
    splitConversations(rawDir,rawDir,14)
    fitCSVConversations(rawDir,processedDir, 14, 6, 32)
    getTranscriptParagraphsJson(mainDir)
    jsonDecomposeSentencesHighlight(mainDir + '/transcript_sentences',mainDir + '/transcript_sentences',kws)
    # Process main directory files
    reconstruirDialogos(rawDir, processedDir,rebuiltDir)
    df_res = process_directory_mac_price_def(rebuiltDir,rebuiltDir,topics_col='topics_sequence')