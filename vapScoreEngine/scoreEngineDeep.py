from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import ast


import glob
import json
import os
from difflib import SequenceMatcher
from sklearn.preprocessing import QuantileTransformer

import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from segmentationModel.fittingDeep import fitCSVConversations

from lang.VapLangUtils import normalize_text, get_kws, word_count, \
    correctCommonTranscriptionMistakes, splitConversations

from setup.MatrixSetup import remove_connectors
from utils.VapFunctions import measureDbAplitude, measure_speed_classification
from utils.VapUtils import jsonDecompose, get_data_from_name, jsonDecomposeSentencesHighlight, jsonTranscriptionToCsv, getTranscriptParagraphsJson

import pandas as pd
import numpy as np

from segmentationModel.textPostprocessing import reconstruirDialogos, process_directory_mac_price_def
from vapScoreEngine.dfUtils import calculate_confidence_scores_per_topic, df_getWordRate, generateConvDataframe



# ---------------------------
# FIX recomendado: reemplaza eval por literal_eval
# ---------------------------
def _safe_parse_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []

# ---------------------------
# Orquestador limpio
# ---------------------------
def process_directory_conversations_with_memory_v2(
    mainDir: str,
    rawDir: str,
    processedDir: str,
    rebuiltDir: str,
    kws: List[str],
    *,
    max_words_split: int = 14,
    window_size: int = 14,
    stride: int = 6,
    max_length: int = 32,
    do_paragraph_json: bool = True,
    do_highlight: bool = True,
    do_move_isolated_on_error: bool = True,  # si tu fitCSVConversations lo usa internamente
) -> List[Dict[str, Any]]:
    """
    Pipeline:
      1) JSON -> CSV raw
      2) Split long fragments (raw -> raw_split)
      3) Sliding windows + clasificación (raw_split -> processed)
      4) (Opcional) JSON paragraphs + highlight
      5) Reconstrucción de diálogo (raw_split + processed -> rebuilt)
      6) Marcar MAC_DEF/PRECIO_DEF (rebuilt -> rebuilt_marked)
      7) Features finales por archivo (+ memoria)
    Retorna lista de dicts: uno por archivo, con sus dataframes y metadatos.
    """

    mainDir = str(Path(mainDir))
    rawDir = Path(rawDir)
    processedDir = Path(processedDir)
    rebuiltDir = Path(rebuiltDir)

    rawDir.mkdir(parents=True, exist_ok=True)
    processedDir.mkdir(parents=True, exist_ok=True)
    rebuiltDir.mkdir(parents=True, exist_ok=True)

    # Subcarpetas por etapa (evita pisar)
    raw_split_dir = rawDir.parent / (rawDir.name + "_split")
    raw_split_dir.mkdir(parents=True, exist_ok=True)

    rebuilt_marked_dir = rebuiltDir.parent / (rebuiltDir.name + "_marked")
    rebuilt_marked_dir.mkdir(parents=True, exist_ok=True)

    # 1) JSON -> CSV raw
    jsonTranscriptionToCsv(mainDir, str(rawDir))

    # 2) Split (raw -> raw_split)
    splitConversations(str(rawDir), str(raw_split_dir), max_words_split)

    # 3) Windows + clasificación (raw_split -> processed)
    fitCSVConversations(str(raw_split_dir), str(processedDir), window_size, stride, max_length)

    # 4) Paragraph JSON + highlight (opcional)
    if do_paragraph_json:
        getTranscriptParagraphsJson(mainDir)

    # 5) Reconstrucción (raw_split + processed -> rebuilt)
    reconstruirDialogos(str(raw_split_dir), str(processedDir), str(rebuiltDir))

    # 6) Marcar MAC/PRECIO (rebuilt -> rebuilt_marked)
    process_directory_mac_price_def(str(rebuiltDir), str(rebuilt_marked_dir), topics_col="topics_sequence")

    # 7) Build resultados finales
    results: List[Dict[str, Any]] = []

    def _process_csv_dir(dir_path: Path, tag: str):
        csvs = sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
        for fp in tqdm(csvs, desc=f"Generating conv features [{tag}]"):
            try:
                classifiedConv, aggregated, duration = generateConvDataframe(str(fp))
                results.append({
                    "source": tag,
                    "csv_path": str(fp),
                    "file_stem": fp.stem,
                    "classified": classifiedConv,
                    "aggregated": aggregated,
                    "duration_sec": duration,
                })
            except Exception as e:
                print(f"[features] Error en {fp.name}: {e}")

    _process_csv_dir(rebuilt_marked_dir, tag="rebuilt_marked")

    # Memory folder (si existe) adentro de rebuilt_marked_dir o rebuiltDir: soporta ambos
    mem1 = rebuilt_marked_dir / "memory"
    mem2 = rebuiltDir / "memory"

    if mem1.exists():
        _process_csv_dir(mem1, tag="memory_marked")
    elif mem2.exists():
        _process_csv_dir(mem2, tag="memory_unmarked")

    return results