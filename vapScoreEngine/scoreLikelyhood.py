#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
score_likelihood_matrix.py

Objetivo
--------
1) Mantener TU pipeline actual:
   jsonTranscriptionToCsv -> splitConversations -> fitCSVConversations -> reconstruirDialogos -> process_directory_mac_price_def
   y luego generateConvDataframe (por cada csv en rebuiltDir y /memory).

2) Medir LIKELIHOOD por conversación vs MATRIZ (name/cluster) usando técnicas mejores:
   - Fuzzy matching (RapidFuzz, robusto a typos)
   - Levenshtein normalizado (opcional; RapidFuzz ya aproxima)
   - Similaridad semántica con embeddings (SentenceTransformers) (opcional pero recomendado)
   - Reporte por conversación, por cluster obligatorio y global.

3) Clusters obligatorios (según tu mensaje):
   - "Pregunta de aceptación"
   - "Precio"
   - "Términos de ley"

Salida
------
- likelyhood_by_call.xlsx  (1 fila por conversación)
- likelyhood_evidence_by_call.xlsx (evidencias top por conversación por cluster)
- (opcional) spans_by_call.xlsx (si tu generateConvDataframe trae spans/segmentos)

Dependencias opcionales
----------------------
pip install rapidfuzz sentence-transformers torch

Si NO tienes sentence-transformers, el script se ejecuta SOLO con fuzzy.

NOTA
----
Este script NO modifica tu segmentador: solo agrega scoring vs matriz sobre los
diálogos reconstruidos y su metadata.
"""

import os
import re
import json
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from segmentationModel.fittingDeep import fitCSVConversations
from lang.VapLangUtils import correctCommonTranscriptionMistakes, splitConversations
from utils.VapUtils import (
    jsonTranscriptionToCsv,
    getTranscriptParagraphsJson,
    jsonDecomposeSentencesHighlight,
)
from segmentationModel.textPostprocessing import reconstruirDialogos, process_directory_mac_price_def
from vapScoreEngine.dfUtils import generateConvDataframe

# ====== Matching mejorado ======
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

# ====== Embeddings semánticos (opcional) ======
try:
    from sentence_transformers import SentenceTransformer
    import torch
except Exception:
    SentenceTransformer = None
    torch = None


# =========================
# CONFIG
# =========================

# Clusters obligatorios (tal como vienen en la matriz)
REQUIRED_CLUSTERS = [    "SALUDO",
    "PERFILAMIENTO",
    "PRODUCTO",
    "CONFIRMACION MONITOREO",
    "LEY RETRACTO",
    "TERMINOS LEGALES",
    "TRATAMIENTO DATOS",      
    "MAC",
    "MAC REFUERZO",
    "PRECIO",
    "CONFIRMACION DATOS",
    "CONFORMIDAD",
    "ATENCION",
    "DESPEDIDA"]

# Modelo embeddings (multilingüe). Cámbialo si quieres uno en español puro.
DEFAULT_EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Pesos para score combinado
W_FUZZY = 0.55
W_SEM   = 0.45  # si no hay embeddings, se ignora

# Thresholds recomendados (ajusta con datos)
THRESH_WARN = 0.80  # bajo esto: "warning"
THRESH_OK   = 0.88  # sobre esto: "ok"

# Ventaneo para comparar dentro de conversación (en palabras)
WINDOW_WORDS = 18
WINDOW_STRIDE = 6

# Cuántas evidencias guardar por cluster
TOPK_EVIDENCE = 3


# =========================
# Utils: texto
# =========================

def _norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = correctCommonTranscriptionMistakes(s)
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize_words(s: str) -> List[str]:
    s = _norm_text(s)
    return [w for w in s.split(" ") if w]

def build_windows(text: str, window_words: int = WINDOW_WORDS, stride: int = WINDOW_STRIDE) -> List[str]:
    """
    Construye ventanas de texto a nivel palabra para hacer matching robusto.
    Si el texto es corto, devuelve [texto].
    """
    words = _tokenize_words(text)
    if not words:
        return []
    if len(words) <= window_words:
        return [" ".join(words)]
    out = []
    for i in range(0, len(words) - window_words + 1, stride):
        out.append(" ".join(words[i:i + window_words]))
    # incluir cola
    tail = " ".join(words[-window_words:])
    if out and out[-1] != tail:
        out.append(tail)
    return out


# =========================
# Matriz (name/cluster)
# =========================

def load_matrix(matrix_path: str) -> pd.DataFrame:
    """
    Espera columnas: name, cluster
    """
    df = pd.read_csv(matrix_path) if matrix_path.lower().endswith(".csv") else pd.read_excel(matrix_path)
    if not {"name", "cluster"}.issubset(df.columns):
        raise ValueError("La matriz debe tener columnas: name, cluster")
    df = df.dropna(subset=["name", "cluster"]).copy()
    df["name_norm"] = df["name"].apply(_norm_text)
    df["cluster_norm"] = df["cluster"].astype(str).str.strip()
    return df

def matrix_by_cluster(df_matrix: pd.DataFrame) -> Dict[str, List[str]]:
    d: Dict[str, List[str]] = {}
    for cl, sub in df_matrix.groupby("cluster_norm"):
        d[cl] = sub["name_norm"].tolist()
    return d


# =========================
# Fuzzy scoring
# =========================

def fuzzy_score(a: str, b: str) -> float:
    """
    Score 0..1 usando rapidfuzz (token_set_ratio + partial_ratio).
    Si rapidfuzz no está, fallback a difflib ratio (menos recomendado).
    """
    a = _norm_text(a)
    b = _norm_text(b)
    if not a or not b:
        return 0.0

    if fuzz is not None:
        s1 = fuzz.token_set_ratio(a, b) / 100.0
        s2 = fuzz.partial_ratio(a, b) / 100.0
        # mezcla robusta
        return float(max(s1, 0.85 * s2))
    else:
        from difflib import SequenceMatcher
        return float(SequenceMatcher(None, a, b).ratio())


# =========================
# Embeddings semánticos (opcional)
# =========================

@dataclass
class SemanticIndex:
    model_name: str
    model: Any
    # dict cluster -> (refs_texts, refs_emb [N,D], centroid [D])
    refs: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]

def build_semantic_index(refs_by_cluster: Dict[str, List[str]], model_name: str = DEFAULT_EMB_MODEL) -> Optional[SemanticIndex]:
    if SentenceTransformer is None:
        return None

    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    model = SentenceTransformer(model_name, device=device)

    refs: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]] = {}
    for cl, phrases in refs_by_cluster.items():
        phrases = [p for p in phrases if p]
        if not phrases:
            continue
        emb = model.encode(phrases, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        centroid = emb.mean(axis=0)
        # re-normalizar centroid
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        refs[cl] = (phrases, emb, centroid)

    return SemanticIndex(model_name=model_name, model=model, refs=refs)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))

def semantic_score(text: str, sem_index: SemanticIndex, cluster: str, topk: int = 5) -> Tuple[float, Optional[str]]:
    """
    Retorna (score 0..1, best_ref_phrase)
    - compara embedding(text) contra centroid y top-k refs (max)
    """
    text = _norm_text(text)
    if not text or cluster not in sem_index.refs:
        return 0.0, None

    phrases, emb_refs, centroid = sem_index.refs[cluster]
    q = sem_index.model.encode([text], normalize_embeddings=True, show_progress_bar=False)
    q = np.asarray(q[0], dtype=np.float32)

    # centroid score
    sc_cent = float(np.dot(q, centroid))
    # max ref score (aprox topk por dot-product)
    sims = emb_refs @ q  # [N]
    if sims.size == 0:
        return sc_cent, None
    best_idx = int(np.argmax(sims))
    best_sc = float(sims[best_idx])
    best_phrase = phrases[best_idx]

    # mezcla: prioriza el mejor match
    return float(max(sc_cent, best_sc)), best_phrase


# =========================
# Scoring por conversación
# =========================

@dataclass
class Evidence:
    cluster: str
    score_fuzzy: float
    score_sem: float
    score_combined: float
    best_window: str
    best_ref: Optional[str]

def score_text_against_cluster(
    conversation_text: str,
    cluster: str,
    refs_by_cluster: Dict[str, List[str]],
    sem_index: Optional[SemanticIndex] = None,
    window_words: int = WINDOW_WORDS,
    stride: int = WINDOW_STRIDE,
    topk_evidence: int = TOPK_EVIDENCE
) -> Tuple[float, List[Evidence]]:
    """
    Devuelve:
      - score_combined_max (0..1) del cluster en la conversación
      - lista de evidencias (top-k)
    """
    refs = refs_by_cluster.get(cluster, [])
    if not refs:
        return 0.0, []

    windows = build_windows(conversation_text, window_words=window_words, stride=stride)
    if not windows:
        return 0.0, []

    evidences: List[Evidence] = []

    # pre-cálculo: para fuzzy, recorrer refs es caro si hay miles. Aquí tu matriz parece moderada.
    # Si crece, se puede indexar por ngrams / BM25.
    for w in windows:
        # fuzzy: mejor ref
        best_fuzzy = 0.0
        best_ref_fuzzy = None
        for r in refs:
            sc = fuzzy_score(w, r)
            if sc > best_fuzzy:
                best_fuzzy = sc
                best_ref_fuzzy = r

        # semántico (opcional)
        best_sem = 0.0
        best_ref_sem = None
        if sem_index is not None:
            best_sem, best_ref_sem = semantic_score(w, sem_index, cluster=cluster)

        # combinado
        if sem_index is None:
            comb = best_fuzzy
            ref_final = best_ref_fuzzy
        else:
            comb = (W_FUZZY * best_fuzzy) + (W_SEM * best_sem)
            # evidencia: prioriza el ref que explique mejor
            ref_final = best_ref_sem if best_sem >= best_fuzzy else best_ref_fuzzy

        evidences.append(Evidence(
            cluster=cluster,
            score_fuzzy=float(best_fuzzy),
            score_sem=float(best_sem),
            score_combined=float(comb),
            best_window=w,
            best_ref=ref_final
        ))

    evidences.sort(key=lambda e: e.score_combined, reverse=True)
    top_evs = evidences[:topk_evidence]
    best = float(top_evs[0].score_combined) if top_evs else 0.0
    return best, top_evs


def presence_flag(score: float) -> str:
    if score >= THRESH_OK:
        return "ok"
    if score >= THRESH_WARN:
        return "warn"
    return "missing"


# =========================
# Tu función crítica: NO romper
# =========================

def process_directory_conversations_with_memory(mainDir, rawDir, processedDir, rebuiltDir, kws):
    """
    Mantengo tu lógica, pero NO asumo que kws sea lista/df:
    - jsonDecomposeSentencesHighlight espera kws: aquí tú le pasas keywords_df
      (en tu pipeline actual). Lo dejo tal cual.
    """
    dataframes = []

    jsonTranscriptionToCsv(mainDir, rawDir)
    splitConversations(rawDir, rawDir, 14)

    # segmentación por ventanas (tu modelo)
    fitCSVConversations(rawDir, processedDir, 14, 6, 32)

    # genera transcript_sentences + highlight
    getTranscriptParagraphsJson(mainDir)
    jsonDecomposeSentencesHighlight(
        os.path.join(mainDir, "transcript_sentences"),
        os.path.join(mainDir, "transcript_sentences"),
        kws
    )

    # reconstrucción: AQUÍ quedan los diálogos a calificar
    reconstruirDialogos(rawDir, processedDir, rebuiltDir)

    # tus features mac/price/def, etc.
    _ = process_directory_mac_price_def(rebuiltDir, rebuiltDir, topics_col="topics_sequence")

    # cargar todo lo reconstruido
    files = [f for f in os.listdir(rebuiltDir) if f.endswith(".csv")]
    for filename in tqdm(files, desc=f"generateConvDataframe: {rebuiltDir}"):
        filepath = os.path.join(rebuiltDir, filename)
        try:
            processed_df = generateConvDataframe(filepath)
            dataframes.append(processed_df)
        except Exception as e:
            print(f"Error al procesar {filepath}: {e}")

    # memory
    memory_dir = os.path.join(rebuiltDir, "memory")
    if os.path.exists(memory_dir):
        mem_files = [f for f in os.listdir(memory_dir) if f.endswith(".csv")]
        for filename in tqdm(mem_files, desc="generateConvDataframe: /memory"):
            filepath = os.path.join(memory_dir, filename)
            try:
                processed_df = generateConvDataframe(filepath)
                dataframes.append(processed_df)
            except Exception as e:
                print(f"Error con memory file {filename}: {e}")

    return dataframes


# =========================
# Obtener texto conversación desde generateConvDataframe
# =========================

def extract_conversation_text(df_conv: pd.DataFrame) -> str:
    """
    generateConvDataframe NO lo has pegado, entonces hago un extractor robusto:
    intenta múltiples columnas típicas.

    Prioridad:
    - si existe 'text_string': usarla
    - si existe 'text' y está a nivel de filas: concatenar
    - si existe 'conversation': tomar primera fila si es str o lista
    - else: concatenar columnas string relevantes
    """
    # caso típico de tus dfs por tema: text_string
    if "text_string" in df_conv.columns:
        # puede venir repetida por fila -> concateno
        vals = df_conv["text_string"].dropna().astype(str).tolist()
        if vals:
            return " ".join(vals)

    # si es transcript por líneas
    if "text" in df_conv.columns:
        vals = df_conv["text"].dropna().astype(str).tolist()
        if vals:
            return " ".join(vals)

    if "conversation" in df_conv.columns:
        v = df_conv["conversation"].iloc[0]
        if isinstance(v, (list, tuple, np.ndarray)):
            return " ".join([str(x) for x in v])
        return str(v)

    # fallback: concat de strings
    cols = [c for c in df_conv.columns if df_conv[c].dtype == object]
    parts = []
    for c in cols[:8]:
        parts.extend(df_conv[c].dropna().astype(str).tolist())
    return " ".join(parts)


# =========================
# Orquestador score_camp (LIKELIHOOD vs MATRIZ)
# =========================

def score_camp_likelihood_matrix(
    campaign_directory: str,
    campaign_id: str,
    topics_combined_df: pd.DataFrame,
    matrix_path: str,
    out_dir: Optional[str] = None,
    use_semantic: bool = True,
    emb_model_name: str = DEFAULT_EMB_MODEL
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    - corre tu pipeline hasta rebuiltDir (vía process_directory_conversations_with_memory)
    - calcula likelihood por conversación para clusters obligatorios
    - devuelve:
        calls_df: 1 fila por llamada con scores y flags
        evidence_df: evidencias top-k por cluster por llamada
    """
    if out_dir is None:
        out_dir = os.path.join(campaign_directory, "misc")
    os.makedirs(out_dir, exist_ok=True)

    routeRawCsvTranscripts = campaign_directory + campaign_id.replace("/", "") + "_RAW"
    routeRawCsvGraded     = campaign_directory + campaign_id.replace("/", "") + "_PROCESSED"
    routeRawCsvRebuilt    = campaign_directory + campaign_id.replace("/", "") + "_RECONS"

    # 1) pipeline actual (NO romper)
    conv_dfs = process_directory_conversations_with_memory(
        campaign_directory,
        routeRawCsvTranscripts,
        routeRawCsvGraded,
        routeRawCsvRebuilt,
        topics_combined_df
    )
    print(f"TOTAL de conversaciones procesadas: {len(conv_dfs)}")

    # 2) matriz
    df_matrix = load_matrix(matrix_path)
    refs_by_cluster = matrix_by_cluster(df_matrix)

    # 3) semantic index (opcional)
    sem_index = None
    if use_semantic:
        sem_index = build_semantic_index(refs_by_cluster, model_name=emb_model_name)
        if sem_index is None:
            print("[WARN] sentence-transformers no disponible; se usará solo fuzzy.")
    else:
        print("[INFO] use_semantic=False; se usará solo fuzzy.")

    # 4) por conversación
    rows_calls = []
    rows_evidence = []

    for df_conv in tqdm(conv_dfs, desc="Scoring likelihood vs matriz"):
        # intentar detectar file_name
        file_name = None
        if "file_name" in df_conv.columns:
            try:
                file_name = str(df_conv["file_name"].iloc[0])
            except Exception:
                file_name = None

        conversation_text = extract_conversation_text(df_conv)
        conversation_text_norm = _norm_text(conversation_text)

        # métricas de apoyo si existen en df_conv (no obligatorias)
        transcript_conf = float(df_conv["mean_transcript_confidence"].iloc[0]) if "mean_transcript_confidence" in df_conv.columns else np.nan
        topic_conf      = float(df_conv["topic_mean_conf"].iloc[0]) if "topic_mean_conf" in df_conv.columns else np.nan
        wpm             = float(df_conv["topic_words_p_m"].iloc[0]) if "topic_words_p_m" in df_conv.columns else np.nan

        # likelihood clusters obligatorios
        cluster_scores = {}
        cluster_flags = {}
        for cl in REQUIRED_CLUSTERS:
            sc, evs = score_text_against_cluster(
                conversation_text_norm,
                cluster=cl,
                refs_by_cluster=refs_by_cluster,
                sem_index=sem_index,
                window_words=WINDOW_WORDS,
                stride=WINDOW_STRIDE,
                topk_evidence=TOPK_EVIDENCE
            )
            cluster_scores[cl] = sc
            cluster_flags[cl] = presence_flag(sc)

            for rank, ev in enumerate(evs, start=1):
                rows_evidence.append({
                    "file_name": file_name,
                    "campaign_id": campaign_id,
                    "cluster": cl,
                    "rank": rank,
                    "score_combined": ev.score_combined,
                    "score_fuzzy": ev.score_fuzzy,
                    "score_sem": ev.score_sem,
                    "best_window": ev.best_window,
                    "best_ref_phrase": ev.best_ref,
                })

        # score global de cumplimiento (promedio de obligatorios)
        required_mean = float(np.mean([cluster_scores[c] for c in REQUIRED_CLUSTERS]))

        rows_calls.append({
            "file_name": file_name,
            "campaign_id": campaign_id,
            "conversation_len_chars": len(conversation_text_norm),
            "transcript_confidence": transcript_conf,
            "topic_confidence": topic_conf,
            "speed_wpm": wpm,

            # scores por cluster obligatorio
            "likelihood_pregunta_aceptacion": cluster_scores.get("Pregunta de aceptación", 0.0),
            "flag_pregunta_aceptacion": cluster_flags.get("Pregunta de aceptación", "missing"),

            "likelihood_precio": cluster_scores.get("Precio", 0.0),
            "flag_precio": cluster_flags.get("Precio", "missing"),

            "likelihood_terminos_ley": cluster_scores.get("Términos de ley", 0.0),
            "flag_terminos_ley": cluster_flags.get("Términos de ley", "missing"),

            "likelihood_required_mean": required_mean,
            "required_all_ok": int(all(cluster_flags[c] == "ok" for c in REQUIRED_CLUSTERS)),
            "required_any_missing": int(any(cluster_flags[c] == "missing" for c in REQUIRED_CLUSTERS)),
        })

    calls_df = pd.DataFrame(rows_calls)
    evidence_df = pd.DataFrame(rows_evidence)

    # 5) export
    calls_xlsx = os.path.join(out_dir, "likelyhood_by_call.xlsx")
    ev_xlsx    = os.path.join(out_dir, "likelyhood_evidence_by_call.xlsx")
    calls_df.to_excel(calls_xlsx, index=False)
    evidence_df.to_excel(ev_xlsx, index=False)
    print(f"[OK] Export calls: {calls_xlsx}")
    print(f"[OK] Export evidence: {ev_xlsx}")

    return calls_df, evidence_df


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign_directory", type=str, required=True, help="Directorio base campaña (termina en /)")
    ap.add_argument("--campaign_id", type=str, required=True, help="ID campaña para nombres de carpetas")
    ap.add_argument("--matrix_path", type=str, required=True, help="CSV/XLSX con columnas name,cluster")
    ap.add_argument("--topics_matrix_path", type=str, required=True, help="Tu matriz/df para kws (topics_combined_df) usada por pipeline actual")
    ap.add_argument("--out_dir", type=str, default=None, help="Salida (default campaign_directory/misc)")
    ap.add_argument("--no_semantic", action="store_true", help="Deshabilita embeddings (solo fuzzy)")
    ap.add_argument("--emb_model", type=str, default=DEFAULT_EMB_MODEL, help="Modelo sentence-transformers")
    args = ap.parse_args()

    # topics_combined_df: en tu pipeline viene como df, aquí lo cargamos
    # (Si ya lo tienes en memoria, puedes invocar score_camp_likelihood_matrix directamente)
    topics_df = pd.read_csv(args.topics_matrix_path) if args.topics_matrix_path.lower().endswith(".csv") else pd.read_excel(args.topics_matrix_path)

    score_camp_likelihood_matrix(
        campaign_directory=args.campaign_directory,
        campaign_id=args.campaign_id,
        topics_combined_df=topics_df,
        matrix_path=args.matrix_path,
        out_dir=args.out_dir,
        use_semantic=not args.no_semantic,
        emb_model_name=args.emb_model
    )
