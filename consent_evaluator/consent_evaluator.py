# -*- coding: utf-8 -*-
"""consent_evaluator.py (refactor sin AWS)

Este módulo procesa transcritos (.json) **desde carpeta local** y evalúa, vía LLM:
- Participación (cliente/agente) (calculada localmente)
- Preguntas abiertas detectadas (AGENTE vs BANCO_PREGUNTAS)
- MAC_camuflado
- Cercanía con el guion
- Preguntas del CLIENTE y calidad de respuesta del AGENTE

Nuevo API (para orquestador):
    import consent_evaluator
    df = consent_evaluator.evaluate_consent(transcripts_path, script)

Dónde:
- transcripts_path: ruta local a una carpeta con archivos .json de transcritos (se analizan todos).
- script: texto del guion a evaluar (se inyecta en el prompt como GUI0N_CLARO_2025).

Salida:
- DataFrame con las mismas columnas que el Excel consolidado previo (pero sin escribir ni leer de S3).

Notas:
- Se eliminó TODO lo relacionado con AWS/S3 (boto3, URIs, lectura/escritura de Excel en S3, etc.).
- OpenAI sigue siendo necesario (misma lógica).
"""

from __future__ import annotations

from dotenv import load_dotenv
import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from openai import OpenAI

# ==========================
# Config
# ==========================

load_dotenv()

PROCESS_MODE = os.getenv("PROCESS_MODE", "all")  # "sample" | "all"
PROCESS_ONLY_N = int(os.getenv("PROCESS_ONLY_N", "2"))

OPENAI_API_KEY_VC = os.getenv("OPENAI_API_KEY_VC")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # fallback razonable

# === Diarización contextual (parámetros de control) ===
DIARIZATION_MAX_ATTEMPTS = 1  # Intentos máximos de diarización contextual
DIARIZATION_TRIGGERED_COUNT = 0  # contador global de transcritos diarizados contextualmente (por participación 100/0)

# ==========================
# OpenAI helpers
# ==========================

def openai_client() -> OpenAI:
    if not OPENAI_API_KEY_VC:
        raise RuntimeError("OPENAI_API_KEY no está configurada en variables de entorno.")
    return OpenAI(api_key=OPENAI_API_KEY_VC)

# ==========================
# Utilidades locales
# ==========================

def list_local_json_files(transcripts_path: str, *, mode: str = PROCESS_MODE, only_n: int = PROCESS_ONLY_N) -> List[Path]:
    root = Path(transcripts_path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"transcripts_path no existe o no es carpeta: {transcripts_path}")

    files = sorted([p for p in root.rglob("*.json") if p.is_file()])
    if mode != "all":
        files = files[: max(0, int(only_n))]
    return files

def read_local_json(path: Path) -> Dict[str, Any]:
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
        return json.loads(txt)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parseando JSON: {path.name} -> {e}") from e

def _basename_without_ext(path_or_name: str) -> str:
    return path_or_name.split("-all")[0]+"-all.mp3"  

def extract_call_date_from_filename(filename: str) -> Optional[str]:
    """
    Intenta extraer fecha de nombre con formatos típicos:
      - YYYY-MM-DD
      - YYYYMMDD
    Devuelve 'YYYY-MM-DD' o None.
    """
    base = _basename_without_ext(filename)

    m = re.search(r"(20\d{2})[-_\.](\d{2})[-_\.](\d{2})", base)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mo}-{d}"

    m = re.search(r"(20\d{2})(\d{2})(\d{2})", base)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mo}-{d}"

    return None

def extract_agent_id_from_filename(filename: str) -> str | None:
    """
    Mantiene la lógica previa (antes era 'extract_agent_id_from_key'):
    split por '_' y tomar parts[4] (índice 4), luego cortar por '-' y tomar el primer fragmento.
    Ej: '..._3183599255-all_transcript_paragraphs.json' -> '3183599255'
    """
    try:
        base = os.path.basename(filename)
        parts = base.split("_")
        if len(parts) < 6:
            return None
        raw = parts[4]
        first = raw.split("-")[0]
        agent_id = "".join(ch for ch in first if ch.isdigit()) or first.strip()
        return agent_id if agent_id else None
    except Exception:
        return None

# ==========================
# Cálculo de silencios
# ==========================

class SilenceInfo(NamedTuple):
    max_silence: float | None
    exceeded_60s: bool
    prev_end: float | None
    next_start: float | None

def _float_or_none(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def _iter_utterances_text_level(call_json: Dict[str, Any]):
    try:
        channels = call_json.get("results", {}).get("channels", [])
        if not channels:
            return []

        alts = channels[0].get("alternatives", [])
        if not alts:
            return []

        paragraphs = alts[0].get("paragraphs", [])
        utterances = []
        for p in paragraphs:
            for s in p.get("sentences", []):
                st = _float_or_none(s.get("start"))
                en = _float_or_none(s.get("end", s.get("stop")))
                if st is not None and en is not None:
                    utterances.append({"start": st, "end": en, "text": s.get("text", "")})
        utterances.sort(key=lambda u: u["start"])
        return utterances
    except Exception:
        return []

def compute_max_silence_from_text(call_json: Dict[str, Any]) -> SilenceInfo:
    utts = _iter_utterances_text_level(call_json)
    if len(utts) < 2:
        return SilenceInfo(None, False, None, None)

    max_gap = -1.0
    prev_end_for_max = None
    next_start_for_max = None

    prev_end = utts[0]["end"]
    for i in range(1, len(utts)):
        cur_start = utts[i]["start"]
        gap = cur_start - prev_end
        if gap > max_gap:
            max_gap = gap
            prev_end_for_max = prev_end
            next_start_for_max = cur_start
        prev_end = utts[i]["end"]

    max_val = max_gap if max_gap >= 0 else None
    exceeded = (max_val is not None) and (max_val > 60.0)
    return SilenceInfo(max_val, exceeded, prev_end_for_max, next_start_for_max)

# ==========================
# Participación por speaker (palabras y tiempo)
# ==========================

def _iter_paragraphs(call_json: Dict[str, Any]):
    """
    Devuelve un generador de (speaker_id, sentence_dict)
    donde sentence_dict tiene 'text', 'start', 'end'.
    """
    try:
        channels = call_json.get("results", {}).get("channels", [])
        if not channels:
            return
        alts = channels[0].get("alternatives", [])
        if not alts:
            return
        paragraphs = alts[0].get("paragraphs", [])
        for p in paragraphs:
            spk = p.get("speaker", None)
            for s in p.get("sentences", []):
                yield spk, {
                    "text": s.get("text", "") or "",
                    "start": _float_or_none(s.get("start")),
                    "end": _float_or_none(s.get("end", s.get("stop")))
                }
    except Exception:
        return

def compute_participation_metrics(call_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcula:
      - palabras por speaker y % sobre el total (USANDO num_words directo del JSON si existe)
      - tiempo hablado por speaker (segundos) y % sobre el total
    """
    words_by_speaker: dict[int, int] = {}
    time_by_speaker: dict[int, float] = {}

    # === 1) Palabras por speaker usando num_words (prioritario) ===
    try:
        channels = call_json.get("results", {}).get("channels", [])
        if channels:
            alts = channels[0].get("alternatives", [])
            if alts:
                paragraphs = alts[0].get("paragraphs", [])
                for p in paragraphs:
                    sentences = p.get("sentences", [])
                    sentence_has_speaker = any(
                        isinstance(s.get("speaker", None), (int, float))
                        or (isinstance(s.get("speaker", None), str) and str(s.get("speaker")).isdigit())
                        for s in sentences
                    )

                    if sentence_has_speaker:
                        for s in sentences:
                            s_spk = s.get("speaker", p.get("speaker", None))
                            if s_spk is None:
                                continue
                            if isinstance(s_spk, str) and s_spk.isdigit():
                                s_spk = int(s_spk)

                            n_sentence = s.get("num_words", None)
                            if isinstance(n_sentence, (int, float)):
                                words_by_speaker[s_spk] = words_by_speaker.get(s_spk, 0) + int(n_sentence)
                            else:
                                txt = s.get("text", "") or ""
                                words_by_speaker[s_spk] = words_by_speaker.get(s_spk, 0) + (len(txt.split()) if txt else 0)

                            st = _float_or_none(s.get("start"))
                            en = _float_or_none(s.get("end", s.get("stop")))
                            if st is not None and en is not None and en >= st:
                                time_by_speaker[s_spk] = time_by_speaker.get(s_spk, 0.0) + float(en - st)
                    else:
                        spk = p.get("speaker", None)
                        if spk is None:
                            continue
                        if isinstance(spk, str) and spk.isdigit():
                            spk = int(spk)

                        n_paragraph = p.get("num_words", None)
                        if isinstance(n_paragraph, (int, float)):
                            words_by_speaker[spk] = words_by_speaker.get(spk, 0) + int(n_paragraph)
                        else:
                            acc_sent_words = 0
                            for s in sentences:
                                n_sentence = s.get("num_words", None)
                                if isinstance(n_sentence, (int, float)):
                                    acc_sent_words += int(n_sentence)
                                else:
                                    txt = s.get("text", "") or ""
                                    acc_sent_words += (len(txt.split()) if txt else 0)
                            words_by_speaker[spk] = words_by_speaker.get(spk, 0) + acc_sent_words

                        st = _float_or_none(p.get("start"))
                        en = _float_or_none(p.get("end", p.get("stop")))
                        if st is not None and en is not None and en >= st:
                            time_by_speaker[spk] = time_by_speaker.get(spk, 0.0) + float(en - st)

    except Exception:
        pass

    # === 2) Fallback para palabras si no se encontró ningún num_words ===
    if not words_by_speaker:
        for spk, sent in _iter_paragraphs(call_json) or []:
            if spk is None:
                continue
            tokens = re.findall(r"\w+(?:'\w+)?", sent["text"], flags=re.UNICODE)
            n_words = len(tokens)
            words_by_speaker[spk] = words_by_speaker.get(spk, 0) + n_words

            st = sent["start"]; en = sent["end"]
            dur = (en - st) if (st is not None and en is not None and en >= st) else 0.0
            time_by_speaker[spk] = time_by_speaker.get(spk, 0.0) + float(dur)

    total_words = sum(words_by_speaker.values())
    total_time = sum(time_by_speaker.values())

    def pct_map(raw: dict, total: float) -> dict:
        if not total:
            return {k: 0.0 for k in raw}
        return {k: round((v / total) * 100.0, 2) for k, v in raw.items()}

    metrics = {
        "palabras": {
            "por_speaker": {str(k): v for k, v in sorted(words_by_speaker.items())},
            "totales": total_words,
            "porcentaje": {str(k): v for k, v in sorted(pct_map(words_by_speaker, float(total_words)).items())}
        },
        "tiempo_seg": {
            "por_speaker": {str(k): round(v, 3) for k, v in sorted(time_by_speaker.items())},
            "totales": round(total_time, 3),
            "porcentaje": {str(k): v for k, v in sorted(pct_map(time_by_speaker, total_time).items())}
        }
    }
    return metrics

# Banco de preguntas de calidad para "preguntas_detectadas"
PREGUNTAS_BANCO = [
    "¿Cómo cree que estos servicios podrían integrarse en su rutina diaria?",
    "¿Qué beneficios específicos espera obtener de este servicio?",
    "¿Cómo se compara este servicio con otros que ha utilizado anteriormente?",
    "¿Qué preocupaciones tiene sobre la implementación de este servicio?",
    "¿Cómo podría este servicio mejorar su calidad de vida o la de su familia?",
    "¿Qué aspectos del servicio le generan más curiosidad?",
    "¿Cómo se imagina utilizando este servicio en situaciones cotidianas?",
    "¿Qué le gustaría saber más sobre cómo funciona este servicio?",
    "¿Qué experiencias pasadas le hacen considerar este servicio como una opción viable?",
    "¿Cómo cree que este servicio podría resolver problemas que ha enfrentado antes?",
    "¿Qué le ha llamado más la atención de los servicios?",
    "¿Tiene alguna pregunta o inquietud sobre lo que le he comentado?",
    "¿Qué servicios cree que utilizaría más?",
    "¿Cómo cree que estos servicios podrían beneficiarle?"
]

def _linear_ratio(value: float, threshold: float) -> float:
    if value is None:
        return 0.0
    if value <= 0:
        return 0.0
    if value >= threshold:
        return 1.0
    return float(value) / float(threshold)

def compute_participacion_compuesta(participation_dict: dict, cliente_role_id: int | None) -> dict:
    if cliente_role_id is None:
        return {
            "cliente_pct_palabras": 0.0, "cliente_pct_tiempo": 0.0,
            "cliente_pct_promedio": 0.0, "puntaje_0_a_1": 0.0
        }
    cid = str(cliente_role_id)
    try:
        pct_words = participation_dict["palabras"]["porcentaje"].get(cid, 0.0)
        pct_time  = participation_dict["tiempo_seg"]["porcentaje"].get(cid, 0.0)
    except Exception:
        pct_words = 0.0
        pct_time  = 0.0
    avg_pct = round((pct_words + pct_time) / 2.0, 2)
    score01 = _linear_ratio(avg_pct, 15.0)
    return {
        "cliente_pct_palabras": pct_words,
        "cliente_pct_tiempo": pct_time,
        "cliente_pct_promedio": avg_pct,
        "puntaje_0_a_1": round(score01, 4)
    }

def compute_participacion_ponderada(participacion_score_01: float,
                                    respuestas_score_01: float | None,
                                    total_pregs_cliente: int) -> float:
    if not isinstance(total_pregs_cliente, (int, float)) or total_pregs_cliente <= 0:
        return max(0.0, min(1.0, float(participacion_score_01 or 0.0)))
    rs = 0.0 if respuestas_score_01 is None else float(respuestas_score_01)
    rs = max(0.0, min(1.0, rs))
    ps = max(0.0, min(1.0, float(participacion_score_01 or 0.0)))
    return round(0.5 * ps + 0.5 * rs, 4)

def compute_preguntas_score(n_pregs: int | None) -> float:
    if not isinstance(n_pregs, (int, float)):
        return 0.0
    if n_pregs <= 0:
        return 0.0
    if n_pregs >= 4:
        return 1.0
    return float(n_pregs) / 4.0

def build_venta_consciente_score(
    participacion_puntaje_0a1: float,
    preguntas_puntaje_0a1: float,
    mac_puntaje_0a1: float,
    cercania_puntaje_0a1: float
) -> dict:
    PESO_PART = 0.30
    PESO_PREG = 0.30
    PESO_MAC  = 0.20
    PESO_CER  = 0.20
    final_01 = (
        participacion_puntaje_0a1 * PESO_PART +
        preguntas_puntaje_0a1     * PESO_PREG +
        mac_puntaje_0a1           * PESO_MAC  +
        cercania_puntaje_0a1      * PESO_CER
    )
    return {
        "puntaje_final_0_a_1": round(final_01, 4),
        "desglose": {
            "participacion_del_cliente": {"peso": PESO_PART, "puntaje_0_a_1": round(participacion_puntaje_0a1, 4)},
            "preguntas_detectadas":     {"peso": PESO_PREG, "puntaje_0_a_1": round(preguntas_puntaje_0a1, 4)},
            "MAC_camuflado":            {"peso": PESO_MAC,  "puntaje_0_a_1": round(mac_puntaje_0a1, 4)},
            "cercania_con_guion":       {"peso": PESO_CER,  "puntaje_0_a_1": round(cercania_puntaje_0a1, 4)},
        }
    }

# ==========================
# Prompts
# ==========================

super_prompt = f"""
Eres auditor/a de calidad de contact center para CLARO. Recibirás:
1) DATOS_PRECALCULADOS con el máximo silencio (segundos) ya calculado (NO recalcules).
2) GUI0N_CLARO_2025 (resumen del guion oficial).
3) TRANSCRITO_JSON de la llamada.
4) BANCO_PREGUNTAS: una lista de preguntas de referencia del AGENTE para comparar por significado, no por igualdad literal.

BANCO_PREGUNTAS = {json.dumps(PREGUNTAS_BANCO, ensure_ascii=False)}

DEBES IDENTIFICAR quién es AGENTE y quién es CLIENTE (speaker 0/1) con evidencia textual.
Si no hay evidencia suficiente, deja null y explica en 'evidencia'.

IMPORTANTE:
- Tu evaluación SOLO debe cubrir estos ítems:
  (A) todas las "preguntas_detectadas" del AGENTE (comparadas con BANCO_PREGUNTAS) // El conteo solo debe mostrar la cantidad de preguntas detectadas, no las similitudes.
  (B) "MAC_camuflado".
  (C) "cercania_con_guion".
  (D) "preguntas_cliente_y_respuestas": detecta las PREGUNTAS que hace el CLIENTE y evalúa si el AGENTE las respondió completa y coherentemente.
- El ítem (1) de participación del cliente (porcentaje) NO lo calculas tú; lo hace el sistema con datos previos.

Tu salida debe ser ÚNICAMENTE un JSON válido con este esquema EXACTO:

{{
  "tiempos": {{
    "max_espera_segundos": number|null,
    "excedio_60s": boolean
  }},
  "identificacion_speakers": {{
    "agente_speaker_id": 0|1|null,
    "cliente_speaker_id": 0|1|null
  }},
  "datos_cliente": {{
    "nombre_completo": string|null,
    "numero_documento": string|null,
    "correo": string|null
  }},
  "evaluacion": {{
    "ajuste_al_guion": {{
      "resumen_ejecutivo": string
    }},
    "preguntas_detectadas": {{
      "conteo": number,
      "coincidencias": [
        {{ "texto_en_llamada": string, "parecido_a": string }}
      ]
    }},
    "MAC_camuflado": {{
      "frase_de_cierre_agente": string|null,
      "es_camuflado": boolean,
      "extras_detectados": [string],
      "similitud_0_a_1": number,
      "pregunta_de_confirmación": string|null,
      "respuesta_cliente_a_pregunta_de_confirmacion": string|null,
      "timestamp_confirmacion": number|null

    }},
    "cercania_con_guion": {{
      "puntaje_0_a_1": number
    }},
    "preguntas_cliente_y_respuestas": {{
      "total_preguntas_cliente": number,
      "respondidas_bien": number,
      "detalle": [
        {{
          "pregunta_cliente": string,
          "tipo": "simple" | "elaborada",
          "respuesta_agente": string|null,
          "respuesta_adecuada": boolean,
          "razon": string
        }}
      ]
    }}
  }}
}}

REGLAS:
- Idioma: español. Cita literal en evidencias.
- Para “preguntas_cliente_y_respuestas”:
  - Marca como "simple" si la pregunta es de precio, sí/no, dato directo; "elaborada" si requiere explicación.
  - "respuesta_adecuada" = true si la respuesta del AGENTE contesta completa y coherentemente la pregunta del CLIENTE (según su tipo).
- “tiempos” debe copiar EXACTAMENTE los DATOS_PRECALCULADOS.
- NO inventes datos de cliente; extrae lo que exista textual en el transcrito (si está ofuscado, respétalo).
- En muchas ocasiones, antes de que el agente pregunte al cliente por su "numero_documento", el agente seguramente dirá los primeros digitos y luego le pedirá al cliente confirmar el resto, si eso ocurre, concatena esos primeros digitos con los demás que diga el cliente.
- En las recomendaciones_breves has sugerencias sobre lo que el agente puede mejorar en la llamada, que sean concretas pero dicientes.
- “tiempos” debe copiar EXACTAMENTE el valor de “max_espera_segundos” y “excedio_60s” entregado en DATOS_PRECALCULADOS.

- Para “preguntas_detectadas” (AGENTE vs BANCO_PREGUNTAS):
  - Considera como **pregunta del AGENTE** cualquier enunciado interrogativo o acto de habla equivalente (aunque no lleve “¿...?” explícito) que busque información, confirme, sondee necesidades o valide aceptación.
  - Compara **por significado** con los ítems de BANCO_PREGUNTAS (sinónimos, paráfrasis, inversión de orden, cambios menores). No exijas igualdad literal.
  - Para cada pregunta del AGENTE que sea semánticamente similar a alguna del BANCO_PREGUNTAS, agrega una entrada en "coincidencias" con:
      - "texto_en_llamada": la pregunta tal como aparece (cita breve literal, saneada si es necesario),
      - "parecido_a": la **pregunta del BANCO_PREGUNTAS** más cercana en significado.
  - **No** incluyas duplicados obvios del mismo turno; si el AGENTE reformula inmediatamente la misma pregunta, cuenta **una sola** coincidencia.
  - El campo "conteo" = número total de coincidencias listadas.
  - Evita registrar preguntas de trámite puramente logístico (p. ej., “¿me copia por favor?”, “¿me escucha?”, “¿aló?”) salvo que correspondan claramente a una intención del BANCO_PREGUNTAS.

- Para “MAC_camuflado”:
      -1) 'frase_de_cierre_agente' es la misma pregunta de confirmación del servicio: es donde se le pregunta al cliente si desea adquirir el producto/servicio.
      -2) trata personalizaciones triviales (nombre) como aceptables; marca “extras” cuando agreguen sesgo o parafernalia (ej.: “para el bienestar de su familia”) pegado a la pregunta de cierre o justo antes de la pregunta.
      -3) la "pregunta de confirmación" debe ser la pregunta que el agente le hace al cliente para que el cliente confirme que desea adquirir el producto. es importante que se muestre textualmente lo que se dijo. esta pregunta de confirmación es la que está definida en el guion de operación, justo después de el texto "CONFIRMACIÓN DE VENTA", y aunque puede variar ligeramente en cada llamada debe tener la misma pragmática y significado.
      -3.1) Si leyendo lo que consideras como pregunta de confirmación no estás seguro, revisa el contexto inmediato anterior y posterior para asegurarte de que es efectivamente la pregunta de confirmación del servicio y agrega ese contexto anterior y posterior a la cita textual que entregues en "pregunta de confirmación".
      -4) la "respuesta_cliente" debe ser la respuesta textual del cliente a la pregunta de confirmación.
      -5) si no se detecta claramente la pregunta de confirmación o la respuesta del cliente, deja esos campos como null.
      -6) "timestamp_confirmacion" es el tiempo (en segundos) en que se produce la pregunta de confirmación dentro de la llamada (si no se puede determinar, deja null). Ese tiempo se encuentra en el campo "start" de la "sentence" o "paragraph" donde se produce la pregunta de confirmación.
- Devuelve SOLO el JSON final, sin comentarios ni explicaciones.
"""


def build_diarization_prompt(call_json: Dict[str, Any]) -> str:
    attached = json.dumps(call_json, ensure_ascii=False)
    instrucciones = (
        "Corrige la diarización de este transcrito detectando qué partes corresponden al AGENTE (0) y al CLIENTE (1). "
        "Devuelve el *MISMO JSON*, sin alterar texto, tiempos ni num_words. "
        "Cambia únicamente los campos 'speaker' de cada 'paragraph', puede que consideres que en un mismo 'paragraph' hay más de un 'speaker', en ese caso el 'speaker será el predominante'. "
        "Responde con SOLO el JSON final."
    )
    return instrucciones + "\n\n# TRANSCRITO_JSON\n" + attached

def extract_json_block(text: str) -> dict:
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        return json.loads(fence.group(1))

    start = text.find("{")
    if start != -1:
        stack = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                stack += 1
            elif text[i] == "}":
                stack -= 1
                if stack == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        pass
    return json.loads(text)  # último intento directo (puede lanzar)

def diarize_transcript_with_llm(client: OpenAI, model: str, call_json: Dict[str, Any]) -> Dict[str, Any]:
    try:
        prompt = build_diarization_prompt(call_json)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Responde exactamente según las instrucciones del usuario."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.01,
        )
        raw = resp.choices[0].message.content
        fixed = extract_json_block(raw)
        return fixed if isinstance(fixed, dict) else call_json
    except Exception as e:
        print(f"⚠️ Diarización: error al invocar LLM ({e}). Se mantiene transcrito original.")
        return call_json

# ==========================
# Helpers varios
# ==========================

def _safe_get(d, path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def is_unbalanced_100_0(participation: dict) -> bool:
    try:
        pw = participation["palabras"]["porcentaje"]
        pt = participation["tiempo_seg"]["porcentaje"]
        def _flag(pdict):
            vals = list((pdict or {}).values())
            if not vals:
                return False
            if len(vals) == 1:
                return abs(float(vals[0]) - 100.0) < 1e-6
            return (abs(max(vals) - 100.0) < 1e-6) and (min(vals) == 0.0)
        return _flag(pw) or _flag(pt)
    except Exception:
        return False

def build_full_prompt(call_json: Dict[str, Any], base_prompt: str,
                      max_silence: float | None, exceeded_60s: bool,
                      script_text: str) -> str:
    attached = json.dumps(call_json, ensure_ascii=False)
    datos_precalc = {
        "max_espera_segundos": max_silence,
        "excedio_60s": bool(exceeded_60s)
    }
    return (
        base_prompt.rstrip()
        + "\n\n# DATOS_PRECALCULADOS\n"
        + json.dumps(datos_precalc, ensure_ascii=False)
        + "\n\n# GUI0N_CLARO_2025\n"
        + (script_text or "")
        + "\n\n# TRANSCRITO_JSON\n"
        + attached
    )

def _json_str(x) -> str:
    try:
        if x is None:
            return ""
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return ""

def _strip_brackets_text(text: str) -> str:
    try:
        if not text:
            return ""
        return re.sub(r"[\[\]\{\}]", "", str(text))
    except Exception:
        return str(text or "")

def _apply_monthly_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Se mantiene el mismo cálculo, pero aplicado al DF retornado (batch local)
    vc_col = "porcentaje_venta_consciente"
    df[vc_col] = pd.to_numeric(df[vc_col], errors="coerce")

    promedio_general = df[vc_col].mean(skipna=True)

    if pd.notna(promedio_general) and promedio_general != 0:
        df["diferencia_%_vs_promedio"] = ((df[vc_col] - promedio_general) / promedio_general) * 100
    else:
        df["diferencia_%_vs_promedio"] = None

    df["promedio_general"] = promedio_general
    df["diferencia_%_vs_promedio"] = df["diferencia_%_vs_promedio"].round(2)
    try:
        df["promedio_general"] = df["promedio_general"].round(0).astype("Int64")
    except Exception:
        pass

    cols = list(df.columns)
    for tail_col in ["diferencia_%_vs_promedio", "promedio_general"]:
        if tail_col in cols:
            cols.remove(tail_col)

    if "Nombre llamada" in cols and vc_col in cols:
        cols.remove(vc_col)
        insert_pos = cols.index("Nombre llamada") + 1
        cols = cols[:insert_pos] + [vc_col] + cols[insert_pos:]

    cols = [c for c in cols if c not in ["cliente_tipo_documento", "max_espera_minutos", "preguntas_detectadas_json", "cliente_pct_promedio_participacion"]]
    return df[cols + ["diferencia_%_vs_promedio", "promedio_general"]]

def _role_pct_promedio(role_obj: dict | None) -> float:
    p = role_obj or {}
    pw = float((p.get("palabras") or {}).get("porcentaje", 0.0) or 0.0)
    pt = float((p.get("tiempo_seg") or {}).get("porcentaje", 0.0) or 0.0)
    return round((pw + pt) / 2.0, 2)

def _role_counts_and_time_min(role_obj: dict | None) -> tuple[int, float]:
    p = role_obj or {}
    n_pal = int((p.get("palabras") or {}).get("conteo", 0) or 0)
    segs  = float((p.get("tiempo_seg") or {}).get("conteo", 0.0) or 0.0)
    mins  = round(segs / 60.0, 1)
    return n_pal, mins

# ==========================
# Core API
# ==========================

def evaluate_consent(transcripts_path: str, script: str, *, mode: str = PROCESS_MODE, only_n: int = PROCESS_ONLY_N) -> pd.DataFrame:
    """
    Evalúa todos los JSON en `transcripts_path` (recursivo) y retorna un DataFrame con los resultados.
    No escribe archivos. No usa AWS.

    Uso:
        df = consent_evaluator.evaluate_consent("/ruta/a/transcripts", guion_texto)
    """
    files = list_local_json_files(transcripts_path, mode=mode, only_n=only_n)
    if not files:
        return pd.DataFrame()

    client = openai_client()
    base_prompt = super_prompt

    processed_date_str = datetime.now().strftime("%Y-%m-%d")
    rows: list[dict] = []

    print(f"\n🚀 Iniciando procesamiento local: {len(files)} archivo(s). Modo='{mode}'.")
    for i, fpath in enumerate(files, 1):
        try:
            print(f"[LOAD] Leyendo JSON: {fpath}")
            call = read_local_json(fpath)

            participation = compute_participation_metrics(call)

            # --- Diarización contextual previa si se detecta 100/0 ---
            attempts = 0
            while is_unbalanced_100_0(participation) and attempts < DIARIZATION_MAX_ATTEMPTS:
                print("[DIAR] 100/0 detectado. Enviando transcrito a LLM para diarización contextual...")
                call_fixed = diarize_transcript_with_llm(client, OPENAI_MODEL, call)

                global DIARIZATION_TRIGGERED_COUNT
                DIARIZATION_TRIGGERED_COUNT += 1

                participation = compute_participation_metrics(call_fixed)
                if not is_unbalanced_100_0(participation):
                    call = call_fixed
                    print("[DIAR] Diarización aplicada. Recalculadas métricas de participación.")
                    break
                attempts += 1

            if is_unbalanced_100_0(participation):
                print("⚠️ [DIAR] Diarización no resolvió 100/0. Se continúa con participación 100%/0% para este archivo.")

            agent_id = extract_agent_id_from_filename(fpath.name)
            # sin mapping externo, usamos agent_id como "Nombre agente" (mantiene columna y utilidad)
            agent_name = agent_id

            sil = compute_max_silence_from_text(call)
            full_prompt = build_full_prompt(call, base_prompt, sil.max_silence, sil.exceeded_60s, script_text=script)

            print(f"[PROCESS] {i}/{len(files)} -> Enviando a OpenAI...")
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Responde exactamente según las instrucciones del usuario."},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.01,
            )
            raw_text = resp.choices[0].message.content

            try:
                result = extract_json_block(raw_text)
            except Exception as e:
                print(f"❌ No se pudo extraer JSON de la respuesta para {fpath.name}: {e}")
                continue

            if isinstance(result, dict):
                result["nombre agente"] = agent_name
            else:
                result = {"llm_result": result, "nombre agente": agent_name}

            result["participacion_speakers"] = participation

            ag_id = _safe_get(result, ["identificacion_speakers", "agente_speaker_id"], None)
            cl_id = _safe_get(result, ["identificacion_speakers", "cliente_speaker_id"], None)

            def _role_view(part, role_id: int):
                return {
                    "palabras": {
                        "conteo": part["palabras"]["por_speaker"].get(str(role_id), 0),
                        "porcentaje": part["palabras"]["porcentaje"].get(str(role_id), 0.0),
                    },
                    "tiempo_seg": {
                        "conteo": part["tiempo_seg"]["por_speaker"].get(str(role_id), 0.0),
                        "porcentaje": part["tiempo_seg"]["porcentaje"].get(str(role_id), 0.0),
                    },
                }

            if isinstance(ag_id, int) or (isinstance(ag_id, str) and ag_id.isdigit()):
                role_id = int(ag_id)
                agente_view = _role_view(participation, role_id)
            else:
                agente_view = None

            if isinstance(cl_id, int) or (isinstance(cl_id, str) and cl_id.isdigit()):
                role_id = int(cl_id)
                cliente_view = _role_view(participation, role_id)
            else:
                cliente_view = None

            result["participacion_roles"] = {"agente": agente_view, "cliente": cliente_view}

            cliente_role_id = int(cl_id) if (isinstance(cl_id, int) or (isinstance(cl_id, str) and cl_id.isdigit())) else None
            part_compuesta = compute_participacion_compuesta(participation, cliente_role_id)

            llm_preguntas = _safe_get(result, ["evaluacion", "preguntas_detectadas"], {}) or {}
            preguntas_conteo = llm_preguntas.get("conteo", 0)
            preguntas_puntaje_01 = round(compute_preguntas_score(preguntas_conteo), 4)

            llm_mac = _safe_get(result, ["evaluacion", "MAC_camuflado"], {}) or {}
            mac_sim_01 = float(llm_mac.get("similitud_0_a_1", 0.0) or 0.0)
            mac_puntaje_01 = max(0.0, min(1.0, mac_sim_01))

            llm_cercania = _safe_get(result, ["evaluacion", "cercania_con_guion", "puntaje_0_a_1"], 0.0)
            cercania_puntaje_01 = max(0.0, min(1.0, float(llm_cercania or 0.0)))

            llm_qna_cli = _safe_get(result, ["evaluacion", "preguntas_cliente_y_respuestas"], {}) or {}
            total_cli_pregs = int(llm_qna_cli.get("total_preguntas_cliente", 0) or 0)
            respondidas_bien = int(llm_qna_cli.get("respondidas_bien", 0) or 0)
            respuestas_score_01 = (respondidas_bien / total_cli_pregs) if total_cli_pregs > 0 else None

            participacion_efectiva_01 = compute_participacion_ponderada(
                participacion_score_01=part_compuesta["puntaje_0_a_1"],
                respuestas_score_01=respuestas_score_01,
                total_pregs_cliente=total_cli_pregs
            )

            venta_consciente = build_venta_consciente_score(
                participacion_puntaje_0a1=participacion_efectiva_01,
                preguntas_puntaje_0a1=preguntas_puntaje_01,
                mac_puntaje_0a1=mac_puntaje_01,
                cercania_puntaje_0a1=cercania_puntaje_01
            )

            result["venta_consciente"] = {
                "puntaje_final_0_a_1": venta_consciente["puntaje_final_0_a_1"],
                "desglose": venta_consciente["desglose"],
                "participacion_del_cliente": {
                    "cliente_pct_palabras": part_compuesta["cliente_pct_palabras"],
                    "cliente_pct_tiempo": part_compuesta["cliente_pct_tiempo"],
                    "cliente_pct_promedio": part_compuesta["cliente_pct_promedio"],
                    "regla_aplicada": "sin_preguntas=>30%" if (total_cli_pregs == 0) else "con_preguntas=>15%+15%",
                    "puntaje_0_a_1_efectivo": participacion_efectiva_01
                },
                "preguntas_cliente": {
                    "total": total_cli_pregs,
                    "respondidas_bien": respondidas_bien,
                    "puntaje_0_a_1": (None if respuestas_score_01 is None else round(respuestas_score_01, 4)),
                    "detalle_llm": llm_qna_cli
                },
                "preguntas_detectadas": {
                    "conteo": preguntas_conteo,
                    "puntaje_0_a_1": preguntas_puntaje_01,
                    "detalle_llm": llm_preguntas
                },
                "MAC_camuflado": {"similitud_0_a_1": mac_puntaje_01, "detalle_llm": llm_mac},
                "cercania_con_guion": {"puntaje_0_a_1": cercania_puntaje_01}
            }

            # --- Construir fila DF (mismas columnas del Excel previo) ---
            nombre_llamada = _basename_without_ext(fpath.name)
            call_date_str = extract_call_date_from_filename(fpath.name) or ""

            tiempos       = _safe_get(result, ["tiempos"], {}) or {}
            datos_cliente = _safe_get(result, ["datos_cliente"], {}) or {}
            evaluacion    = _safe_get(result, ["evaluacion"], {}) or {}
            eval_preg     = _safe_get(evaluacion, ["preguntas_detectadas"], {}) or {}
            eval_mac      = _safe_get(evaluacion, ["MAC_camuflado"], {}) or {}
            eval_cerc     = _safe_get(evaluacion, ["cercania_con_guion"], {}) or {}
            part_roles    = _safe_get(result, ["participacion_roles"], {}) or {}

            part_agente  = part_roles.get("agente")
            part_cliente = part_roles.get("cliente")

            ag_pct_prom = _role_pct_promedio(part_agente)
            cl_pct_prom = _role_pct_promedio(part_cliente)
            ag_palabras, ag_tiempo_min = _role_counts_and_time_min(part_agente)
            cl_palabras, cl_tiempo_min = _role_counts_and_time_min(part_cliente)

            venta_consciente_pct = None
            try:
                vc_01 = float(result.get("venta_consciente", {}).get("puntaje_final_0_a_1", None))
                if vc_01 is not None:
                    venta_consciente_pct = int(round(vc_01 * 100, 0))
            except Exception:
                venta_consciente_pct = None

            cercania_pct = None
            try:
                cer_01 = float(eval_cerc.get("puntaje_0_a_1", None))
                if cer_01 is not None:
                    cercania_pct = int(round(cer_01 * 100, 0))
            except Exception:
                cercania_pct = None

            if total_cli_pregs == 0:
                cliente_q_resueltas_pct = "No aplica"
                cliente_q_detalle = "No aplica"
            else:
                cliente_q_resueltas_pct = int(round(float(respuestas_score_01 or 0.0) * 100, 0))
                cliente_q_detalle = _strip_brackets_text(_json_str(llm_qna_cli.get("detalle", [])))

            n_pregs_agente = int(eval_preg.get("conteo", 0) or 0)
            detalle_pregs_agente_dict = {k: v for k, v in (eval_preg or {}).items() if k != "conteo"}
            detalle_pregs_agente_txt = _strip_brackets_text(_json_str(detalle_pregs_agente_dict))

            fila = {
                "Nombre llamada": nombre_llamada,
                "Numero de Cabina": agent_name,

                "porcentaje_venta_consciente": venta_consciente_pct,

                "Cliente nombre": datos_cliente.get("nombre_completo"),
                "Cliente numero documento": datos_cliente.get("numero_documento"),
                "Cliente correo": datos_cliente.get("correo"),

                "% de participacion agente": ag_pct_prom,
                "% participacion cliente": cl_pct_prom,

                "Agente palabrasconteo": ag_palabras,
                "Agente tiempo mins": ag_tiempo_min,
                "Cliente palabras conteo": cl_palabras,
                "Cliente tiempo mins": cl_tiempo_min,

                "% cercania con guion": cercania_pct,

                "Evaluacion json": _safe_get(evaluacion, ["ajuste_al_guion", "resumen_ejecutivo"], ""),

                "# de preguntas agente": n_pregs_agente,
                "Detalle preguntas agente": detalle_pregs_agente_txt,

                "Eficiencia de respuesta del agente ": cliente_q_resueltas_pct,
                "Detalle preguntas resueltas cliente": cliente_q_detalle,

                "% de Similitud MAC camuflado": eval_mac.get("similitud_0_a_1"),
                "Detalles MAC camuflado": _strip_brackets_text(_json_str({k: v for k, v in (eval_mac or {}).items() if k != "similitud_0_a_1"})),

                "fecha llamada": call_date_str,
                "fecha procesado": processed_date_str,
            }

            rows.append(fila)

        except Exception as e:
            print(f"❌ Error procesando {fpath.name}: {e}")

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty:
        df = _apply_monthly_summary_columns(df)

    print(f"Total llamadas diarizadas:{DIARIZATION_TRIGGERED_COUNT}")
    df.to_excel("consent_evaluation_output.xlsx", index=False)
    return df


# Backward compatible entrypoint 
def process_transcripts_consent_evaluator(transcripts_path: str, script: str, mode: str = PROCESS_MODE, only_n: int = PROCESS_ONLY_N) -> pd.DataFrame:
    """
    Docstring for process_transcripts
    :param transcripts_path: is the path to the transcripts
    :type transcripts_path: str
    :param script: Is the script text, it depends on the use case
    :type script: str
    :param mode: This function mode, can be 'all' or 'sample', defaults to 'all'. 'sample' processes only a subset of files and its useful for testing.
    :type mode: str
    :param only_n: If mode is 'sample', this parameter indicates how many files to process, defaults to 2.
    :type only_n: int
    :return: Dataframe with the evaluation of the consent evaluation for each transcript.
    :rtype: DataFrame
    """
    return evaluate_consent(transcripts_path, script, mode=mode, only_n=only_n)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Consent evaluator (local, sin AWS).")
    ap.add_argument("transcripts_path", help="Carpeta local con JSONs")
    ap.add_argument("--script_file", help="Ruta a un .txt con el guion", required=False)
    ap.add_argument("--script_text", help="Texto del guion (si no usa --script_file)", required=False)
    ap.add_argument("--mode", default=PROCESS_MODE, choices=["all", "sample"])
    ap.add_argument("--only_n", type=int, default=PROCESS_ONLY_N)
    args = ap.parse_args()

    if args.script_file:
        script_txt = Path(args.script_file).read_text(encoding="utf-8", errors="replace")
    else:
        script_txt = args.script_text or ""

    df_out = evaluate_consent(args.transcripts_path, script_txt, mode=args.mode, only_n=args.only_n)
    print(df_out.head(5).to_string(index=False))
