"""
schema.py — Contratos de datos para scoreEngineV2

Tres entidades principales:
  - CallRecord     : una fila por llamada (evolución de MAT_CALLS_THIS_CAMPAIGN)
  - TopicRecord    : una fila por tópico por llamada (evolución de values_per_topic_for_all_convs)
  - CampaignStats  : una fila por tópico a nivel campaña (evolución de statistics)

Convenciones:
  - Todos los scores continuos están en rango [0, 1] salvo que se indique _10 (rango [0, 10]).
  - Campos con prefijo score_d* son las cinco dimensiones del modelo MDCL.
  - None indica "no calculado aún", distinto de 0.0 (calculado, valor cero).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, asdict
from typing import TYPE_CHECKING, Optional

# numpy y pandas se importan solo cuando se invocan los métodos que los necesitan
# (to_series, from_series, from_mat_dataframe), para que el módulo importe
# limpio incluso en entornos de test sin esas dependencias.
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


# ---------------------------------------------------------------------------
# Pesos del modelo MDCL — fuente única de verdad
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS: dict[str, float] = {
    "d1_compliance":  0.30,
    "d2_script":      0.25,
    "d3_mac_quality": 0.20,
    "d4_engagement":  0.15,
    "d5_efficiency":  0.10,
}

SCORE_VERSION = "2.0.0"


# ---------------------------------------------------------------------------
# CallRecord
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    """
    Registro completo de una llamada calificada.

    Campos agrupados por sección:
      1. Identidad
      2. Transcripción y audio
      3. Estadísticas por speaker
      4. Keywords
      5. MAC / Precio
      6. Momentos de compliance
      7. Estructura conversacional
      8. Puntajes por dimensión (D1–D5)
      9. Score final y trazabilidad
    """

    # ── 1. IDENTIDAD ────────────────────────────────────────────────────────
    file_name:   str = ""
    DATE_TIME:   Optional[str] = None
    LEAD_ID:     Optional[str] = None
    AGENT_ID:    Optional[str] = None
    CLIENT_ID:   Optional[str] = None

    # ── 2. TRANSCRIPCIÓN Y AUDIO ────────────────────────────────────────────
    transcript:       str   = ""
    # conversation y speaker_order se mantienen como str/list por compatibilidad
    # con el pipeline existente; no participan en el scoring.
    conversation:     Optional[object] = None   # list[str] serializado
    speaker_order:    Optional[object] = None   # list[int] serializado

    confidence_score: float = 0.0   # confianza media del ASR (0–1)
    noauditable:      bool  = False  # True si confidence_score < 0.9

    TMO:              float = 0.0   # duración total en minutos
    silence_ratio:    Optional[float] = None  # % de duración sin actividad de voz (0–1)

    # ── 3. ESTADÍSTICAS POR SPEAKER ─────────────────────────────────────────
    agent_word_count:          int   = 0
    client_word_count:         int   = 0
    agent_participation:       float = 0.0   # ratio palabras agente / total (0–1)
    agent_wpm:                 Optional[float] = None  # palabras/minuto del agente
    client_wpm:                Optional[float] = None  # palabras/minuto del cliente
    agent_max_monologue_words: Optional[int]   = None  # monólogo más largo (# palabras)
    turn_count:                Optional[int]   = None  # número de cambios de turno
    client_question_count:     Optional[int]   = None  # preguntas del cliente detectadas

    # ── 4. KEYWORDS ──────────────────────────────────────────────────────────
    count_must_have: int   = 0
    count_forbidden: int   = 0
    must_have_rate:  float = 0.0   # % de keywords obligatorias presentes (0–100)
    forbidden_rate:  float = 0.0   # % de keywords prohibidas presentes (0–100)

    # ── 5. MAC / PRECIO ──────────────────────────────────────────────────────
    best_mac_window:         Optional[str]   = None
    best_mac_likelihood:     float           = 0.0   # similitud normalizada (0–1)
    best_price_likelihood:   float           = 0.0
    mac_times_said:          int             = 0     # veces que apareció tópico MAC
    price_times_said:        int             = 0
    mac_warn:                bool            = True  # True si likelihood < tolerancia
    price_warn:              bool            = True
    mac_completeness_score:  Optional[float] = None  # % atributos MAC cubiertos (0–1)

    # ── 6. MOMENTOS DE COMPLIANCE ────────────────────────────────────────────
    # Detectado (bool, compatible con modelo anterior)
    mvd_detected:       bool = False   # CONFIRMACION DATOS presente
    terms_detected:     bool = False   # TERMINOS LEGALES presente
    mac_r_detected:     bool = False   # MAC REFUERZO presente
    igs_comp_detected:  bool = False   # mención de producto integral

    # Calidad semántica del momento (nuevo — None = no calculado)
    mvd_quality_score:     Optional[float] = None  # similitud vs plantilla MVD (0–1)
    terms_coverage_score:  Optional[float] = None  # % cláusulas cubiertas (0–1)
    mac_r_quality_score:   Optional[float] = None  # similitud vs plantilla refuerzo (0–1)

    # ── 7. ESTRUCTURA CONVERSACIONAL ─────────────────────────────────────────
    # Lista de labels en el orden temporal en que ocurrieron
    topic_order_observed:  Optional[list[str]] = None
    # Kendall-tau normalizado vs secuencia canónica de la campaña (0–1, 1=perfecto)
    topic_order_score:     Optional[float] = None
    # En qué fracción de la llamada apareció el primer MAC (0–1)
    time_to_first_mac_pct: Optional[float] = None
    opening_detected:      Optional[bool]  = None
    closing_detected:      Optional[bool]  = None

    # ── 8. PUNTAJES POR DIMENSIÓN (0–10) ────────────────────────────────────
    score_d1_compliance:  Optional[float] = None  # compliance & requisitos legales
    score_d2_script:      Optional[float] = None  # adherencia al guión
    score_d3_mac_quality: Optional[float] = None  # calidad de presentación MAC
    score_d4_engagement:  Optional[float] = None  # engagement y comunicación
    score_d5_efficiency:  Optional[float] = None  # eficiencia de la llamada

    # ── 9. SCORE FINAL Y TRAZABILIDAD ────────────────────────────────────────
    score:         float = 0.0
    score_version: str   = SCORE_VERSION

    # ── MÉTODOS ──────────────────────────────────────────────────────────────

    def compute_final_score(self) -> float:
        """
        Calcula el score final como suma ponderada de las cinco dimensiones.
        Solo opera sobre dimensiones ya calculadas (no None).
        Retorna el score y lo asigna a self.score.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        dim_map = {
            "d1_compliance":  self.score_d1_compliance,
            "d2_script":      self.score_d2_script,
            "d3_mac_quality": self.score_d3_mac_quality,
            "d4_engagement":  self.score_d4_engagement,
            "d5_efficiency":  self.score_d5_efficiency,
        }

        for dim_key, dim_score in dim_map.items():
            if dim_score is not None:
                w = DIMENSION_WEIGHTS[dim_key]
                weighted_sum += dim_score * w
                total_weight  += w

        if total_weight == 0.0:
            self.score = 0.0
        else:
            # Normaliza por el peso acumulado para no penalizar dimensiones aún no calculadas
            raw = weighted_sum / total_weight
            self.score = round(max(0.0, min(10.0, raw)), 4)

        return self.score

    def dimensions_ready(self) -> list[str]:
        """Retorna los nombres de las dimensiones ya calculadas (no None)."""
        ready = []
        for key in DIMENSION_WEIGHTS:
            attr = f"score_{key}"
            if getattr(self, attr) is not None:
                ready.append(key)
        return ready

    def to_dict(self) -> dict:
        """Serializa el registro a dict plano (compatible con pd.DataFrame)."""
        return asdict(self)

    def to_series(self):
        """Serializa el registro a pd.Series (una fila de DataFrame)."""
        import pandas as pd
        return pd.Series(self.to_dict())

    @classmethod
    def from_series(cls, row) -> "CallRecord":
        """
        Construye un CallRecord desde una fila del DataFrame legacy
        (MAT_CALLS_THIS_CAMPAIGN / MAT_VOLUMES).

        Mapea los nombres de columnas del modelo anterior a los campos nuevos.
        Campos nuevos que no existan en la fila quedan en su valor por defecto.
        """
        import math
        known = {f.name for f in fields(cls)}

        # Mapeo de nombres legacy → nombres nuevos
        LEGACY_MAP: dict[str, str] = {
            "mvd":      "mvd_detected",
            "terms":    "terms_detected",
            "mac_r":    "mac_r_detected",
            "igs_comp": "igs_comp_detected",
        }

        kwargs: dict = {}
        for col, val in row.items():
            target = LEGACY_MAP.get(col, col)
            if target in known:
                # Convierte NaN a None para campos opcionales
                if isinstance(val, float) and math.isnan(val):
                    kwargs[target] = None
                else:
                    kwargs[target] = val

        # noauditable: si no viene en la fila, se deriva de confidence_score
        if "noauditable" not in kwargs:
            conf = kwargs.get("confidence_score", 0.0) or 0.0
            kwargs["noauditable"] = float(conf) < 0.9

        kwargs.setdefault("score_version", SCORE_VERSION)
        return cls(**kwargs)

    @classmethod
    def from_mat_dataframe(cls, df) -> list["CallRecord"]:
        """
        Construye una lista de CallRecord desde el DataFrame MAT completo.
        Útil para migrar el pipeline existente sin reescribir todo de golpe.
        """
        return [cls.from_series(row) for _, row in df.iterrows()]


# ---------------------------------------------------------------------------
# TopicRecord
# ---------------------------------------------------------------------------

@dataclass
class TopicRecord:
    """
    Registro de un tópico dentro de una llamada.
    Una llamada tiene N TopicRecords (uno por label detectado).
    """

    # Identidad
    file_name:    str = ""
    final_label:  str = ""

    # Timing
    topic_start:        float = 0.0
    topic_end:          float = 0.0
    time_centroid:      float = 0.0
    time_centroid_pct:  float = 0.0   # centroide / duración total

    # Contenido
    text_string:             str   = ""
    topic_num_words:         int   = 0
    topic_words_p_m:         float = 0.0
    topic_occurrence:        int   = 0
    velocity_classification: str   = ""

    # Calidad de transcripción
    topic_mean_conf:        float = 0.0
    topic_max_conf:         float = 0.0
    noauditable_transcript: bool  = False

    # Flags de compliance (compatibles con modelo anterior: 0/1)
    mvd:       int = 0
    terms:     int = 0
    mac_r:     int = 0
    igs_comp:  int = 0

    # Calidad semántica del tópico (nuevo)
    topic_semantic_score:   Optional[float] = None  # similitud vs plantilla (0–1)
    topic_keyword_coverage: Optional[float] = None  # % keywords esperadas presentes (0–1)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_series(self):
        import pandas as pd
        return pd.Series(self.to_dict())

    @classmethod
    def from_series(cls, row) -> "TopicRecord":
        import math
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for col, val in row.items():
            if col in known:
                kwargs[col] = None if (isinstance(val, float) and math.isnan(val)) else val
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# CampaignStats
# ---------------------------------------------------------------------------

@dataclass
class CampaignStats:
    """
    Estadísticas agregadas de un tópico a nivel de campaña completa.
    Una campaña tiene N CampaignStats (uno por label de tópico).
    """

    final_label:  str = ""

    # Posición temporal (existente)
    mean_centroid:  float = 0.0
    std_centroid:   float = 0.0
    min_centroid:   float = 0.0
    max_centroid:   float = 0.0

    # Calidad de transcripción (existente)
    mean_conf:       float = 0.0
    mean_max_conf:   float = 0.0

    # Velocidad y palabras (existente)
    mean_words_p_m:  float = 0.0
    mean_num_words:  float = 0.0
    topic_frequency: int   = 0

    # Posición esperada en el guión (nuevo)
    expected_position_pct:  Optional[float] = None  # 0–1, según guión canónico
    order_deviation_mean:   Optional[float] = None  # desviación media observado vs esperado

    def to_dict(self) -> dict:
        return asdict(self)

    def to_series(self):
        import pandas as pd
        return pd.Series(self.to_dict())

    @classmethod
    def from_series(cls, row) -> "CampaignStats":
        import math
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for col, val in row.items():
            if col in known:
                kwargs[col] = None if (isinstance(val, float) and math.isnan(val)) else val
        return cls(**kwargs)

    @classmethod
    def from_statistics_dataframe(cls, df) -> list["CampaignStats"]:
        return [cls.from_series(row) for _, row in df.iterrows()]


# ---------------------------------------------------------------------------
# Helpers de conversión DataFrame ↔ lista de dataclasses
# ---------------------------------------------------------------------------

def records_to_dataframe(records: list):
    """Convierte una lista de dataclasses (CallRecord / TopicRecord / CampaignStats) a DataFrame."""
    import pandas as pd
    return pd.DataFrame([asdict(r) for r in records])


def callrecords_from_dataframe(df: pd.DataFrame) -> list[CallRecord]:
    return CallRecord.from_mat_dataframe(df)


def topicrecords_from_dataframe(df: pd.DataFrame) -> list[TopicRecord]:
    return [TopicRecord.from_series(row) for _, row in df.iterrows()]


def campaignstats_from_dataframe(df: pd.DataFrame) -> list[CampaignStats]:
    return CampaignStats.from_statistics_dataframe(df)
