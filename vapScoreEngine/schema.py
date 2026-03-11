"""
schema.py — Contratos de datos para scoreEngineV3

Tres entidades principales:
  - CallRecord     : una fila por llamada
  - TopicRecord    : una fila por tópico por llamada
  - CampaignStats  : una fila por tópico a nivel campaña

Modelo de calificación MDCL v3 — cinco dimensiones ordenadas por impacto:

  D1 · mac_venta   (40 %) — Pregunta de Activación (MAC) y confirmación de venta.
       Es el indicador más crítico: sin un SÍ explícito o reforzado, la venta
       no existe. Captura la calidad del MAC, la respuesta del cliente y el
       correcto uso del MAC REFUERZO cuando la aceptación fue ambigua.

  D2 · compliance  (30 %) — Completitud de momentos regulatorios y legales.
       Mide si el agente cubrió todos los puntos obligatorios de la llamada
       (Términos Legales, Tratamiento de Datos, Ley de Retracto, Confirmación
       de Monitoreo, Confirmación de Datos, Precio explícito).

  D3 · script      (15 %) — Adherencia al guión y secuencia correcta.
       Penaliza tópicos faltantes y el desorden en el flujo de la llamada
       (Saludo → Perfilamiento → Producto → Conformidad → MAC → Términos →
       Despedida).

  D4 · engagement  (10 %) — Calidad comunicativa del agente.
       Velocidad de habla, balance de participación, preguntas del cliente,
       cantidad de turnos de conversación.

  D5 · audio       ( 5 %) — Calidad técnica del audio / transcripción ASR.
       Penaliza baja confianza de transcripción y exceso de silencios.

Convenciones:
  - Scores continuos en [0, 1] salvo sufijo _10 (rango [0, 10]).
  - Campos score_d* son las cinco dimensiones; score es el resultado ponderado.
  - None = "no calculado aún", distinto de 0.0 (calculado con valor cero).
  - mac_client_response y mac_flow_outcome son strings literales definidos
    abajo como constantes para facilitar comparaciones sin importar enums.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, asdict
from typing import TYPE_CHECKING, Optional, Literal

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Versión y pesos del modelo MDCL v3 — fuente única de verdad
# ---------------------------------------------------------------------------

SCORE_VERSION = "3.0.0"

DIMENSION_WEIGHTS: dict[str, float] = {
    "d1_mac_venta":  0.40,   # Pregunta de Activación + confirmación de venta
    "d2_compliance": 0.30,   # Completitud de momentos legales y regulatorios
    "d3_script":     0.15,   # Adherencia al guión y estructura conversacional
    "d4_engagement": 0.10,   # Calidad comunicativa del agente
    "d5_audio":      0.05,   # Calidad técnica del audio / transcripción ASR
}

# ---------------------------------------------------------------------------
# Momentos de compliance obligatorios para calcular compliance_completeness.
# El orden refleja el guión canónico esperado.
# ---------------------------------------------------------------------------

CANONICAL_SCRIPT_ORDER: list[str] = [
    "SALUDO",
    "CONFIRMACION MONITOREO",
    "PERFILAMIENTO",
    "PRODUCTO",
    "CONFORMIDAD",
    "TRATAMIENTO DATOS",
    "MAC",
    "MAC REFUERZO",          # sólo si aplica
    "PRECIO",
    "TERMINOS LEGALES",
    "LEY RETRACTO",
    "CONFIRMACION DATOS",
    "ATENCION",
    "DESPEDIDA",
]

# Subconjunto de tópicos cuya presencia es obligatoria (compliance duro).
# Ausencia de cualquiera de estos resta en D2.
REQUIRED_COMPLIANCE_TOPICS: list[str] = [
    "TERMINOS LEGALES",
    "TRATAMIENTO DATOS",
    "LEY RETRACTO",
    "CONFIRMACION MONITOREO",
    "PRECIO",
    "CONFIRMACION DATOS",
]

# ---------------------------------------------------------------------------
# Literales para el flujo MAC
# ---------------------------------------------------------------------------

# Tipo de respuesta del cliente ante la Pregunta de Activación.
MacClientResponse = Literal[
    "EXPLICITA",    # "Sí, perfecto", "Sí, de acuerdo", "Ok, sí" → venta directa
    "AMBIGUA",      # "Claro", "Correcto", "Muy bien", "Ok" → requiere MAC REFUERZO
    "AUSENTE",      # El cliente no respondió o la respuesta no fue detectada
]

# Resultado final del flujo de activación.
MacFlowOutcome = Literal[
    "VENTA_CONFIRMADA",   # MAC con respuesta EXPLICITA, sin necesidad de refuerzo
    "VENTA_REFUERZO",     # Respuesta AMBIGUA + MAC REFUERZO efectivo
    "NO_CONFIRMADA",      # MAC hecho pero sin respuesta afirmativa clara al final
    "NO_MAC",             # El MAC no fue detectado en la llamada
]


# ---------------------------------------------------------------------------
# CallRecord
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    """
    Registro completo de una llamada calificada.

    Secciones:
      1. Identidad
      2. Transcripción y audio
      3. Estadísticas por speaker
      4. Keywords (palabras clave obligatorias / prohibidas)
      5. MAC / Precio — calidad y flujo de activación
      6. Momentos de compliance — detección y calidad semántica
      7. Estructura conversacional — orden y cobertura de tópicos
      8. Puntajes por dimensión D1–D5  (escala 0–10)
      9. Score final y trazabilidad
    """

    # ── 1. IDENTIDAD ────────────────────────────────────────────────────────
    file_name:  str           = ""
    DATE_TIME:  Optional[str] = None
    LEAD_ID:    Optional[str] = None
    AGENT_ID:   Optional[str] = None
    CLIENT_ID:  Optional[str] = None

    # ── 2. TRANSCRIPCIÓN Y AUDIO ────────────────────────────────────────────
    transcript:       str            = ""
    conversation:     Optional[object] = None   # list[str] serializado
    speaker_order:    Optional[object] = None   # list[int] serializado

    confidence_score: float          = 0.0    # confianza media ASR (0–1)
    noauditable:      bool           = False  # True si confidence_score < 0.9

    TMO:           float          = 0.0   # duración total en minutos
    silence_ratio: Optional[float] = None  # fracción sin actividad de voz (0–1)

    # ── 3. ESTADÍSTICAS POR SPEAKER ─────────────────────────────────────────
    agent_word_count:          int            = 0
    client_word_count:         int            = 0
    # Fracción de palabras del agente sobre el total (0–1).
    # El rango óptimo para ventas es ~0.55–0.70; por encima de 0.85 se penaliza.
    agent_participation:       float          = 0.0
    agent_wpm:                 Optional[float] = None  # palabras/minuto del agente
    client_wpm:                Optional[float] = None  # palabras/minuto del cliente
    agent_max_monologue_words: Optional[int]   = None  # monólogo más largo
    turn_count:                Optional[int]   = None  # cambios de turno
    client_question_count:     Optional[int]   = None  # preguntas detectadas del cliente

    # ── 4. KEYWORDS ──────────────────────────────────────────────────────────
    count_must_have: int   = 0
    count_forbidden: int   = 0
    must_have_rate:  float = 0.0   # % de keywords obligatorias presentes (0–100)
    forbidden_rate:  float = 0.0   # % de keywords prohibidas presentes (0–100)

    # ── 5. MAC / PRECIO — calidad y flujo de activación ──────────────────────
    # Texto de la ventana con mayor similitud al guión MAC esperado.
    best_mac_window:       Optional[str]   = None
    # Similitud normalizada del mejor fragmento MAC detectado (0–1).
    best_mac_likelihood:   float           = 0.0
    best_price_likelihood: float           = 0.0
    mac_times_said:        int             = 0   # ocurrencias del tópico MAC
    price_times_said:      int             = 0
    mac_warn:              bool            = True  # True si likelihood < umbral
    price_warn:            bool            = True
    # Proporción de atributos esperados del MAC que fueron mencionados (0–1).
    mac_completeness_score: Optional[float] = None

    # Tipo de respuesta que dio el cliente a la Pregunta de Activación.
    # "EXPLICITA" → SÍ claro (ej. "Sí, perfecto", "Sí, de acuerdo")
    # "AMBIGUA"   → aceptación débil (ej. "Claro", "Ok", "Muy bien")
    # "AUSENTE"   → sin respuesta detectada
    mac_client_response: str = "AUSENTE"  # ver MacClientResponse

    # Resultado final del flujo MAC (MAC → respuesta → [MAC REFUERZO] → resultado).
    # "VENTA_CONFIRMADA" → MAC con SÍ explícito
    # "VENTA_REFUERZO"   → ambiguo + MAC REFUERZO con confirmación
    # "NO_CONFIRMADA"    → no se obtuvo confirmación clara
    # "NO_MAC"           → tópico MAC no detectado
    mac_flow_outcome: str = "NO_MAC"  # ver MacFlowOutcome

    # True si el agente debió aplicar MAC REFUERZO (respuesta ambigua al MAC).
    mac_r_triggered: bool = False
    # True si el MAC REFUERZO obtuvo una respuesta de aceptación.
    mac_r_effective: bool = False

    # ── 6. MOMENTOS DE COMPLIANCE ────────────────────────────────────────────
    # — Detección binaria por tópico SUBTAG —
    saludo_detected:                bool = False
    perfilamiento_detected:         bool = False
    producto_detected:              bool = False
    conformidad_detected:           bool = False
    confirmacion_monitoreo_detected: bool = False
    tratamiento_datos_detected:     bool = False   # Aviso de Privacidad
    ley_retracto_detected:          bool = False
    mac_detected:                   bool = False   # alias de mac_times_said > 0
    mac_r_detected:                 bool = False   # MAC REFUERZO presente
    precio_detected:                bool = False
    confirmacion_datos_detected:    bool = False   # MVD: correo / datos del cliente
    conformidad_atencion_detected:  bool = False   # Línea de atención informada
    despedida_detected:             bool = False

    # Número de momentos de compliance obligatorio que están presentes.
    # Se cuenta sobre REQUIRED_COMPLIANCE_TOPICS (máximo = len(REQUIRED_COMPLIANCE_TOPICS)).
    compliance_moment_count: int           = 0
    # Fracción de momentos obligatorios cubiertos (0–1).
    # 1.0 = todos los tópicos de REQUIRED_COMPLIANCE_TOPICS detectados.
    compliance_completeness: float         = 0.0

    # — Calidad semántica de momentos clave (similitud vs plantilla, 0–1) —
    terms_coverage_score:   Optional[float] = None  # % cláusulas legales cubiertas
    tratamiento_quality:    Optional[float] = None  # similitud vs plantilla aviso
    mac_r_quality_score:    Optional[float] = None  # similitud vs plantilla refuerzo
    mvd_quality_score:      Optional[float] = None  # similitud vs plantilla MVD

    # Compatibilidad con modelo anterior (0/1 int flags)
    mvd_detected:      bool = False   # alias de confirmacion_datos_detected
    terms_detected:    bool = False   # alias de TERMINOS LEGALES
    igs_comp_detected: bool = False   # mención de proveedor (IGS Asistencia)

    # ── 7. ESTRUCTURA CONVERSACIONAL ─────────────────────────────────────────
    # Secuencia temporal de labels detectados en la llamada.
    topic_order_observed: Optional[list[str]] = None
    # Kendall-tau normalizado vs CANONICAL_SCRIPT_ORDER (0–1; 1 = orden perfecto).
    topic_order_score:    Optional[float]      = None
    # Fracción de la llamada transcurrida hasta el primer MAC (0–1).
    time_to_first_mac_pct: Optional[float]     = None
    # Cobertura de tópicos: fracción de tópicos esperados que fueron detectados.
    script_completeness:  Optional[float]      = None  # (0–1)
    opening_detected:     Optional[bool]       = None
    closing_detected:     Optional[bool]       = None

    # ── 8. PUNTAJES POR DIMENSIÓN (0–10) ────────────────────────────────────
    # D1 — Pregunta de Activación y confirmación de venta (peso 40 %)
    #      Combina: mac_completeness_score, mac_flow_outcome, mac_r_effective,
    #               best_mac_likelihood, best_price_likelihood.
    score_d1_mac_venta:  Optional[float] = None

    # D2 — Completitud de momentos legales y regulatorios (peso 30 %)
    #      Combina: compliance_completeness, calidad semántica de cada momento.
    score_d2_compliance: Optional[float] = None

    # D3 — Adherencia al guión y estructura (peso 15 %)
    #      Combina: script_completeness, topic_order_score.
    score_d3_script:     Optional[float] = None

    # D4 — Calidad comunicativa del agente (peso 10 %)
    #      Combina: agent_participation, agent_wpm, turn_count, client_question_count.
    score_d4_engagement: Optional[float] = None

    # D5 — Calidad técnica del audio / transcripción (peso 5 %)
    #      Combina: confidence_score, silence_ratio.
    score_d5_audio:      Optional[float] = None

    # ── 9. SCORE FINAL Y TRAZABILIDAD ────────────────────────────────────────
    score:         float = 0.0
    score_version: str   = SCORE_VERSION

    # ── MÉTODOS ──────────────────────────────────────────────────────────────

    def compute_final_score(self) -> float:
        """
        Calcula el score final como suma ponderada de las cinco dimensiones.
        Solo opera sobre dimensiones ya calculadas (no None).
        Normaliza por el peso acumulado de las dimensiones disponibles para
        no penalizar dimensiones que aún no fueron calculadas.
        Retorna el score (0–10) y lo asigna a self.score.
        """
        total_weight  = 0.0
        weighted_sum  = 0.0

        dim_map = {
            "d1_mac_venta":  self.score_d1_mac_venta,
            "d2_compliance": self.score_d2_compliance,
            "d3_script":     self.score_d3_script,
            "d4_engagement": self.score_d4_engagement,
            "d5_audio":      self.score_d5_audio,
        }

        for dim_key, dim_score in dim_map.items():
            if dim_score is not None:
                w = DIMENSION_WEIGHTS[dim_key]
                weighted_sum += dim_score * w
                total_weight  += w

        if total_weight == 0.0:
            self.score = 0.0
        else:
            raw = weighted_sum / total_weight
            self.score = round(max(0.0, min(10.0, raw)), 4)

        return self.score

    def compute_compliance_completeness(self) -> float:
        """
        Calcula compliance_moment_count y compliance_completeness a partir
        de los flags individuales de detección.
        Retorna la fracción (0–1) y actualiza self.
        """
        flag_map: dict[str, bool] = {
            "TERMINOS LEGALES":       self.terms_detected,
            "TRATAMIENTO DATOS":      self.tratamiento_datos_detected,
            "LEY RETRACTO":           self.ley_retracto_detected,
            "CONFIRMACION MONITOREO": self.confirmacion_monitoreo_detected,
            "PRECIO":                 self.precio_detected,
            "CONFIRMACION DATOS":     self.confirmacion_datos_detected,
        }
        count = sum(1 for v in flag_map.values() if v)
        self.compliance_moment_count = count
        self.compliance_completeness = round(count / len(REQUIRED_COMPLIANCE_TOPICS), 4)
        return self.compliance_completeness

    def dimensions_ready(self) -> list[str]:
        """Retorna los nombres de las dimensiones ya calculadas (valor no None)."""
        return [
            key for key in DIMENSION_WEIGHTS
            if getattr(self, f"score_{key}") is not None
        ]

    def to_dict(self) -> dict:
        """Serializa el registro a dict plano (compatible con pd.DataFrame)."""
        return asdict(self)

    def to_series(self):
        """Serializa el registro a pd.Series."""
        import pandas as pd
        return pd.Series(self.to_dict())

    @classmethod
    def from_series(cls, row) -> "CallRecord":
        """
        Construye un CallRecord desde una fila del DataFrame legacy
        (MAT_CALLS_THIS_CAMPAIGN / MAT_VOLUMES).

        Mapea nombres de columnas del modelo anterior a los campos nuevos.
        Campos nuevos que no existan en la fila quedan en su valor por defecto.
        """
        import math
        known = {f.name for f in fields(cls)}

        # Columnas del modelo v2 → nombres v3
        LEGACY_MAP: dict[str, str] = {
            # compliance flags v2 → v3
            "mvd":                 "mvd_detected",
            "terms":               "terms_detected",
            "mac_r":               "mac_r_detected",
            "igs_comp":            "igs_comp_detected",
            # score dimensiones v2 → v3 (renombradas)
            "score_d1_compliance": "score_d2_compliance",
            "score_d2_script":     "score_d3_script",
            "score_d3_mac_quality": "score_d1_mac_venta",
            "score_d5_efficiency": "score_d5_audio",
        }

        kwargs: dict = {}
        for col, val in row.items():
            target = LEGACY_MAP.get(col, col)
            if target in known:
                kwargs[target] = None if (isinstance(val, float) and math.isnan(val)) else val

        if "noauditable" not in kwargs:
            conf = float(kwargs.get("confidence_score", 0.0) or 0.0)
            kwargs["noauditable"] = conf < 0.9

        kwargs.setdefault("score_version", SCORE_VERSION)
        return cls(**kwargs)

    @classmethod
    def from_mat_dataframe(cls, df) -> list["CallRecord"]:
        """Construye una lista de CallRecord desde el DataFrame MAT completo."""
        return [cls.from_series(row) for _, row in df.iterrows()]


# ---------------------------------------------------------------------------
# TopicRecord
# ---------------------------------------------------------------------------

@dataclass
class TopicRecord:
    """
    Registro de un tópico dentro de una llamada.
    Una llamada tiene N TopicRecords (uno por label detectado en tiempo).
    """

    # Identidad
    file_name:   str = ""
    final_label: str = ""   # uno de SUBTAGS

    # Timing
    topic_start:       float = 0.0
    topic_end:         float = 0.0
    time_centroid:     float = 0.0
    time_centroid_pct: float = 0.0   # centroide / duración total (0–1)

    # Contenido
    text_string:             str   = ""
    topic_num_words:         int   = 0
    topic_words_p_m:         float = 0.0
    topic_occurrence:        int   = 0
    velocity_classification: str   = ""   # "LENTO" | "NORMAL" | "RAPIDO"

    # Calidad de transcripción ASR
    topic_mean_conf:        float = 0.0
    topic_max_conf:         float = 0.0
    noauditable_transcript: bool  = False

    # Flags de compliance heredados del modelo anterior (0/1, compatibilidad)
    mvd:       int = 0   # CONFIRMACION DATOS
    terms:     int = 0   # TERMINOS LEGALES
    mac_r:     int = 0   # MAC REFUERZO
    igs_comp:  int = 0   # mención proveedor

    # Calidad semántica del fragmento (0–1)
    topic_semantic_score:   Optional[float] = None  # similitud vs plantilla del tópico
    topic_keyword_coverage: Optional[float] = None  # % keywords esperadas presentes

    # Respuesta del cliente detectada en este fragmento (solo para MAC / MAC REFUERZO)
    # Permite reconstruir el mac_flow_outcome a nivel de llamada.
    client_response_in_fragment: Optional[str] = None  # MacClientResponse o None

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
    Una campaña tiene N CampaignStats (uno por label de tópico distinto).
    """

    final_label: str = ""   # uno de SUBTAGS

    # Posición temporal normalizada en la llamada (centroide / duración)
    mean_centroid: float = 0.0
    std_centroid:  float = 0.0
    min_centroid:  float = 0.0
    max_centroid:  float = 0.0

    # Calidad de transcripción ASR
    mean_conf:     float = 0.0
    mean_max_conf: float = 0.0

    # Velocidad y extensión
    mean_words_p_m: float = 0.0
    mean_num_words: float = 0.0
    topic_frequency: int  = 0   # cuántas veces apareció en toda la campaña

    # Posición esperada según CANONICAL_SCRIPT_ORDER (0–1)
    expected_position_pct: Optional[float] = None
    # Desviación media entre posición observada y esperada (en fracción de llamada)
    order_deviation_mean:  Optional[float] = None

    # Tasa de presencia en llamadas de la campaña (topic_frequency / n_calls)
    presence_rate: Optional[float] = None

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


def callrecords_from_dataframe(df) -> list[CallRecord]:
    return CallRecord.from_mat_dataframe(df)


def topicrecords_from_dataframe(df) -> list[TopicRecord]:
    return [TopicRecord.from_series(row) for _, row in df.iterrows()]


def campaignstats_from_dataframe(df) -> list[CampaignStats]:
    return CampaignStats.from_statistics_dataframe(df)
