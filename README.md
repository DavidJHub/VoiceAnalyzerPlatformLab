# VoiceAnalyzerPlatform

VoiceAnalyzerPlatform (VAP) es una plataforma de auditoría asistida para evaluar llamadas telefónicas. 

## Características principales

- **Preprocesamiento de audio**: limpieza, normalización y detección de actividad de voz para mejorar la calidad antes de transcribir.
- **Transcripción**: integración con Deepgram y un motor local (VapEngine) para obtener STT.
- **Calificación de llamadas**: modelos de clasificación determinan calidad, razones de rechazo y metadatos relevantes.
- **Gestión de campañas**: las métricas resultantes se almacenan en S3 y en MySQL para generar KPIs por sponsor y campaña.

## Estructura del repositorio

- `audio/` – utilidades de procesamiento de señales de audio.
- `auditableSelector/` – modelos y lógica de clasificación para determinar si una llamada es auditable y sus causas de rechazo.
- `config/` – configuración de logger, dependencias y scripts de despliegue.
- `database/` – capas de acceso a datos: conexión a MySQL y manejo de objetos en S3.
- `docs/` – documentación extendida del proyecto.
- `lambdas/` – funciones pensadas para ejecutarse en AWS Lambda.
- `segmentationModel/` – entrenamiento y realimentación del modelo de segmentación de texto.
- `setup/` – configuración de campañas, matrices de tópicos y manejo de memoria.
- `scripts/` – utilidades de soporte para renombrar o depurar audios.
- `training/` – scripts para descargar matrices y ajustar nuevos tópicos o modelos.
- `transcript/` – funciones de transcripción por lotes y configuración de motores.
- `utils/` – funciones auxiliares para métricas y gestión de campañas.
- `workflow/` – automatización de infraestructura (p. ej. apagado de instancias).

## Uso básico

```bash
python main.py <prefijo_campaña> <días_atrás> <modo>
```

El modo puede ser `write_auto`, `write_manual` o `dry_run`.
La ejecución procesa los audios de la campaña, genera transcripciones, calcula métricas y actualiza la base de datos.

## Documentación adicional

En la carpeta `docs/` se incluyen:

- `modules.md`: descripción detallada de cada módulo.
- `er_diagrams.md`: explicación de las principales tablas y relaciones de la base de datos.



docs/modules.md
New
+77
-0

# Documentación por módulo

Esta sección describe los módulos principales que componen VoiceAnalyzerPlatform.

## audio

Herramientas para el manejo y preprocesamiento de archivos de audio. Incluye:
- **AudioPreprocessing.py**: filtra ruido, detecta actividad de voz y normaliza señales.
- **ConcatCalls.py**: concatena segmentos de audio.
- **measureActivity.py**: mide niveles de actividad y volumen.

## auditableSelector

Contiene modelos que determinan si una llamada es auditable y las razones de rechazo.
- **main.py**: punto de entrada para la clasificación de calidad.
- **affectedClassifier.py**: genera motivos de rechazo combinando diversas métricas.
- **audioMetrics.py**, **confidenceSelection.py**, **speechSegmentation.py**: cálculo de características de voz, selección por confianza y segmentación del discurso.

## config

Archivos de configuración y utilidades de ejecución.
- **logger.py**: redirige la salida estándar a archivos de log.
- **Dockerfile** y **requirements.txt**: definen el entorno de ejecución.

## database

Capa de acceso a datos.
- **SQLDataManager.py**: consultas e inserciones en MySQL para KPIs, recomendaciones y gráficos.
- **SQLCampaignQuality.py**: descargas y combinaciones de tablas para obtener calidad de campaña.
- **S3Loader.py**: carga de archivos hacia/desde buckets de S3.
- **dbConfig.py**: parámetros de conexión a las bases de datos.

## lambdas

Scripts pensados para ejecutarse como funciones Lambda en AWS.
- **ETIQUETAS.py**: renombrado de archivos dentro de buckets S3 según reglas de etiquetado.

## segmentationModel

Entrenamiento y ajuste de modelos de segmentación textual.
- **feedbackSystem.py**: incorpora retroalimentación etiquetada para refinar el modelo.
- **fitting.py** e **hyperTunning.py**: optimizan el modelo de clasificación.

## setup

Inicialización de campañas y estructuras de datos.
- **CampaignSetup.py**: prepara rutas y parámetros de campaña.
- **MatrixSetup.py**: construye la matriz de tópicos y palabras permitidas/prohibidas.
- **vapOrchester.py** y **vapDealer.py**: rutinas para gestionar ejecuciones masivas.

## scripts

Utilidades aisladas para tareas puntuales (renombrado de archivos, reportes, etc.).

## training

Scripts de entrenamiento y actualización de modelos.
- **fitNewTopics.py** y **mapTopics.py**: permiten incorporar nuevos tópicos a las matrices existentes.

## transcript

Servicios de transcripción de audio.
- **VapTranscript.py**: integra Deepgram y Vosk para obtener texto y diarización.
- **batchTranscript.py**: procesa grandes volúmenes de audios en modo batch.

## utils

Funciones auxiliares comunes a varias partes del sistema.
- **VapFunctions.py**: manejo de directorios en S3 y medición de volumen.
- **VapUtils.py**: utilidades para extraer datos de nombres de archivos, métricas y JSON.
- **campaignMetrics.py**: cálculo de número de archivos y duraciones.

## workflow

Automatización de infraestructura.
- **stopInstances.py**: apaga instancias de procesamiento según reglas definidas.

