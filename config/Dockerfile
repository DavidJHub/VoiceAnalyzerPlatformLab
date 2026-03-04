# ---------- 1) Imagen base con Python 3.11 ----------
FROM python:3.11-slim AS base

# ---------- 2) Variables de directorio ----------
ENV APP_DIR=/app \
    LOG_DIR=/logs \
    PROC_DIR=/process

# ---------- 3) Dependencias de sistema ----------
# - ffmpeg: se instala desde apt, así evitas descargar el .zip
# - unzip, wget … por si más adelante los necesitas
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg unzip wget curl \
 && rm -rf /var/lib/apt/lists/*

# ---------- 4) Crear carpetas de trabajo ----------
RUN mkdir -p $APP_DIR $LOG_DIR $PROC_DIR
WORKDIR $APP_DIR

# ---------- 5) Instalar requirements ----------
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# ---------- 6) Descargar recursos NLTK y spaCy ----------
RUN python -m nltk.downloader stopwords \
 && python -m spacy download es_core_news_sm

# ---------- 7) Copiar código fuente ----------
COPY . .

# ---------- 8) Variables de entorno opcionales ----------


# ---------- 9) Puerto / comando de arranque ----------
EXPOSE 8000
CMD ["python", "main.py"]
