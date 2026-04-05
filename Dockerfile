FROM python:3.12-slim AS base

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir .[production]

# Copy application code
COPY speaker_verify/ speaker_verify/
COPY demo/ demo/
COPY configs/ configs/

# Non-root user
RUN useradd --create-home appuser
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/v1/health/live')"

EXPOSE 5000

ENV ORION_VOICE_HOST=0.0.0.0 \
    ORION_VOICE_PORT=5000 \
    ORION_VOICE_PRODUCTION=true \
    ORION_VOICE_LOG_FORMAT=json \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "demo/server.py"]
