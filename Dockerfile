FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-cpu.txt ./
RUN pip install --no-cache-dir -r requirements-cpu.txt

COPY review_service.py pipeline.py inference_backend.py state_store.py weights_manifest.json ./

RUN useradd -m -u 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data

USER appuser

VOLUME ["/data"]

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INFERENCE_MODE=http \
    REVIEW_DATA_DIR=/data \
    ALLOW_NON_LOOPBACK=1 \
    MAX_PDF_PAGES=3500 \
    LARGE_PDF_POLICY=reject \
    JOB_WORKERS=2

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/ml/health', timeout=3).read()"]

CMD ["python", "review_service.py", "serve", "--host", "0.0.0.0", "--port", "8765", "--data-dir", "/data"]
