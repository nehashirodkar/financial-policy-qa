# Production container image — runnable on AWS ECS, Fargate, or any container host.
# For AWS Lambda deployment use deploy/Dockerfile instead.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (mmh3 and BM25 need build tools on slim image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Layer Python dependencies first for better cache hits
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# App code
COPY app/ ./app/
COPY bm25_params.json* ./

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
