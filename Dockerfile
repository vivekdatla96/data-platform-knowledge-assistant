# Data Platform Knowledge Assistant: offline demo image.
# The embedding model is downloaded at build time so the container starts without network access.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/opt/models \
    DPKA_CORPUS_DIR=/app/data/corpus \
    DPKA_INDEX_PATH=/data/index.sqlite

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[bedrock]" \
 && python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/models')"

COPY data ./data

RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app /data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

# Incremental ingest on every start (a no-op when nothing changed), then serve.
CMD ["sh", "-c", "dpka ingest && dpka serve --host 0.0.0.0 --port 8000"]
