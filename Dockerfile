FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY inference_worker/ inference_worker/

RUN pip install --no-cache-dir .

RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

EXPOSE 7861

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7861/api/status')"

CMD ["grid-inference-worker"]
