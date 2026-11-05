# One image for the API, the dashboard and the demo replayer.
# The trained model is not baked in: mount it at /app/models (see docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    NIDS_HOME=/app \
    PATH="/app/.venv/bin:$PATH"

# libgomp: OpenMP runtime needed by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --extra dashboard --no-install-project
COPY nids ./nids
COPY config ./config
COPY data/sample ./data/sample
RUN uv sync --frozen --no-dev --extra dashboard

RUN useradd --uid 1000 --create-home nids && mkdir -p /app/models /app/reports /app/logs \
    && chown -R nids /app/models /app/reports /app/logs
USER nids

EXPOSE 8000 5001
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "nids.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
