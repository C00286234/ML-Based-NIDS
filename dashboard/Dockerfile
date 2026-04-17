# ==============================================================================
# NIDS Dashboard Docker Image
# ==============================================================================
# Multi-stage build for the Flask dashboard that proxies to the NIDS API.
#
# Build: docker build -t nids-dashboard -f dashboard/Dockerfile .
# Run:   docker run -p 5001:5001 -e NIDS_API_URL=http://host.docker.internal:8000 nids-dashboard
# ==============================================================================

FROM python:3.11-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY dashboard/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ==============================================================================
# Production stage
# ==============================================================================
FROM python:3.11-slim AS production

LABEL maintainer="FYP-NIDS-ML Project"
LABEL description="NIDS Dashboard (Flask frontend + SSE proxy)"
LABEL version="2.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

RUN groupadd --gid 1000 dash && \
    useradd --uid 1000 --gid dash --shell /bin/bash --create-home dash

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=dash:dash dashboard/ ./dashboard/

USER dash
WORKDIR /app/dashboard

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/health')" || exit 1

# gthread workers let a small number of threads handle long-lived SSE connections.
CMD ["gunicorn", "--bind", "0.0.0.0:5001", \
     "--worker-class", "gthread", \
     "--workers", "1", \
     "--threads", "8", \
     "--timeout", "0", \
     "app:app"]
