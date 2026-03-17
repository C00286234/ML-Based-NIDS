# ==============================================================================
# NIDS API Docker Image
# ==============================================================================
# Multi-stage build for optimized production image
#
# Build: docker build -t nids-api .
# Run:   docker run -p 8000:8000 nids-api
# ==============================================================================

FROM python:3.11-slim AS builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ==============================================================================
# Production stage
# ==============================================================================
FROM python:3.11-slim AS production

# Labels
LABEL maintainer="ML-BASED-NIDS Project"
LABEL description="Network Intrusion Detection System API"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd --gid 1000 nids && \
    useradd --uid 1000 --gid nids --shell /bin/bash --create-home nids

# Create app directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application files
COPY --chown=nids:nids api/ ./api/
COPY --chown=nids:nids models/ ./models/

# Create logs directory
RUN mkdir -p /app/api/logs && chown -R nids:nids /app/api/logs

# Switch to non-root user
USER nids

# Set working directory to api/ so local imports work (auth.py, rate_limit.py)
WORKDIR /app/api

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application (now from /app/api directory)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
