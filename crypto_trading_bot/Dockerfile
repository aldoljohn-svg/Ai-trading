# syntax=docker/dockerfile:1
# ===========================================================================
# Multi-stage build.  The runtime image runs as a non-root user and contains
# no build toolchain.
# ===========================================================================

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build dependencies for the scientific wheels that lack a manylinux build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt


# ---------------------------------------------------------------------------

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    TZ=UTC

# libgomp is needed at runtime by LightGBM; curl is used by HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin --uid 10001 bot

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=bot:bot app/ ./app/
COPY --chown=bot:bot scripts/ ./scripts/
COPY --chown=bot:bot config.yaml ./
COPY --chown=bot:bot requirements.txt ./

# Writable state.  These are mounted as volumes in docker-compose.
RUN mkdir -p /app/data /app/logs /app/models \
    && chown -R bot:bot /app

USER bot

EXPOSE 8000

# The dashboard's /health endpoint is the liveness signal.  It returns 503
# when a component is in ERROR, so an unhealthy bot is visible to the
# orchestrator rather than silently degraded.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# tini reaps zombies and forwards SIGTERM so shutdown stays graceful.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "app.main"]
