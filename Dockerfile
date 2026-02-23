# Confucius Code Agent (CCA) - Development & Runtime Container
# Multi-architecture (amd64, arm64) - targets DGX Spark1 (ARM64)
#
# Build:  docker compose -f cca-compose.yml build
# Run:    docker compose -f cca-compose.yml run --rm cca
# Shell:  docker compose -f cca-compose.yml run --rm cca bash
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Confucius Code Agent"
LABEL org.opencontainers.image.description="Meta's CCA agent framework for AI software engineering"
LABEL org.opencontainers.image.source="https://github.com/seli-equinix/cca-swebench"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies (infra expert needs SSH, Docker CLI, network tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    sshpass \
    openssh-client \
    docker.io \
    iputils-ping \
    dnsutils \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CCA package (editable for development)
COPY pyproject.toml setup.py ./
COPY confucius/ confucius/
COPY scripts/ scripts/
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 cca && \
    mkdir -p /workspace /home/cca/.confucius && \
    chown -R cca:cca /app /workspace /home/cca/.confucius

USER cca

# Default: interactive REPL
ENTRYPOINT ["confucius"]
CMD ["code", "--verbose"]
