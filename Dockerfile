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

# System dependencies — organized by role.
# Everything installed directly (no mounts) for portability.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # ── Base development ──
    git \
    curl \
    wget \
    build-essential \
    nodejs \
    npm \
    # ── Remote access (infra expert) ──
    sshpass \
    openssh-client \
    rsync \
    # ── Container management ──
    docker.io \
    # ── Process & system diagnostics ──
    procps \
    # ── Network diagnostics ──
    iproute2 \
    iputils-ping \
    dnsutils \
    net-tools \
    netcat-openbsd \
    traceroute \
    # ── Database CLIs ──
    redis-tools \
    # ── Data processing ──
    jq \
    bc \
    # ── Archive/compression ──
    zip \
    unzip \
    # ── Disk/storage inspection ──
    lsof \
    # ── File utilities ──
    tree \
    file \
    # ── Build tools & compilers ──
    cmake \
    # ── Bridge & firewall (infra) ──
    bridge-utils \
    # ── Hardware diagnostics (infra, may need privileged) ──
    dmidecode \
    lm-sensors \
    # ── Certificate management ──
    certbot \
    && rm -rf /var/lib/apt/lists/*

# Install tree-sitter CLI (needed for PowerShell grammar generation)
RUN npm install -g tree-sitter-cli@0.20.8

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Python development tools — agent can lint, test, format user code in /workspace
RUN pip install --no-cache-dir \
    pytest \
    mypy \
    ruff \
    black \
    flake8 \
    isort \
    coverage \
    uv

# PowerShell 7 — agent can execute .ps1 scripts in /workspace
# Multi-arch: detects ARM64 vs AMD64 at build time
ARG PS_VERSION=7.5.4
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ]; then PS_ARCH="linux-arm64"; \
    else PS_ARCH="linux-x64"; fi && \
    curl -fsSL "https://github.com/PowerShell/PowerShell/releases/download/v${PS_VERSION}/powershell-${PS_VERSION}-${PS_ARCH}.tar.gz" \
      -o /tmp/pwsh.tar.gz && \
    mkdir -p /opt/microsoft/powershell/7 && \
    tar xzf /tmp/pwsh.tar.gz -C /opt/microsoft/powershell/7 && \
    chmod +x /opt/microsoft/powershell/7/pwsh && \
    ln -s /opt/microsoft/powershell/7/pwsh /usr/local/bin/pwsh && \
    rm /tmp/pwsh.tar.gz && \
    pwsh -v

# Go programming language
ARG GO_VERSION=1.24.1
RUN ARCH=$(dpkg --print-architecture) && \
    curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-${ARCH}.tar.gz" \
      -o /tmp/go.tar.gz && \
    tar -C /usr/local -xzf /tmp/go.tar.gz && \
    ln -s /usr/local/go/bin/go /usr/local/bin/go && \
    rm /tmp/go.tar.gz && \
    go version

# Rust toolchain (cargo + rustc) — install to /opt so all users can access
ENV RUSTUP_HOME=/opt/rust/rustup \
    CARGO_HOME=/opt/rust/cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable --profile minimal && \
    chmod -R a+rX /opt/rust && \
    ln -s /opt/rust/cargo/bin/cargo /usr/local/bin/cargo && \
    ln -s /opt/rust/cargo/bin/rustc /usr/local/bin/rustc && \
    cargo --version && rustc --version

# Node.js ecosystem tools — yarn, pnpm, bun, TypeScript, Context Hub
RUN npm install -g yarn pnpm typescript @aisuite/chub && \
    yarn --version && pnpm --version && tsc --version && chub --cli-version

# Bun — fast JavaScript runtime (multi-arch)
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ]; then BUN_ARCH="aarch64"; \
    else BUN_ARCH="x64"; fi && \
    curl -fsSL "https://github.com/oven-sh/bun/releases/latest/download/bun-linux-${BUN_ARCH}.zip" \
      -o /tmp/bun.zip && \
    unzip -o /tmp/bun.zip -d /tmp/bun && \
    mv /tmp/bun/bun-linux-${BUN_ARCH}/bun /usr/local/bin/bun && \
    chmod +x /usr/local/bin/bun && \
    rm -rf /tmp/bun /tmp/bun.zip && \
    bun --version

# yq — YAML/XML/TOML processor
RUN pip install --no-cache-dir yq

# Build tree-sitter language grammars (Python, Bash, PowerShell, YAML, Markdown)
COPY confucius/server/code_intelligence/build_languages.py /tmp/build_languages.py
RUN python3 /tmp/build_languages.py && rm /tmp/build_languages.py

# Pre-build context-hub documentation registry (Nutanix SDK docs)
COPY docs/context-hub/content/ /app/context-hub/content/
RUN chub build /app/context-hub/content -o /app/context-hub/dist

# Install CCA package
COPY pyproject.toml ./
COPY confucius/ confucius/
COPY scripts/ scripts/
RUN pip install --no-cache-dir .

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 cca && \
    mkdir -p /workspace /home/cca/.confucius /home/cca/.chub && \
    printf 'sources:\n  - name: default\n    url: https://cdn.aichub.org/v1\n  - name: local\n    path: /app/context-hub/dist\n' \
      > /home/cca/.chub/config.yaml && \
    chown -R cca:cca /app /workspace /home/cca/.confucius /home/cca/.chub

USER cca

# Default: HTTP server
ENTRYPOINT ["confucius"]
CMD ["--port", "8500"]
