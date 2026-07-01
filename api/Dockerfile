# api/Dockerfile
# 3-stage build: py-builder → bin-fetcher → runtime
# Build context: project root (.) — needed so COPY eval/ works.
#
# TARGETARCH mapping (BuildKit auto-injects):
#   arm64 (Apple Silicon / Graviton) → aarch64-unknown-linux-gnu
#   amd64 (x86_64, Fly.io)           → x86_64-unknown-linux-gnu

# ────────────────────────────────────────────────────────────────────────────
# Stage 1: py-builder — install Python deps into a venv
# ────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS py-builder
WORKDIR /build

RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ────────────────────────────────────────────────────────────────────────────
# Stage 2: bin-fetcher — download arch-matched gfix + mergiraf binaries
# ────────────────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS bin-fetcher
ARG TARGETARCH

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Map Docker TARGETARCH → Rust target triple
# Used for both gfix (ameyypawar/gfix) and mergiraf (codeberg.org/mergiraf/mergiraf).
# mergiraf is GPL-3.0; it is invoked as a subprocess only — never linked — so
# the LGPL/GPL boundary is maintained. See NOTICE (6b) for the full attribution.
RUN set -e; \
    case "${TARGETARCH}" in \
        arm64) TRIPLE=aarch64-unknown-linux-gnu ;; \
        amd64) TRIPLE=x86_64-unknown-linux-gnu  ;; \
        *)     echo "unsupported TARGETARCH: ${TARGETARCH}" && exit 1 ;; \
    esac; \
    mkdir /out; \
    \
    # gfix 0.1.0-alpha.6
    curl -fsSL --retry 3 --retry-delay 2 \
        "https://github.com/ameyypawar/gfix/releases/download/v0.1.0-alpha.6/gfix-v0.1.0-alpha.6-${TRIPLE}.tar.gz" \
        | tar xz -C /out gfix; \
    \
    # mergiraf 0.17.0  (GPL-3.0 subprocess boundary — not linked)
    curl -fsSL --retry 3 --retry-delay 2 \
        "https://codeberg.org/mergiraf/mergiraf/releases/download/v0.17.0/mergiraf_${TRIPLE}.tar.gz" \
        | tar xz -C /out mergiraf; \
    \
    chmod +x /out/gfix /out/mergiraf

# ────────────────────────────────────────────────────────────────────────────
# Stage 3: runtime
# ────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps from venv
COPY --from=py-builder /venv /venv
ENV PATH=/venv/bin:$PATH

# Arch-matched binaries
COPY --from=bin-fetcher /out/gfix /out/mergiraf /usr/local/bin/

# Application code + eval corpus (seed.py reads /eval/golden_set.jsonl)
COPY api/ /app/
COPY eval/ /eval/

WORKDIR /app

# Pre-download the embedding model so the container runs offline and keyless.
# Model is cached to /root/.cache (committed as an image layer).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

ENV GITFIX_ALLOW_ANY_REPO=1
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
