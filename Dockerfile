# =============================================================================
# Stage 1: builder
# Install dependencies with build tools, then discard them from the final image.
# We use python:3.11-slim (glibc) not alpine — fastembed/onnxruntime ships
# pre-built glibc wheels only; alpine (musl) requires a multi-hour source build.
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# =============================================================================
# Stage 2: runtime
# Lean image — no compiler, no root user, just the app and its packages.
# =============================================================================
FROM python:3.11-slim AS runtime

# Non-root user for least-privilege execution
RUN useradd --no-create-home --shell /bin/false appuser

WORKDIR /app

# Carry over installed packages from the builder stage
COPY --from=builder /install /usr/local

# Application code and pre-built corpus
# Layout must mirror the repo so DATA_PATH resolves correctly:
#   /app/api/search.py  →  /app/data/chunks.json
COPY api/search.py ./api/search.py
COPY data/chunks.json ./data/chunks.json
COPY models/ ./models/

USER appuser

EXPOSE 8000

# Model is pre-bundled — no download needed. Start-period can be shorter now.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "api.search:app", "--host", "0.0.0.0", "--port", "8000"]
