"""
Python backend for semantic search — production-grade FastAPI implementation.

Local development:
    python api/search.py
    # or: uvicorn api.search:app --reload

Docker:
    docker compose up

Vercel deployment:
    Vercel picks up the `handler` name (Mangum wraps the ASGI app).
    The `lifespan="off"` is required — Vercel does not call ASGI lifespan events.

Environment variables:
    CORS_ORIGINS   Comma-separated allowed origins. Default: "*"
    MIN_SCORE      Minimum cosine similarity threshold. Default: 0.55
    RATE_LIMIT     SlowAPI limit string. Default: "30/minute"
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import AsyncGenerator

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mangum import Mangum
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("search")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "chunks.json",
)
MIN_SCORE: float = float(os.environ.get("MIN_SCORE", "0.55"))
RATE_LIMIT: str = os.environ.get("RATE_LIMIT", "30/minute")
CORS_ORIGINS: list[str] = os.environ.get("CORS_ORIGINS", "*").split(",")

# ---------------------------------------------------------------------------
# Thread-safe lazy-loaded globals
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_model = None
_corpus: list[dict] | None = None
_corpus_matrix: np.ndarray | None = None


def get_model():
    """Return the embedding model, loading it once on first call."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from fastembed import TextEmbedding

                logger.info("Loading embedding model BAAI/bge-small-en-v1.5 ...")
                _model = TextEmbedding("BAAI/bge-small-en-v1.5")
                logger.info("Embedding model loaded.")
    return _model


def get_corpus() -> tuple[list[dict], np.ndarray]:
    """Return (corpus, normalised_matrix), loading from disk once on first call."""
    global _corpus, _corpus_matrix
    if _corpus is None:
        with _lock:
            if _corpus is None:
                if not os.path.exists(DATA_PATH):
                    raise FileNotFoundError(
                        f"Corpus not found at {DATA_PATH}. "
                        "Run: python scripts/ingest.py"
                    )
                logger.info("Loading corpus from %s ...", DATA_PATH)
                with open(DATA_PATH, encoding="utf-8") as f:
                    data: list[dict] = json.load(f)

                vecs = np.array(
                    [item["embedding"] for item in data], dtype=np.float32
                )
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                # Assign matrix before corpus so the outer `if _corpus is None`
                # guard remains valid as the single sentinel.
                _corpus_matrix = vecs / (norms + 1e-10)
                _corpus = data
                logger.info("Corpus loaded: %d chunks.", len(_corpus))
    return _corpus, _corpus_matrix


# ---------------------------------------------------------------------------
# Core search logic
# ---------------------------------------------------------------------------


def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """Return up to `top_k` results above MIN_SCORE for the given query."""
    model = get_model()
    corpus, matrix = get_corpus()

    query_emb = list(model.embed([query]))[0].astype(np.float32)
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    scores = matrix @ query_norm

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "text": corpus[i]["text"],
            "page": corpus[i]["page"],
            "source": corpus[i]["source"],
            "score": float(scores[i]),
        }
        for i in top_indices
        if float(scores[i]) >= MIN_SCORE
    ]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

app = FastAPI(title="TSRF Semantic Search", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Startup preload (works on both uvicorn lifespan and Vercel cold-starts)
# Vercel does NOT invoke ASGI lifespan, so we trigger loading on the first
# real request via middleware instead of @app.on_event("startup").
# ---------------------------------------------------------------------------

_startup_done = False
_startup_lock = threading.Lock()


@app.middleware("http")
async def preload_on_first_request(request: Request, call_next):
    global _startup_done
    if not _startup_done:
        with _startup_lock:
            if not _startup_done:
                try:
                    get_model()
                    get_corpus()
                    _startup_done = True
                except Exception:
                    logger.exception("Startup preload failed — will retry on next request.")
    return await call_next(request)


# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank or whitespace-only")
        return v


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
@app.get("/api/health")
async def health():
    """Readiness probe — reports whether corpus and model are loaded."""
    return {
        "status": "ok",
        "corpus_loaded": _corpus is not None,
        "model_loaded": _model is not None,
    }


@app.post("/search")
@limiter.limit(RATE_LIMIT)
async def search(request: Request, body: SearchRequest):
    """Stream semantic search results as Server-Sent Events."""

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            results = semantic_search(body.query, body.top_k)

            if not results:
                yield f"data: {json.dumps({'type': 'no_knowledge'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'start', 'total': len(results)})}\n\n"

            for i, result in enumerate(results):
                yield f"data: {json.dumps({'type': 'result', 'index': i, **result})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except FileNotFoundError as e:
            logger.error("Corpus not found: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception:
            logger.exception("Search failed for query: %r", body.query)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Search failed'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Vercel serverless handler (Mangum wraps the ASGI app)
# ---------------------------------------------------------------------------

handler = Mangum(app, lifespan="off")

# ---------------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run("search:app", host="0.0.0.0", port=port, reload=False)
