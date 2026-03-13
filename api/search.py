"""
Python backend for semantic search.

Vercel deployment:
    Vercel's Python runtime only supports BaseHTTPRequestHandler — not ASGI/Mangum.
    The `handler` class below is what Vercel invokes for every request.
    Model + corpus are loaded at module level so they are ready before the first
    request (Lambda INIT phase), eliminating first-request latency.

Docker / local:
    Uses FastAPI + uvicorn (rate limiting, validation, etc.)
    python api/search.py          # runs uvicorn
    docker compose up

Environment variables:
    CORS_ORIGINS   Comma-separated allowed origins. Default: "*"
    MIN_SCORE      Minimum cosine similarity threshold. Default: 0.55
    RATE_LIMIT     SlowAPI limit string (Docker only). Default: "30/minute"
"""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler
from typing import AsyncGenerator

import numpy as np

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
_BUNDLED_MODELS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
)
MIN_SCORE: float = float(os.environ.get("MIN_SCORE", "0.55"))
CORS_ORIGINS: str = os.environ.get("CORS_ORIGINS", "*")
RATE_LIMIT: str = os.environ.get("RATE_LIMIT", "30/minute")

# ---------------------------------------------------------------------------
# Module-level initialisation
# On Vercel/Lambda, module init runs in the INIT phase — before the first
# request. Loading here means the first request pays no setup cost.
# On plain uvicorn the effect is the same: ready before serving traffic.
# ---------------------------------------------------------------------------

logger.info("Initialising model and corpus ...")

from fastembed import TextEmbedding  # noqa: E402

_model = TextEmbedding(
    "BAAI/bge-small-en-v1.5",
    cache_dir=os.environ.get("FASTEMBED_CACHE_PATH", _BUNDLED_MODELS),
)
logger.info("Model loaded.")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Corpus not found at {DATA_PATH}. Run: python scripts/ingest.py"
    )
with open(DATA_PATH, encoding="utf-8") as _f:
    _corpus: list[dict] = json.load(_f)

_vecs = np.array([item["embedding"] for item in _corpus], dtype=np.float32)
_norms = np.linalg.norm(_vecs, axis=1, keepdims=True)
_corpus_matrix: np.ndarray = _vecs / (_norms + 1e-10)
del _vecs, _norms  # free the unnormalised copy

logger.info("Corpus loaded: %d chunks.", len(_corpus))

# ---------------------------------------------------------------------------
# Core search logic (shared by Vercel handler and FastAPI app)
# ---------------------------------------------------------------------------


def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """Return up to `top_k` results above MIN_SCORE for the given query."""
    query_emb = list(_model.embed([query]))[0].astype(np.float32)
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    scores = _corpus_matrix @ query_norm

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "text": _corpus[i]["text"],
            "page": _corpus[i]["page"],
            "source": _corpus[i]["source"],
            "score": float(scores[i]),
        }
        for i in top_indices
        if float(scores[i]) >= MIN_SCORE
    ]


# ---------------------------------------------------------------------------
# Vercel handler — BaseHTTPRequestHandler (the only format Vercel supports)
# ---------------------------------------------------------------------------


class handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info(fmt, *args)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", CORS_ORIGINS)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_error(self, status: int, message: str):
        body = json.dumps({"error": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_event(self, data: dict):
        msg = f"data: {json.dumps(data)}\n\n".encode()
        self.wfile.write(msg)
        self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        body = json.dumps({"status": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            query = str(body.get("query", "")).strip()
            top_k = min(max(int(body.get("top_k", 5)), 1), 10)
        except (json.JSONDecodeError, ValueError, TypeError):
            self._json_error(400, "Invalid request body")
            return

        if not query:
            self._json_error(400, "Query is required")
            return

        if len(query) > 500:
            self._json_error(400, "Query too long (max 500 characters)")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()

        try:
            results = semantic_search(query, top_k)

            if not results:
                self._send_event({"type": "no_knowledge"})
                return

            self._send_event({"type": "start", "total": len(results)})
            for i, result in enumerate(results):
                self._send_event({"type": "result", "index": i, **result})
            self._send_event({"type": "done"})

        except Exception:
            logger.exception("Search failed for query: %r", query)
            self._send_event({"type": "error", "message": "Search failed"})


# ---------------------------------------------------------------------------
# FastAPI app — used by Docker / uvicorn only
# ---------------------------------------------------------------------------


def _make_fastapi_app():
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field, field_validator
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

    _app = FastAPI(title="TSRF Semantic Search", version="1.0.0")
    _app.state.limiter = limiter
    _app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS.split(","),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    class SearchRequest(BaseModel):
        query: str = Field(..., min_length=1, max_length=500)
        top_k: int = Field(default=5, ge=1, le=10)

        @field_validator("query")
        @classmethod
        def strip_and_validate(cls, v: str) -> str:
            v = v.strip()
            if not v:
                raise ValueError("query must not be blank")
            return v

    @_app.get("/health")
    async def health():
        return {"status": "ok"}

    @_app.post("/search")
    @limiter.limit(RATE_LIMIT)
    async def search(request: Request, body: SearchRequest):
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
            except Exception:
                logger.exception("Search failed for query: %r", body.query)
                yield f"data: {json.dumps({'type': 'error', 'message': 'Search failed'})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return _app


# ---------------------------------------------------------------------------
# Local development / Docker entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import uvicorn

    app = _make_fastapi_app()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    logger.info("Starting uvicorn on http://0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
