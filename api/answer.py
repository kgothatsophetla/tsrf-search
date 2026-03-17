"""
Context-aware answer endpoint.

Performs semantic search over the TSRF Q&A corpus and returns the
pre-written answer for the best-matching FAQ question.

No external API calls are made — all processing is fully local.
The embeddings model and corpus are loaded at module level (Lambda INIT
phase on Vercel) to avoid first-request latency.

Environment variables:
    CORS_ORIGINS   Comma-separated allowed origins. Default: "*"
    MIN_SCORE      Minimum cosine similarity threshold. Default: 0.55
"""

import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler
from typing import Optional, List

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("answer")

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

# ---------------------------------------------------------------------------
# Module-level initialisation
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
del _vecs, _norms

logger.info("Corpus loaded: %d items.", len(_corpus))

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """Return up to `top_k` results above MIN_SCORE for the given query."""
    query_emb = list(_model.embed([query]))[0].astype(np.float32)
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    scores = _corpus_matrix @ query_norm
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx, i in enumerate(top_indices):
        if float(scores[i]) < MIN_SCORE:
            continue
        entry = {
            "text": _corpus[i]["text"],
            "page": _corpus[i]["page"],
            "source": _corpus[i]["source"],
            "score": float(scores[i]),
            "index": idx,
        }
        # Include the matched question if this is a Q&A corpus entry
        if "question" in _corpus[i]:
            entry["question"] = _corpus[i]["question"]
        results.append(entry)

    return results


_FOLLOWUP_PREFIXES = (
    "what about", "and ", "also ", "tell me more",
    "more about", "how about", "what of",
    "can you explain", "elaborate on",
)


def enrich_query(query: str, history: list[dict]) -> str:
    """
    Detect follow-up queries and expand them with context from the most recent
    matched corpus question in history.

    A query is treated as a follow-up if it is very short (≤4 words) or starts
    with a conversational continuation phrase. In that case the most recent
    matched_question from history is appended so the embedding search has enough
    context to find the right FAQ entry.
    """
    if not history:
        return query

    q_lower = query.lower().strip()
    is_followup = len(q_lower.split()) <= 4 or any(
        q_lower.startswith(p) for p in _FOLLOWUP_PREFIXES
    )

    if not is_followup:
        return query

    for entry in reversed(history):
        context = entry.get("matched_question", "")
        if context:
            enriched = f"{query} {context}".strip()[:500]
            logger.info("Enriched query %r -> %r", query, enriched)
            return enriched

    return query


def generate_answer(query: str, top_k: int = 5, history: Optional[List[dict]] = None):
    """
    Generator yielding SSE-compatible dicts:
      {"type": "sources", "results": [...]}   — matched corpus entries
      {"type": "answer",  "text": "..."}      — answer text
      {"type": "done"}
    or:
      {"type": "no_knowledge"}
    """
    effective_query = enrich_query(query, history or [])
    results = semantic_search(effective_query, top_k)

    if not results:
        yield {"type": "no_knowledge"}
        return

    yield {"type": "sources", "results": results}

    # Stream the answer 3 characters at a time — natural typewriter feel
    text = results[0]["text"]
    accumulated = ""
    for i in range(0, len(text), 3):
        accumulated += text[i:i + 3]
        yield {"type": "answer", "text": accumulated}
        time.sleep(0.012)

    yield {"type": "done"}


# ---------------------------------------------------------------------------
# Vercel handler
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
            history_raw = body.get("history", [])
            history = history_raw[-3:] if isinstance(history_raw, list) else []
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
            for event in generate_answer(query, top_k, history):
                self._send_event(event)
        except Exception:
            logger.exception("Answer generation failed for query: %r", query)
            self._send_event({"type": "error", "message": "Answer generation failed"})


# ---------------------------------------------------------------------------
# FastAPI app — used by Docker / local dev (uvicorn)
# ---------------------------------------------------------------------------


def _make_fastapi_app():
    from typing import AsyncGenerator

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field, field_validator
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    RATE_LIMIT = os.environ.get("RATE_LIMIT", "30/minute")
    limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

    _app = FastAPI(title="TSRF Answer API", version="1.0.0")
    _app.state.limiter = limiter
    _app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS.split(","),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    class HistoryEntryModel(BaseModel):
        question: str = Field(default="", max_length=500)
        matched_question: str = Field(default="", max_length=500)

    class AnswerRequest(BaseModel):
        query: str = Field(..., min_length=1, max_length=500)
        top_k: int = Field(default=5, ge=1, le=10)
        history: list[HistoryEntryModel] = Field(default_factory=list)

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

    @_app.post("/answer")
    async def answer(body: AnswerRequest):
        history_dicts = [h.model_dump() for h in body.history[-3:]]

        async def event_stream() -> AsyncGenerator[str, None]:
            for event in generate_answer(body.query, body.top_k, history_dicts):
                yield f"data: {json.dumps(event)}\n\n"

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
