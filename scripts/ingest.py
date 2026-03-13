"""
Run this script once before deploying to generate the embedded chunks.
Usage: python scripts/ingest.py
"""

from __future__ import annotations

import json
import os
import sys

import fitz
import numpy as np

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def extract_pages(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def make_chunks(text: str) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) > 30:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_corpus(pdf_path: str) -> list[dict]:
    pages = extract_pages(pdf_path)
    corpus = []
    for p in pages:
        for chunk in make_chunks(p["text"]):
            corpus.append({
                "text": chunk,
                "page": p["page"],
                "source": os.path.basename(pdf_path),
            })
    return corpus


def embed_corpus(corpus: list[dict]) -> list[dict]:
    from fastembed import TextEmbedding

    print(f"Loading model: {MODEL_NAME}")
    model = TextEmbedding(MODEL_NAME)

    texts = [item["text"] for item in corpus]
    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = list(model.embed(texts))
    for i, item in enumerate(corpus):
        item["embedding"] = embeddings[i].tolist()

    return corpus


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    out_path = os.path.join(data_dir, "chunks.json")

    pdf_files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        print(f"Error: no PDF files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s):")
    for p in pdf_files:
        print(f"  {os.path.basename(p)}")

    corpus: list[dict] = []
    for pdf_path in pdf_files:
        chunks = build_corpus(pdf_path)
        print(f"  {os.path.basename(pdf_path)}: {len(chunks)} chunks")
        corpus.extend(chunks)

    print(f"Total chunks: {len(corpus)}")

    corpus = embed_corpus(corpus)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    print(f"Saved {len(corpus)} embedded chunks to {out_path}")


if __name__ == "__main__":
    main()
