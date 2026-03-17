"""
Run this script once before deploying to generate the embedded chunks.
Usage: python scripts/ingest.py

For FAQ-structured PDFs the script detects question-answer boundaries and
creates one entry per Q&A pair, embedding the *question* so that semantic
search matches user queries against FAQ questions rather than answer text.

For non-FAQ documents it falls back to regular word-based chunking.
"""

from __future__ import annotations

import json
import os
import re
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


def extract_qa_pairs(pages: list[dict], source: str) -> list[dict]:
    """
    Detect Q&A structure in a FAQ document.

    Looks for lines whose entire content is a question (ends with '?').
    Returns a list of {question, text, page, source} dicts, where:
      - question  = the FAQ question (used for embedding / matching)
      - text      = the answer (displayed to the user)

    Returns an empty list if no clear Q&A structure is found, so the
    caller can fall back to regular word-based chunking.
    """
    # Build a single text blob while tracking where each page starts
    full_text = ""
    page_positions: list[tuple[int, int]] = []  # (char_start, page_num)
    for p in pages:
        page_positions.append((len(full_text), p["page"]))
        full_text += p["text"] + "\n"

    # A question is a line whose trimmed content ends with '?' and starts
    # with an uppercase letter (avoids mid-sentence '?' matches).
    question_pattern = re.compile(r"^([A-Z][^\n]*\?)\s*$", re.MULTILINE)
    matches = list(question_pattern.finditer(full_text))

    if len(matches) < 3:
        return []  # Not a FAQ — caller should use regular chunking

    pairs: list[dict] = []
    for i, match in enumerate(matches):
        question = match.group(1).strip()

        # Answer = text between this question and the next one
        ans_start = match.end()
        ans_end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        answer = full_text[ans_start:ans_end]

        # Normalise whitespace while preserving bullet structure
        answer = re.sub(r" {2,}", " ", answer)        # collapse extra spaces
        answer = re.sub(r"\n{3,}", "\n\n", answer)    # cap consecutive blank lines
        answer = answer.strip()

        if len(answer) < 10:
            continue

        # Map question position back to a page number
        page_num = 1
        for pos, pnum in page_positions:
            if pos <= match.start():
                page_num = pnum

        pairs.append(
            {
                "question": question,
                "text": answer,
                "page": page_num,
                "source": source,
            }
        )

    return pairs


def strip_repeated_lines(pages: list[dict]) -> list[dict]:
    """
    Remove lines that appear on 3 or more pages — these are almost always
    page headers or footers that pollute chunk embeddings.
    """
    if len(pages) < 3:
        return pages

    # Count how many pages each stripped line appears on
    from collections import Counter
    line_page_count: Counter = Counter()
    for p in pages:
        seen = set()
        for line in p["text"].splitlines():
            line = line.strip()
            if line and line not in seen:
                line_page_count[line] += 1
                seen.add(line)

    repeated = {line for line, count in line_page_count.items() if count >= 3}
    if not repeated:
        return pages

    cleaned = []
    for p in pages:
        lines = [l for l in p["text"].splitlines() if l.strip() not in repeated]
        cleaned.append({"page": p["page"], "text": "\n".join(lines).strip()})
    return cleaned


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
    source = os.path.basename(pdf_path)
    pages = extract_pages(pdf_path)

    # Try Q&A extraction first
    qa_pairs = extract_qa_pairs(pages, source)
    if qa_pairs:
        print(f"  {source}: {len(qa_pairs)} Q&A pairs extracted")
        return qa_pairs

    # Strip repeated headers/footers before chunking
    pages = strip_repeated_lines(pages)

    # Fall back to regular word-based chunking
    corpus = []
    for p in pages:
        for chunk in make_chunks(p["text"]):
            corpus.append({"text": chunk, "page": p["page"], "source": source})
    print(f"  {source}: {len(corpus)} text chunks (no Q&A structure detected)")
    return corpus


def embed_corpus(corpus: list[dict]) -> list[dict]:
    from fastembed import TextEmbedding

    print(f"Loading model: {MODEL_NAME}")
    model = TextEmbedding(MODEL_NAME)

    # For Q&A pairs embed the question so user queries match FAQ questions.
    # For regular chunks embed the text as before.
    texts_to_embed = [item.get("question", item["text"]) for item in corpus]
    print(f"Generating embeddings for {len(texts_to_embed)} items...")

    embeddings = list(model.embed(texts_to_embed))
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
        corpus.extend(chunks)

    print(f"Total items: {len(corpus)}")

    corpus = embed_corpus(corpus)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    print(f"Saved {len(corpus)} embedded items to {out_path}")


if __name__ == "__main__":
    main()
