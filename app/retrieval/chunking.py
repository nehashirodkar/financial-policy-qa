"""
Document chunking with configurable strategies.

Supports three chunking strategies (each benchmarked via scripts/compare_chunking.py):
  - fixed-size       : uniform chunk_size with overlap (simple, fast, baseline)
  - hierarchical     : splits by structural separators first (headings → paragraphs → sentences)
  - sentence-window  : small chunks with adjacent-sentence context windows for precision

Also supports document-type-aware sizing:
  - regulation  → 256 tokens   (precise regulatory text)
  - policy      → 512 tokens   (medium-length policies)
  - report      → 1024 tokens  (long annual reports)
"""

import re
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from app.config import config

# ── Document-type-aware sizing ─────────────────────────────────────────────────
CHUNK_CONFIG = {
    "regulation": {"chunk_size": 256, "chunk_overlap": 32},
    "policy":     {"chunk_size": 512, "chunk_overlap": 64},
    "report":     {"chunk_size": 1024, "chunk_overlap": 128},
    "default":    {"chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP},
}

# Supported chunking strategies
CHUNKING_STRATEGIES = ("fixed-size", "hierarchical", "sentence-window")


# ── Strategy 1: Fixed-size ─────────────────────────────────────────────────────
def _fixed_size_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Uniform-size chunks — fastest, simplest baseline."""
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" ",
    )
    return splitter.split_text(text)


# ── Strategy 2: Hierarchical ───────────────────────────────────────────────────
def _hierarchical_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Splits by structural separators first (headings → paragraphs → sentences).
    Preserves document structure, better for regulatory/policy docs with sections."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# ── Strategy 3: Sentence-window ────────────────────────────────────────────────
def _sentence_window_chunks(text: str, window_size: int = 3) -> list[str]:
    """Each chunk is one sentence plus `window_size` neighboring sentences.
    High precision retrieval with local context."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []
    chunks = []
    for i in range(len(sentences)):
        start = max(0, i - window_size)
        end = min(len(sentences), i + window_size + 1)
        chunks.append(" ".join(sentences[start:end]))
    return chunks


# ── Unified API ────────────────────────────────────────────────────────────────
def get_splitter(doc_type: str = "default") -> RecursiveCharacterTextSplitter:
    """Back-compat: returns the default hierarchical splitter for a doc type."""
    cfg = CHUNK_CONFIG.get(doc_type, CHUNK_CONFIG["default"])
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_document(
    text: str,
    doc_type: str = "default",
    metadata: dict | None = None,
    strategy: str = "hierarchical",
) -> list[dict]:
    """
    Split a document using the requested strategy, then attach metadata.

    Args:
        text: raw document text.
        doc_type: regulation | policy | report | default (controls chunk_size).
        metadata: dict of extra fields (source, date, etc.) to stamp on every chunk.
        strategy: fixed-size | hierarchical | sentence-window.

    Returns:
        list of {"text": str, "metadata": dict} dicts.
    """
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Use one of {CHUNKING_STRATEGIES}.")

    cfg = CHUNK_CONFIG.get(doc_type, CHUNK_CONFIG["default"])
    chunk_size, chunk_overlap = cfg["chunk_size"], cfg["chunk_overlap"]

    if strategy == "fixed-size":
        raw_chunks = _fixed_size_chunks(text, chunk_size, chunk_overlap)
    elif strategy == "hierarchical":
        raw_chunks = _hierarchical_chunks(text, chunk_size, chunk_overlap)
    else:  # sentence-window
        raw_chunks = _sentence_window_chunks(text, window_size=3)

    meta = metadata.copy() if metadata else {}
    meta["doc_type"] = doc_type
    meta["chunking_strategy"] = strategy
    return [
        {"text": chunk, "metadata": {**meta, "chunk_index": i}}
        for i, chunk in enumerate(raw_chunks)
    ]
