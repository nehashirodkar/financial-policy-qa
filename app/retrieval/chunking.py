"""
Document-type-aware chunking strategy.

Different financial document types benefit from different chunk sizes:
  - Regulatory text  → smaller, precise chunks (256 tokens)
  - Policy documents → medium chunks (512 tokens)
  - Annual reports   → larger chunks (1024 tokens)
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import config

CHUNK_CONFIG = {
    "regulation": {"chunk_size": 256, "chunk_overlap": 32},
    "policy":     {"chunk_size": 512, "chunk_overlap": 64},
    "report":     {"chunk_size": 1024, "chunk_overlap": 128},
    "default":    {"chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP},
}


def get_splitter(doc_type: str = "default") -> RecursiveCharacterTextSplitter:
    cfg = CHUNK_CONFIG.get(doc_type, CHUNK_CONFIG["default"])
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_document(text: str, doc_type: str = "default", metadata: dict | None = None) -> list[dict]:
    """
    Split a document into chunks and attach metadata.

    Returns a list of dicts with keys: text, metadata.
    """
    splitter = get_splitter(doc_type)
    chunks = splitter.split_text(text)
    meta = metadata or {}
    meta["doc_type"] = doc_type
    return [{"text": chunk, "metadata": {**meta, "chunk_index": i}} for i, chunk in enumerate(chunks)]
