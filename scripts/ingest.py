"""
Document ingestion script.

Usage:
    python scripts/ingest.py --path ./data --doc-type policy

Steps:
  1. Load .txt / .md / .pdf files from a directory
  2. Chunk with document-type-aware splitter
  3. Embed via Bedrock Titan
  4. Fit BM25 on the corpus
  5. Upsert dense + sparse vectors into Pinecone
"""

import argparse
import json
import logging
import uuid
from pathlib import Path

import boto3
import pdfplumber
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from app.config import config
from app.retrieval.chunking import chunk_document

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _extract_pdf_text(fp: Path) -> str:
    """Extract plain text from a PDF, joining all pages."""
    pages = []
    with pdfplumber.open(fp) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def load_files(path: str) -> list[dict]:
    root = Path(path)
    files = []
    for fp in root.rglob("*"):
        if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            if fp.suffix.lower() == ".pdf":
                text = _extract_pdf_text(fp)
            else:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                files.append({"filename": fp.name, "text": text})
            else:
                logger.warning("Skipping empty file: %s", fp.name)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", fp.name, exc)
    logger.info("Loaded %d files from %s", len(files), path)
    return files


def embed(texts: list[str]) -> list[list[float]]:
    client = boto3.client(
        "bedrock-runtime",
        region_name=config.AWS_REGION,
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    )
    embeddings = []
    for text in texts:
        body = json.dumps({"inputText": text})
        resp = client.invoke_model(
            modelId=config.BEDROCK_EMBEDDING_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())
        embeddings.append(result["embedding"])
    return embeddings


def ensure_index(pc: Pinecone, dimension: int = 1536):
    existing = [idx.name for idx in pc.list_indexes()]
    if config.PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=dimension,
            metric="dotproduct",          # required for hybrid search
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info("Created Pinecone index '%s'", config.PINECONE_INDEX_NAME)


def main(path: str, doc_type: str, batch_size: int = 50):
    files = load_files(path)
    if not files:
        logger.error("No files found. Exiting.")
        return

    # 1. Chunk all documents
    all_chunks = []
    for f in files:
        chunks = chunk_document(f["text"], doc_type=doc_type, metadata={"source": f["filename"]})
        all_chunks.extend(chunks)
    logger.info("Total chunks: %d", len(all_chunks))

    texts = [c["text"] for c in all_chunks]

    # 2. Fit BM25 on the full corpus and save for later inference
    bm25 = BM25Encoder()
    bm25.fit(texts)
    bm25.dump("bm25_params.json")
    logger.info("BM25 fitted and saved to bm25_params.json")

    sparse_vecs = bm25.encode_documents(texts)

    # 3. Embed with Bedrock
    logger.info("Embedding %d chunks (this may take a while)...", len(texts))
    dense_vecs = embed(texts)

    # 4. Upsert into Pinecone
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    ensure_index(pc, dimension=len(dense_vecs[0]))
    index = pc.Index(config.PINECONE_INDEX_NAME)

    vectors = []
    for i, (chunk, dense, sparse) in enumerate(zip(all_chunks, dense_vecs, sparse_vecs)):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": dense,
            "sparse_values": sparse,
            "metadata": {**chunk["metadata"], "text": chunk["text"]},
        })

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        logger.info("Upserted batch %d/%d", i // batch_size + 1, -(-len(vectors) // batch_size))

    logger.info("Ingestion complete. %d vectors upserted.", len(vectors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Directory containing documents")
    parser.add_argument("--doc-type", default="policy", choices=["policy", "regulation", "report"])
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    main(args.path, args.doc_type, args.batch_size)
