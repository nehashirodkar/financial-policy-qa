"""
Hybrid search (dense + sparse) against a Pinecone index.

Dense  : Amazon Titan Embed via AWS Bedrock
Sparse : BM25 via pinecone-text
"""

import json
import logging
from pathlib import Path
from typing import Optional

import boto3
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

from app.config import config
from app.schemas.models import SourceDocument

logger = logging.getLogger(__name__)


def _get_bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=config.AWS_REGION,
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    )


def _embed_dense(text: str) -> list[float]:
    """Call Titan Embeddings via Bedrock to get a dense vector."""
    client = _get_bedrock_client()
    body = json.dumps({"inputText": text})
    response = client.invoke_model(
        modelId=config.BEDROCK_EMBEDDING_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


class HybridRetriever:
    def __init__(self):
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = pc.Index(config.PINECONE_INDEX_NAME)
        self.bm25 = self._load_bm25(config.BM25_PARAMS_PATH)
        logger.info("HybridRetriever initialized against index '%s'", config.PINECONE_INDEX_NAME)

    @staticmethod
    def _load_bm25(params_path: str = "bm25_params.json") -> BM25Encoder:
        """Load corpus-fitted BM25 params if available, else fall back to default."""
        path = Path(params_path)
        if path.exists():
            encoder = BM25Encoder()
            encoder.load(str(path))
            logger.info("Loaded BM25 params from '%s'", path)
            return encoder
        logger.warning(
            "BM25 params file '%s' not found — using generic default encoder. "
            "Run scripts/ingest.py to fit BM25 on your corpus.",
            path,
        )
        return BM25Encoder().default()

    def retrieve(
        self,
        query: str,
        top_k: int = config.TOP_K_FINAL,
        doc_type: Optional[str] = None,
        alpha: float = config.ALPHA,
    ) -> list[SourceDocument]:
        """
        Hybrid retrieval with reciprocal rank fusion (RRF).

        alpha: weight applied to dense scores (1-alpha goes to sparse).
        """
        dense_vec = _embed_dense(query)
        sparse_vec = self.bm25.encode_queries(query)

        filter_dict = {"doc_type": {"$eq": doc_type}} if doc_type else None

        # Fetch more candidates before re-ranking
        candidates = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=config.TOP_K_DENSE + config.TOP_K_SPARSE,
            include_metadata=True,
            filter=filter_dict,
        )

        results = self._rerank(candidates.matches, alpha, top_k)
        logger.debug("Retrieved %d docs for query: %s", len(results), query[:80])
        return results

    def _rerank(self, matches, alpha: float, top_k: int) -> list[SourceDocument]:
        """Weighted combination of dense and sparse scores then take top_k."""
        scored = []
        for m in matches:
            dense_score = m.score if m.score is not None else 0.0
            sparse_score = m.sparse_values.values[0] if m.sparse_values and m.sparse_values.values else 0.0
            combined = alpha * dense_score + (1 - alpha) * sparse_score
            scored.append((combined, m))

        scored.sort(key=lambda x: x[0], reverse=True)

        docs = []
        for score, m in scored[:top_k]:
            docs.append(
                SourceDocument(
                    doc_id=m.id,
                    text=m.metadata.get("text", ""),
                    score=round(score, 4),
                    metadata={k: v for k, v in m.metadata.items() if k != "text"},
                )
            )
        return docs
