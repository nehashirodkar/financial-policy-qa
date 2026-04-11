"""
Single-chain baseline (sequential, no LangGraph parallelism).

Implements the same retrieve → summarize → validate pipeline as the
multi-agent system but as a plain sequential function call — no parallel
fan-out, no graph overhead.  Used by benchmark.py to establish the
latency baseline that the multi-agent system is compared against.

Usage (standalone):
    python scripts/baseline.py --query "What is the maximum LTV ratio?"
"""

import argparse
import json
import logging
import time

import boto3

from app.config import config
from app.guardrails.validators import validate_and_clean
from app.retrieval.pinecone_client import HybridRetriever
from app.tracing.langsmith import compute_token_cost

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial policy analyst. Answer the user's question
using ONLY the provided context documents. Be precise and cite the document IDs.
If the answer cannot be determined from the context, say so explicitly.
Respond in plain English — no markdown."""


def _build_prompt(query: str, docs: list[dict]) -> str:
    context = "\n\n".join(
        f"[DOC {i+1} | id={d['doc_id']} | score={d['score']}]\n{d['text']}"
        for i, d in enumerate(docs)
    )
    return f"Context:\n{context}\n\nQuestion: {query}"


def run_baseline(
    query: str,
    top_k: int = 5,
    doc_type: str | None = None,
) -> dict:
    """
    Sequential single-chain baseline.

    Steps (all blocking, no parallelism):
      1. Retrieve
      2. Summarize
      3. Validate

    Returns the same state-dict shape as the multi-agent coordinator so
    benchmark.py can compare apples-to-apples.
    """
    timings: dict[str, float] = {}

    # ── 1. Retrieve ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    retriever = HybridRetriever()
    docs = retriever.retrieve(query=query, top_k=top_k, doc_type=doc_type)
    docs_dicts = [d.model_dump() for d in docs]
    timings["retrieval_agent"] = round((time.perf_counter() - t0) * 1000, 2)

    # ── 2. Summarize ─────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    if not docs_dicts:
        raw_answer = "No relevant documents found."
        token_cost: dict = {}
    else:
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name=config.AWS_REGION,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        )
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": _build_prompt(query, docs_dicts)}],
        }
        response = bedrock.invoke_model(
            modelId=config.BEDROCK_MODEL_ID,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        body = json.loads(response["body"].read())
        raw_answer = body["content"][0]["text"]
        usage = body.get("usage", {})
        token_cost = compute_token_cost(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            model=config.BEDROCK_MODEL_ID,
        )
    timings["summarization_agent"] = round((time.perf_counter() - t1) * 1000, 2)

    # ── 3. Validate ──────────────────────────────────────────────────────────
    t2 = time.perf_counter()
    source_ids = [d["doc_id"] for d in docs_dicts]
    validated = validate_and_clean(raw_answer, source_ids)
    timings["validation_agent"] = round((time.perf_counter() - t2) * 1000, 2)

    return {
        "query": query,
        "retrieved_docs": docs_dicts,
        "raw_answer": raw_answer,
        "validated_answer": validated["answer"],
        "pii_detected": validated["pii_detected"],
        "token_cost": token_cost,
        "node_latencies": timings,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-chain baseline query")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--doc-type", default=None)
    args = parser.parse_args()

    result = run_baseline(args.query, top_k=args.top_k, doc_type=args.doc_type)
    total_ms = sum(result["node_latencies"].values())
    print(f"\nAnswer: {result['validated_answer']}")
    print(f"\nNode latencies: {result['node_latencies']}")
    print(f"Total sequential latency: {total_ms:.1f} ms")
