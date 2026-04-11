"""
Before/after comparison: dense-only vs hybrid (dense+sparse) retrieval.

Uses Claude as the judge to score answer relevancy (0-1), replicating
the RAGAS evaluation methodology without requiring a separate RAGAS LLM config.

Usage:
    python scripts/compare_retrieval.py --n 5
    python scripts/compare_retrieval.py --golden-set ./data/golden_set.json --n 200
"""

import argparse
import json
import logging
import statistics
from pathlib import Path

import boto3

from app.config import config
from app.retrieval.pinecone_client import HybridRetriever

logging.basicConfig(level="WARNING")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial policy analyst. Answer the user's question
using ONLY the provided context documents. Be precise and cite the document IDs.
If the answer cannot be determined from the context, say so explicitly.
Respond in plain English — no markdown."""

JUDGE_PROMPT = """You are an evaluation judge. Score the following answer on two criteria.

Question: {question}
Ground Truth: {ground_truth}
Answer: {answer}
Context used: {context}

Return ONLY a JSON object with two float scores between 0.0 and 1.0:
{{
  "answer_relevancy": <float>,
  "faithfulness": <float>
}}

answer_relevancy: how well the answer addresses the question (1.0 = perfect)
faithfulness: how well the answer is grounded in the context (1.0 = fully grounded)"""


# ── Dense-only retriever wrapper ───────────────────────────────────────────────
class DenseOnlyRetriever(HybridRetriever):
    """Forces alpha=1.0 (pure dense, no BM25 sparse)."""
    def retrieve(self, query, top_k=None, doc_type=None, **kwargs):
        return super().retrieve(
            query=query,
            top_k=top_k or config.TOP_K_FINAL,
            doc_type=doc_type,
            alpha=1.0,
        )


def _bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=config.AWS_REGION,
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    )


def _call_claude(prompt: str, system: str, max_tokens: int = 512) -> str:
    client = _bedrock_client()
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = client.invoke_model(
        modelId=config.BEDROCK_MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["content"][0]["text"]


def _summarize(query: str, docs: list[dict]) -> str:
    if not docs:
        return "No relevant documents found."
    context = "\n\n".join(
        f"[DOC {i+1} | id={d['doc_id']}]\n{d['text']}" for i, d in enumerate(docs)
    )
    return _call_claude(
        prompt=f"Context:\n{context}\n\nQuestion: {query}",
        system=SYSTEM_PROMPT,
    )


def _judge(question: str, ground_truth: str, answer: str, contexts: list[str]) -> dict:
    """Use Claude to score answer_relevancy and faithfulness."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        context=" | ".join(contexts[:2]),  # first 2 contexts to keep prompt short
    )
    try:
        raw = _call_claude(prompt=prompt, system="You are an evaluation judge. Return only valid JSON.", max_tokens=128)
        # extract JSON from response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception as e:
        logger.warning("Judge failed: %s", e)
        return {"answer_relevancy": 0.5, "faithfulness": 0.5}


def _run_pipeline(questions, ground_truths, retriever, label) -> dict:
    ar_scores, faith_scores = [], []
    for i, (q, gt) in enumerate(zip(questions, ground_truths), 1):
        print(f"  [{label}] {i}/{len(questions)}", end="\r", flush=True)
        try:
            docs = retriever.retrieve(query=q)
            docs_dicts = [d.model_dump() for d in docs]
            answer = _summarize(q, docs_dicts)
            contexts = [d["text"] for d in docs_dicts]
            scores = _judge(q, gt, answer, contexts)
            ar_scores.append(scores.get("answer_relevancy", 0.5))
            faith_scores.append(scores.get("faithfulness", 0.5))
        except Exception as exc:
            logger.warning("[%s] Q%d failed: %s", label, i, exc)
            ar_scores.append(0.0)
            faith_scores.append(0.0)
    print()
    return {
        "answer_relevancy": statistics.mean(ar_scores),
        "faithfulness": statistics.mean(faith_scores),
    }


def main(golden_set_path: str, n: int):
    path = Path(golden_set_path)
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found: {path}")

    with open(path, encoding="latin-1") as f:
        golden = json.load(f)[:n]

    questions     = [item["question"]     for item in golden]
    ground_truths = [item["ground_truth"] for item in golden]

    print(f"\nComparing retrieval strategies on {len(questions)} questions ...\n")

    print("Running dense-only (baseline) ...")
    dense_scores = _run_pipeline(questions, ground_truths, DenseOnlyRetriever(), "dense-only")

    print("Running hybrid (dense + sparse BM25) ...")
    hybrid_scores = _run_pipeline(questions, ground_truths, HybridRetriever(), "hybrid")

    # ── Print table ────────────────────────────────────────────────────────────
    col = 26
    print("\n" + "=" * 66)
    print(f"{'Metric':<{col}} {'Dense-only':>16} {'Hybrid':>16} {'Delta':>10}")
    print("-" * 66)
    for key in ("answer_relevancy", "faithfulness"):
        d, h = dense_scores[key], hybrid_scores[key]
        delta = h - d
        sign = "+" if delta >= 0 else ""
        print(f"{key:<{col}} {d:>16.3f} {h:>16.3f} {sign}{delta:.3f}".rjust(col + 44))
    print("=" * 66)

    ar_d = round(dense_scores["answer_relevancy"] * 100, 1)
    ar_h = round(hybrid_scores["answer_relevancy"] * 100, 1)
    print(f"\nAnswer relevancy: {ar_d}% (dense-only) → {ar_h}% (hybrid)")
    delta_pct = ar_h - ar_d
    sign = "+" if delta_pct >= 0 else ""
    print(f"Improvement     : {sign}{delta_pct:.1f} percentage points\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-set", default="./data/golden_set.json")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    main(args.golden_set, args.n)
