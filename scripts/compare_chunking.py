"""
Chunking strategy comparison benchmark.

Measures how the three chunking strategies (fixed-size, hierarchical,
sentence-window) affect:
  - chunk count
  - average chunk length
  - end-to-end RAG answer quality (answer_relevancy + faithfulness via Claude judge)

Usage:
    python scripts/compare_chunking.py --n 5
    python scripts/compare_chunking.py --docs-dir ./data/documents --n 10
"""

import argparse
import json
import logging
import statistics
from pathlib import Path

import boto3

from app.config import config
from app.retrieval.chunking import CHUNKING_STRATEGIES, chunk_document

logging.basicConfig(level="WARNING")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial policy analyst. Answer using ONLY the
provided context. Cite document IDs. If the answer cannot be determined, say so."""

JUDGE_PROMPT = """Score this answer on two criteria.

Question: {question}
Ground Truth: {ground_truth}
Answer: {answer}
Context: {context}

Return ONLY a JSON object:
{{
  "answer_relevancy": <float 0.0-1.0>,
  "faithfulness": <float 0.0-1.0>,
  "context_recall": <float 0.0-1.0>
}}"""


def _bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=config.AWS_REGION,
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    )


def _call_claude(prompt: str, system: str, max_tokens: int = 512) -> str:
    client = _bedrock_client()
    response = client.invoke_model(
        modelId=config.BEDROCK_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["content"][0]["text"]


def _load_corpus(docs_dir: str) -> list[dict]:
    """Load all .txt docs from a directory as raw text."""
    out = []
    for path in Path(docs_dir).glob("*.txt"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            out.append({"id": path.stem, "text": f.read()})
    return out


def _top_k_chunks(query: str, chunks: list[dict], k: int = 5) -> list[dict]:
    """Dumb keyword-overlap retrieval so we can isolate the chunking effect
    without the vector DB confounding the comparison."""
    q_tokens = set(query.lower().split())
    scored = []
    for c in chunks:
        c_tokens = set(c["text"].lower().split())
        overlap = len(q_tokens & c_tokens)
        scored.append((overlap, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


def _answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant context."
    context = "\n\n".join(f"[{c['metadata'].get('chunk_index','?')}] {c['text']}" for c in chunks)
    return _call_claude(f"Context:\n{context}\n\nQuestion: {query}", SYSTEM_PROMPT)


def _judge(question: str, ground_truth: str, answer: str, context: str) -> dict:
    try:
        raw = _call_claude(
            JUDGE_PROMPT.format(question=question, ground_truth=ground_truth,
                                answer=answer, context=context[:2000]),
            system="You are an evaluation judge. Return only valid JSON.",
            max_tokens=128,
        )
        start = raw.find("{"); end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception as e:
        logger.warning("Judge failed: %s", e)
        return {"answer_relevancy": 0.5, "faithfulness": 0.5, "context_recall": 0.5}


def benchmark_strategy(strategy: str, corpus: list[dict], questions: list[dict]) -> dict:
    # 1. Chunk the corpus
    all_chunks = []
    for doc in corpus:
        all_chunks.extend(chunk_document(doc["text"], doc_type="policy",
                                         metadata={"source": doc["id"]},
                                         strategy=strategy))

    chunk_lengths = [len(c["text"]) for c in all_chunks]

    # 2. For each question, retrieve top-k chunks via keyword overlap, answer, judge
    ar_scores, faith_scores, recall_scores = [], [], []
    for i, q in enumerate(questions, 1):
        print(f"  [{strategy}] Q {i}/{len(questions)}", end="\r", flush=True)
        retrieved = _top_k_chunks(q["question"], all_chunks, k=5)
        context = "\n\n".join(c["text"] for c in retrieved)
        try:
            answer = _answer(q["question"], retrieved)
            scores = _judge(q["question"], q["ground_truth"], answer, context)
            ar_scores.append(scores.get("answer_relevancy", 0.5))
            faith_scores.append(scores.get("faithfulness", 0.5))
            recall_scores.append(scores.get("context_recall", 0.5))
        except Exception as e:
            logger.warning("Q %d failed (%s): %s", i, strategy, e)
    print()

    return {
        "strategy": strategy,
        "num_chunks": len(all_chunks),
        "avg_chunk_len": round(statistics.mean(chunk_lengths), 1) if chunk_lengths else 0,
        "answer_relevancy": round(statistics.mean(ar_scores), 3) if ar_scores else 0,
        "faithfulness":     round(statistics.mean(faith_scores), 3) if faith_scores else 0,
        "context_recall":   round(statistics.mean(recall_scores), 3) if recall_scores else 0,
    }


def main(docs_dir: str, golden_set: str, n: int):
    corpus = _load_corpus(docs_dir)
    if not corpus:
        raise SystemExit(f"No .txt documents found in {docs_dir}")

    with open(golden_set, encoding="latin-1") as f:
        questions = json.load(f)[:n]

    print(f"\nBenchmarking {len(CHUNKING_STRATEGIES)} chunking strategies "
          f"on {len(questions)} questions, corpus of {len(corpus)} docs ...\n")

    results = [benchmark_strategy(s, corpus, questions) for s in CHUNKING_STRATEGIES]

    # ── Print table ────────────────────────────────────────────────────────────
    col = 18
    print("\n" + "=" * 96)
    print(f"{'Strategy':<{col}} {'#Chunks':>10} {'AvgLen':>8} "
          f"{'Relevancy':>12} {'Faithfulness':>14} {'ContextRecall':>14}")
    print("-" * 96)
    for r in results:
        print(f"{r['strategy']:<{col}} {r['num_chunks']:>10} {r['avg_chunk_len']:>8.1f} "
              f"{r['answer_relevancy']:>12.3f} {r['faithfulness']:>14.3f} {r['context_recall']:>14.3f}")
    print("=" * 96)

    # ── Winner ─────────────────────────────────────────────────────────────────
    best = max(results, key=lambda r: r["context_recall"])
    baseline = next(r for r in results if r["strategy"] == "fixed-size")
    if baseline["context_recall"] > 0:
        delta = (best["context_recall"] - baseline["context_recall"]) / baseline["context_recall"] * 100
        print(f"\nBest strategy: {best['strategy']} "
              f"(context_recall +{delta:.1f}% vs fixed-size baseline)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir",   default="./data/documents")
    parser.add_argument("--golden-set", default="./data/golden_set.json")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    main(args.docs_dir, args.golden_set, args.n)
