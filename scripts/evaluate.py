"""
RAGAS evaluation script.

Usage:
    python scripts/evaluate.py --golden-set ./data/golden_set.json

Expects golden_set.json to be a list of:
  { "question": "...", "ground_truth": "...", "contexts": ["...", ...] }
"""

import argparse
import json
import logging

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_recall, context_precision

from app.agents.coordinator import run_query

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def main(golden_set_path: str):
    with open(golden_set_path, encoding="utf-8") as f:
        golden = json.load(f)

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in golden:
        q = item["question"]
        result = run_query(query=q)

        questions.append(q)
        answers.append(result.get("validated_answer", ""))
        contexts.append([d["text"] for d in result.get("retrieved_docs", [])])
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    scores = evaluate(
        dataset,
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
    )

    logger.info("RAGAS scores:\n%s", scores)
    print(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-set", required=True, help="Path to golden_set.json")
    args = parser.parse_args()
    main(args.golden_set)
