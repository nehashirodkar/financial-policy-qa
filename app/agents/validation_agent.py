"""
Validation specialist agent.

Runs the raw answer through Guardrails AI to:
  - Validate structure
  - Redact PII
"""

import logging
from app.guardrails.validators import validate_and_clean
from app.tracing.langsmith import trace_node, compute_hallucination_score

logger = logging.getLogger(__name__)


@trace_node("validation_agent")
def validation_agent(state: dict) -> dict:
    """
    Graph node: validates and cleans the raw answer.

    Input state keys : raw_answer, retrieved_docs
    Output state keys: validated_answer, pii_detected
    """
    raw_answer = state.get("raw_answer", "")
    docs = state.get("retrieved_docs", [])
    source_ids = [d["doc_id"] for d in docs]

    result = validate_and_clean(raw_answer, source_ids)

    hallucination_score = compute_hallucination_score(result["answer"], docs)

    logger.info(
        "Validation complete | pii_detected=%s | confidence=%.2f | hallucination_score=%.2f",
        result.get("pii_detected"),
        result.get("confidence", 0),
        hallucination_score,
    )
    return {
        "validated_answer": result["answer"],
        "pii_detected": result["pii_detected"],
        "hallucination_score": hallucination_score,
    }
