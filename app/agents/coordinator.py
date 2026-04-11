"""
Coordinator agent.

Receives the user query, enriches the graph state, and hands off to the
LangGraph workflow which runs retrieval → (summarization ‖ validation) in
parallel, then merges results.
"""

import logging
from app.graph.workflow import build_graph

logger = logging.getLogger(__name__)

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(query: str, top_k: int = 5, doc_type: str | None = None) -> dict:
    """
    Entry point called by the FastAPI handler.

    Returns a dict with keys: validated_answer, retrieved_docs, token_cost,
    node_latencies, pii_detected.
    """
    initial_state = {
        "query": query,
        "top_k": top_k,
        "doc_type": doc_type,
        "retrieved_docs": [],
        "raw_answer": "",
        "validated_answer": "",
        "token_cost": {},
        "node_latencies": {},
        "pii_detected": False,
        "hallucination_score": -1.0,
    }

    graph = get_graph()
    logger.info("Coordinator dispatching query: %s", query[:80])
    final_state = graph.invoke(initial_state)
    logger.info("Coordinator received final state")
    return final_state
