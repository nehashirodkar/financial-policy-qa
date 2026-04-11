"""
Shared state schema for the LangGraph multi-agent workflow.
"""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    query: str
    top_k: int
    doc_type: str | None
    retrieved_docs: list
    raw_answer: str
    validated_answer: str
    token_cost: dict
    node_latencies: dict
    pii_detected: bool
    hallucination_score: float
