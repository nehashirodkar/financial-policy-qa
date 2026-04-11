"""
Retrieval specialist agent.

Receives the user query from the graph state and performs hybrid
(dense + sparse) search against Pinecone.
"""

import logging
from app.retrieval.pinecone_client import HybridRetriever
from app.tracing.langsmith import trace_node

logger = logging.getLogger(__name__)

_retriever = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


@trace_node("retrieval_agent")
def retrieval_agent(state: dict) -> dict:
    """
    Graph node: retrieves relevant documents.

    Input state keys : query, top_k, doc_type
    Output state keys: retrieved_docs
    """
    query = state["query"]
    top_k = state.get("top_k", 5)
    doc_type = state.get("doc_type")

    retriever = _get_retriever()
    docs = retriever.retrieve(query=query, top_k=top_k, doc_type=doc_type)

    logger.info("Retrieved %d documents", len(docs))
    return {"retrieved_docs": [d.model_dump() for d in docs]}
