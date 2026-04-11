"""
Unit tests for individual agent nodes.
All external calls (Bedrock, Pinecone) are mocked.
"""

from unittest.mock import patch, MagicMock
import pytest

from app.agents.retrieval_agent import retrieval_agent
from app.agents.summarization_agent import summarization_agent
from app.agents.validation_agent import validation_agent


# ── Retrieval agent ────────────────────────────────────────────────────────────
@patch("app.agents.retrieval_agent._get_retriever")
def test_retrieval_agent_returns_docs(mock_get_retriever):
    mock_doc = MagicMock()
    mock_doc.model_dump.return_value = {
        "doc_id": "doc1", "text": "Sample policy text.", "score": 0.92, "metadata": {}
    }
    mock_get_retriever.return_value.retrieve.return_value = [mock_doc]

    state = {"query": "What is the refund policy?", "top_k": 3, "doc_type": "policy"}
    result = retrieval_agent(state)

    assert "retrieved_docs" in result
    assert len(result["retrieved_docs"]) == 1
    assert result["retrieved_docs"][0]["doc_id"] == "doc1"


# ── Summarization agent ────────────────────────────────────────────────────────
@patch("boto3.client")
def test_summarization_agent_generates_answer(mock_boto):
    mock_response = {
        "content": [{"text": "The refund policy allows returns within 30 days."}],
        "usage": {"input_tokens": 200, "output_tokens": 50},
    }
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {
        "body": MagicMock(read=lambda: __import__("json").dumps(mock_response).encode())
    }
    mock_boto.return_value = mock_client

    state = {
        "query": "What is the refund policy?",
        "retrieved_docs": [
            {"doc_id": "doc1", "text": "Refunds are accepted within 30 days.", "score": 0.9, "metadata": {}}
        ],
    }
    result = summarization_agent(state)

    assert "raw_answer" in result
    assert "30 days" in result["raw_answer"]
    assert result["token_cost"]["output_tokens"] == 50


# ── Validation agent ───────────────────────────────────────────────────────────
@patch("app.agents.validation_agent.compute_hallucination_score", return_value=0.95)
@patch("app.agents.validation_agent.validate_and_clean")
def test_validation_agent_redacts_pii(mock_validate, mock_hallucination):
    mock_validate.return_value = {
        "answer": "Contact [EMAIL] for refunds.",
        "confidence": 0.88,
        "sources_used": ["doc1"],
        "pii_detected": True,
    }

    state = {
        "raw_answer": "Contact john@acme.com for refunds.",
        "retrieved_docs": [{"doc_id": "doc1", "text": "...", "score": 0.9, "metadata": {}}],
    }
    result = validation_agent(state)

    assert result["pii_detected"] is True
    assert "[EMAIL]" in result["validated_answer"]
    assert result["hallucination_score"] == 0.95
