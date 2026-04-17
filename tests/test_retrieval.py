"""
Unit tests for retrieval utilities.
"""

import pytest

from app.retrieval.chunking import chunk_document, get_splitter, CHUNKING_STRATEGIES


def test_chunk_document_policy():
    text = "This is a policy. " * 100
    chunks = chunk_document(text, doc_type="policy")
    assert len(chunks) > 1
    assert all("text" in c for c in chunks)
    assert all(c["metadata"]["doc_type"] == "policy" for c in chunks)


def test_chunk_document_regulation_smaller_chunks():
    text = "Regulation text. " * 200
    policy_chunks = chunk_document(text, doc_type="policy")
    reg_chunks = chunk_document(text, doc_type="regulation")
    # Regulation chunks should be smaller → more of them
    assert len(reg_chunks) >= len(policy_chunks)


def test_chunk_metadata_preserved():
    text = "Sample financial document text. " * 50
    meta = {"source": "annual_report_2024.txt"}
    chunks = chunk_document(text, doc_type="report", metadata=meta)
    for c in chunks:
        assert c["metadata"]["source"] == "annual_report_2024.txt"
        assert "chunk_index" in c["metadata"]


# ── Chunking strategy tests ────────────────────────────────────────────────────
@pytest.mark.parametrize("strategy", CHUNKING_STRATEGIES)
def test_chunking_strategies_produce_chunks(strategy):
    text = "First sentence here. Second sentence here. Third sentence here. " * 30
    chunks = chunk_document(text, doc_type="policy", strategy=strategy)
    assert len(chunks) > 0
    assert all(c["metadata"]["chunking_strategy"] == strategy for c in chunks)


def test_sentence_window_preserves_neighbors():
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    chunks = chunk_document(text, strategy="sentence-window")
    # Middle sentence chunk should contain neighboring sentences too
    assert any("Sentence two" in c["text"] and "Sentence four" in c["text"] for c in chunks)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        chunk_document("some text", strategy="random-strategy")
