"""
Unit tests for retrieval utilities.
"""

from app.retrieval.chunking import chunk_document, get_splitter


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
