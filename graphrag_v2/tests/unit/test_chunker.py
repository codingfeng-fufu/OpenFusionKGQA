"""Tests for document chunking."""

from graphrag_v2.document.chunker import chunk_documents
from graphrag_v2.document.models import SourceDocument


def test_chunker_preserves_provenance():
    document = SourceDocument(
        doc_id="doc_1",
        source_path="/tmp/doc.md",
        title="doc.md",
        text="GraphRAG uses graphs for grounded retrieval.",
    )

    text_units = chunk_documents([document], chunk_size=10, chunk_overlap=2)

    assert len(text_units) >= 1
    assert text_units[0].doc_id == "doc_1"
    assert text_units[0].source_path == "/tmp/doc.md"
    assert text_units[0].chunk_index == 0
    assert text_units[0].chunk_id.startswith("chunk_")
    assert text_units[0].n_tokens > 0


def test_chunk_ids_are_stable():
    document = SourceDocument(
        doc_id="doc_1",
        source_path="/tmp/doc.md",
        title="doc.md",
        text="GraphRAG uses graphs for grounded retrieval.",
    )

    first = chunk_documents([document], chunk_size=10, chunk_overlap=2)
    second = chunk_documents([document], chunk_size=10, chunk_overlap=2)

    assert [unit.chunk_id for unit in first] == [unit.chunk_id for unit in second]


def test_chunker_rejects_invalid_overlap():
    document = SourceDocument(
        doc_id="doc_1",
        source_path="/tmp/doc.md",
        title="doc.md",
        text="text",
    )

    try:
        chunk_documents([document], chunk_size=10, chunk_overlap=10)
    except ValueError as error:
        assert "chunk_overlap" in str(error)
    else:
        raise AssertionError("Expected ValueError")
