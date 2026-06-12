"""Text chunking utilities."""

from __future__ import annotations

import hashlib

import tiktoken

from graphrag_v2.document.models import SourceDocument, TextUnit


def chunk_documents(
    documents: list[SourceDocument],
    chunk_size: int,
    chunk_overlap: int,
    encoding_model: str = "cl100k_base",
) -> list[TextUnit]:
    """Split documents into text units with stable provenance."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    encoding = _get_encoding(encoding_model)
    text_units: list[TextUnit] = []

    for document in documents:
        tokens = encoding.encode(document.text)
        if not tokens and document.text == "":
            continue

        for chunk_index, chunk_tokens in enumerate(_iter_token_chunks(tokens, chunk_size, chunk_overlap)):
            chunk_text = encoding.decode(chunk_tokens)
            text_units.append(
                TextUnit(
                    chunk_id=_stable_chunk_id(document.doc_id, chunk_index, chunk_text),
                    doc_id=document.doc_id,
                    source_path=document.source_path,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    n_tokens=len(chunk_tokens),
                    metadata={"title": document.title},
                )
            )

    return text_units


def _iter_token_chunks(
    tokens: list[int],
    chunk_size: int,
    chunk_overlap: int,
) -> list[list[int]]:
    chunks: list[list[int]] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokens[start:end])
        if end >= len(tokens):
            break
        start = end - chunk_overlap
    return chunks


def _get_encoding(encoding_model: str):
    try:
        return tiktoken.get_encoding(encoding_model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _stable_chunk_id(doc_id: str, chunk_index: int, chunk_text: str) -> str:
    content = f"{doc_id}:{chunk_index}:{chunk_text}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"chunk_{digest}"
