"""Data models for document processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceDocument:
    """Input document with stable provenance."""

    doc_id: str
    source_path: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextUnit:
    """Chunked text unit with source provenance."""

    chunk_id: str
    doc_id: str
    source_path: str
    chunk_index: int
    text: str
    n_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentScanRecord:
    """One input file decision from document scanning."""

    source_path: str
    title: str
    extension: str
    size_bytes: int
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class DocumentScanResult:
    """Documents plus a manifest of scan decisions."""

    input_path: str
    unsupported_file_policy: str
    documents: list[SourceDocument] = field(default_factory=list)
    records: list[DocumentScanRecord] = field(default_factory=list)

    @property
    def num_files(self) -> int:
        return len(self.records)

    @property
    def num_included_files(self) -> int:
        return _count_status(self.records, "included")

    @property
    def num_ignored_files(self) -> int:
        return _count_status(self.records, "ignored")

    @property
    def num_rejected_files(self) -> int:
        return _count_status(self.records, "rejected")

    @property
    def num_empty_documents(self) -> int:
        return sum(1 for document in self.documents if not document.text.strip())

    def to_manifest(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "unsupported_file_policy": self.unsupported_file_policy,
            "records": [
                {
                    "source_path": record.source_path,
                    "title": record.title,
                    "extension": record.extension,
                    "size_bytes": record.size_bytes,
                    "status": record.status,
                    "reason": record.reason,
                }
                for record in self.records
            ],
            "num_files": self.num_files,
            "num_included_files": self.num_included_files,
            "num_ignored_files": self.num_ignored_files,
            "num_rejected_files": self.num_rejected_files,
            "num_empty_documents": self.num_empty_documents,
        }


def _count_status(records: list[DocumentScanRecord], status: str) -> int:
    return sum(1 for record in records if record.status == status)
