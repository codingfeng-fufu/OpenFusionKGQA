"""Document loader for text, Markdown, and PDF files."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

from graphrag_v2.document.models import (
    DocumentScanRecord,
    DocumentScanResult,
    SourceDocument,
)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
SUPPORTED_UNSUPPORTED_FILE_POLICIES = {"ignore", "warn", "fail"}


def load_documents(input_path: str | Path, encoding: str = "utf-8") -> list[SourceDocument]:
    """Load supported documents from a file or directory."""
    scan = scan_documents(input_path=input_path, encoding=encoding)
    _raise_if_rejected_scan(scan)
    return scan.documents


def scan_documents(
    input_path: str | Path,
    encoding: str = "utf-8",
    unsupported_file_policy: str = "ignore",
    max_file_size_bytes: int | None = None,
    max_document_count: int | None = None,
) -> DocumentScanResult:
    """Scan input files, load included documents, and keep every decision."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    policy = unsupported_file_policy.strip().lower()
    if policy not in SUPPORTED_UNSUPPORTED_FILE_POLICIES:
        valid = ", ".join(sorted(SUPPORTED_UNSUPPORTED_FILE_POLICIES))
        raise ValueError(f"unsupported_file_policy must be one of: {valid}")
    if max_file_size_bytes is not None and max_file_size_bytes < 1:
        raise ValueError("max_file_size_bytes must be at least 1")
    if max_document_count is not None and max_document_count < 1:
        raise ValueError("max_document_count must be at least 1")

    if path.is_file():
        files = [path]
    else:
        files = sorted(file for file in path.rglob("*") if file.is_file())

    records: list[DocumentScanRecord] = []
    documents: list[SourceDocument] = []
    ignored_unsupported = 0

    for file in files:
        extension = file.suffix.lower()
        size_bytes = file.stat().st_size
        if extension not in SUPPORTED_EXTENSIONS:
            status = "rejected" if path.is_file() or policy == "fail" else "ignored"
            if status == "ignored":
                ignored_unsupported += 1
            records.append(
                _scan_record(
                    file,
                    status=status,
                    reason="unsupported_extension",
                    size_bytes=size_bytes,
                )
            )
            continue

        if max_file_size_bytes is not None and size_bytes > max_file_size_bytes:
            records.append(
                _scan_record(
                    file,
                    status="rejected",
                    reason="file_too_large",
                    size_bytes=size_bytes,
                )
            )
            continue

        if max_document_count is not None and len(documents) >= max_document_count:
            records.append(
                _scan_record(
                    file,
                    status="rejected",
                    reason="max_document_count_exceeded",
                    size_bytes=size_bytes,
                )
            )
            continue

        try:
            document = _load_file(file, encoding)
        except Exception as exc:
            records.append(
                _scan_record(
                    file,
                    status="rejected",
                    reason=f"load_error:{exc.__class__.__name__}",
                    size_bytes=size_bytes,
                )
            )
            continue

        documents.append(document)
        records.append(
            _scan_record(
                file,
                status="included",
                reason="empty_text" if not document.text.strip() else None,
                size_bytes=size_bytes,
            )
        )

    if policy == "warn" and ignored_unsupported:
        warnings.warn(
            f"Ignored {ignored_unsupported} unsupported input file(s).",
            UserWarning,
            stacklevel=2,
        )

    return DocumentScanResult(
        input_path=str(path.resolve()),
        unsupported_file_policy=policy,
        documents=documents,
        records=records,
    )


def _load_file(path: Path, encoding: str) -> SourceDocument:
    if path.suffix.lower() == ".pdf":
        text, metadata = _read_pdf_text(path)
    else:
        text = path.read_text(encoding=encoding)
        metadata = {"extension": path.suffix.lower()}
    source_path = str(path.resolve())
    return SourceDocument(
        doc_id=_stable_doc_id(source_path),
        source_path=source_path,
        title=path.name,
        text=text,
        metadata=metadata,
    )


def _stable_doc_id(source_path: str) -> str:
    digest = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:16]
    return f"doc_{digest}"


def _read_pdf_text(path: Path) -> tuple[str, dict]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF input requires the optional 'pypdf' package to be installed."
        ) from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    empty_pages = 0
    for index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[Page {index}]\n{page_text.strip()}")
        else:
            empty_pages += 1
    metadata = {
        "extension": ".pdf",
        "pdf_page_count": len(reader.pages),
        "pdf_extracted_pages": len(pages),
        "pdf_empty_pages": empty_pages,
    }
    return "\n\n".join(pages), metadata


def _scan_record(
    path: Path,
    *,
    status: str,
    reason: str | None,
    size_bytes: int,
) -> DocumentScanRecord:
    return DocumentScanRecord(
        source_path=str(path.resolve()),
        title=path.name,
        extension=path.suffix.lower(),
        size_bytes=size_bytes,
        status=status,
        reason=reason,
    )


def _raise_if_rejected_scan(scan: DocumentScanResult) -> None:
    if scan.num_rejected_files == 0:
        return
    reasons = sorted(
        {
            record.reason or "unknown"
            for record in scan.records
            if record.status == "rejected"
        }
    )
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    if reasons == ["unsupported_extension"]:
        rejected = next(
            record for record in scan.records if record.status == "rejected"
        )
        extension = rejected.extension or "<none>"
        raise ValueError(
            f"Unsupported input file extension: {extension}. "
            f"Supported extensions: {supported}"
        )
    raise ValueError(
        f"Rejected {scan.num_rejected_files} input file(s): {', '.join(reasons)}. "
        f"Supported extensions: {supported}"
    )
