"""Tests for document loading."""

from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from graphrag_v2.document.loader import load_documents, scan_documents


def test_load_single_txt_file(temp_dir: Path):
    path = temp_dir / "sample.txt"
    path.write_text("GraphRAG text", encoding="utf-8")

    documents = load_documents(path)

    assert len(documents) == 1
    assert documents[0].title == "sample.txt"
    assert documents[0].text == "GraphRAG text"
    assert documents[0].source_path == str(path.resolve())


def test_load_single_md_file(temp_dir: Path):
    path = temp_dir / "sample.md"
    path.write_text("# GraphRAG", encoding="utf-8")

    documents = load_documents(path)

    assert len(documents) == 1
    assert documents[0].metadata["extension"] == ".md"


def test_load_directory_recursively_and_ignore_unsupported(temp_dir: Path):
    nested = temp_dir / "nested"
    nested.mkdir()
    (temp_dir / "b.md").write_text("B", encoding="utf-8")
    (nested / "a.txt").write_text("A", encoding="utf-8")
    (temp_dir / "ignore.csv").write_text("ignored", encoding="utf-8")

    documents = load_documents(temp_dir)

    assert [doc.title for doc in documents] == ["b.md", "a.txt"]
    assert all(doc.doc_id.startswith("doc_") for doc in documents)


def test_load_unsupported_single_file_raises(temp_dir: Path):
    path = temp_dir / "sample.csv"
    path.write_text("GraphRAG,csv", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported input file extension"):
        load_documents(path)


def test_load_single_pdf_file(temp_dir: Path):
    path = temp_dir / "sample.pdf"
    _write_minimal_text_pdf(path, "GraphRAG PDF text")

    documents = load_documents(path)

    assert len(documents) == 1
    assert documents[0].title == "sample.pdf"
    assert documents[0].metadata["extension"] == ".pdf"
    assert "[Page 1]" in documents[0].text
    assert "GraphRAG PDF text" in documents[0].text
    assert documents[0].metadata["pdf_page_count"] == 1
    assert documents[0].metadata["pdf_extracted_pages"] == 1


def test_load_empty_directory(temp_dir: Path):
    assert load_documents(temp_dir) == []


def test_scan_warns_for_unsupported_files(temp_dir: Path):
    (temp_dir / "doc.txt").write_text("GraphRAG", encoding="utf-8")
    (temp_dir / "ignored.csv").write_text("unsupported", encoding="utf-8")

    with pytest.warns(UserWarning, match="Ignored 1 unsupported"):
        scan = scan_documents(temp_dir, unsupported_file_policy="warn")

    assert scan.num_files == 2
    assert scan.num_included_files == 1
    assert scan.num_ignored_files == 1
    assert scan.num_rejected_files == 0
    ignored = next(record for record in scan.records if record.status == "ignored")
    assert ignored.reason == "unsupported_extension"


def test_scan_fail_policy_marks_unsupported_files_rejected(temp_dir: Path):
    (temp_dir / "doc.txt").write_text("GraphRAG", encoding="utf-8")
    (temp_dir / "bad.csv").write_text("unsupported", encoding="utf-8")

    scan = scan_documents(temp_dir, unsupported_file_policy="fail")

    assert scan.num_included_files == 1
    assert scan.num_rejected_files == 1
    rejected = next(record for record in scan.records if record.status == "rejected")
    assert rejected.reason == "unsupported_extension"


def test_scan_tracks_empty_documents(temp_dir: Path):
    path = temp_dir / "empty.txt"
    path.write_text("   ", encoding="utf-8")

    scan = scan_documents(path)

    assert scan.num_included_files == 1
    assert scan.num_empty_documents == 1
    assert scan.records[0].status == "included"
    assert scan.records[0].reason == "empty_text"


def test_scan_rejects_oversized_files(temp_dir: Path):
    path = temp_dir / "large.txt"
    path.write_text("GraphRAG", encoding="utf-8")

    scan = scan_documents(path, max_file_size_bytes=1)

    assert scan.documents == []
    assert scan.num_rejected_files == 1
    assert scan.records[0].reason == "file_too_large"


def test_scan_rejects_corrupt_pdf(temp_dir: Path):
    path = temp_dir / "bad.pdf"
    path.write_text("not a pdf", encoding="utf-8")

    scan = scan_documents(path)

    assert scan.documents == []
    assert scan.num_rejected_files == 1
    assert scan.records[0].status == "rejected"
    assert scan.records[0].reason.startswith("load_error:")


def _write_minimal_text_pdf(path: Path, text: str) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 200 Td ({text}) Tj ET".encode("utf-8"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)
