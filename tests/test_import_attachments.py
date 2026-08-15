import json
import unicodedata

import pytest
from pypdf import PdfWriter

from sejong_rag.import_attachments import (
    build_manifest,
    import_pdf,
    load_expected_attachments,
    normalize_filename,
)


FILENAME = "2026학년도 등록금 안내.pdf"


def write_detail(directory, filename=FILENAME, document_id="notice3-1", attach_no="9"):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "document_id": document_id,
        "title": "등록금 공지",
        "source_url": "https://www.sejong.ac.kr/kor/intro/notice3.do?mode=view&articleNo=1",
        "attachments": [
            {
                "filename": filename,
                "download_url": "https://www.sejong.ac.kr/kor/intro/notice3.do?mode=download&articleNo=1&attachNo=9",
                "attach_no": attach_no,
                "file_type": "pdf",
                "version_date": None,
            }
        ],
    }
    (directory / f"{document_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def write_pdf(path):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with path.open("wb") as output:
        writer.write(output)


def test_normalize_filename_handles_unicode_and_spaces():
    decomposed = unicodedata.normalize("NFD", FILENAME)
    assert normalize_filename(decomposed) == normalize_filename("  " + FILENAME + "  ")


def test_load_expected_builds_parent_and_attachment_ids(tmp_path):
    details = tmp_path / "details"
    write_detail(details)

    expected = load_expected_attachments(details)

    assert expected[0]["attachment_id"] == "notice3-1-attachment-9"
    assert expected[0]["parent_document_id"] == "notice3-1"


def test_manifest_changes_from_missing_to_ready(tmp_path):
    details = tmp_path / "details"
    inputs = tmp_path / "manual-input"
    write_detail(details)
    expected = load_expected_attachments(details)

    assert build_manifest(expected, inputs)[0]["status"] == "missing"
    write_pdf(inputs / FILENAME)
    assert build_manifest(expected, inputs)[0]["status"] == "ready"


def test_import_pdf_extracts_pages_and_detects_unchanged(tmp_path):
    details = tmp_path / "details"
    inputs = tmp_path / "manual-input"
    write_detail(details)
    inputs.mkdir()
    write_pdf(inputs / FILENAME)
    item = build_manifest(load_expected_attachments(details), inputs)[0]
    output = tmp_path / "data"

    first = import_pdf(item, output)
    second = import_pdf(item, output)

    assert first["change_status"] == "new"
    assert first["page_count"] == 1
    assert second["change_status"] == "unchanged"
    assert (output / "processed/pdf/notice3-1-attachment-9.jsonl").exists()
    assert (output / "metadata/imported-attachments/notice3-1-attachment-9.json").exists()


def test_import_pdf_rejects_fake_pdf(tmp_path):
    fake = tmp_path / FILENAME
    fake.write_text("PDF가 아님", encoding="utf-8")
    item = {
        "local_path": str(fake),
        "attachment_id": "notice3-1-attachment-9",
    }

    with pytest.raises(ValueError, match="PDF 서명"):
        import_pdf(item, tmp_path / "output")


def test_duplicate_expected_filename_requires_parent_subdirectories(tmp_path):
    details = tmp_path / "details"
    inputs = tmp_path / "manual-input"
    write_detail(details, document_id="notice3-1", attach_no="9")
    write_detail(details, document_id="notice3-2", attach_no="10")
    expected = load_expected_attachments(details)
    rows = build_manifest(expected, inputs)
    assert len(rows) == 2
    assert all(row["filename_is_duplicated"] for row in rows)
    assert all(row["status"] == "missing" for row in rows)
    target = inputs / "notice3-1" / FILENAME
    target.parent.mkdir(parents=True)
    write_pdf(target)
    rows = build_manifest(expected, inputs)
    assert rows[0]["status"] == "ready"
    assert rows[1]["status"] == "missing"
