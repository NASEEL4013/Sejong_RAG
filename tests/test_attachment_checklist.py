from sejong_rag.attachment_checklist import build_checklist


def test_checklist_includes_links_paths_and_completion():
    item = {
        "attachment_id": "notice3-1-attachment-9",
        "parent_document_id": "notice3-1",
        "parent_title": "수강편람 안내",
        "source_url": "https://www.sejong.ac.kr/notice",
        "download_url": "https://www.sejong.ac.kr/download",
        "filename": "수강편람.pdf",
        "file_type": "pdf",
    }
    text = build_checklist(
        [item],
        {"notice3-1": {"academic_year": 2026, "semester": 2, "version_status": "current_candidate"}},
        {"notice3-1-attachment-9"},
        __import__("pathlib").Path("data/manual-input"),
    )
    assert "[x] **수강편람.pdf**" in text
    assert "https://www.sejong.ac.kr/download" in text
    assert "data/manual-input/notice3-1/수강편람.pdf" in text
    assert "`current_candidate`" in text


def test_imported_attachment_ids_includes_auto_download_metadata(tmp_path):
    from sejong_rag.attachment_checklist import imported_attachment_ids
    directory = tmp_path / "metadata/downloaded-attachments"
    directory.mkdir(parents=True)
    (directory / "notice3-1-attachment-9.json").write_text("{}", encoding="utf-8")
    assert "notice3-1-attachment-9" in imported_attachment_ids(tmp_path)
