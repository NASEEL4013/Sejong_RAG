import hashlib

import pytest

from sejong_rag.download_attachments import (
    already_downloaded,
    attachment_paths,
    download_all,
    stream_download,
    validate_attachment_url,
)


def item():
    return {
        "attachment_id": "notice3-1-attachment-9",
        "parent_document_id": "notice3-1",
        "parent_title": "공지",
        "source_url": "https://www.sejong.ac.kr/kor/intro/notice3.do?mode=view&articleNo=1",
        "download_url": "https://www.sejong.ac.kr/kor/intro/notice3.do?mode=download&articleNo=1&attachNo=9",
        "attach_no": "9",
        "filename": "안내.pdf",
        "file_type": "pdf",
    }


class FakeResponse:
    headers = {"content-length": "7", "content-type": "application/pdf"}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield b"%PDF-"
        yield b"ok"


def test_validate_attachment_url_rejects_external_or_wrong_attach_number():
    validate_attachment_url(item())
    wrong = {**item(), "download_url": "https://example.com/a?mode=download&attachNo=9"}
    with pytest.raises(ValueError):
        validate_attachment_url(wrong)
    wrong_number = {**item(), "download_url": item()["download_url"].replace("attachNo=9", "attachNo=8")}
    with pytest.raises(ValueError):
        validate_attachment_url(wrong_number)


def test_stream_download_hashes_and_atomically_finishes(tmp_path):
    destination = tmp_path / "file.pdf"

    digest, size, content_type = stream_download(
        item()["download_url"], destination, requester=lambda *a, **k: FakeResponse()
    )

    assert destination.read_bytes() == b"%PDF-ok"
    assert digest == hashlib.sha256(b"%PDF-ok").hexdigest()
    assert size == 7
    assert content_type == "application/pdf"
    assert not destination.with_suffix(".pdf.part").exists()


def test_stream_download_rejects_declared_oversize(tmp_path):
    with pytest.raises(ValueError, match="크기 제한"):
        stream_download(
            item()["download_url"], tmp_path / "file.pdf", max_bytes=6,
            requester=lambda *a, **k: FakeResponse(),
        )


def test_download_all_saves_file_and_metadata_with_fake_downloader(tmp_path):
    def fake_downloader(url, destination, max_bytes):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"content")
        return hashlib.sha256(b"content").hexdigest(), 7, "application/pdf"

    results = download_all([item()], tmp_path, delay_seconds=0, downloader=fake_downloader)

    raw, metadata = attachment_paths(item(), tmp_path)
    assert results[0]["status"] == "downloaded"
    assert raw.exists() and metadata.exists()
    assert already_downloaded(item(), tmp_path)


def test_real_downloader_requires_ten_second_delay():
    with pytest.raises(ValueError, match="10초 이상"):
        download_all([item()], delay_seconds=1)
