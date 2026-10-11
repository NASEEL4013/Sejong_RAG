import json
from pathlib import Path

import pytest

from sejong_rag.ingest import parse_notice_html
from sejong_rag.notice_images import process_images, validate_image_url
from sejong_rag.unified_chunk import make_chunks, load_jsonl
from test_unified_chunk import FakeTokenizer

URL = "https://www.sejong.ac.kr/kor/intro/notice3.do?mode=view&articleNo=1"
HTML = '''<div class="b-view-common-box"><p class="b-title">신청 안내</p>
<ul class="b-info"><li>학사지원과</li><li class="b-date">2026.08.01</li></ul>
<div class="b-content-box"><img src="/images/notice.png" alt="신청 일정"></div></div>'''


def setup_notice(tmp_path):
    details = tmp_path / "metadata/details"
    details.mkdir(parents=True)
    (details / "notice3-1.json").write_text(json.dumps({
        "document_id": "notice3-1", "title": "신청 안내", "source_url": URL, "body": ""}), encoding="utf-8")
    raw = tmp_path / "raw/notices"
    raw.mkdir(parents=True)
    (raw / "notice3-1.html").write_text(HTML, encoding="utf-8")


def fake_download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test image")


def test_image_only_notice_keeps_absolute_url_and_alt():
    notice = parse_notice_html(HTML, URL)
    assert notice.body == ""
    assert notice.body_images == [{"url": "https://www.sejong.ac.kr/images/notice.png", "alt": "신청 일정"}]


@pytest.mark.parametrize("url", ["http://www.sejong.ac.kr/a", "https://evil.test/a",
    "https://www.sejong.ac.kr:8443/a", "file:///etc/passwd", "data:image/png;base64,x",
    "https://name@www.sejong.ac.kr/a"])
def test_rejects_nonofficial_images(url):
    with pytest.raises(ValueError):
        validate_image_url(url)


def test_saved_html_backfill_resume_and_review_only_chunks(tmp_path):
    setup_notice(tmp_path)
    calls = []
    def reader(path):
        calls.append(path)
        return "신청 기간 8월 14일", 93.0
    first = process_images(tmp_path, downloader=fake_download, reader=reader)
    second = process_images(tmp_path, downloader=fake_download, reader=reader)
    assert first["processed"] == 1 and second["processed"] == 0
    assert len(calls) == 1
    output = tmp_path / "processed/unified-chunks"
    make_chunks(tmp_path, output, FakeTokenizer())
    trusted, review = load_jsonl(output / "trusted.jsonl"), load_jsonl(output / "review.jsonl")
    assert len(trusted) == len(review) == 1
    assert trusted[0]["evidence_warning"]
    assert "8월 14일" not in trusted[0]["text"]
    assert "8월 14일" in review[0]["text"]
    assert review[0]["parent_document_id"] == "notice3-1"
    assert review[0]["locators"][0]["image"] == 1


def test_failed_ocr_is_reported_without_text(tmp_path):
    setup_notice(tmp_path)
    result = process_images(tmp_path, downloader=fake_download, reader=lambda path: ("", None))
    assert result["failed"] == 1
    report = json.loads((tmp_path / "processed/notice-images/notice3-1.json").read_text())
    assert report["images"][0]["status"] == "failed"
    assert report["images"][0]["text"] == ""


def test_dry_run_makes_no_requests_or_outputs(tmp_path):
    setup_notice(tmp_path)
    result = process_images(tmp_path, dry_run=True, downloader=lambda *args: pytest.fail("network"))
    assert result["images"] == 1
    assert not (tmp_path / "processed").exists()


def test_refresh_reprocesses_replaced_image(tmp_path):
    setup_notice(tmp_path)
    process_images(tmp_path, downloader=fake_download, reader=lambda path: ("옛 일정", 90))
    def replacement(url, path):
        path.write_bytes(b"replaced image")
    result = process_images(tmp_path, refresh=True, downloader=replacement, reader=lambda path: ("정정 일정", 90))
    assert result["processed"] == 1
    report = json.loads((tmp_path / "processed/notice-images/notice3-1.json").read_text())
    assert report["images"][0]["text"] == "정정 일정"


def test_download_rejects_redirect_and_non_image(monkeypatch, tmp_path):
    from sejong_rag.notice_images import download_image
    class Response:
        status_code = 302
        headers = {"Content-Type": "image/png"}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
    def request(*args, **kwargs):
        assert kwargs["allow_redirects"] is False
        return Response()
    monkeypatch.setattr("sejong_rag.notice_images.requests.get", request)
    path = tmp_path / "image.img"
    with pytest.raises(ValueError, match="이동"):
        download_image("https://www.sejong.ac.kr/a", path)
    Response.status_code = 200
    Response.headers = {"Content-Type": "text/html"}
    with pytest.raises(ValueError, match="형식"):
        download_image("https://www.sejong.ac.kr/a", path)
    assert not path.exists()


def test_small_screenshot_uses_sparse_text_fallback(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from sejong_rag.notice_images import ocr_image
    modes = []
    def command(args):
        modes.append(args[-2])
        return SimpleNamespace(stdout="empty" if args[-2] == "3" else "recognized")
    monkeypatch.setattr("sejong_rag.notice_images.run_command", command)
    monkeypatch.setattr("sejong_rag.notice_images.parse_tesseract_tsv",
                        lambda text: ("신청" if text == "recognized" else "", 90))
    assert ocr_image(tmp_path / "small.png")[0] == "신청"
    assert modes == ["3", "11"]


def test_offline_ocr_never_downloads_missing_source(tmp_path):
    setup_notice(tmp_path)
    result=process_images(tmp_path,offline=True,downloader=lambda *args:pytest.fail('network'))
    assert result['failed']==1 and result['processed']==0
