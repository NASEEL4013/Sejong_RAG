from dataclasses import replace

import pytest

from sejong_rag.collect_details import (
    collect_details,
    content_hash,
    save_detail,
    select_candidates,
    validate_detail_url,
)
from sejong_rag.ingest import Attachment, Notice


URL = "https://www.sejong.ac.kr/kor/intro/notice7.do?mode=view&articleNo=123"


def candidate(document_id="notice7-123", priority="high", categories=None):
    return {
        "document_id": document_id,
        "source_url": URL,
        "title": "국가장학금 안내",
        "priority": priority,
        "categories": categories or ["scholarship"],
        "matched_keywords": ["장학"],
        "classification_method": "title_keyword_v1",
    }


def notice(body="신청하세요"):
    return Notice(
        document_id="notice7-123",
        article_no="123",
        title="국가장학금 안내",
        department="학생지원과",
        published_at="2026.08.01",
        source_url=URL,
        board="scholarship_notice",
        body=body,
        attachments=[
            Attachment("안내.pdf", URL + "&mode=download&attachNo=9", "9", "pdf", None)
        ],
    )


def test_select_candidates_filters_without_discarding_source_payload():
    normal = candidate("notice7-124", "normal")
    normal["source_url"] = URL.replace("123", "124")
    payload = {"items": [candidate(), normal]}

    selected = select_candidates(payload, {"high"}, {"scholarship"}, max_items=5)

    assert [item["document_id"] for item in selected] == ["notice7-123"]
    assert len(payload["items"]) == 2


def test_validate_detail_url_rejects_download_and_external_urls():
    with pytest.raises(ValueError):
        validate_detail_url("https://example.com/kor/intro/notice7.do?mode=view&articleNo=1")
    with pytest.raises(ValueError):
        validate_detail_url(
            "https://www.sejong.ac.kr/kor/intro/notice7.do?mode=download&articleNo=1"
        )


def test_content_hash_changes_only_when_notice_content_changes():
    original = notice()
    assert content_hash(original) == content_hash(replace(original, source_url=URL + "&x=1"))
    assert content_hash(original) != content_hash(replace(original, body="수정 내용"))


def test_save_detail_detects_new_unchanged_and_updated(tmp_path):
    assert save_detail(notice(), b"first", candidate(), tmp_path) == "new"
    assert save_detail(notice(), b"second markup", candidate(), tmp_path) == "unchanged"
    assert save_detail(notice("수정 내용"), b"third", candidate(), tmp_path) == "updated"


def test_collect_details_uses_fetcher_without_downloading_attachment(tmp_path):
    calls = []

    def fake_fetcher(url):
        calls.append(url)
        return notice(), b"<html>detail</html>"

    results = collect_details([candidate()], tmp_path, delay_seconds=0, fetcher=fake_fetcher)

    assert calls == [URL]
    assert results[0]["attachment_count"] == 1
    assert (tmp_path / "metadata/details/notice7-123.json").exists()
    assert not (tmp_path / "raw/attachments").exists()


def test_real_detail_fetch_requires_robots_delay():
    with pytest.raises(ValueError, match="10초 이상"):
        collect_details([candidate()], delay_seconds=1)


def test_select_candidates_allows_year_catalog_scale():
    payload = {"items": []}
    assert select_candidates(payload, max_items=500) == []
    with pytest.raises(ValueError, match="1~500"):
        select_candidates(payload, max_items=501)
