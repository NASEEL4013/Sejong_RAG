from dataclasses import asdict

import pytest

from sejong_rag.list_notices import (
    NoticeListItem,
    collect_board,
    compare_catalogs,
    page_url,
    parse_notice_list_html,
)


BOARD_URL = "https://www.sejong.ac.kr/kor/intro/notice3.do"
LIST_HTML = """
<div class="b-list-common-wrap"><table><tbody>
<tr class="b-top-box">
  <td class="b-num-box">공지</td>
  <td class="b-td-title b-td-left">
    <div class="b-title-box"><a href="?mode=view&amp;articleNo=100&amp;article.offset=0&amp;articleLimit=10"><span class="b-title">조기졸업 안내</span></a><p class="b-file">첨부파일</p></div>
  </td><td>학사지원과</td><td>2026.08.04</td><td>100</td>
</tr>
<tr>
  <td class="b-num-box">9</td>
  <td class="b-td-title b-td-left">
    <div class="b-title-box"><a href="?mode=view&amp;articleNo=101"><span class="b-title">수강신청 안내</span></a></div>
  </td><td>학사지원과</td><td>2026.08.07</td><td>50</td>
</tr>
</tbody></table></div>
"""


def test_parse_notice_list_extracts_metadata():
    items = parse_notice_list_html(LIST_HTML, BOARD_URL, "academic_notice")

    assert [item.article_no for item in items] == ["100", "101"]
    assert items[0].document_id == "notice3-100"
    assert items[0].has_attachment is True
    assert items[0].is_pinned is True
    assert items[1].source_url.startswith(BOARD_URL + "?mode=view")
    assert items[1].has_attachment is False


def test_parse_notice_list_rejects_missing_table():
    with pytest.raises(ValueError, match="공지 목록 행"):
        parse_notice_list_html("<html></html>", BOARD_URL, "academic_notice")


def test_page_url_sets_offset_without_losing_query():
    url = page_url(BOARD_URL + "?search=x", 20, 10)
    assert "search=x" in url
    assert "article.offset=20" in url
    assert "articleLimit=10" in url


def test_collect_board_deduplicates_pinned_posts_across_pages():
    called = []

    def fake_fetcher(url):
        called.append(url)
        return LIST_HTML, url

    items = collect_board(
        "academic_notice", BOARD_URL, pages=2, delay_seconds=0, fetcher=fake_fetcher
    )

    assert len(items) == 2
    assert "article.offset=10" in called[1]


def test_compare_catalogs_finds_new_updated_and_unchanged():
    base = NoticeListItem(
        "notice3-100", "100", "academic_notice", "제목", "학사지원과",
        "2026.08.04", BOARD_URL + "?mode=view&articleNo=100", True, False,
    )
    changed = {**asdict(base), "title": "수정된 제목"}
    same = asdict(base)
    new = {**asdict(base), "document_id": "notice3-101", "article_no": "101"}

    result = compare_catalogs([same], [changed, new])
    assert result == {
        "new": ["notice3-101"],
        "updated": ["notice3-100"],
        "unchanged": [],
    }


def test_collect_board_rejects_excessive_pages():
    with pytest.raises(ValueError, match="1~100"):
        collect_board("academic_notice", BOARD_URL, pages=101)


def test_real_fetch_requires_robots_crawl_delay():
    with pytest.raises(ValueError, match="10초 이상"):
        collect_board("academic_notice", BOARD_URL, pages=1, delay_seconds=1)
