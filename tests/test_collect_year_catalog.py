import pytest

from sejong_rag.collect_year_catalog import collect_year_board, published_year


URL = "https://www.sejong.ac.kr/kor/intro/notice3.do"


def html(rows):
    body = []
    for article, date, title, pinned in rows:
        body.append(f"""
        <tr class="{'b-top-box' if pinned else ''}">
          <td class="b-num-box">{'공지' if pinned else article}</td>
          <td class="b-td-title"><div class="b-title-box"><a href="?mode=view&amp;articleNo={article}"><span class="b-title">{title}</span></a></div></td>
          <td>학사지원과</td><td>{date}</td><td>1</td>
        </tr>""")
    return "<div class='b-list-common-wrap'><table><tbody>" + "".join(body) + "</tbody></table></div>"


def test_published_year_validates_format():
    assert published_year("2026.08.01") == 2026
    with pytest.raises(ValueError):
        published_year("2026-08-01")


def test_collect_year_stops_after_older_regular_page_and_deduplicates_pinned():
    pages = [
        html([("100", "2026.08.01", "고정", True), ("101", "2026.07.01", "현재", False)]),
        html([("100", "2026.08.01", "고정", True), ("90", "2025.12.31", "과거", False)]),
    ]
    calls = []

    def fake_fetcher(url):
        calls.append(url)
        return pages[len(calls) - 1], url

    items, count = collect_year_board(
        "academic_notice", URL, 2026, delay_seconds=0, fetcher=fake_fetcher
    )
    assert count == 2
    assert {item.article_no for item in items} == {"100", "101"}


def test_collect_year_rejects_non_descending_page():
    page = html([
        ("1", "2026.01.01", "하나", False),
        ("2", "2026.02.01", "둘", False),
    ])
    with pytest.raises(ValueError, match="내림차순"):
        collect_year_board(
            "academic_notice", URL, 2026, delay_seconds=0,
            fetcher=lambda url: (page, url),
        )


@pytest.mark.parametrize("delay", [-1, float("nan"), float("inf")])
def test_collection_rejects_invalid_delay(delay):
    with pytest.raises(ValueError, match="0 이상"):
        collect_year_board("academic_notice", URL, 2026, delay_seconds=delay)


def test_zero_delay_does_not_sleep(monkeypatch):
    monkeypatch.setattr("sejong_rag.collect_year_catalog.time.sleep",
                        lambda seconds: pytest.fail("zero delay must not sleep"))
    page = html([("90", "2025.12.31", "과거", False)])
    items, count = collect_year_board("academic_notice", URL, 2026,
        delay_seconds=0, wait_before_first=True, fetcher=lambda url: (page, url))
    assert items == [] and count == 1
