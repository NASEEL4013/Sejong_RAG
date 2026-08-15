"""세종대학교 공식 공지 목록을 발견하고 이전 목록과 변경점을 비교한다."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from sejong_rag.ingest import USER_AGENT


BOARD_URLS = {
    "general_notice": "https://www.sejong.ac.kr/kor/intro/notice1.do",
    "academic_notice": "https://www.sejong.ac.kr/kor/intro/notice3.do",
    "scholarship_notice": "https://www.sejong.ac.kr/kor/intro/notice7.do",
}
DEFAULT_CATALOG = Path("data/catalog/notice-index.json")
MIN_CRAWL_DELAY_SECONDS = 10.0


@dataclass(frozen=True)
class NoticeListItem:
    document_id: str
    article_no: str
    board: str
    title: str
    department: str
    published_at: str
    source_url: str
    has_attachment: bool
    is_pinned: bool


def _query_value(url: str, name: str) -> str | None:
    values = parse_qs(urlparse(url).query).get(name)
    return values[0] if values else None


def page_url(board_url: str, offset: int, article_limit: int = 10) -> str:
    """기존 query를 보존하면서 목록 페이지 offset을 명시한다."""
    if offset < 0 or article_limit <= 0:
        raise ValueError("offset은 0 이상, article_limit은 1 이상이어야 합니다.")
    parsed = urlparse(board_url)
    query = parse_qs(parsed.query)
    query.update(
        {
            "mode": ["list"],
            "article.offset": [str(offset)],
            "articleLimit": [str(article_limit)],
        }
    )
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def parse_notice_list_html(
    html: str, board_url: str, board: str
) -> list[NoticeListItem]:
    """게시판 표 안의 공지 행만 구조화한다."""
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select(".b-list-common-wrap tbody tr")
    if not rows:
        raise ValueError("공지 목록 행(.b-list-common-wrap tbody tr)을 찾지 못했습니다.")

    expected = urlparse(board_url)
    items: list[NoticeListItem] = []
    seen: set[str] = set()
    for row in rows:
        link = row.select_one(".b-title-box a[href]")
        if link is None:
            continue
        source_url = urljoin(board_url, link["href"])
        parsed = urlparse(source_url)
        article_no = _query_value(source_url, "articleNo")
        if (
            not article_no
            or _query_value(source_url, "mode") != "view"
            or parsed.hostname != expected.hostname
            or parsed.path != expected.path
            or article_no in seen
        ):
            continue

        title_node = link.select_one(".b-title")
        cells = row.select(":scope > td")
        mobile = row.select_one(".b-m-con")
        department_node = cells[2] if len(cells) >= 4 else None
        date_node = cells[3] if len(cells) >= 4 else None
        if department_node is None and mobile is not None:
            department_node = mobile.select_one(".b-writer")
        if date_node is None and mobile is not None:
            date_node = mobile.select_one(".b-date")
        if title_node is None or department_node is None or date_node is None:
            raise ValueError(f"게시물 {article_no}의 제목, 부서 또는 등록일이 없습니다.")

        items.append(
            NoticeListItem(
                document_id=f"{Path(expected.path).stem}-{article_no}",
                article_no=article_no,
                board=board,
                title=title_node.get_text(" ", strip=True),
                department=department_node.get_text(" ", strip=True),
                published_at=date_node.get_text(" ", strip=True),
                source_url=source_url,
                has_attachment=row.select_one(".b-file") is not None,
                is_pinned="b-top-box" in (row.get("class") or []),
            )
        )
        seen.add(article_no)
    return items


def fetch_list_page(url: str, timeout: int = 30) -> tuple[str, str]:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return response.text, response.url


def collect_board(
    board: str,
    board_url: str,
    pages: int = 1,
    article_limit: int = 10,
    delay_seconds: float = MIN_CRAWL_DELAY_SECONDS,
    fetcher: Callable[[str], tuple[str, str]] = fetch_list_page,
) -> list[NoticeListItem]:
    """지정한 수의 목록 페이지만 천천히 순회한다."""
    if not 1 <= pages <= 100:
        raise ValueError("pages는 서버 보호를 위해 1~100만 허용합니다.")
    if delay_seconds < 0:
        raise ValueError("delay_seconds는 0 이상이어야 합니다.")
    if fetcher is fetch_list_page and delay_seconds < MIN_CRAWL_DELAY_SECONDS:
        raise ValueError("학교 robots.txt에 따라 실제 요청 간격은 10초 이상이어야 합니다.")

    by_id: dict[str, NoticeListItem] = {}
    for page_number in range(pages):
        if page_number and delay_seconds:
            time.sleep(delay_seconds)
        requested = page_url(board_url, page_number * article_limit, article_limit)
        html, final_url = fetcher(requested)
        for item in parse_notice_list_html(html, final_url, board):
            by_id[item.document_id] = item
    return list(by_id.values())


def item_signature(item: dict[str, object]) -> tuple[object, ...]:
    return tuple(
        item.get(name)
        for name in ("title", "department", "published_at", "source_url", "has_attachment")
    )


def compare_catalogs(
    previous: list[dict[str, object]], current: list[dict[str, object]]
) -> dict[str, list[str]]:
    """현재 수집 범위 안에서 새 항목과 목록 메타데이터 변경을 찾는다."""
    old = {str(item["document_id"]): item for item in previous}
    result = {"new": [], "updated": [], "unchanged": []}
    for item in current:
        document_id = str(item["document_id"])
        if document_id not in old:
            result["new"].append(document_id)
        elif item_signature(old[document_id]) != item_signature(item):
            result["updated"].append(document_id)
        else:
            result["unchanged"].append(document_id)
    return result


def load_previous_catalog(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("items", []))


def save_catalog(
    path: Path,
    items: list[NoticeListItem],
    changes: dict[str, list[str]],
    board_pages: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "board_pages": board_pages,
        "item_count": len(items),
        "changes": changes,
        "items": [asdict(item) for item in items],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="세종대학교 공식 공지 목록을 수집합니다.")
    parser.add_argument(
        "--board",
        action="append",
        choices=sorted(BOARD_URLS),
        help="반복 지정 가능. 생략하면 학사공지와 장학공지를 수집합니다.",
    )
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--delay", type=float, default=MIN_CRAWL_DELAY_SECONDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()

    selected = args.board or ["academic_notice", "scholarship_notice"]
    previous = load_previous_catalog(args.output)
    all_items: list[NoticeListItem] = []
    for board_number, board in enumerate(selected):
        if board_number and args.delay:
            time.sleep(args.delay)
        all_items.extend(
            collect_board(board, BOARD_URLS[board], pages=args.pages, delay_seconds=args.delay)
        )
    current = [asdict(item) for item in all_items]
    changes = compare_catalogs(previous, current)
    save_catalog(args.output, all_items, changes, {board: args.pages for board in selected})
    print(
        json.dumps(
            {
                "output": str(args.output),
                "item_count": len(all_items),
                "new": len(changes["new"]),
                "updated": len(changes["updated"]),
                "unchanged": len(changes["unchanged"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
