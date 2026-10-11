"""게시일을 기준으로 특정 연도의 공식 공지 목록 전체를 수집한다."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sejong_rag.list_notices import (
    BOARD_URLS,
    MIN_CRAWL_DELAY_SECONDS,
    NoticeListItem,
    fetch_list_page,
    page_url,
    parse_notice_list_html,
)

DEFAULT_OUTPUT = Path("data/catalog/notices-2026.json")


def published_date(value: str) -> datetime:
    try:
        return datetime.strptime(value.strip(), "%Y.%m.%d")
    except ValueError as error:
        raise ValueError(f"지원하지 않는 공지 게시일 형식입니다: {value!r}") from error


def published_year(value: str) -> int:
    return published_date(value).year


def collect_year_board(
    board: str,
    board_url: str,
    year: int,
    article_limit: int = 10,
    max_pages: int = 100,
    delay_seconds: float = MIN_CRAWL_DELAY_SECONDS,
    fetcher: Callable[[str], tuple[str, str]] = fetch_list_page,
    wait_before_first: bool = False,
) -> tuple[list[NoticeListItem], int]:
    """일반 게시물의 날짜가 목표 연도보다 과거가 될 때까지 순회한다."""
    if not 2000 <= year <= 2100:
        raise ValueError("year는 2000~2100만 허용합니다.")
    if not 1 <= max_pages <= 500:
        raise ValueError("max_pages는 서버 보호를 위해 1~500만 허용합니다.")
    if not math.isfinite(delay_seconds) or delay_seconds < 0:
        raise ValueError("delay_seconds는 유한한 0 이상의 값이어야 합니다.")

    selected: dict[str, NoticeListItem] = {}
    pages_fetched = 0
    reached_older = False
    for page_number in range(max_pages):
        if delay_seconds and (page_number or wait_before_first):
            time.sleep(delay_seconds)
        requested = page_url(board_url, page_number * article_limit, article_limit)
        html, final_url = fetcher(requested)
        pages_fetched += 1
        items = parse_notice_list_html(html, final_url, board)
        regular = [item for item in items if not item.is_pinned]
        if not regular:
            raise ValueError(f"{board} {page_number + 1}페이지에 일반 게시물이 없습니다.")

        regular_dates = [published_date(item.published_at) for item in regular]
        regular_years = [value.year for value in regular_dates]
        if regular_dates != sorted(regular_dates, reverse=True):
            raise ValueError(
                f"{board} 목록이 게시일 내림차순이 아닙니다: "
                f"{[value.strftime('%Y.%m.%d') for value in regular_dates]}"
            )
        for item in items:
            if published_year(item.published_at) == year:
                selected[item.document_id] = item

        if max(regular_years) < year:
            reached_older = True
            break
    if not reached_older:
        raise RuntimeError(
            f"{board}가 {max_pages}페이지 안에서 {year - 1}년 이전 구간에 도달하지 못했습니다."
        )
    return list(selected.values()), pages_fetched


def collect_year_catalog(
    year: int,
    boards: list[str],
    max_pages: int = 100,
    delay_seconds: float = MIN_CRAWL_DELAY_SECONDS,
    fetcher: Callable[[str], tuple[str, str]] = fetch_list_page,
) -> tuple[list[NoticeListItem], dict[str, int]]:
    all_items: dict[str, NoticeListItem] = {}
    page_counts = {}
    for number, board in enumerate(boards):
        items, pages = collect_year_board(
            board,
            BOARD_URLS[board],
            year,
            max_pages=max_pages,
            delay_seconds=delay_seconds,
            fetcher=fetcher,
            wait_before_first=number > 0,
        )
        page_counts[board] = pages
        for item in items:
            all_items[item.document_id] = item
    return list(all_items.values()), page_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="특정 게시 연도의 공식 공지 목록 전체 수집")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument(
        "--board", action="append", choices=sorted(BOARD_URLS),
        help="반복 지정 가능. 기본값은 학사공지와 장학공지",
    )
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--delay", type=float, default=MIN_CRAWL_DELAY_SECONDS,
                        help="목록 요청 사이 대기 초. 명시적으로 0을 주면 대기 없이 순차 요청")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    boards = args.board or ["academic_notice", "scholarship_notice"]
    output = args.output or Path(f"data/catalog/notices-{args.year}.json")
    items, page_counts = collect_year_catalog(
        args.year, boards, args.max_pages, args.delay
    )
    payload = {
        "year": args.year,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "boards": boards,
        "page_counts": page_counts,
        "item_count": len(items),
        "items": [asdict(item) for item in items],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "item_count": len(items), "page_counts": page_counts
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
