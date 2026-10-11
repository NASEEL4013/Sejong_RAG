"""분류된 공지 후보의 허용된 상세 HTML과 첨부 메타데이터를 수집한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from sejong_rag.classify_notices import DEFAULT_OUTPUT as DEFAULT_CLASSIFIED_CATALOG
from sejong_rag.ingest import Notice, fetch_notice
from sejong_rag.list_notices import BOARD_URLS, MIN_CRAWL_DELAY_SECONDS

DEFAULT_OUTPUT_DIR = Path("data")
ALLOWED_BOARD_PATHS = {urlparse(url).path for url in BOARD_URLS.values()}


def select_candidates(
    payload: dict[str, object],
    priorities: set[str] | None = None,
    categories: set[str] | None = None,
    max_items: int = 5,
) -> list[dict[str, object]]:
    """원본 순서를 유지하며 조건에 맞는 상세 수집 후보를 선택한다."""
    if not 1 <= max_items <= 500:
        raise ValueError("max_items는 서버 보호를 위해 1~500만 허용합니다.")
    selected = []
    for item in payload.get("items", []):
        if priorities and str(item.get("priority")) not in priorities:
            continue
        item_categories = {str(value) for value in item.get("categories", [])}
        if categories and not categories.intersection(item_categories):
            continue
        validate_detail_url(str(item.get("source_url", "")))
        selected.append(item)
        if len(selected) >= max_items:
            break
    return selected


def validate_detail_url(url: str) -> None:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "www.sejong.ac.kr"
        or parsed.path not in ALLOWED_BOARD_PATHS
        or query.get("mode") != ["view"]
        or not query.get("articleNo")
    ):
        raise ValueError(f"허용된 세종대학교 공지 상세 URL이 아닙니다: {url}")


def content_hash(notice: Notice) -> str:
    """조회수·화면 장식을 제외한 실제 공지 내용의 변경 감지 해시다."""
    value = {
        "title": notice.title,
        "department": notice.department,
        "published_at": notice.published_at,
        "body": notice.body,
        "attachments": [asdict(item) for item in notice.attachments],
        "body_images": notice.body_images,
    }
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def save_detail(
    notice: Notice,
    raw_html: bytes,
    classification: dict[str, object],
    output_dir: Path,
) -> str:
    """상세 HTML과 구조화 메타데이터를 저장하고 변경 상태를 반환한다."""
    raw_path = output_dir / "raw" / "notices" / f"{notice.document_id}.html"
    metadata_path = output_dir / "metadata" / "details" / f"{notice.document_id}.json"
    new_hash = content_hash(notice)
    old_hash = None
    if metadata_path.exists():
        old_hash = json.loads(metadata_path.read_text(encoding="utf-8")).get("content_hash")
    status = "new" if old_hash is None else "unchanged" if old_hash == new_hash else "updated"

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw_html)
    metadata = {
        **asdict(notice),
        "attachments": [asdict(item) for item in notice.attachments],
        "categories": classification.get("categories", []),
        "priority": classification.get("priority"),
        "matched_keywords": classification.get("matched_keywords", []),
        "classification_method": classification.get("classification_method"),
        "content_hash": new_hash,
        "change_status": status,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "raw_html_path": str(raw_path),
        "attachment_acquisition": "metadata_only_manual_import_required",
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return status


def collect_details(
    candidates: list[dict[str, object]],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    delay_seconds: float = MIN_CRAWL_DELAY_SECONDS,
    fetcher: Callable[[str], tuple[Notice, bytes]] = fetch_notice,
) -> list[dict[str, object]]:
    """후보 상세 페이지를 정책 간격으로 요청하되 첨부파일은 요청하지 않는다."""
    if delay_seconds < 0:
        raise ValueError("delay_seconds는 0 이상이어야 합니다.")
    if fetcher is fetch_notice and delay_seconds < MIN_CRAWL_DELAY_SECONDS:
        raise ValueError("학교 robots.txt에 따라 실제 요청 간격은 10초 이상이어야 합니다.")

    results = []
    for number, candidate in enumerate(candidates):
        if number and delay_seconds:
            time.sleep(delay_seconds)
        url = str(candidate["source_url"])
        validate_detail_url(url)
        notice, raw_html = fetcher(url)
        if notice.document_id != candidate["document_id"]:
            raise ValueError(
                f"목록과 상세 문서 ID가 다릅니다: {candidate['document_id']} != {notice.document_id}"
            )
        status = save_detail(notice, raw_html, candidate, output_dir)
        results.append(
            {
                "document_id": notice.document_id,
                "status": status,
                "title": notice.title,
                "attachment_count": len(notice.attachments),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="분류된 공지의 상세 HTML만 수집하고 첨부파일은 메타데이터로 기록합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_CLASSIFIED_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--priority", action="append", choices=["high", "normal"])
    parser.add_argument("--category", action="append")
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--delay", type=float, default=MIN_CRAWL_DELAY_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--refresh", action="store_true",
        help="이미 상세 메타데이터가 있는 공지도 다시 요청",
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    candidates = select_candidates(
        payload,
        priorities=set(args.priority or ["high"]),
        categories=set(args.category or []),
        max_items=args.max_items,
    )
    if not args.refresh:
        detail_dir = args.output_dir / "metadata" / "details"
        candidates = [
            item for item in candidates
            if not (detail_dir / f"{item['document_id']}.json").exists()
        ]
    if args.dry_run:
        print(
            json.dumps(
                [
                    {
                        "document_id": item["document_id"],
                        "title": item["title"],
                        "categories": item["categories"],
                        "source_url": item["source_url"],
                    }
                    for item in candidates
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    results = collect_details(candidates, args.output_dir, args.delay)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
