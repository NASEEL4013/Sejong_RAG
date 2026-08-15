"""학교 허가 범위에서 공식 첨부파일을 10초 간격으로 자동 다운로드한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import parse_qs, urlparse

import requests

from sejong_rag.import_attachments import DEFAULT_DETAILS_DIR, load_expected_attachments
from sejong_rag.attachment_checklist import imported_attachment_ids
from sejong_rag.ingest import USER_AGENT
from sejong_rag.list_notices import BOARD_URLS
from sejong_rag.list_notices import MIN_CRAWL_DELAY_SECONDS

DEFAULT_OUTPUT_DIR = Path("data")
MAX_FILE_BYTES = 200 * 1024 * 1024
KNOWN_BOARD_PATHS = {urlparse(url).path for url in BOARD_URLS.values()}
AUTHORIZATION_BASIS = (
    "학교 측이 개인 학습 목적 자동 다운로드를 허용했으며 요청 간 10초 준수를 요구함"
)


def validate_attachment_url(item: dict[str, object]) -> None:
    url = str(item.get("download_url", ""))
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "www.sejong.ac.kr"
        or parsed.path not in KNOWN_BOARD_PATHS
        or query.get("mode") != ["download"]
        or query.get("articleNo") is None
        or query.get("attachNo") != [str(item.get("attach_no"))]
    ):
        raise ValueError(f"공식 상세 메타데이터와 일치하지 않는 첨부 URL입니다: {url}")


def attachment_paths(item: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    suffix = "." + str(item.get("file_type") or "bin").lower().lstrip(".")
    parent = str(item["parent_document_id"])
    attachment_id = str(item["attachment_id"])
    raw_path = output_dir / "raw" / "auto-attachments" / parent / f"{attachment_id}{suffix}"
    metadata_path = output_dir / "metadata" / "downloaded-attachments" / f"{attachment_id}.json"
    return raw_path, metadata_path


def already_downloaded(item: dict[str, object], output_dir: Path) -> bool:
    raw_path, metadata_path = attachment_paths(item, output_dir)
    return (raw_path.is_file() and metadata_path.is_file()) or (
        str(item["attachment_id"]) in imported_attachment_ids(output_dir)
    )


def stream_download(
    url: str,
    destination: Path,
    max_bytes: int = MAX_FILE_BYTES,
    requester: Callable[..., object] = requests.get,
) -> tuple[str, int, str | None]:
    """응답을 임시 파일에 쓰고 성공 시에만 최종 이름으로 바꾼다."""
    response = requester(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=120,
        stream=True,
    )
    response.raise_for_status()
    declared = response.headers.get("content-length")
    if declared and int(declared) > max_bytes:
        raise ValueError(f"첨부파일이 크기 제한을 초과합니다: {declared} bytes")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with temporary.open("wb") as output:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if not block:
                    continue
                size += len(block)
                if size > max_bytes:
                    raise ValueError(f"첨부파일이 크기 제한 {max_bytes} bytes를 초과합니다.")
                digest.update(block)
                output.write(block)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), size, response.headers.get("content-type")


def save_download_metadata(
    item: dict[str, object],
    raw_path: Path,
    metadata_path: Path,
    content_hash: str,
    size: int,
    content_type: str | None,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "document_id": item["attachment_id"],
        "parent_document_id": item["parent_document_id"],
        "parent_title": item["parent_title"],
        "source_url": item["source_url"],
        "attachment_url": item["download_url"],
        "attach_no": item["attach_no"],
        "filename": item["filename"],
        "file_type": item["file_type"],
        "raw_path": str(raw_path),
        "content_hash": content_hash,
        "size_bytes": size,
        "content_type": content_type,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "acquisition_method": "school_authorized_educational_auto_download",
        "authorization_basis": AUTHORIZATION_BASIS,
        "minimum_request_interval_seconds": MIN_CRAWL_DELAY_SECONDS,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def download_all(
    items: Iterable[dict[str, object]],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    delay_seconds: float = MIN_CRAWL_DELAY_SECONDS,
    max_bytes: int = MAX_FILE_BYTES,
    downloader: Callable[..., tuple[str, int, str | None]] = stream_download,
) -> list[dict[str, object]]:
    if delay_seconds < MIN_CRAWL_DELAY_SECONDS and downloader is stream_download:
        raise ValueError("학교 허가 조건에 따라 실제 다운로드 간격은 10초 이상이어야 합니다.")

    results = []
    request_count = 0
    for item in items:
        validate_attachment_url(item)
        raw_path, metadata_path = attachment_paths(item, output_dir)
        if already_downloaded(item, output_dir):
            results.append({"attachment_id": item["attachment_id"], "status": "skipped"})
            continue
        if request_count:
            time.sleep(delay_seconds)
        content_hash, size, content_type = downloader(
            str(item["download_url"]), raw_path, max_bytes=max_bytes
        )
        request_count += 1
        save_download_metadata(
            item, raw_path, metadata_path, content_hash, size, content_type
        )
        results.append(
            {
                "attachment_id": item["attachment_id"],
                "status": "downloaded",
                "size_bytes": size,
                "sha256": content_hash,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="학교 허가 조건에 따라 공식 첨부를 최소 10초 간격으로 다운로드합니다."
    )
    parser.add_argument("--details-dir", type=Path, default=DEFAULT_DETAILS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--delay", type=float, default=MIN_CRAWL_DELAY_SECONDS)
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    items = load_expected_attachments(args.details_dir)
    pending = [item for item in items if not already_downloaded(item, args.output_dir)]
    selected = pending if args.all else pending[: args.max_items]
    if args.dry_run:
        print(json.dumps(selected, ensure_ascii=False, indent=2))
        return
    results = download_all(selected, args.output_dir, args.delay)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
