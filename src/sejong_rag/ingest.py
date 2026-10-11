"""세종대학교 공지와 PDF 첨부파일을 수집하는 첫 번째 파이프라인."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


DEFAULT_NOTICE_URL = (
    "https://www.sejong.ac.kr/kor/intro/notice3.do"
    "?mode=view&articleNo=891485&article.offset=0&articleLimit=10"
)
USER_AGENT = "SejongUniversityRAG/0.1 (educational project)"

BOARD_BY_PATH = {
    "notice1.do": "general_notice",
    "notice3.do": "academic_notice",
    "notice7.do": "scholarship_notice",
}


@dataclass(frozen=True)
class Attachment:
    filename: str
    download_url: str
    attach_no: str | None
    file_type: str
    version_date: str | None


@dataclass(frozen=True)
class Notice:
    document_id: str
    article_no: str
    title: str
    department: str
    published_at: str
    source_url: str
    board: str
    body: str
    attachments: list[Attachment]
    body_images: list[dict[str, str]] = field(default_factory=list)


def _clean_text(element) -> str:
    """HTML 요소의 문장을 읽기 쉬운 줄 단위 텍스트로 정리한다."""
    lines = (line.strip() for line in element.get_text("\n").splitlines())
    return "\n".join(line for line in lines if line)


def _query_value(url: str, name: str) -> str | None:
    values = parse_qs(urlparse(url).query).get(name)
    return values[0] if values else None


def _version_date(filename: str) -> str | None:
    match = re.search(r"(?<!\d)(20\d{6})(?!\d)", filename)
    if not match:
        return None
    raw = match.group(1)
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def parse_notice_html(html: str, source_url: str) -> Notice:
    """세종대학교 공지 상세 HTML을 구조화된 데이터로 변환한다."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one(".b-view-common-box")
    if container is None:
        raise ValueError("공지 상세 영역(.b-view-common-box)을 찾지 못했습니다.")

    title_node = container.select_one(".b-title")
    info_nodes = container.select(".b-info > li")
    date_node = container.select_one(".b-info .b-date")
    body_node = container.select_one(".b-content-box")
    if not title_node or not info_nodes or not date_node or not body_node:
        raise ValueError("공지 제목, 부서, 게시일 또는 본문을 찾지 못했습니다.")

    article_no = _query_value(source_url, "articleNo")
    if not article_no:
        hidden = container.select_one('input[name="articleNo"]')
        article_no = hidden.get("value") if hidden else None
    if not article_no:
        raise ValueError("게시물 번호(articleNo)를 찾지 못했습니다.")

    attachments: list[Attachment] = []
    for link in container.select(".b-file-box a[href]"):
        filename = link.get_text(" ", strip=True)
        download_url = urljoin(source_url, link["href"])
        suffix = Path(filename).suffix.lower().lstrip(":.") or "unknown"
        attachments.append(
            Attachment(
                filename=filename,
                download_url=download_url,
                attach_no=_query_value(download_url, "attachNo"),
                file_type=suffix,
                version_date=_version_date(filename),
            )
        )

    board_page = Path(urlparse(source_url).path).name
    board = BOARD_BY_PATH.get(board_page)
    if board is None:
        raise ValueError(f"지원하지 않는 세종대학교 게시판입니다: {board_page}")

    return Notice(
        document_id=f"{Path(board_page).stem}-{article_no}",
        article_no=article_no,
        title=title_node.get_text(" ", strip=True),
        department=info_nodes[0].get_text(" ", strip=True),
        published_at=date_node.get_text(" ", strip=True),
        source_url=source_url,
        board=board,
        body=_clean_text(body_node),
        attachments=attachments,
        body_images=[{"url": urljoin(source_url, image.get("data-src") or image.get("src", "")),
                      "alt": image.get("alt", "")} for image in body_node.select("img")
                     if image.get("data-src") or image.get("src")],
    )


def fetch_notice(url: str, timeout: int = 30) -> tuple[Notice, bytes]:
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return parse_notice_html(response.text, response.url), response.content


def _safe_filename(filename: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z가-힣._() -]+", "_", filename).strip()
    if not safe or safe in {".", ".."}:
        raise ValueError(f"안전하지 않은 파일명입니다: {filename!r}")
    return safe


def download_attachment(attachment: Attachment, destination: Path) -> str:
    """첨부파일을 저장하고 내용 변경 감지용 SHA-256 해시를 반환한다."""
    host = urlparse(attachment.download_url).hostname
    if host != "www.sejong.ac.kr":
        raise ValueError(f"허용되지 않은 첨부파일 호스트입니다: {host}")

    response = requests.get(
        attachment.download_url,
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return hashlib.sha256(response.content).hexdigest()


def extract_pdf_pages(pdf_path: Path) -> Iterable[dict[str, object]]:
    """PDF 텍스트를 페이지 단위로 추출한다."""
    reader = PdfReader(str(pdf_path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        yield {
            "page": page_number,
            "text": text,
            "text_length": len(text),
            "extraction_ok": bool(text),
        }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ingest_notice(url: str, output_dir: Path) -> dict[str, object]:
    """공지 하나와 그 안의 수강편람 PDF를 수집한다."""
    notice, raw_html = fetch_notice(url)
    collected_at = datetime.now(timezone.utc).isoformat()

    notice_path = output_dir / "raw" / "notices" / f"{notice.document_id}.html"
    notice_path.parent.mkdir(parents=True, exist_ok=True)
    notice_path.write_bytes(raw_html)

    attachment_records: list[dict[str, object]] = []
    selected = [
        item
        for item in notice.attachments
        if item.file_type == "pdf" and "수강편람" in item.filename
    ]
    if not selected:
        raise ValueError("공지에서 이름에 '수강편람'이 포함된 PDF를 찾지 못했습니다.")

    for attachment in selected:
        filename = _safe_filename(attachment.filename)
        raw_path = output_dir / "raw" / "attachments" / filename
        content_hash = download_attachment(attachment, raw_path)
        attachment_id = f"{notice.document_id}-attachment-{attachment.attach_no}"
        processed_path = output_dir / "processed" / "pdf" / f"{attachment_id}.jsonl"
        processed_path.parent.mkdir(parents=True, exist_ok=True)

        page_count = 0
        empty_pages = 0
        with processed_path.open("w", encoding="utf-8") as output:
            for page in extract_pdf_pages(raw_path):
                page_count += 1
                empty_pages += int(not page["extraction_ok"])
                record = {
                    "document_id": attachment_id,
                    "parent_document_id": notice.document_id,
                    "source_url": notice.source_url,
                    "attachment_url": attachment.download_url,
                    "filename": attachment.filename,
                    **page,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

        attachment_records.append(
            {
                **asdict(attachment),
                "document_id": attachment_id,
                "parent_document_id": notice.document_id,
                "content_hash": content_hash,
                "raw_path": str(raw_path),
                "processed_path": str(processed_path),
                "page_count": page_count,
                "empty_text_pages": empty_pages,
            }
        )

    metadata = {
        **asdict(notice),
        "attachments": [asdict(item) for item in notice.attachments],
        "processed_attachments": attachment_records,
        "collected_at": collected_at,
        "raw_html_path": str(notice_path),
    }
    metadata_path = output_dir / "metadata" / f"{notice.document_id}.json"
    _write_json(metadata_path, metadata)

    return {
        "metadata_path": str(metadata_path),
        "title": notice.title,
        "attachment_count": len(notice.attachments),
        "processed_pdf_count": len(attachment_records),
        "processed_attachments": attachment_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="세종대학교 공지와 수강편람 PDF를 수집합니다."
    )
    parser.add_argument("--url", default=DEFAULT_NOTICE_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    result = ingest_notice(args.url, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
