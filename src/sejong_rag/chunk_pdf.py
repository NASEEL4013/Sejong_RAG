"""저장된 수강편람 PDF를 레이아웃 텍스트와 검색용 chunk로 변환한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pypdf import PdfReader

from sejong_rag.chunking import chunk_page_text


DEFAULT_PDF = Path("data/raw/attachments/2026-2학기 수강편람_20260730.pdf")
DEFAULT_DOCUMENT_ID = "notice3-891485-attachment-236669"
DEFAULT_PARENT_ID = "notice3-891485"
DEFAULT_SOURCE_URL = (
    "https://www.sejong.ac.kr/kor/intro/notice3.do"
    "?mode=view&articleNo=891485&article.offset=0&articleLimit=10"
)
DEFAULT_ATTACHMENT_URL = (
    "https://www.sejong.ac.kr/kor/intro/notice3.do"
    "?mode=download&articleNo=891485&attachNo=236669"
)


def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    document_id: str,
    parent_document_id: str,
    source_url: str,
    attachment_url: str,
    chunk_size: int = 1600,
    chunk_overlap: int = 200,
) -> dict[str, object]:
    """PDF를 페이지별 레이아웃 텍스트와 chunk JSONL로 저장한다."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF를 찾을 수 없습니다: {pdf_path}")

    pages_path = output_dir / "layout-pages" / f"{document_id}.jsonl"
    chunks_path = output_dir / "chunks" / f"{document_id}.jsonl"
    pages_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    chunk_count = 0
    empty_pages = 0

    with (
        pages_path.open("w", encoding="utf-8") as pages_output,
        chunks_path.open("w", encoding="utf-8") as chunks_output,
    ):
        for page_number, page in enumerate(reader.pages, start=1):
            plain_text = (page.extract_text() or "").strip()
            layout_text = (
                page.extract_text(extraction_mode="layout") or ""
            ).strip()
            search_text = layout_text or plain_text
            empty_pages += int(not search_text)

            page_record = {
                "document_id": document_id,
                "parent_document_id": parent_document_id,
                "page": page_number,
                "text": search_text,
                "plain_text": plain_text,
                "extraction_method": (
                    "pypdf_layout" if layout_text else "pypdf_plain"
                ),
                "text_length": len(search_text),
                "extraction_ok": bool(search_text),
                "source_url": source_url,
                "attachment_url": attachment_url,
                "filename": pdf_path.name,
            }
            pages_output.write(
                json.dumps(page_record, ensure_ascii=False) + "\n"
            )

            for page_chunk_index, text in enumerate(
                chunk_page_text(
                    search_text,
                    max_chars=chunk_size,
                    overlap_chars=chunk_overlap,
                    context_lines=5,
                ),
                start=1,
            ):
                chunk_count += 1
                chunk_record = {
                    "chunk_id": (
                        f"{document_id}-page-{page_number}-chunk-"
                        f"{page_chunk_index}"
                    ),
                    "document_id": document_id,
                    "parent_document_id": parent_document_id,
                    "page": page_number,
                    "page_chunk_index": page_chunk_index,
                    "text": text,
                    "text_length": len(text),
                    "source_url": source_url,
                    "attachment_url": attachment_url,
                    "filename": pdf_path.name,
                    "academic_year": 2026,
                    "semester": "fall",
                    "department": "software",
                    "category": "course_guide",
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
                chunks_output.write(
                    json.dumps(chunk_record, ensure_ascii=False) + "\n"
                )

    return {
        "pdf_path": str(pdf_path),
        "pages_path": str(pages_path),
        "chunks_path": str(chunks_path),
        "page_count": len(reader.pages),
        "empty_pages": empty_pages,
        "chunk_count": chunk_count,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="수강편람 PDF를 레이아웃 텍스트와 chunk로 변환합니다."
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--document-id", default=DEFAULT_DOCUMENT_ID)
    parser.add_argument("--parent-document-id", default=DEFAULT_PARENT_ID)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--attachment-url", default=DEFAULT_ATTACHMENT_URL)
    parser.add_argument("--chunk-size", type=int, default=1600)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    result = process_pdf(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        document_id=args.document_id,
        parent_document_id=args.parent_document_id,
        source_url=args.source_url,
        attachment_url=args.attachment_url,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
