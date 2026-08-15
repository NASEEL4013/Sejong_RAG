"""임베딩 모델의 토큰 한도에 맞춰 페이지 JSONL을 다시 청킹한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from sejong_rag.chunk_pdf import DEFAULT_DOCUMENT_ID
from sejong_rag.embed_chunks import DEFAULT_MODEL

DEFAULT_PAGES = Path(
    "data/processed/layout-pages/notice3-891485-attachment-236669.jsonl"
)
DEFAULT_OUTPUT = Path(
    "data/processed/token-chunks/notice3-891485-attachment-236669.jsonl"
)


def window_token_spans(
    offsets: list[tuple[int, int]],
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, int, int]]:
    """토큰 offset을 겹치는 원문 문자 범위로 변환한다."""
    if max_tokens <= 0:
        raise ValueError("max_tokens는 1 이상이어야 합니다.")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens는 0 이상 max_tokens 미만이어야 합니다.")
    meaningful = [(start, end) for start, end in offsets if end > start]
    if not meaningful:
        return []

    spans = []
    step = max_tokens - overlap_tokens
    for token_start in range(0, len(meaningful), step):
        window = meaningful[token_start : token_start + max_tokens]
        if not window:
            break
        spans.append((window[0][0], window[-1][1], len(window)))
        if token_start + max_tokens >= len(meaningful):
            break
    return spans


def load_jsonl(path: Path) -> Iterable[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"페이지 JSONL을 찾을 수 없습니다: {path}")
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}의 {line_number}번째 줄이 올바른 JSON이 아닙니다."
                ) from error


def token_chunk_pages(
    pages_path: Path,
    output_path: Path,
    tokenizer: object,
    max_tokens: int = 448,
    overlap_tokens: int = 64,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_count = 0
    maximum_tokens = 0
    page_count = 0

    with output_path.open("w", encoding="utf-8") as output:
        for page in load_jsonl(pages_path):
            page_count += 1
            text = str(page.get("text", "")).strip()
            if not text:
                continue
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
            )
            offsets = [tuple(item) for item in encoded["offset_mapping"]]
            spans = window_token_spans(offsets, max_tokens, overlap_tokens)
            for page_chunk_index, (start, end, token_count) in enumerate(
                spans, start=1
            ):
                chunk_count += 1
                maximum_tokens = max(maximum_tokens, token_count)
                chunk_text = text[start:end].strip()
                record = {
                    "chunk_id": (
                        f"{page['document_id']}-page-{page['page']}-"
                        f"token-chunk-{page_chunk_index}"
                    ),
                    "document_id": page["document_id"],
                    "parent_document_id": page["parent_document_id"],
                    "page": page["page"],
                    "page_chunk_index": page_chunk_index,
                    "text": chunk_text,
                    "text_length": len(chunk_text),
                    "token_count": token_count,
                    "source_url": page["source_url"],
                    "attachment_url": page["attachment_url"],
                    "filename": page["filename"],
                    "academic_year": 2026,
                    "semester": "fall",
                    "department": "software",
                    "category": "course_guide",
                    "chunk_method": "e5_token_window",
                    "chunk_max_tokens": max_tokens,
                    "chunk_overlap_tokens": overlap_tokens,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "pages_path": str(pages_path),
        "chunks_path": str(output_path),
        "page_count": page_count,
        "chunk_count": chunk_count,
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
        "maximum_actual_tokens": maximum_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E5 토큰 기준 PDF 페이지 청킹")
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=448)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    result = token_chunk_pages(
        args.pages,
        args.output,
        tokenizer,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
