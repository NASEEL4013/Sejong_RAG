"""JSONL chunk를 대상으로 하는 단순 키워드 검색 baseline."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_CHUNKS = Path(
    "data/processed/chunks/notice3-891485-attachment-236669.jsonl"
)
TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")
STOPWORDS = {
    "은", "는", "이", "가", "을", "를", "의", "에", "에서",
    "으로", "로", "와", "과", "그리고", "또는", "무엇", "뭐야",
    "알려줘", "어떻게", "되나요", "인가요",
}
PARTICLE_SUFFIXES = (
    "에서는", "으로는", "에게서", "에서", "으로", "에게",
    "부터", "까지", "의", "은", "는", "을", "를",
)


def normalize_token(token: str) -> str:
    """명확한 한국어 조사만 제거해 검색어 형태를 맞춘다."""
    token = token.lower()
    for suffix in PARTICLE_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 2:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    """한국어·영문·숫자 토큰을 뽑고 간단히 정규화한다."""
    tokens = [
        normalize_token(token)
        for token in TOKEN_PATTERN.findall(text)
    ]
    return [
        token
        for token in tokens
        if token not in STOPWORDS and (len(token) > 1 or token.isdigit())
    ]


def load_chunks(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"chunk 파일을 찾을 수 없습니다: {path}")

    chunks = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            chunks.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path}의 {line_number}번째 줄이 올바른 JSON이 아닙니다."
            ) from error
    return chunks


def build_document_frequency(
    tokenized_documents: Iterable[list[str]],
) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    for tokens in tokenized_documents:
        frequencies.update(set(tokens))
    return frequencies


def search_chunks(
    chunks: list[dict[str, object]],
    query: str,
    top_k: int = 5,
) -> list[dict[str, object]]:
    """질문 토큰의 TF-IDF 가중 합으로 chunk 순위를 계산한다."""
    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    query_tokens = list(dict.fromkeys(tokenize(query)))
    if not query_tokens:
        return []

    tokenized = [tokenize(str(chunk.get("text", ""))) for chunk in chunks]
    document_frequency = build_document_frequency(tokenized)
    document_count = len(chunks)
    results: list[dict[str, object]] = []

    for chunk, tokens in zip(chunks, tokenized):
        term_frequency = Counter(tokens)
        matched_terms = [
            term for term in query_tokens if term_frequency[term] > 0
        ]
        if not matched_terms:
            continue

        term_scores = {}
        for term in matched_terms:
            inverse_document_frequency = (
                math.log(
                    (document_count + 1)
                    / (document_frequency[term] + 1)
                )
                + 1
            )
            term_scores[term] = inverse_document_frequency * (
                1 + math.log(term_frequency[term])
            )

        coverage = len(matched_terms) / len(query_tokens)
        score = sum(term_scores.values()) * (1 + coverage)
        results.append(
            {
                "score": round(score, 6),
                "coverage": round(coverage, 4),
                "matched_terms": matched_terms,
                "term_scores": {
                    key: round(value, 6)
                    for key, value in term_scores.items()
                },
                "chunk": chunk,
            }
        )

    results.sort(
        key=lambda item: (
            -float(item["score"]),
            int(item["chunk"].get("page", 0)),
            str(item["chunk"].get("chunk_id", "")),
        )
    )
    return results[:top_k]


def make_snippet(
    text: str,
    matched_terms: list[str],
    limit: int = 500,
) -> str:
    """첫 일치어 주변을 터미널에서 읽기 좋은 길이로 자른다."""
    positions = [
        text.lower().find(term)
        for term in matched_terms
        if text.lower().find(term) >= 0
    ]
    anchor = positions[0] if positions else 0
    start = max(0, anchor - 120)
    snippet = text[start : start + limit].strip()
    return re.sub(r"\n{3,}", "\n\n", snippet)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="세종대학교 수강편람 chunk 키워드 검색"
    )
    parser.add_argument("query", help="검색할 질문 또는 키워드")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    results = search_chunks(chunks, args.query, top_k=args.top_k)
    if not results:
        print("일치하는 chunk를 찾지 못했습니다.")
        return

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]
        print(f"\n[{rank}] score={result['score']} page={chunk['page']}")
        print(f"matched={', '.join(result['matched_terms'])}")
        print(f"chunk_id={chunk['chunk_id']}")
        print(f"source={chunk['source_url']}")
        print("-" * 72)
        print(
            make_snippet(
                str(chunk["text"]),
                list(result["matched_terms"]),
            )
        )


if __name__ == "__main__":
    main()
