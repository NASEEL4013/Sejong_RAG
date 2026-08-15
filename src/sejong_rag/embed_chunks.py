"""청크를 E5 임베딩으로 변환해 로컬 NumPy 인덱스를 만든다."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sejong_rag.keyword_search import DEFAULT_CHUNKS, load_chunks

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_INDEX = Path("data/index/notice3-891485-e5-small.npz")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_passages(chunks: list[dict[str, object]]) -> list[str]:
    return [f"passage: {chunk.get('text', '')}" for chunk in chunks]


def token_length_stats(model: object, passages: list[str]) -> dict[str, int]:
    tokenizer = model.tokenizer
    lengths = [
        len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        for text in passages
    ]
    limit = int(model.max_seq_length)
    return {
        "minimum": min(lengths, default=0),
        "maximum": max(lengths, default=0),
        "model_limit": limit,
        "truncated_count": sum(length > limit for length in lengths),
    }


def save_index(
    output: Path,
    embeddings: np.ndarray,
    chunks: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        chunks=np.asarray(
            [json.dumps(chunk, ensure_ascii=False) for chunk in chunks]
        ),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="수강편람 chunk 임베딩 생성")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    chunks = load_chunks(args.chunks)
    if not chunks:
        raise ValueError("임베딩할 chunk가 없습니다.")

    print(f"모델 로딩: {args.model}")
    model = SentenceTransformer(args.model, device="cpu")
    passages = build_passages(chunks)
    stats = token_length_stats(model, passages)
    print(
        "토큰 길이: "
        f"min={stats['minimum']}, max={stats['maximum']}, "
        f"limit={stats['model_limit']}, "
        f"잘림 예상={stats['truncated_count']}/{len(chunks)}"
    )

    embeddings = model.encode(
        passages,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    metadata = {
        "model": args.model,
        "dimension": int(embeddings.shape[1]),
        "chunk_count": len(chunks),
        "normalized": True,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "source_chunks": str(args.chunks),
        "source_sha256": file_sha256(args.chunks),
        "token_lengths": stats,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_index(args.output, embeddings, chunks, metadata)
    print(f"저장 완료: {args.output}")
    print(f"벡터 행렬: {embeddings.shape[0]} x {embeddings.shape[1]}")


if __name__ == "__main__":
    main()
