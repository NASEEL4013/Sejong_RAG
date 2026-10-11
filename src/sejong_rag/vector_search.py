"""저장된 임베딩과 질문 벡터의 코사인 유사도로 청크를 검색한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sejong_rag.embed_chunks import DEFAULT_INDEX, file_sha256
from sejong_rag.keyword_search import make_snippet


def load_index(path: Path) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"벡터 인덱스를 찾을 수 없습니다: {path}")
    with np.load(path, allow_pickle=False) as data:
        embeddings = data["embeddings"]
        chunks = [json.loads(item) for item in data["chunks"].tolist()]
        metadata = json.loads(data["metadata"].item())
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("벡터 행과 chunk 수가 일치하지 않습니다.")
    return embeddings, chunks, metadata


def rank_by_similarity(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int,
) -> list[tuple[int, float]]:
    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if embeddings.ndim != 2 or embeddings.shape[1] != query.shape[0]:
        raise ValueError("문서 벡터와 질문 벡터의 차원이 다릅니다.")
    scores = embeddings @ query
    order = np.argsort(-scores, kind="stable")[: min(top_k, len(scores))]
    return [(int(index), float(scores[index])) for index in order]


def chunk_locations(chunk):
    locators = chunk.get('locators') or [s.get('locator', {}) for s in chunk.get('source_spans', [])]
    if not locators and 'page' in chunk:
        locators = [{'page': chunk['page']}]
    values = []
    for loc in locators:
        label = ', '.join(f'{key}={loc[key]}' for key in
            ('zip_member','page','sheet','table','row','slide','section','image') if key in loc)
        if label and label not in values: values.append(label)
    return '; '.join(values[:4]) or 'document'


def main() -> None:
    parser = argparse.ArgumentParser(description="수강편람 벡터 검색")
    parser.add_argument("query", help="자연어 질문")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    embeddings, chunks, metadata = load_index(args.index)
    source = Path(str(metadata["source_chunks"]))
    if source.is_file() and file_sha256(source) != metadata["source_sha256"]:
        raise ValueError("chunk 파일이 임베딩 생성 후 변경됐습니다. 인덱스를 다시 만드세요.")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(metadata["model"]), device="cpu", local_files_only=args.local_files_only)
    query_embedding = model.encode(
        f"{metadata['query_prefix']}{args.query}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    results = rank_by_similarity(embeddings, query_embedding, args.top_k)

    for rank, (index, score) in enumerate(results, start=1):
        chunk = chunks[index]
        print(f"\n[{rank}] cosine={score:.6f} location={chunk_locations(chunk)}")
        print(f"chunk_id={chunk['chunk_id']}")
        print(f"title={chunk.get('document_title', '')}")
        print(f"source={chunk['source_url']}")
        for alias in chunk.get('source_aliases', []):
            print(f"linked_notice={alias.get('document_title', '')} date={alias.get('published_at', '')} source={alias.get('source_url', '')}")
        print("-" * 72)
        print(make_snippet(str(chunk["text"]), [], limit=500))


if __name__ == "__main__":
    main()
