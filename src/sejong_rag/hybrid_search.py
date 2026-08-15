"""키워드 순위와 임베딩 순위를 RRF로 결합하는 하이브리드 검색."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from sejong_rag.keyword_search import make_snippet, search_chunks
from sejong_rag.vector_search import load_index, rank_by_similarity

DEFAULT_TOKEN_INDEX = Path("data/index/notice3-891485-e5-small-token448.npz")


def expand_query_for_keyword(query: str) -> str:
    """사용자 표현을 학교 문서에서 쓰는 최소한의 용어로 확장한다."""
    expanded = query
    if re.search(r"다시\s*(듣|들|수강)", query):
        expanded += " 재수강"
    if re.search(r"(최고|최대)\s*성적", query):
        expanded += " 성적 제한"
    return expanded


def reciprocal_rank_fusion(
    vector_ids: list[str],
    keyword_ids: list[str],
    rrf_k: int = 60,
) -> list[dict[str, object]]:
    """서로 다른 점수 단위 대신 두 결과의 순위를 결합한다."""
    if rrf_k <= 0:
        raise ValueError("rrf_k는 1 이상이어야 합니다.")

    fused: dict[str, dict[str, object]] = {}
    for source, identifiers in (
        ("vector_rank", vector_ids),
        ("keyword_rank", keyword_ids),
    ):
        for rank, chunk_id in enumerate(identifiers, start=1):
            item = fused.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "vector_rank": None,
                    "keyword_rank": None,
                    "rrf_score": 0.0,
                },
            )
            if item[source] is None:
                item[source] = rank
                item["rrf_score"] = float(item["rrf_score"]) + 1 / (
                    rrf_k + rank
                )

    results = list(fused.values())
    results.sort(
        key=lambda item: (
            -float(item["rrf_score"]),
            min(
                rank
                for rank in (item["vector_rank"], item["keyword_rank"])
                if rank is not None
            ),
            str(item["chunk_id"]),
        )
    )
    return results


def hybrid_search(
    chunks: list[dict[str, object]],
    embeddings: object,
    query_embedding: object,
    query: str,
    top_k: int = 5,
    candidate_k: int = 100,
    rrf_k: int = 60,
) -> list[dict[str, object]]:
    if top_k <= 0 or candidate_k <= 0:
        raise ValueError("top_k와 candidate_k는 1 이상이어야 합니다.")

    vector_results = rank_by_similarity(embeddings, query_embedding, candidate_k)
    keyword_query = expand_query_for_keyword(query)
    keyword_results = search_chunks(chunks, keyword_query, top_k=candidate_k)
    vector_ids = [str(chunks[index]["chunk_id"]) for index, _ in vector_results]
    keyword_ids = [str(result["chunk"]["chunk_id"]) for result in keyword_results]
    fused = reciprocal_rank_fusion(vector_ids, keyword_ids, rrf_k=rrf_k)

    by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    vector_scores = {
        str(chunks[index]["chunk_id"]): score for index, score in vector_results
    }
    keyword_scores = {
        str(result["chunk"]["chunk_id"]): float(result["score"])
        for result in keyword_results
    }
    for item in fused:
        chunk_id = str(item["chunk_id"])
        item["chunk"] = by_id[chunk_id]
        item["vector_score"] = vector_scores.get(chunk_id)
        item["keyword_score"] = keyword_scores.get(chunk_id)
        item["expanded_query"] = keyword_query
    return fused[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="수강편람 하이브리드 검색")
    parser.add_argument("query", help="자연어 질문")
    parser.add_argument("--index", type=Path, default=DEFAULT_TOKEN_INDEX)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()

    embeddings, chunks, metadata = load_index(args.index)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(metadata["model"]), device="cpu")
    query_embedding = model.encode(
        f"{metadata['query_prefix']}{args.query}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    results = hybrid_search(
        chunks,
        embeddings,
        query_embedding,
        args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )

    print(f"keyword query: {expand_query_for_keyword(args.query)}")
    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]
        print(
            f"\n[{rank}] rrf={result['rrf_score']:.6f} "
            f"vector_rank={result['vector_rank']} "
            f"keyword_rank={result['keyword_rank']} page={chunk['page']}"
        )
        print(f"chunk_id={chunk['chunk_id']}")
        print(f"source={chunk['source_url']}")
        print("-" * 72)
        print(make_snippet(str(chunk["text"]), [], limit=500))


if __name__ == "__main__":
    main()
