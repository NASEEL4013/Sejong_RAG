"""검색 결과와 같은 페이지의 인접 chunk를 RAG 근거 문맥으로 조립한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from sejong_rag.hybrid_search import DEFAULT_TOKEN_INDEX, hybrid_search
from sejong_rag.vector_search import load_index

DEFAULT_RETRIEVAL_TOP_K = 3
DEFAULT_MAX_CONTEXT_CHUNKS = 6
DEFAULT_MAX_CONTEXT_CHARS = 6_000


def select_context_chunks(
    results: list[dict[str, object]],
    all_chunks: list[dict[str, object]],
    neighbor_window: int = 1,
    max_chunks: int = DEFAULT_MAX_CONTEXT_CHUNKS,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> list[dict[str, object]]:
    """검색 chunk를 우선하고 같은 페이지의 앞뒤 chunk를 중복 없이 추가한다."""
    if neighbor_window < 0:
        raise ValueError("neighbor_window는 0 이상이어야 합니다.")
    if max_chunks <= 0 or max_chars <= 0:
        raise ValueError("max_chunks와 max_chars는 1 이상이어야 합니다.")

    by_location = {
        (
            str(chunk.get("document_id", "")),
            int(chunk.get("page", -1)),
            int(chunk.get("page_chunk_index", -1)),
        ): chunk
        for chunk in all_chunks
    }
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    used_chars = 0

    def add(chunk: dict[str, object], relation: str, retrieval_rank: int) -> bool:
        nonlocal used_chars
        chunk_id = str(chunk["chunk_id"])
        if chunk_id in selected_ids:
            return True
        text_length = len(str(chunk.get("text", "")))
        if selected and used_chars + text_length > max_chars:
            return False
        if len(selected) >= max_chunks:
            return False
        enriched = dict(chunk)
        enriched["context_relation"] = relation
        enriched["retrieval_rank"] = retrieval_rank
        selected.append(enriched)
        selected_ids.add(chunk_id)
        used_chars += text_length
        return True

    for retrieval_rank, result in enumerate(results, start=1):
        chunk = result["chunk"]
        if not isinstance(chunk, dict):
            continue
        if not add(chunk, "retrieved", retrieval_rank):
            break
        document_id = str(chunk.get("document_id", ""))
        page = int(chunk.get("page", -1))
        index = int(chunk.get("page_chunk_index", -1))
        for distance in range(1, neighbor_window + 1):
            for neighbor_index in (index - distance, index + distance):
                neighbor = by_location.get((document_id, page, neighbor_index))
                if neighbor is not None:
                    add(neighbor, "neighbor", retrieval_rank)
    return selected


def format_context(chunks: list[dict[str, object]]) -> str:
    """LLM과 사람이 출처를 구분할 수 있는 근거 블록 문자열을 만든다."""
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[근거 {number}]",
                    f"페이지: {chunk['page']}",
                    f"청크 ID: {chunk['chunk_id']}",
                    f"선택 이유: {chunk['context_relation']}",
                    f"출처: {chunk['source_url']}",
                    "내용:",
                    str(chunk["text"]),
                ]
            )
        )
    return "\n\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG용 검색 근거 문맥 생성")
    parser.add_argument("query", help="자연어 질문")
    parser.add_argument("--index", type=Path, default=DEFAULT_TOKEN_INDEX)
    parser.add_argument("--retrieval-top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--neighbor-window", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CONTEXT_CHUNKS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
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
        top_k=args.retrieval_top_k,
        candidate_k=args.candidate_k,
    )
    context_chunks = select_context_chunks(
        results,
        chunks,
        neighbor_window=args.neighbor_window,
        max_chunks=args.max_chunks,
        max_chars=args.max_chars,
    )
    print(f"질문: {args.query}")
    print(f"검색 결과: {len(results)}개")
    print(f"최종 context: {len(context_chunks)}개 chunk")
    print(f"본문 문자 수: {sum(len(str(c['text'])) for c in context_chunks)}")
    print("=" * 72)
    print(format_context(context_chunks))


if __name__ == "__main__":
    main()
