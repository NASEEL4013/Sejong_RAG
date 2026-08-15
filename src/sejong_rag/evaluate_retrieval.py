"""질문별 정답 chunk 순위로 하이브리드 검색 품질을 평가한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sejong_rag.hybrid_search import DEFAULT_TOKEN_INDEX, hybrid_search
from sejong_rag.vector_search import load_index

DEFAULT_EVAL = Path("eval/questions.jsonl")


def normalize_space(text: str) -> str:
    return " ".join(text.split()).lower()


def load_cases(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"평가 질문 파일을 찾을 수 없습니다: {path}")
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identifiers = [str(case["id"]) for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("평가 질문 id가 중복됐습니다.")
    return cases


def relevant_chunk_ids(
    chunks: list[dict[str, object]], case: dict[str, object]
) -> set[str]:
    page = int(case["expected_page"])
    phrases = [normalize_space(str(item)) for item in case["required_phrases"]]
    relevant = set()
    for chunk in chunks:
        text = normalize_space(str(chunk.get("text", "")))
        if int(chunk.get("page", -1)) == page and all(
            phrase in text for phrase in phrases
        ):
            relevant.add(str(chunk["chunk_id"]))
    return relevant


def metrics_from_ranks(ranks: list[int | None], hit_k: int = 5) -> dict[str, float]:
    if not ranks:
        raise ValueError("평가할 순위가 없습니다.")
    hits = sum(rank is not None and rank <= hit_k for rank in ranks)
    reciprocal_sum = sum(1 / rank for rank in ranks if rank is not None)
    return {
        f"hit_rate@{hit_k}": hits / len(ranks),
        "mrr": reciprocal_sum / len(ranks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="하이브리드 검색 자동 평가")
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--index", type=Path, default=DEFAULT_TOKEN_INDEX)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=100)
    args = parser.parse_args()

    embeddings, chunks, metadata = load_index(args.index)
    cases = load_cases(args.eval)
    ground_truth = {
        str(case["id"]): relevant_chunk_ids(chunks, case) for case in cases
    }
    unresolved = [identifier for identifier, ids in ground_truth.items() if not ids]
    if unresolved:
        raise ValueError(
            "정답 문구를 포함한 chunk를 찾지 못했습니다: " + ", ".join(unresolved)
        )

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(metadata["model"]), device="cpu")
    ranks: list[int | None] = []
    rows = []
    for case in cases:
        query = str(case["query"])
        query_embedding = model.encode(
            f"{metadata['query_prefix']}{query}",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        results = hybrid_search(
            chunks,
            embeddings,
            query_embedding,
            query,
            top_k=args.candidate_k,
            candidate_k=args.candidate_k,
        )
        relevant = ground_truth[str(case["id"])]
        rank = next(
            (
                position
                for position, result in enumerate(results, start=1)
                if str(result["chunk_id"]) in relevant
            ),
            None,
        )
        ranks.append(rank)
        rows.append((str(case["id"]), rank, int(case["expected_page"]), query))

    for identifier, rank, page, query in rows:
        hit = rank is not None and rank <= args.top_k
        print(
            f"{'HIT ' if hit else 'MISS'} id={identifier} "
            f"rank={rank or '-'} page={page} query={query}"
        )
    metrics = metrics_from_ranks(ranks, hit_k=args.top_k)
    print("\n질문 수:", len(cases))
    print(f"Hit Rate@{args.top_k}: {metrics[f'hit_rate@{args.top_k}']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")


if __name__ == "__main__":
    main()
