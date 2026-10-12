"""평가 질문을 Gemini RAG에 보내 핵심값·인용·출처와 token을 기록한다."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from sejong_rag.context_builder import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_CHUNKS,
    DEFAULT_RETRIEVAL_TOP_K,
    select_context_chunks,
)
from sejong_rag.evaluate_retrieval import DEFAULT_EVAL, load_cases
from sejong_rag.gemini_rag import (
    DEFAULT_GENERATION_MODEL,
    generate_with_gemini,
    get_api_key,
)
from sejong_rag.hybrid_search import DEFAULT_TOKEN_INDEX, hybrid_search
from sejong_rag.prompt_builder import build_messages
from sejong_rag.vector_search import load_index

DEFAULT_EXPECTATIONS = Path("eval/generation_expectations.jsonl")
DEFAULT_OUTPUT = Path(f"data/evaluation/{DEFAULT_GENERATION_MODEL}-answers.jsonl")


def load_expectations(path: Path) -> dict[str, list[list[str]]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {str(row["id"]): row["answer_concepts"] for row in rows}
    if len(result) != len(rows):
        raise ValueError("생성 평가 expectation id가 중복됐습니다.")
    return result


def evaluate_answer(
    answer: str,
    expected_page: int,
    answer_concepts: list[list[str]],
) -> dict[str, object]:
    normalized = " ".join(answer.lower().split())
    concept_results = [
        any(alternative.lower() in normalized for alternative in alternatives)
        for alternatives in answer_concepts
    ]
    citation_blocks = re.findall(r"\[([^\]]+)\]", answer)
    cited_pages = {
        int(page)
        for block in citation_blocks
        for page in re.findall(r"p\.?\s*(\d+)", block, re.IGNORECASE)
    }
    return {
        "concept_results": concept_results,
        "answer_values_ok": all(concept_results),
        "page_citation_ok": expected_page in cited_pages,
        "official_source_ok": "https://www.sejong.ac.kr/" in answer,
    }


def metric_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        raise ValueError("평가 결과가 없습니다.")
    count = len(rows)
    return {
        "answer_value_accuracy": sum(bool(r["answer_values_ok"]) for r in rows) / count,
        "page_citation_rate": sum(bool(r["page_citation_ok"]) for r in rows) / count,
        "official_source_rate": sum(bool(r["official_source_ok"]) for r in rows) / count,
        "all_automatic_checks_rate": sum(
            bool(r["answer_values_ok"])
            and bool(r["page_citation_ok"])
            and bool(r["official_source_ok"])
            for r in rows
        ) / count,
    }


def usage_value(usage: Any, name: str) -> int | None:
    value = getattr(usage, name, None) if usage is not None else None
    return int(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description="15문항 Gemini RAG 생성 평가")
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--expectations", type=Path, default=DEFAULT_EXPECTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--index", type=Path, default=DEFAULT_TOKEN_INDEX)
    parser.add_argument("--retrieval-top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--neighbor-window", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CONTEXT_CHUNKS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    args = parser.parse_args()

    api_key = get_api_key()
    cases = load_cases(args.eval)
    expectations = load_expectations(args.expectations)
    missing = [str(case["id"]) for case in cases if str(case["id"]) not in expectations]
    if missing:
        raise ValueError("생성 평가 expectation이 없습니다: " + ", ".join(missing))

    embeddings, chunks, metadata = load_index(args.index)
    from google import genai
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(str(metadata["model"]), device="cpu")
    client = genai.Client(api_key=api_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.output.open("w", encoding="utf-8") as output:
        for number, case in enumerate(cases, start=1):
            identifier = str(case["id"])
            query = str(case["query"])
            query_embedding = embedding_model.encode(
                f"{metadata['query_prefix']}{query}",
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            results = hybrid_search(
                chunks, embeddings, query_embedding, query,
                top_k=args.retrieval_top_k, candidate_k=args.candidate_k,
            )
            context = select_context_chunks(
                results, chunks,
                neighbor_window=args.neighbor_window,
                max_chunks=args.max_chunks,
                max_chars=args.max_chars,
            )
            messages = build_messages(query, context)
            answer, usage = generate_with_gemini(client, messages, model=args.model)
            checks = evaluate_answer(
                answer, int(case["expected_page"]), expectations[identifier]
            )
            row = {
                "id": identifier,
                "query": query,
                "expected_page": case["expected_page"],
                "answer": answer,
                **checks,
                "prompt_tokens": usage_value(usage, "prompt_token_count"),
                "answer_tokens": usage_value(usage, "candidates_token_count"),
                "total_tokens": usage_value(usage, "total_token_count"),
                "context_chunk_ids": [chunk["chunk_id"] for chunk in context],
                "manual_review": None,
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            passed = all(
                bool(row[key])
                for key in ("answer_values_ok", "page_citation_ok", "official_source_ok")
            )
            print(
                f"[{number}/{len(cases)}] {'PASS' if passed else 'CHECK'} "
                f"{identifier} tokens={row['total_tokens']}"
            )

    summary = metric_summary(rows)
    print("\n결과 파일:", args.output)
    for name, value in summary.items():
        print(f"{name}: {value:.4f}")
    print("total_tokens:", sum(int(row["total_tokens"] or 0) for row in rows))


if __name__ == "__main__":
    main()
