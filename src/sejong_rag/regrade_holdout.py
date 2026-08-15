"""저장된 홀드아웃 답변과 재시도 결과를 최신 기준으로 비용 없이 재채점한다."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from sejong_rag.evaluate_generation import load_expectations
from sejong_rag.evaluate_unified import load_cases
from sejong_rag.evaluate_unified_generation import evaluate_unified_answer,summary
from sejong_rag.vector_search import load_index
DEFAULT_BASE=Path("data/evaluation/holdout-gemini-answers.jsonl")
DEFAULT_RETRY=Path("data/evaluation/holdout-gemini-retry.jsonl")
DEFAULT_OUTPUT=Path("data/evaluation/holdout-gemini-final.jsonl")

def load_rows(path: Path) -> dict[str,dict[str,object]]:
    if not path.is_file(): return {}
    return {str(row["id"]):row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}

def synthetic_chunk(chunk_id: str) -> dict[str,object] | None:
    if chunk_id.startswith("verified-credit-") or chunk_id.startswith("verified-graduation-"):
        return {"chunk_id":chunk_id,"parent_document_id":"notice3-891086"}
    return None

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--base",type=Path,default=DEFAULT_BASE); p.add_argument("--retry",type=Path,default=DEFAULT_RETRY); p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); a=p.parse_args()
    cases=load_cases(Path("eval/holdout_questions.jsonl")); by_case={str(c["id"]):c for c in cases}; expectations=load_expectations(Path("eval/holdout_generation_expectations.jsonl"))
    rows=load_rows(a.base); rows.update(load_rows(a.retry)); _,chunks,_=load_index(Path("data/index/sejong-2026-trusted-e5-small.npz")); by_chunk={str(c["chunk_id"]):c for c in chunks}
    final=[]
    for case in cases:
        identifier=str(case["id"]); row=dict(rows[identifier]); context=[]
        for chunk_id in row.get("context_chunk_ids",[]):
            chunk=by_chunk.get(str(chunk_id)) or synthetic_chunk(str(chunk_id))
            if chunk: context.append(chunk)
        checks=evaluate_unified_answer(str(row.get("answer","")),expectations[identifier],context,str(case["expected_document_id"]))
        row.update(checks); row.pop("error",None); final.append(row)
    a.output.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in final),encoding="utf-8")
    print(json.dumps(summary(final),ensure_ascii=False,indent=2))
    for row in final:
        if not all(bool(row[key]) for key in ("answer_values_ok","expected_source_in_context","expected_source_cited","citation_numbers_valid")): print("CHECK",row["id"],row["concept_results"],row["expected_source_cited"],row["citation_numbers_valid"])
if __name__=="__main__": main()
