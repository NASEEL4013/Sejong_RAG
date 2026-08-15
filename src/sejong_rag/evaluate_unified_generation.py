"""통합 홀드아웃 질문의 Gemini 답변·근거 인용을 평가한다."""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
from sejong_rag.evaluate_generation import load_expectations, usage_value
from sejong_rag.evaluate_unified import load_cases
from sejong_rag.gemini_rag import DEFAULT_GENERATION_MODEL, generate_with_gemini, get_api_key
from sejong_rag.graduation_requirements import credit_structured_chunk, structured_chunk
from sejong_rag.prompt_builder import build_messages
from sejong_rag.unified_context import select_unified_context
from sejong_rag.unified_search import DEFAULT_TRUSTED, search_indexes
from sejong_rag.vector_search import load_index

DEFAULT_EVAL=Path("eval/holdout_questions.jsonl")
DEFAULT_EXPECTATIONS=Path("eval/holdout_generation_expectations.jsonl")
DEFAULT_OUTPUT=Path("data/evaluation/holdout-gemini-answers.jsonl")

def normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()

def parent_id(chunk: dict[str, object]) -> str:
    return str(chunk.get("parent_document_id") or chunk.get("document_id") or "")

def evaluate_unified_answer(answer: str, concepts: list[list[str]], context: list[dict[str, object]], expected_document_id: str) -> dict[str, object]:
    text=normalize(answer)
    concept_results=[any(normalize(option) in text for option in alternatives) for alternatives in concepts]
    expected_numbers={i for i,chunk in enumerate(context,1) if parent_id(chunk)==expected_document_id}
    citation_blocks = re.findall(r"\[([^]\n]+)\]", answer)
    cited = {int(value) for block in citation_blocks for value in re.findall(r"(?:근거\s*)?(\d+)", block)}
    return {
        "concept_results":concept_results,
        "answer_values_ok":all(concept_results),
        "expected_source_in_context":bool(expected_numbers),
        "expected_source_cited":bool(cited.intersection(expected_numbers)),
        "citation_numbers_valid":all(1<=n<=len(context) for n in cited),
        "cited_numbers":sorted(cited),
        "expected_evidence_numbers":sorted(expected_numbers),
    }

def summary(rows: list[dict[str, object]]) -> dict[str,float]:
    completed=[r for r in rows if not r.get("error")]
    if not completed: return {"completed":0.0}
    keys=("answer_values_ok","expected_source_in_context","expected_source_cited","citation_numbers_valid")
    result={key:sum(bool(r[key]) for r in completed)/len(completed) for key in keys}
    result["all_checks"]=sum(all(bool(r[key]) for key in keys) for r in completed)/len(completed)
    result["completed"]=float(len(completed)); result["errors"]=float(len(rows)-len(completed))
    return result

def main() -> None:
    p=argparse.ArgumentParser(description="통합 홀드아웃 Gemini 생성 평가")
    p.add_argument("--eval",type=Path,default=DEFAULT_EVAL); p.add_argument("--expectations",type=Path,default=DEFAULT_EXPECTATIONS); p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--model",default=DEFAULT_GENERATION_MODEL); p.add_argument("--limit",type=int); p.add_argument("--only",help="쉼표로 구분한 평가 ID만 실행"); p.add_argument("--delay",type=float,default=1.0); p.add_argument("--resume",action="store_true")
    a=p.parse_args(); cases=load_cases(a.eval); expectations=load_expectations(a.expectations)
    if a.only:
        selected={value.strip() for value in a.only.split(",") if value.strip()}
        cases=[case for case in cases if str(case["id"]) in selected]
    if a.limit: cases=cases[:a.limit]
    missing=[str(c["id"]) for c in cases if str(c["id"]) not in expectations]
    if missing: raise ValueError("생성 expectation 누락: "+", ".join(missing))
    done={}
    if a.resume and a.output.is_file():
        done={r["id"]:r for r in (json.loads(line) for line in a.output.read_text(encoding="utf-8").splitlines() if line.strip())}
    _,chunks,_=load_index(DEFAULT_TRUSTED)
    from google import genai
    client=genai.Client(api_key=get_api_key()); rows=[]
    for number,case in enumerate(cases,1):
        identifier=str(case["id"])
        if identifier in done: rows.append(done[identifier]); print(f"[{number}/{len(cases)}] SKIP {identifier}"); continue
        try:
            results=search_indexes(str(case["query"]),DEFAULT_TRUSTED,top_k=5,candidate_k=300)
            context=select_unified_context(results,chunks)
            for extra in (credit_structured_chunk(str(case["query"])),structured_chunk(str(case["query"]))):
                if extra is not None: context.append(extra)
            answer,usage=generate_with_gemini(client,build_messages(str(case["query"]),context),model=a.model,max_output_tokens=1200)
            checks=evaluate_unified_answer(answer,expectations[identifier],context,str(case["expected_document_id"]))
            row={"id":identifier,"query":case["query"],"answer":answer,**checks,"context_chunk_ids":[c["chunk_id"] for c in context],"prompt_tokens":usage_value(usage,"prompt_token_count"),"answer_tokens":usage_value(usage,"candidates_token_count"),"total_tokens":usage_value(usage,"total_token_count"),"manual_review":None}
            passed=all(bool(row[k]) for k in ("answer_values_ok","expected_source_in_context","expected_source_cited","citation_numbers_valid"))
            print(f"[{number}/{len(cases)}] {'PASS' if passed else 'CHECK'} {identifier}")
        except Exception as exc:
            row={"id":identifier,"query":case["query"],"error":f"{type(exc).__name__}: {exc}"}; print(f"[{number}/{len(cases)}] ERROR {identifier}: {exc}")
        rows.append(row); done[identifier]=row
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text("".join(json.dumps(done[str(c["id"])],ensure_ascii=False)+"\n" for c in cases if str(c["id"]) in done),encoding="utf-8")
        if a.delay: time.sleep(a.delay)
    print(json.dumps(summary(rows),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
