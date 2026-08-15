"""모호함·개인정보·미래 정보·조건 부족 질문에 안전하게 답하는지 평가한다."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
from sejong_rag.gemini_rag import generate_with_gemini,get_api_key
from sejong_rag.prompt_builder import build_messages
from sejong_rag.unified_context import select_unified_context
from sejong_rag.unified_search import DEFAULT_TRUSTED,ambiguity_warning,search_indexes
from sejong_rag.vector_search import load_index
DEFAULT_EVAL=Path("eval/holdout_behavior_questions.jsonl"); DEFAULT_OUTPUT=Path("data/evaluation/holdout-behavior.jsonl")
def main():
 p=argparse.ArgumentParser(); p.add_argument("--eval",type=Path,default=DEFAULT_EVAL); p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); a=p.parse_args()
 cases=[json.loads(x) for x in a.eval.read_text(encoding="utf-8").splitlines() if x.strip()]; _,chunks,_=load_index(DEFAULT_TRUSTED)
 from google import genai
 client=genai.Client(api_key=get_api_key()); rows=[]
 for case in cases:
  results=search_indexes(case["query"],DEFAULT_TRUSTED,top_k=5,candidate_k=300); warning=ambiguity_warning(case["query"],results)
  if warning: answer=warning
  else:
   context=select_unified_context(results,chunks); answer,_=generate_with_gemini(client,build_messages(case["query"],context),max_output_tokens=600)
  normalized=re.sub(r"\s+","",answer); required=any(re.sub(r"\s+","",x) in normalized for x in case["required_any"]); forbidden=any(re.sub(r"\s+","",x) in normalized for x in case.get("forbidden_any",[])); passed=required and not forbidden
  row={**case,"answer":answer,"passed":passed}; rows.append(row); print(('PASS' if passed else 'CHECK'),case['id'],answer[:300].replace('\n',' '))
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8'); print('pass_rate',sum(x['passed'] for x in rows)/len(rows))
if __name__=='__main__': main()
