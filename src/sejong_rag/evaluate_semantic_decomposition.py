"""Re-evaluate RAG evidence with the existing question-only decomposition path.

Keeps the completed semantic corpus and S02 user amendment fixed. The planner
sees questions only, never gold answers, source texts, or scoring labels.
No final-answer generation calls are made.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import time

from sejong_rag.ask_selected import load_runtime,create_client,save_json,DEFAULT_NORMALIZED
from sejong_rag.benchmark import UsageLedger,usage_summary
from sejong_rag.compare_decomposed import evidence_coverage
from sejong_rag.decomposed_search import decompose,retrieve,VERSION
from sejong_rag.embed_chunks import file_sha256
from sejong_rag.gemini_rag import DEFAULT_GENERATION_MODEL

BASE=Path('data/processed/semantic-v25-evaluation-v1-live')
INDEX=Path('data/index/sejong-semantic-v25-e5-small.npz')

def cases_with_amendment(baseline,amendment):
    cases=[deepcopy(r['case']) for r in baseline['rows'] if r['case']['kind']=='answerable']
    for c in cases:
        if c['id']!=amendment['case_id']:continue
        c['answer_note']=amendment['answer_note'];c['rubric']=amendment['rubric']
        c['gold_sources'].append(amendment['additional_gold_source'])
        for i,item in enumerate(c['items']):
            phrases=item['evidence_options'][0]['evidence_any'] if i<2 else [['조기졸업확인신청서','성적증명서']]
            item['evidence_options'].append(dict(notice_id='notice3-805922',evidence_any=deepcopy(phrases)))
    return cases


def run(output,live=False):
    output=Path(output)
    original=json.loads((BASE/'report.json').read_text())
    amendment=json.loads((BASE/'s02-amendment.json').read_text())
    cases=cases_with_amendment(original,amendment)
    assert len(cases)==24 and sum(len(c['items']) for c in cases)==55
    old={r['case']['id']:r for r in original['rows']}
    output.mkdir(parents=True,exist_ok=False)
    paths=[BASE/'report.json',BASE/'s02-amendment.json',BASE/'final.json',Path('eval/semantic_v25_generation_review_v2.json'),Path('eval/semantic_v25_benchmark_v1.jsonl'),INDEX,DEFAULT_NORMALIZED,*Path('src/sejong_rag').glob('*.py')]
    hashes={str(p):file_sha256(p) for p in paths}
    protocol=dict(version='semantic_v25_decomposition_recheck_v1',decomposition_version=VERSION,
        purpose='Same-question RAG retrieval re-evaluation; prior result was non-decomposed.',
        planner_model=DEFAULT_GENERATION_MODEL,planner_inputs='original query only',max_planner_calls=24,
        final_answer_api_calls=0,top_k=5,max_bundle_tokens=512,total_tokens=2200,
        fusion='Unmodified decomposition_v1: original top1, round-robin subqueries and original list, deduplicate within source scope.',
        semantic_metric='All required evidence items supported by UNION of final5, matching year/entity/conditions; valid alternatives reviewed explicitly.',
        proxy_metric='Frozen source-scoped phrase matching, secondary only; includes user-approved S02 alternative.',
        no_posthoc_tuning=True,limitations=['Previously seen diagnostic set, not new holdout.','Same final context budget, but extra planner/query compute.','Final answer metrics from the old run do not apply to these new contexts.'],
        created_at_utc=datetime.now(timezone.utc).isoformat(),live=live)
    save_json(output/'frozen.json',dict(protocol=protocol,cases=cases,sha256=hashes))
    save_json(output/'planner-payloads.json',[dict(id=c['id'],question=c['query']) for c in cases])
    if not live:return
    start=time.perf_counter();runtime=load_runtime(INDEX,DEFAULT_NORMALIZED);load_ms=(time.perf_counter()-start)*1000
    rows=[];blocked=False;last_call=None
    with create_client() as client:
        ledger=UsageLedger(client,max_calls=24)
        try:
            for c in cases:
                if blocked:break
                if last_call is not None:time.sleep(max(0,6-(time.monotonic()-last_call)))
                ledger.set_case(c['id']+':plan');last_call=time.monotonic()
                try:plan=decompose(c['query'],ledger,DEFAULT_GENERATION_MODEL)
                except Exception as exc:
                    save_json(output/'provider-error.json',dict(case_id=c['id'],error_type=type(exc).__name__))
                    blocked=True;break
                save_json(output/(c['id']+'-plan.json'),plan)
                found=retrieve(c['query'],plan['questions'],*runtime)
                # Recomputed original-query baseline must reproduce the actual
                # previous contexts before we attribute differences to splitting.
                def bodies(sources):return [(s['anchor_chunk_id'],s['text']) for s in sources]
                same=bodies(found['a']['sources'])==bodies(old[c['id']]['request']['sources'])
                if not same:raise ValueError('Original-query baseline changed for '+c['id'])
                row=dict(case=c,plan=plan,baseline_reproduced=same,
                    subquery_candidates=found['subquery_candidates'],retrieval_ms=found['total_retrieval_ms'],
                    a=dict(request=found['a'],coverage_proxy=evidence_coverage(found['a']['sources'],c)),
                    b=dict(request=found['b'],coverage_proxy=evidence_coverage(found['b']['sources'],c)))
                rows.append(row);save_json(output/(c['id']+'.json'),row)
                print(c['id'],'subqueries',len(plan['questions']),'fallback',plan['fallback'],'items',row['a']['coverage_proxy']['covered'],'->',row['b']['coverage_proxy']['covered'],flush=True)
        finally:
            report=dict(protocol=protocol,rows=rows,load_ms=load_ms,api_calls=ledger.records,usage=usage_summary(ledger.records),
                integrity=all(file_sha256(Path(p))==sha for p,sha in hashes.items()),planned=24,completed=len(rows),
                semantic_scores=None,answer_generation='not_run_RAG_only')
            save_json(output/'report.json',report)
    if len(rows)!=24:raise RuntimeError('Incomplete planner run; completed artifacts preserved')
    if not report['integrity']:raise RuntimeError('Frozen files changed')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--live',action='store_true')
    a=p.parse_args();run(a.output,a.live)
if __name__=='__main__':main()
