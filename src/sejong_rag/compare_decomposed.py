"""Frozen A/B experiment. Live calls are explicit; reports are append-preserved."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from sejong_rag.ask_selected import load_runtime, create_client, save_json, DEFAULT_INDEX, DEFAULT_NORMALIZED
from sejong_rag.benchmark import UsageLedger, usage_summary
from sejong_rag.compare_selected import compact, notice_ids
from sejong_rag.decomposed_search import decompose, retrieve, VERSION
from sejong_rag.embed_chunks import file_sha256
from sejong_rag.gemini_rag import DEFAULT_GENERATION_MODEL
from sejong_rag.selected_generation import generate_answer


def evidence_coverage(bundles, case):
    """Notice-scoped phrase proxy over the UNION of returned bodies.

Each item can require multiple phrases across bundles of the same notice.
This is not semantic verification, and cannot recognize all alternative sources.
"""
    results = []
    for item in case.get('items', []):
        matched = False
        for option in item['evidence_options']:
            texts = [compact(b['body']) for b in bundles if option['notice_id'] in notice_ids(b)]
            if any(all(any(compact(p) in text for text in texts) for p in group)
                   for group in option['evidence_any']):
                matched = True
        results.append(matched)
    return dict(items=results, covered=sum(results), total=len(results),
                complete=all(results) if results else None)


def summarize(rows, calls):
    stats = {}
    for arm in ('a', 'b'):
        completed = [r[arm] for r in rows if arm in r]
        scored = [r['evidence_coverage'] for r in completed if r['evidence_coverage']['total']]
        stats[arm] = dict(prepared=len(completed),
            evidence_items=sum(r['total'] for r in scored),
            covered_items=sum(r['covered'] for r in scored),
            complete_evidence_questions=sum(r['complete'] for r in scored),
            answerable_questions=len(scored),
            generation_completed=sum('result' in r for r in completed),
            semantic_accuracy=None,
            evidence_tokens=sum(r['request']['evidence_tokens_e5'] for r in completed))
    return dict(arms=stats, usage=usage_summary(calls))


def run(cases_path, output_dir, *, live=False):
    if not live:
        raise ValueError('Pass --live to authorize bounded planner and answer API calls')
    cases_path, output_dir = Path(cases_path), Path(output_dir)
    cases = [json.loads(l) for l in cases_path.read_text().splitlines() if l.strip()]
    if not 1 <= len(cases) <= 40 or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Use 1-40 unique cases')
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = [cases_path, Path(DEFAULT_INDEX), Path(DEFAULT_NORMALIZED), Path('eval/decomposition_protocol_v1.json')]
    paths += list(Path('src/sejong_rag').glob('*.py'))
    paths += [Path(s['path']) for c in cases for s in c.get('gold_sources', [])]
    frozen = {str(p): file_sha256(p) for p in sorted(set(paths))}
    settings = dict(version=VERSION, model=DEFAULT_GENERATION_MODEL,
        max_api_calls=len(cases)*5, max_generation_attempts=2,
        top_k=5, max_bundle_tokens=512, total_tokens=2200,
        note='A and B use identical generator and original question. Planner gets question only.')
    save_json(output_dir/'frozen.json', dict(settings=settings, sha256=frozen))
    started=time.perf_counter()
    runtime=load_runtime()
    load_ms=(time.perf_counter()-started)*1000
    rows=[]
    with create_client() as client:
        ledger=UsageLedger(client, max_calls=settings['max_api_calls'])
        try:
            for i, case in enumerate(cases):
                row=dict(id=case['id'], query=case['query'], kind=case['kind'])
                rows.append(row)
                ledger.set_case(case['id']+':plan')
                row['plan']=decompose(case['query'], ledger, settings['model'])
                save_json(output_dir/(case['id']+'.plan.json'),row['plan'])
                retrieval=retrieve(case['query'],row['plan']['questions'],*runtime)
                row['retrieval_ms']=retrieval['total_retrieval_ms']
                row['subquery_candidates']=retrieval['subquery_candidates']
                for arm in ('a','b'):
                    request=retrieval[arm]
                    row[arm]=dict(request=request,
                        evidence_coverage=evidence_coverage(request['sources'],case))
                save_json(output_dir/(case['id']+'.requests.json'), {arm: row[arm]['request'] for arm in ('a','b')})
                # Alternate order to limit a fixed arm/order timing confound.
                for arm in (('a','b') if i%2==0 else ('b','a')):
                    ledger.set_case(case['id']+':'+arm)
                    start=time.perf_counter()
                    row[arm]['result']=generate_answer(ledger,row[arm]['request'],
                        model=settings['model'],max_output_tokens=1600,max_attempts=2)
                    row[arm]['generation_ms']=(time.perf_counter()-start)*1000
                    save_json(output_dir/(case['id']+'.'+arm+'.json'),row[arm])
                save_json(output_dir/(case['id']+'.json'),row)
                print(case['id'], 'items', row['a']['evidence_coverage']['covered'],
                      '->',row['b']['evidence_coverage']['covered'], flush=True)
        except Exception as error:
            save_json(output_dir/'error.json',dict(error_type=type(error).__name__,
                case_id=ledger.case_id, rows=rows, api_calls=ledger.records))
            raise RuntimeError('Experiment stopped; partial results preserved') from None
        finally:
            integrity=all(file_sha256(Path(p))==sha for p,sha in frozen.items())
            save_json(output_dir/'report.json',dict(settings=settings,rows=rows,
                api_calls=ledger.records,load_ms=load_ms,integrity=integrity,
                summary=summarize(rows,ledger.records),
                limitations=['Assistant-authored labels, not human approval.',
                    'Phrase coverage is a proxy, not answer correctness.',
                    'One generation sample per arm; planner and API costs are additional.',
                    'Retrieval timing includes joint batch embedding; not standalone A latency.']))
    if not integrity:
        raise RuntimeError('Frozen inputs changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    run(args.cases,args.output_dir,live=args.live)

if __name__=='__main__':
    main()
