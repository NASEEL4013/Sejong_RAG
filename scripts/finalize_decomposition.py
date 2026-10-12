"""Validate report-bound assistant reviews and summarize the A/B experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import median
from sejong_rag.embed_chunks import file_sha256
from sejong_rag.benchmark import usage_summary


def summarize_review(review):
    report_path=Path(review['report_path'])
    if file_sha256(report_path)!=review['report_sha256']:
        raise ValueError('Report hash mismatch')
    if file_sha256(Path(review['cases_path']))!=review['cases_sha256']:
        raise ValueError('Cases hash mismatch')
    report=json.loads(report_path.read_text())
    cases={c['id']:c for c in map(json.loads,Path(review['cases_path']).read_text().splitlines())}
    rows={r['id']:r for r in report['rows']}
    judgments={r['id']:r for r in review['results']}
    if set(rows)!=set(cases) or set(judgments)!=set(cases) or len(judgments)!=len(review['results']):
        raise ValueError('Incomplete or duplicate review')
    summary={'retrieval_proxy':report['summary']['arms'],'arms':{},'usage_by_stage':{}}
    paired=[i for i in cases if cases[i]['kind']=='answerable' and all('result' in rows[i][a] for a in ['a','b'])]
    for arm in ['a','b']:
        answerable=[i for i in cases if cases[i]['kind']=='answerable']
        behavior=[i for i in cases if cases[i]['kind']!='answerable']
        unsupported=[];supported=0;claims=0
        for ident,row in rows.items():
            result=row[arm].get('result')
            judgment=judgments[ident]['arms'][arm]
            if bool(result)!=judgment['completed']:raise ValueError('Completion mismatch')
            checks=judgment['claim_checks']
            actual=(result or {}).get('claims',[])
            if len(checks)!=len(actual):raise ValueError('Missing claim checks')
            for n,(check,claim) in enumerate(zip(checks,actual),1):
                if check['claim_index']!=n or check['citations']!=claim['citations'] or type(check['supported']) is not bool:
                    raise ValueError('Claim mapping mismatch')
                claims+=1;supported+=check['supported']
                if not check['supported']:unsupported.append(f'{ident}:claim{n}')
        full=lambda i:judgments[i]['arms'][arm]['full_correct'] is True
        grounded=lambda i: full(i) and all(c['supported'] for c in judgments[i]['arms'][arm]['claim_checks']) and not any(not c.get('cited') for c in judgments[i]['arms'][arm]['message_facts'])
        summary['arms'][arm]=dict(answerable=len(answerable),full_correct=sum(full(i) for i in answerable),
            grounded_full_correct=sum(grounded(i) for i in answerable),
            generation_missing=sum('result' not in rows[i][arm] for i in cases),
            paired_answerable=len(paired),paired_full_correct=sum(full(i) for i in paired),
            behavior=len(behavior),behavior_correct=sum(judgments[i]['arms'][arm]['behavior_correct'] is True for i in behavior),
            claims=claims,supported_claims=supported,unsupported_claims=unsupported)
    for stage in ['plan','a','b']:
        calls=[r for r in report['api_calls'] if r['case_id'].endswith(':'+stage)]
        times=[c['elapsed_ms'] for c in calls if c['status']=='returned']
        summary['usage_by_stage'][stage]=dict(usage_summary(calls),response_ms_median=median(times) if times else None)
    summary['planner_fallbacks']=[dict(id=r['id'],reason=r['plan']['fallback']) for r in rows.values() if r['plan']['fallback']]
    summary['identical_input_cases']=[r['id'] for r in review['results'] if r['identical_generation_input']]
    summary['report_integrity']=report.get('integrity')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result={str(path):summarize_review(json.loads(path.read_text())) for path in args.reviews}
    with args.output.open('x') as output:json.dump(result,output,ensure_ascii=False,indent=2);output.write('\n')

if __name__=='__main__':main()
