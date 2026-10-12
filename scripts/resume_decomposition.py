"""Resume frozen decomposition experiment without regenerating completed answers."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from sejong_rag.ask_selected import load_runtime,create_client,save_json
from sejong_rag.benchmark import UsageLedger
from sejong_rag.compare_decomposed import evidence_coverage,summarize
from sejong_rag.decomposed_search import decompose,retrieve
from sejong_rag.embed_chunks import file_sha256
from sejong_rag.selected_generation import generate_answer


class Ledger(UsageLedger):
    def generate_content(self,**kwargs):
        delay=6.0-(time.monotonic()-getattr(self,'last_call',0.0))
        if delay>0:time.sleep(delay)
        self.last_call=time.monotonic()
        try:return super().generate_content(**kwargs)
        except Exception as error:
            code=getattr(error,'code',None)
            if type(code) is int:self.records[-1]['provider_status_code']=code
            raise


def execute(cases,report,runtime,client,output):
    settings=report['settings']
    ledger=Ledger(client,max_calls=settings['max_api_calls'])
    ledger.records=deepcopy(report['api_calls'])
    existing={r['id']:deepcopy(r) for r in report['rows']}
    rows=[]
    failures=0
    for i,case in enumerate(cases):
        row=existing.get(case['id'],dict(id=case['id'],query=case['query'],kind=case['kind']))
        rows.append(row)
        if 'plan' not in row:
            ledger.set_case(case['id']+':plan')
            attempts=sum(c['case_id']==ledger.case_id for c in ledger.records)
            try:
                if attempts:raise ValueError('Prior planner failure')
                row['plan']=decompose(case['query'],ledger,settings['model'])
            except Exception as error:
                row['plan']=dict(questions=[case['query']],fallback='planner_service_error',
                    error_type=type(error).__name__,wall_ms=None)
        if not all(arm in row for arm in ('a','b')):
            result=retrieve(case['query'],row['plan']['questions'],*runtime)
            row['retrieval_ms']=result['total_retrieval_ms']
            row['subquery_candidates']=result['subquery_candidates']
            for arm in ('a','b'):
                row[arm]=dict(request=result[arm],evidence_coverage=evidence_coverage(result[arm]['sources'],case))
        request_path=output/(case['id']+'.requests.json')
        save_json(request_path,dict(plan=row['plan'],a=row['a']['request'],b=row['b']['request']))
        for arm in (('a','b') if i%2==0 else ('b','a')):
            if 'result' in row[arm]:continue
            ledger.set_case(case['id']+':'+arm)
            remaining=2-sum(c['case_id']==ledger.case_id for c in ledger.records)
            started=time.perf_counter()
            while remaining>0:
                try:
                    row[arm]['result']=generate_answer(ledger,row[arm]['request'],model=settings['model'],
                        max_output_tokens=1600,max_attempts=remaining)
                    failures=0
                    row[arm].pop('error_type',None)
                    break
                except Exception as error:
                    row[arm]['error_type']=type(error).__name__
                    failures+=1
                    time.sleep(2)
                    if failures>=3:break
                remaining=2-sum(c['case_id']==ledger.case_id for c in ledger.records)
            row[arm]['generation_ms']=(time.perf_counter()-started)*1000
            save_json(output/(case['id']+'.'+arm+'.json'),row[arm])
        save_json(output/(case['id']+'.json'),row)
        save_json(output/(case['id']+'.checkpoint.json'),dict(settings=settings,rows=rows,
            api_calls=ledger.records,load_ms=report.get('load_ms'),summary=summarize(rows,ledger.records)))
        print(case['id'],'items',row['a']['evidence_coverage']['covered'],'->',row['b']['evidence_coverage']['covered'],flush=True)
        if failures>=3:break
    return dict(settings=settings,rows=rows,api_calls=ledger.records,load_ms=report.get('load_ms'),
        summary=summarize(rows,ledger.records),limitations=report.get('limitations',[]),
        resume_note='Completed plans/answers preserved. Provider-error retries consume the original two-call per-answer budget. Failed planning falls back to original query after one call.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--frozen',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    frozen=json.loads(args.frozen.read_text())
    if not all(file_sha256(Path(p))==sha for p,sha in frozen['sha256'].items()):
        raise ValueError('Frozen code or inputs changed')
    if frozen['sha256'].get(str(args.cases)) != file_sha256(args.cases):
        raise ValueError('Cases differ from freeze')
    report=json.loads(args.report.read_text())
    cases=[json.loads(l) for l in args.cases.read_text().splitlines() if l.strip()]
    args.output_dir.mkdir(parents=True,exist_ok=False)
    save_json(args.output_dir/'resume-frozen.json',dict(source_report=str(args.report),
        report_sha256=file_sha256(args.report),frozen=str(args.frozen),
        script_sha256=file_sha256(Path(__file__))))
    started=time.perf_counter();runtime=load_runtime()
    load_ms=(time.perf_counter()-started)*1000
    with create_client() as client:result=execute(cases,report,runtime,client,args.output_dir)
    result['resume_load_ms']=load_ms
    result['integrity']=all(file_sha256(Path(p))==sha for p,sha in frozen['sha256'].items())
    result['source_report']=str(args.report)
    save_json(args.output_dir/'report.json',result)

if __name__=='__main__':main()
