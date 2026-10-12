"""Resume only provider failures in a frozen replay, within its original two-call budget."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time
from reevaluate_benchmark import PacedLedger
from sejong_rag.ask_selected import create_client, save_json
from sejong_rag.benchmark import usage_summary
from sejong_rag.embed_chunks import file_sha256
from sejong_rag.selected_generation import generate_answer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    report=deepcopy(json.loads(args.report.read_text()))
    frozen=json.loads((args.report.parent/'frozen.json').read_text())
    assert all(file_sha256(Path(p))==h for p,h in frozen['files_sha256'].items())
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for row in report['results']:
        request=json.loads((args.report.parent/f'{row["id"]}-request.json').read_text())
        save_json(args.output_dir/f'{row["id"]}-request.json',request)
    save_json(args.output_dir/'frozen.json',dict(original=frozen,resume_code_sha256=file_sha256(Path(__file__)),
        parent_report_sha256=file_sha256(args.report),note='Completed outputs preserved; retry provider errors only.'))
    ledger=PacedLedger(create_client(),max_calls=64)
    ledger.records=deepcopy(report['api_calls'])
    for row in report['results']:
        if row['generation']['state']=='completed':continue
        remaining=2-sum(c['case_id']==row['id'] for c in ledger.records)
        if remaining<=0:continue
        ledger.set_case(row['id'])
        request=json.loads((args.output_dir/f'{row["id"]}-request.json').read_text())
        started=time.perf_counter()
        prior=deepcopy(row['generation'])
        while remaining>0:
            try:
                result=generate_answer(ledger,request,model=report['settings']['model'],
                    max_output_tokens=report['settings']['max_output_tokens'],max_attempts=remaining)
                row['generation']=dict(state='completed',result=result,prior_failure=prior)
                break
            except Exception as exc:
                row['generation']=dict(state='error',error_type=type(exc).__name__,prior_failure=prior)
                remaining=2-sum(c['case_id']==row['id'] for c in ledger.records)
                if remaining:time.sleep(10)
        row['generation']['wall_ms']=(time.perf_counter()-started)*1000
        report['api_calls']=ledger.records
        save_json(args.output_dir/f'{row["id"]}-retry.json',row['generation'])
        save_json(args.output_dir/f'{row["id"]}-checkpoint.json',report)
        print(row['id'],row['generation']['state'],row['generation'].get('result',{}).get('status'),flush=True)
    report['api_calls']=ledger.records
    report['usage']=usage_summary(ledger.records)
    report['completed_at_utc']=datetime.now(timezone.utc).isoformat()
    report['completed']=all(r['generation']['state']=='completed' for r in report['results'])
    report['integrity']=dict(passed=all(file_sha256(Path(p))==h for p,h in frozen['files_sha256'].items()))
    report['resume']=dict(parent_report=str(args.report),parent_report_sha256=file_sha256(args.report),
        note='Original two-call per-case ceiling includes provider failures; no repeated successful generation.')
    save_json(args.output_dir/'report.json',report)
    print(json.dumps(dict(completed=report['completed'],usage=report['usage']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
