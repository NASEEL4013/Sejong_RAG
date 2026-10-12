"""Replay frozen Top5 requests with a new model; keep historical runs immutable."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from sejong_rag.ask_selected import create_client, save_json
from sejong_rag.benchmark import UsageLedger, usage_summary
from sejong_rag.embed_chunks import file_sha256
from sejong_rag.selected_generation import generate_answer

class PacedLedger(UsageLedger):
    def generate_content(self, **kwargs):
        delay = 6 - (time.monotonic() - getattr(self, 'last_call', 0))
        if delay > 0:
            time.sleep(delay)
        self.last_call = time.monotonic()
        try:
            return super().generate_content(**kwargs)
        except Exception as exc:
            code = getattr(exc, 'code', None)
            if type(code) is int:
                self.records[-1]['provider_status_code'] = code
            raise

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--live', action='store_true')
    args = p.parse_args()
    if not args.live:
        raise ValueError('Use --live to authorize API calls')
    protocol = json.loads(args.protocol.read_text())
    original = json.loads(args.source.read_text())
    cases = [json.loads(x) for x in Path(protocol['cases']).read_text().splitlines() if x.strip()]
    assert [c['id'] for c in cases] == [r['id'] for r in original['results']]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, hashes = [], {}
    paths = [args.source, args.protocol, Path(protocol['cases']), Path(__file__)]
    paths += list(Path('src/sejong_rag').glob('*.py'))
    for case, old in zip(cases, original['results']):
        source = args.source.parent / f'{case["id"]}-request.json'
        request = json.loads(source.read_text())
        assert request['query'] == case['query']
        assert [s['body'] for s in request['sources']] == [s['body'] for s in old['current']['top5']]
        save_json(args.output_dir / source.name, request)
        paths += [source, args.output_dir / source.name]
        row = dict(deepcopy(case), current=deepcopy(old['current']), generation={'state':'not_run'})
        row['current'].pop('retrieval_with_embedding_ms', None)
        rows.append(row)
    for path in paths:
        hashes[str(path.resolve())] = file_sha256(path)
    save_json(args.output_dir / 'frozen.json', dict(settings=protocol, files_sha256=hashes,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        note='Original queries, system/user prompts and Top5 evidence replayed exactly. Only generation model changes.'))
    ledger = PacedLedger(create_client(), max_calls=64)
    report = dict(settings=protocol, results=rows, api_calls=ledger.records,
        source_report_sha256=file_sha256(args.source), started_at_utc=datetime.now(timezone.utc).isoformat())
    failures = 0
    for row in rows:
        ledger.set_case(row['id'])
        request = json.loads((args.output_dir / f'{row["id"]}-request.json').read_text())
        start = time.perf_counter()
        try:
            result = generate_answer(ledger, request, model=protocol['model'],
                max_output_tokens=protocol['max_output_tokens'], max_attempts=2)
            row['generation'] = dict(state='completed', result=result)
            failures = 0
        except Exception as exc:
            row['generation'] = dict(state='error', error_type=type(exc).__name__)
            failures += 1
        row['generation']['wall_ms'] = (time.perf_counter() - start)*1000
        save_json(args.output_dir / f'{row["id"]}-answer.json', row['generation'])
        save_json(args.output_dir / f'{row["id"]}-checkpoint.json', report)
        print(row['id'], row['generation']['state'], row['generation'].get('result',{}).get('status'), flush=True)
        if failures >= 3:
            break
    report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['usage'] = usage_summary(ledger.records)
    report['integrity'] = dict(passed=all(file_sha256(Path(p))==h for p,h in hashes.items()))
    report['completed'] = all(r['generation']['state']=='completed' for r in rows)
    save_json(args.output_dir / 'report.json', report)
    print(json.dumps(dict(completed=report['completed'], usage=report['usage']), ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
