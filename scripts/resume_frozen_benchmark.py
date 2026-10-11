"""Resume only failed/unrun frozen requests; preserve completed answers and budgets."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from sejong_rag.ask_selected import create_client, save_json
from sejong_rag.benchmark import (UsageLedger, check_hashes, grade, render_html,
    summarize, usage_summary)
from sejong_rag.embed_chunks import file_sha256
from sejong_rag.selected_generation import generate_answer


class DiagnosticLedger(UsageLedger):
    def generate_content(self, **kwargs):
        previous_count = len(self.records)
        try:
            return super().generate_content(**kwargs)
        except Exception as exc:
            if len(self.records) > previous_count:
                row = self.records[-1]
                code = getattr(exc, 'code', None)
                row['http_status'] = code if type(code) is int else None
                # Inspect locally, never save/print provider messages, URLs, keys,
                # project IDs, or quota values. Record only coarse categories.
                description = str(exc).lower()
                day = 'perday' in description or 'per day' in description
                minute = 'perminute' in description or 'per minute' in description
                row['quota_scope_hint'] = 'day' if day else 'minute' if minute else 'unknown'
            raise


def resume(base_path, output_dir):
    base_path, output_dir = Path(base_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('A fresh output directory is required.')
    base_hash = file_sha256(base_path)
    report = json.loads(base_path.read_text(encoding='utf-8'))
    if not report.get('live') or not report.get('integrity', {}).get('passed'):
        raise ValueError('Expected a live report with valid frozen inputs.')
    hashes = {**report['files_sha256'], **report['code_sha256'], **report['artifacts_sha256'],
              str(base_path.resolve()): base_hash}
    if not check_hashes(hashes)['passed']:
        raise ValueError('Original frozen input or report changed.')
    settings = report['settings']
    frozen = json.loads((base_path.parent / 'frozen.json').read_text(encoding='utf-8'))
    if settings != frozen['settings']:
        raise ValueError('Report settings differ from the pre-call frozen settings.')
    prior_calls = report['api_calls']
    counts = Counter(row['case_id'] for row in prior_calls)
    pending = [row for row in report['results'] if row['generation']['state'] != 'completed']
    if not pending or len(prior_calls) >= settings['max_api_calls']:
        raise ValueError('No pending cases or no remaining global call budget.')
    if any(counts[row['id']] >= settings['max_attempts'] for row in pending):
        raise ValueError('A pending case has exhausted its predeclared call budget.')
    output_dir.mkdir(parents=True, exist_ok=False)
    requests = {}
    for row in report['results']:
        source = base_path.parent / row['request_path']
        requests[row['id']] = json.loads(source.read_text(encoding='utf-8'))
        destination = output_dir / row['request_path']
        save_json(destination, requests[row['id']])
        hashes[str(destination.resolve())] = file_sha256(destination)
    script_path = Path(__file__).resolve()
    hashes[str(script_path)] = file_sha256(script_path)
    save_json(output_dir / 'resume-code-snapshot.json', {str(script_path): script_path.read_text(encoding='utf-8')})
    save_json(output_dir / 'resume-frozen.json', dict(base_report=str(base_path), base_sha256=base_hash,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), files_sha256=hashes,
        pending_ids=[r['id'] for r in pending], prior_api_calls=len(prior_calls),
        max_api_calls_total=settings['max_api_calls'], max_calls_per_question=settings['max_attempts'],
        policy='Operational retry only. Never regenerate completed answers, relabel cases, rerun retrieval, or change generation settings.'))
    for name in ('resume-code-snapshot.json', 'resume-frozen.json'):
        path = output_dir / name
        hashes[str(path.resolve())] = file_sha256(path)
    if not check_hashes(hashes)['passed']:
        raise ValueError('Frozen input changed before resume.')
    client = ledger = None
    error = None
    try:
        client = create_client()
        ledger = DiagnosticLedger(client, max_calls=settings['max_api_calls'] - len(prior_calls))
        for row in pending:
            identifier = row['id']
            ledger.set_case(identifier)
            previous = deepcopy(row['generation'])
            started = time.perf_counter()
            try:
                result = generate_answer(ledger, requests[identifier], model=settings['model'],
                    max_output_tokens=settings['max_output_tokens'],
                    max_attempts=settings['max_attempts'] - counts[identifier])
                row['generation'] = dict(state='completed', result=result,
                    **grade(row, requests[identifier], result))
            except Exception as exc:
                row['generation'] = dict(state='error')
                error = dict(type=type(exc).__name__, phase='resume_generation', case_id=identifier,
                    message='Frozen-request retry failed. Provider text and credentials are not recorded.')
            finally:
                row['generation']['wall_ms'] = (previous.get('wall_ms') or 0) + (time.perf_counter() - started) * 1000
                row['generation']['previous_state'] = previous
                row['generation']['usage'] = usage_summary([r for r in prior_calls + ledger.records if r['case_id'] == identifier])
                save_json(output_dir / f'{identifier}-generation.json', row['generation'])
            print(identifier, row['generation']['state'], flush=True)
            if error:
                break
    except Exception as exc:
        error = dict(type=type(exc).__name__, phase='resume_client', case_id=None,
                     message='Resume stopped; provider text and credentials are not recorded.')
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                if error is None:
                    error = dict(type=type(exc).__name__, phase='close_client', case_id=None)
    new_calls = deepcopy(ledger.records) if ledger else []
    for call in new_calls:
        call['call_index'] += len(prior_calls)
    calls = prior_calls + new_calls
    report['api_calls'] = calls
    report['error'] = error
    report['resume'] = dict(base_report=str(base_path), base_report_sha256=base_hash,
        regenerated_completed_cases=0, new_calls=len(new_calls), hashes=hashes,
        note='Latency sums active attempt times; between-run investigation/backoff time is excluded.')
    report['integrity'] = check_hashes(hashes)
    report['completed'] = error is None and report['integrity']['passed'] and all(
        row['generation']['state'] == 'completed' for row in report['results'])
    report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    old_latency = report['summary']['latency_ms']
    report['summary'] = summarize(report['results'], report['results'], live=True, api_records=calls,
        load_ms=old_latency['initial_load'], warmup_ms=old_latency['warmup'])
    report['limitations'].append('A provider error interrupted the first run; only failed/unrun fixed requests were resumed, within original per-case and global budgets. The original partial report is preserved.')
    save_json(output_dir / 'report.json', report)
    with (output_dir / 'index.html').open('x', encoding='utf-8') as output:
        output.write(render_html(report))
    print(json.dumps(dict(completed=report['completed'], error=error,
        new_api_calls=[{k: v for k, v in r.items() if k != 'usage'} for r in new_calls if r['status'] == 'error'],
        summary=report['summary']), ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = resume(args.report, args.output_dir)
    if not report['completed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
