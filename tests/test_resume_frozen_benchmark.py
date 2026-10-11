import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/resume_frozen_benchmark.py'
    spec = importlib.util.spec_from_file_location('resume_benchmark_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup(tmp_path, mod):
    original = tmp_path / 'original'
    original.mkdir()
    rows = []
    for identifier, state in [('B01', 'completed'), ('B02', 'error')]:
        request = {'query': identifier, 'messages': [{'role': 'user', 'content': 'frozen evidence only'}], 'sources': []}
        (original / f'{identifier}-request.json').write_text(json.dumps(request))
        rows.append(dict(id=identifier, request_path=f'{identifier}-request.json', kind='answerable',
            generation={'state': state, 'result': {'answer': 'original answer'} if state == 'completed' else {}, 'wall_ms': 10}))
    usage = dict(prompt_token_count=3, candidates_token_count=2, total_token_count=5)
    calls = [dict(case_id='B01', call_index=1, status='returned', usage=usage),
             dict(case_id='B02', call_index=2, status='error', usage={})]
    report = dict(live=True, integrity={'passed': True}, files_sha256={}, code_sha256={},
        artifacts_sha256={str(p.resolve()): mod.file_sha256(p) for p in original.glob('*-request.json')},
        settings={'max_api_calls': 4, 'max_attempts': 2, 'model': 'fake', 'max_output_tokens': 1600},
        results=rows, api_calls=calls, summary={'latency_ms': {'initial_load': 20, 'warmup': 1}}, limitations=[])
    path = original / 'report.json'
    path.write_text(json.dumps(report))
    (original / 'frozen.json').write_text(json.dumps({'settings': report['settings']}))
    return path


def patch_generation(monkeypatch, mod, *, error=None):
    called = []
    def call(**kwargs):
        called.append(kwargs)
        if error: raise error
        return SimpleNamespace(usage_metadata={'prompt_token_count': 5, 'candidates_token_count': 2, 'total_token_count': 7})
    monkeypatch.setattr(mod, 'create_client', lambda: SimpleNamespace(models=SimpleNamespace(generate_content=call), close=lambda: None))
    def generate(client, request, **kwargs):
        assert request['query'] == 'B02'
        assert kwargs == {'model': 'fake', 'max_output_tokens': 1600, 'max_attempts': 1}
        client.models.generate_content(contents=request['messages'], model=kwargs['model'])
        return {'answer': 'new B02 answer', 'status': 'answered', 'claims': []}
    monkeypatch.setattr(mod, 'generate_answer', generate)
    monkeypatch.setattr(mod, 'grade', lambda *args: {})
    monkeypatch.setattr(mod, 'summarize', lambda *args, **kwargs: {'calls': len(kwargs['api_records'])})
    monkeypatch.setattr(mod, 'render_html', lambda report: 'report')
    return called


def test_resume_keeps_completed_answer_and_original_budget(monkeypatch, tmp_path):
    mod = module()
    base = setup(tmp_path, mod)
    before = base.read_bytes()
    called = patch_generation(monkeypatch, mod)
    report = mod.resume(base, tmp_path / 'resumed')
    assert report['completed']
    assert len(called) == 1
    assert len(report['api_calls']) == 3
    assert report['results'][0]['generation']['result']['answer'] == 'original answer'
    assert report['results'][1]['generation']['usage']['missing_count']['total_token_count'] == 1
    assert base.read_bytes() == before


def test_provider_error_is_safely_classified_and_stops(monkeypatch, tmp_path, capsys):
    mod = module()
    base = setup(tmp_path, mod)
    class FakeQuotaError(Exception):
        code = 429
    patch_generation(monkeypatch, mod, error=FakeQuotaError('quotaPerDay secret-test-value'))
    report = mod.resume(base, tmp_path / 'resumed')
    assert not report['completed']
    assert report['api_calls'][-1]['http_status'] == 429
    assert report['api_calls'][-1]['quota_scope_hint'] == 'day'
    assert 'secret-test-value' not in json.dumps(report)
    assert 'secret-test-value' not in capsys.readouterr().out


def test_mutated_frozen_request_blocks_resume(monkeypatch, tmp_path):
    mod = module()
    base = setup(tmp_path, mod)
    called = patch_generation(monkeypatch, mod)
    (base.parent / 'B02-request.json').write_text('modified')
    with pytest.raises(ValueError, match='changed'):
        mod.resume(base, tmp_path / 'resumed')
    assert not called


def test_exhausted_case_is_not_retried(monkeypatch, tmp_path):
    mod = module()
    base = setup(tmp_path, mod)
    value = json.loads(base.read_text())
    value['api_calls'].append(dict(case_id='B02', call_index=3, status='error', usage={}))
    base.write_text(json.dumps(value))
    called = patch_generation(monkeypatch, mod)
    with pytest.raises(ValueError, match='exhausted'):
        mod.resume(base, tmp_path / 'resumed')
    assert not called


def test_limit_error_does_not_mutate_last_success_record():
    mod = module()
    ledger = mod.DiagnosticLedger(SimpleNamespace(), max_calls=1)
    prior = {'status': 'returned', 'usage': {}}
    ledger.records.append(prior.copy())
    with pytest.raises(RuntimeError):
        ledger.generate_content()
    assert ledger.records == [prior]


def test_changed_settings_are_not_used(monkeypatch, tmp_path):
    mod = module()
    base = setup(tmp_path, mod)
    value = json.loads(base.read_text())
    value['settings']['model'] = 'different'
    base.write_text(json.dumps(value))
    called = patch_generation(monkeypatch, mod)
    with pytest.raises(ValueError, match='settings'):
        mod.resume(base, tmp_path / 'resumed')
    assert not called
