import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sejong_rag import benchmark as bm


def case(identifier='B01', kind='answerable'):
    return dict(id=identifier, kind=kind, category='test', query='언제 확인해?',
        expected_status=['answered'] if kind == 'answerable' else ['insufficient_evidence'],
        answer_note='다음 근무일 10시', answer_concepts=[['10시']],
        evidence_options=[dict(notice_id='notice3-123', evidence_any=[['다음 근무일', '10시']])]
            if kind == 'answerable' else [], gold_sources=[], year=2026, rubric='합성 테스트')


def source(number=1, body='다음 근무일 10시 이후'):
    return dict(id=number, parent_document_id='notice3-123', source_aliases=[],
                body=body, text=body, source_url='https://www.sejong.ac.kr/notice', document_title='공지')


def result(status='answered'):
    return dict(status=status, claims=[dict(text='다음 근무일 10시 이후야.', citations=[1])]
        if status == 'answered' else [], message='' if status == 'answered' else '근거가 없어.',
        answer='다음 근무일 10시 이후야. [근거 1]' if status == 'answered' else '근거가 없어.',
        validation_passed=True)


def test_load_cases_rejects_more_than_32(tmp_path):
    path = tmp_path / 'cases.jsonl'
    path.write_text(''.join(json.dumps(case(f'B{i}')) + '\n' for i in range(33)))
    with pytest.raises(ValueError):
        bm.load_cases(path)


@pytest.mark.parametrize('identifier', ['../outside', 'a/b', '', 'a b'])
def test_load_cases_rejects_unsafe_case_id(tmp_path, identifier):
    path = tmp_path / 'cases.jsonl'
    path.write_text(json.dumps(case(identifier)) + '\n')
    with pytest.raises(ValueError):
        bm.load_cases(path)


def test_load_cases_rejects_duplicates(tmp_path):
    path = tmp_path / 'cases.jsonl'
    path.write_text((json.dumps(case()) + '\n') * 2)
    with pytest.raises(ValueError):
        bm.load_cases(path)


def test_load_cases_rejects_empty_gold_phrase(tmp_path):
    item = case()
    item['evidence_options'][0]['evidence_any'] = [['']]
    path = tmp_path / 'cases.jsonl'
    path.write_text(json.dumps(item) + '\n')
    with pytest.raises(ValueError):
        bm.load_cases(path)


def test_retrieval_alias_scope_is_accepted_but_headers_do_not_count():
    item = source(body='다음 근무일 10시 이후')
    item['parent_document_id'] = 'canonical'
    item['source_aliases'] = [{'parent_document_id': 'notice3-123'}]
    assert bm.retrieval_entry([item], case()['evidence_options'])['first_relevant_rank'] == 1
    item['body'] = '언제 확인되는지 설명 없음'
    item['text'] = '다음 근무일 10시 이후'
    assert bm.retrieval_entry([item], case()['evidence_options'])['first_relevant_rank'] is None


def test_grade_does_not_call_lexical_proxy_semantic_truth():
    value = result()
    value['claims'][0]['text'] = '10시 이후에 확인할 수 없다는 뜻이야.'
    value['answer'] = value['claims'][0]['text'] + ' [근거 1]'
    assessed = bm.grade(case(), {'sources': [source()]}, value)
    assert assessed['automatic_proxy_pass'] is True
    # This deliberate false-positive example explains the separate semantic review.
    assert 'semantic_correct' not in assessed


def test_usage_ledger_preserves_first_response_when_second_attempt_raises():
    class Models:
        def __init__(self): self.calls = 0
        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('DO_NOT_SAVE_SECRET')
            return SimpleNamespace(text='bad JSON', usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=2, total_token_count=12))
    client = SimpleNamespace(models=Models())
    ledger = bm.UsageLedger(client)
    ledger.set_case('B01')
    ledger.models.generate_content(model='fake')
    with pytest.raises(RuntimeError):
        ledger.models.generate_content(model='fake')
    usage = bm.usage_summary(ledger.records)
    assert usage['prompt_token_count'] == 10
    assert usage['missing_count']['prompt_token_count'] == 1
    assert usage['complete'] is False
    assert usage['api_call_count'] == 2
    assert 'DO_NOT_SAVE_SECRET' not in json.dumps(ledger.records)


def test_percentiles_linear_interpolation_and_empty():
    assert bm.percentiles([]) == {'count': 0, 'p50': None, 'p95': None}
    value = bm.percentiles([100, 0])
    assert value['p50'] == 50
    assert value['p95'] == 95


def test_summary_offline_never_claims_generation_passes():
    item = case()
    row = dict(item, current=bm.retrieval_entry([source()], item['evidence_options']),
        baseline=bm.retrieval_entry([source()], item['evidence_options']), generation={'state': 'not_run'})
    summary = bm.summarize([row], [item], live=False, api_records=[], load_ms=5, warmup_ms=1)
    assert summary['retrieval']['current']['hit_at_5'] == 1
    assert summary['generation_proxy']['pass_count'] is None
    assert summary['generation_proxy']['pass_rate'] is None
    assert summary['semantic_accuracy'] is summary['faithfulness'] is None
    assert summary['usage']['total_token_count'] is None


def test_safe_error_omits_raw_message():
    value = bm._safe_error(RuntimeError('https://example.test/?key=secret'), 'generation', 'B01')
    assert value['type'] == 'RuntimeError'
    assert 'secret' not in json.dumps(value)


def test_html_escapes_answer_and_question():
    item = case()
    item['query'] = '<script>alert(1)</script>'
    item['generation'] = {'state': 'completed', 'result': {'answer': '<img src=x onerror=alert(1)>'}}
    report = dict(results=[item], summary={}, limitations=[], completed_at_utc='2026-10-11T00:00:00+00:00')
    rendered = bm.render_html(report)
    assert '<script>' not in rendered
    assert '<img src=' not in rendered
    assert '&lt;script&gt;' in rendered


def test_gold_quote_validation_keeps_struck_through_text_without_interpreting_it(tmp_path):
    path = tmp_path / 'notice.html'
    path.write_text('<div class="b-content-box"><s>기존 3명</s> 변경 5명</div>')
    item = case()
    item['gold_sources'] = [dict(path=str(path), notice_id='notice3-123', quotes=['기존 3명', '변경 5명'])]
    assert bm._gold_paths([item]) == {path}


def test_existing_output_is_refused_before_loading_anything(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('must not load runtime')
    monkeypatch.setattr(bm, 'load_runtime', forbidden)
    with pytest.raises(FileExistsError):
        bm.evaluate(tmp_path / 'absent.jsonl', tmp_path)


def test_hash_check_identifies_mutation_and_missing_file(tmp_path):
    one, two = tmp_path / 'one', tmp_path / 'two'
    one.write_text('first')
    two.write_text('second')
    hashes = {str(path): bm.file_sha256(path) for path in (one, two)}
    one.write_text('changed')
    two.unlink()
    assert bm.check_hashes(hashes) == {'passed': False, 'changed_paths': [str(one), str(two)]}
