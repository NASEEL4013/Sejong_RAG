import json
from types import SimpleNamespace

import pytest

from sejong_rag import ask_selected
from sejong_rag.evaluate_selected_generation import grade, summary
from sejong_rag.selected_generation import build_request


def bundle():
    return dict(anchor_chunk_id='c1', chunk_ids=['c1'], text='다음 근무일 오전 10시',
        body='다음 근무일 오전 10시', document_title='2026-2 납부 확인',
        source_url='https://www.sejong.ac.kr/notice', attachment_url=None,
        source_spans=[dict(record_id='r1', start=0, end=12, locator={'page': 6})],
        source_aliases=[dict(document_id='notice3-1', parent_document_id='notice3-1')],
        token_count=12)


def test_prepare_request_preserves_retrieval_and_adds_provenance():
    engine = SimpleNamespace(count=lambda text: len(text),
        chunks=[dict(filename='안내.pdf', quality_tier='ready', quality_flags=['check'])], by_id={'c1': 0})
    calls = []
    engine.search = lambda query, vector, strategy: (calls.append((query, vector, strategy)) or [bundle()])
    model = SimpleNamespace(max_seq_length=512, encode=lambda text, **kwargs: text)
    result = ask_selected.prepare_request(' 언제? ', engine, {'query_prefix': 'query: ', 'model': 'e5', 'source_sha256': 'hash'}, model)
    assert calls == [('언제?', 'query: 언제?', 'vector_context')]
    assert result['sources'][0]['quality_tier'] == 'ready'
    assert result['sources'][0]['filename'] == '안내.pdf'
    assert result['sources'][0]['quality_flags'] == ['check']
    assert result['sources'][0]['locations'] == [{'page': 6}]
    assert result['evidence_tokens_e5'] == 12


def test_long_query_is_rejected_before_embedding():
    engine = SimpleNamespace(count=lambda text: 999)
    model = SimpleNamespace(max_seq_length=512, encode=lambda *a, **k: pytest.fail('must not encode'))
    with pytest.raises(ValueError, match='입력 한도'):
        ask_selected.prepare_request('질문', engine, {'query_prefix': 'query: '}, model)


def test_dry_run_does_not_access_key_or_client(monkeypatch, capsys):
    request = build_request('질문', [bundle()])
    monkeypatch.setattr('sys.argv', ['ask_selected', '질문', '--dry-run'])
    monkeypatch.setattr(ask_selected, 'load_runtime', lambda *args: ())
    monkeypatch.setattr(ask_selected, 'prepare_request', lambda *args: request)
    monkeypatch.setattr(ask_selected, 'create_client', lambda: pytest.fail('external call'))
    ask_selected.main()
    assert json.loads(capsys.readouterr().out)['query'] == '질문'


def test_save_json_refuses_overwrite(tmp_path):
    path = tmp_path / 'request.json'
    ask_selected.save_json(path, {'original': 1})
    with pytest.raises(FileExistsError):
        ask_selected.save_json(path, {'replacement': 2})
    assert json.loads(path.read_text()) == {'original': 1}


def case():
    return dict(expected_status=['answered'], answer_concepts=[['다음 근무일'], ['10시']],
        evidence_options=[dict(notice_id='notice3-1', evidence_any=[['다음 근무일', '10시']])])


def test_generation_grade_checks_cited_body_not_just_citation_id():
    request = build_request('언제?', [bundle()])
    answer = dict(answer='다음 근무일 10시 [근거 1]', status='answered',
                  claims=[dict(text='다음 근무일 10시', citations=[1])], validation_passed=True)
    assert grade(case(), request, answer)['automatic_checks_passed']
    request['sources'][0]['body'] = '실시간 확인 불가'
    assert not grade(case(), request, answer)['checks']['cited_gold_ok']


def test_grade_supports_unknown_and_flags_personal_identifier_request():
    c = dict(expected_status=['needs_clarification'], answer_concepts=[], evidence_options=[])
    result = dict(status='needs_clarification', claims=[], validation_passed=True,
                  answer='어느 학기인지 알려줘.')
    assert grade(c, {'sources': []}, result)['automatic_checks_passed']
    result['answer'] = '학번을 알려줘.'
    assert not grade(c, {'sources': []}, result)['checks']['no_identity_request']


def test_summary_does_not_report_unrun_evaluation_as_pass():
    result = summary([], 12)
    assert result['planned'] == 12 and result['completed'] == 0
    assert result['automatic_pass_rate'] is None
