"""Source-grounded evaluator tests; all documents and questions are synthetic."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sejong_rag import compare_selected as evaluate_module
from sejong_rag.compare_selected import (
    annotate, first_rank, is_evidence, label_options, result_row,
    serialized_context, summarize,
)


def case(**kwargs):
    return dict(id='C1', query='가상의 신청 질문', notice_id='notice1-1',
                evidence_any=[['원래 정답', '필수 조건']], category='synthetic',
                answer_note='가상 정답', **kwargs)


def evidence(body, notice='notice1-1', **kwargs):
    return dict(body=body, text='제목\n' + body, token_count=len(body) + 5,
                source_aliases=[dict(parent_document_id=notice)], **kwargs)


def test_source_scope_cannot_borrow_other_options_phrases():
    options = label_options([case()], {'accepted_alternatives': [
        dict(case_id='C1', notice_id='notice1-2', required_phrase='별도 대체 정답'),
    ]})['C1']
    assert is_evidence('원래 정답 필수 조건', {'notice1-1'}, options)
    assert is_evidence('별도 대체 정답', {'notice1-2'}, options)
    assert not is_evidence('별도 대체 정답', {'notice1-1'}, options)
    assert not is_evidence('원래 정답 필수 조건', {'notice1-2'}, options)
    assert not is_evidence('원래 정답 필수 조건', {'notice1-9'}, options)


def test_additional_notice_ids_reuse_same_label_and_deduplicate_scope():
    options = label_options([case(additional_notice_ids=['notice1-3', 'notice1-1'])])['C1']
    assert [option['notice_id'] for option in options] == ['notice1-1', 'notice1-3']
    assert is_evidence('원래 정답 필수 조건', {'notice1-3'}, options)
    assert not is_evidence('원래 정답', {'notice1-3'}, options)


def test_empty_labels_and_unknown_audit_case_are_rejected():
    for groups in ([], [[]], [['']], [['\n ']], [[123]], ['not-a-group']):
        row = case()
        row['evidence_any'] = groups
        with pytest.raises(ValueError, match='Invalid evidence'):
            label_options([row])
    with pytest.raises(ValueError, match='unknown case'):
        label_options([case()], {'accepted_alternatives': [
            dict(case_id='absent', notice_id='notice1-2', required_phrase='정답'),
        ]})


def test_headers_metadata_and_other_bundles_cannot_complete_evidence():
    options = label_options([case()])['C1']
    items = [evidence('원래 정답', document_title='필수 조건'), evidence('필수 조건')]
    items[0]['text'] = '문서 제목: 필수 조건\n\n원래 정답'
    original = deepcopy(items)
    annotated = annotate(items, options)
    assert not any(item['is_evidence'] for item in annotated)
    assert items == original
    assert annotate([evidence('원래 정답\n필수 조건')], options)[0]['is_evidence']


def test_actual_context_and_original_anchor_are_scored_separately():
    options = label_options([case()])['C1']
    raw = annotate([evidence('원래 정답')], options)
    contexts = annotate([evidence('원래 정답\n필수 조건', context_expanded=True)], options)
    engine = SimpleNamespace(count=lambda text: len(text) + 2)
    row = result_row(engine, raw, contexts, 1.25)
    assert row['raw_anchor_first_rank'] is None
    assert row['context_first_rank'] == 1
    assert row['context_tokens'] == contexts[0]['token_count']
    assert row['serialized_context_tokens'] == len(serialized_context(contexts)) + 2
    assert row['warm_search_ms'] == 1.25


def test_first_rank_and_mrr_at_five_treat_outside_cutoff_as_zero():
    flags = [{'is_evidence': False}] * 5 + [{'is_evidence': True}]
    assert first_rank(flags) is None
    assert first_rank([{'is_evidence': False}, {'is_evidence': True}]) == 2
    rows = [dict(raw_anchor_first_rank=rank, context_first_rank=rank,
                 context_tokens=10, serialized_context_tokens=13,
                 warm_search_ms=2., top5=[dict(context_expanded=False)])
            for rank in (1, 2, None)]
    summary = summarize(rows)
    assert summary['context_hit_at_1'] == pytest.approx(1 / 3)
    assert summary['context_hit_at_5'] == pytest.approx(2 / 3)
    assert summary['context_mrr_at_5'] == pytest.approx((1 + .5 + 0) / 3)
    assert summary['raw_anchor_mrr_at_5'] == summary['context_mrr_at_5']
    assert summarize([]) == {}


def fake_evaluation(tmp_path, monkeypatch, tamper=None):
    """Mock model execution, not labels, scoring, hashing, reports or file checks."""
    cases_file = tmp_path / 'cases.jsonl'
    row = case()
    row['evidence_any'] = [['원래 정답']]
    cases_file.write_text(json.dumps(row, ensure_ascii=False) + '\n')
    audit_file = tmp_path / 'audit.json'
    audit_file.write_text(json.dumps({'accepted_alternatives': [
        dict(case_id='C1', notice_id='notice1-2', required_phrase='대체 정답'),
    ]}, ensure_ascii=False))
    raw_dir = tmp_path / 'raw' / 'notices'
    raw_dir.mkdir(parents=True)
    (raw_dir / 'notice1-1.html').write_text('<p>원래 정답</p>')
    (raw_dir / 'notice1-2.html').write_text('<p>대체 정답</p>')
    normalized = tmp_path / 'normalized.jsonl'
    source = tmp_path / 'selected.jsonl'
    index = tmp_path / 'index.npz'
    for path in (normalized, source, index):
        path.write_text('synthetic fixture only')
    output = tmp_path / 'output'
    chunk = dict(chunk_id='c0', document_id='notice1-2', parent_document_id='notice1-2',
                 source_url='https://example.org/notice1-2', document_title='가상 공지',
                 source_aliases=[dict(parent_document_id='notice1-2')],
                 source_spans=[dict(record_id='r0', start=0, end=5)],
                 text='문서 제목: 가상 공지\n\n대체 정답')

    class Engine:
        chunks = [chunk]
        bodies = ['대체 정답']

        def count(self, text):
            return len(text) + 2

        def rank(self, query, qvector, **kwargs):
            assert (output / 'ground_truth.json').exists()
            return [(0, .9)]

        def search(self, query, qvector, **kwargs):
            return [dict(**chunk, body='대체 정답', token_count=self.count(chunk['text']),
                         chunk_ids=['c0'], context_expanded=False)]

    metadata = dict(source_chunks=str(source), model='fake-no-download', query_prefix='query: ')
    monkeypatch.setattr(evaluate_module, 'load_search', lambda *args: (Engine(), metadata))
    monkeypatch.setattr(evaluate_module, 'version', lambda name: 'synthetic-version')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(set_num_threads=lambda n: None))

    class Model:
        def __init__(self, model, **kwargs):
            assert model == 'fake-no-download'
            assert kwargs['local_files_only'] is True
            self.tokenizer = SimpleNamespace(encode=lambda text: list(text))
            self.max_seq_length = 4096

        def encode(self, queries, **kwargs):
            frozen_path = output / 'ground_truth.json'
            assert frozen_path.exists(), 'Labels must be frozen before even warm-up encoding'
            frozen = json.loads(frozen_path.read_text())
            assert frozen['label_options']['C1'][1]['origin'] == 'pre_existing_audit'
            assert frozen['relevant_original_chunk_ids']['C1'] == ['c0']
            assert all(query.startswith('query: ') for query in queries)
            if tamper == 'input':
                cases_file.write_text(cases_file.read_text() + ' ')
            elif tamper == 'labels':
                frozen_path.write_text(frozen_path.read_text() + ' ')
            return np.array([[1., 0.]] * len(queries))

    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=Model))
    return dict(cases_path=cases_file, index_path=index, normalized_path=normalized,
                output_dir=output, audit_path=audit_file, data_dir=tmp_path)


def test_end_to_end_audit_applies_identically_and_hashes_freeze_before_encoding(tmp_path, monkeypatch):
    kwargs = fake_evaluation(tmp_path, monkeypatch)
    report = evaluate_module.evaluate(**kwargs)
    assert report['baseline_rawvector']['metrics']['context_hit_at_1'] == 1.
    assert set(report['strategy_metrics']) == set(evaluate_module.STRATEGIES)
    assert all(stats['context_hit_at_1'] == 1. for stats in report['strategy_metrics'].values())
    assert all(stats['raw_anchor_hit_at_1'] == 1. for stats in report['strategy_metrics'].values())
    frozen_path = kwargs['output_dir'] / 'ground_truth.json'
    assert report['ground_truth_sha256'] == evaluate_module.file_sha256(frozen_path)
    frozen = json.loads(frozen_path.read_text())
    assert frozen['source_sha256'] == report['input_sha256']
    assert frozen['code_sha256'] == report['code_sha256']
    assert str(kwargs['audit_path'].resolve()) in report['input_sha256']
    assert (kwargs['output_dir'] / 'report.json').exists()
    assert (kwargs['output_dir'] / 'index.html').exists()


@pytest.mark.parametrize('tamper,error', [('input', 'Frozen input/code changed'),
                                         ('labels', 'Ground-truth file changed')])
def test_changed_frozen_inputs_or_labels_prevent_report_publication(tmp_path, monkeypatch, tamper, error):
    kwargs = fake_evaluation(tmp_path, monkeypatch, tamper=tamper)
    with pytest.raises(ValueError, match=error):
        evaluate_module.evaluate(**kwargs)
    assert not (kwargs['output_dir'] / 'report.json').exists()


def test_unsupported_source_label_rejected_before_model_or_search(tmp_path, monkeypatch):
    kwargs = fake_evaluation(tmp_path, monkeypatch)
    (tmp_path / 'raw' / 'notices' / 'notice1-2.html').write_text('<p>다른 내용</p>')
    with pytest.raises(ValueError, match='Unsupported gold label'):
        evaluate_module.evaluate(**kwargs)
    assert not kwargs['output_dir'].exists()
