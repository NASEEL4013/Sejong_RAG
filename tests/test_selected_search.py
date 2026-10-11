"""Offline retrieval invariants using synthetic sources, never evaluation labels."""
from copy import deepcopy
import json

import numpy as np
import pytest

from sejong_rag.selected_search import DEFAULT_STRATEGY, SelectedSearch, expanded_query, period, period_score
from sejong_rag import selected_search


class Characters:
    def __call__(self, text, add_special_tokens=True, **kwargs):
        return {'input_ids': list(range(len(text) + (2 if add_special_tokens else 0)))}


def make_chunk(cid, rid, text, start=0, end=None, notice='notice-a', title='2026-2 안내', aliases=None):
    end = len(text) if end is None else end
    alias = dict(document_id=notice, parent_document_id=notice,
                 document_title=title, source_url=f'https://example.org/{notice}')
    return dict(chunk_id=cid, document_id=notice, parent_document_id=notice,
                document_title=title, source_url=alias['source_url'],
                source_aliases=aliases or [alias],
                source_spans=[dict(record_id=rid, start=start, end=end, locator={'page': 1})],
                text=f'문서 제목: {title}\n\n{text[start:end]}')


def engine_for(texts, *, same_notice=True):
    records = {f'r{i}': {'text': text} for i, text in enumerate(texts)}
    chunks = [make_chunk(f'c{i}', f'r{i}', text,
                         notice='notice-a' if same_notice else f'notice-{i}')
              for i, text in enumerate(texts)]
    vectors = np.array([[1.0 - i * .1, i * .1] for i in range(len(chunks))])
    return SelectedSearch(chunks, vectors, records, Characters())


@pytest.mark.parametrize('text,expected', [
    ('2026년 2학기 등록금', (2026, 2)),
    ('2025학년도 1학기 안내', (2025, 1)),
    ('2026-2 안내', (2026, 2)),
    ('2026-1학기 안내', (2026, 1)),
    ('1학기 안내', (None, 1)),
    ('2025년과 2026년 1학기 및 2학기', (None, None)),
    ('일반 안내', (None, None)),
])
def test_period_handles_explicit_and_ambiguous_periods(text, expected):
    assert period(text) == expected


@pytest.mark.parametrize('query', ['2025년 입학생의 졸업 조건', '2026 입학년도 졸업 기준'])
def test_admission_year_is_not_current_notice_period(query):
    assert period(query, query=True) == (None, None)


def test_period_must_match_in_one_alias_not_across_aliases():
    aliases = [{'document_title': '2026-1 안내'}, {'document_title': '2025-2 안내'}]
    assert period_score('2026년 2학기 문의', aliases) < 0
    aliases.append({'document_title': '2026-2 안내'})
    assert period_score('2026년 2학기 문의', aliases) > 0


@pytest.mark.parametrize('aliases', [[], [{}], [{'document_title': '일반 안내'}],
                                         [{'document_title': '2026년 안내'}]])
def test_unknown_or_partial_period_is_neutral(aliases):
    assert period_score('2026년 2학기 문의', aliases) == 0


def test_unknown_alias_avoids_false_mismatch_penalty():
    aliases = [{'document_title': '2025-1 안내'}, {'document_title': '상시 안내'}]
    assert period_score('2026년 2학기 문의', aliases) == 0


def test_bundle_selects_matching_alias_and_url_together():
    engine = engine_for(['안내 신청 방법', '안내 등록 방법'])
    anchor = engine.chunks[0]
    anchor['source_aliases'] = [
        dict(document_title='2025-1 안내', source_url='https://example.org/old'),
        dict(document_title='2026-2 안내', source_url='https://example.org/matching'),
    ]
    result = engine.bundle([0], '2026년 2학기 문의')
    assert result['document_title'] == '2026-2 안내'
    assert result['source_url'] == 'https://example.org/matching'
    assert result['source_aliases'] == anchor['source_aliases']


def test_changed_alias_header_is_bounded_without_clipping_body():
    engine = engine_for(['안내 신청 방법', '안내 등록 방법'])
    anchor = engine.chunks[0]
    long_title = '2025-1 ' + '길이가 매우 긴 공식 공지 제목 ' * 100
    anchor['source_aliases'] = [
        dict(document_title=long_title, source_url='https://example.org/matching',
             attachment_url='https://example.org/attachment'),
    ]
    result = engine.bundle([0], '2025년 1학기 문의')
    assert engine.count(result['text'].split('\n\n', 1)[0]) <= 224
    assert result['body'] == '안내 신청 방법'
    assert result['document_title'] == long_title
    assert result['attachment_url'] == 'https://example.org/attachment'


def test_body_lexical_index_excludes_repeated_document_title():
    engine = engine_for(['안내 신청 절차', '안내 신청 절차'])
    assert '2026' not in engine.vectorizer.vocabulary_
    assert engine.bodies == ['안내 신청 절차', '안내 신청 절차']


def test_default_preserves_vector_order_with_bounded_context():
    assert DEFAULT_STRATEGY == 'vector_context'
    engine = engine_for(['안내 신청 절차', '안내 등록 절차'])
    result = engine.search('안내 신청', np.array([1., 0.]))
    assert result[0]['anchor_chunk_id'] == 'c0'
    assert all(item['strategy'] == 'vector_context' for item in result)


def test_adjacent_context_never_crosses_record_boundaries():
    first = '안내 신청 대상\n안내 신청 제외 대상'
    boundary = first.index('안내 신청 제외')
    records = {'r0': {'text': first}, 'r1': {'text': '안내 다른 문서 설명'}}
    chunks = [make_chunk('a', 'r0', first, 0, boundary),
              make_chunk('b', 'r0', first, boundary),
              make_chunk('c', 'r1', records['r1']['text'])]
    engine = SelectedSearch(chunks, np.array([[1., 0.], [.5, .5], [.1, .9]]), records, Characters())
    assert engine.neighbors[0] == [1]
    assert engine.neighbors[1] == [0]
    assert engine.neighbors[2] == []
    result = engine.search('안내 신청', np.array([1., 0.]), 'vector_context', top_k=1)
    assert result[0]['chunk_ids'] == ['a', 'b']
    assert {s['record_id'] for s in result[0]['source_spans']} == {'r0'}
    assert result[0]['body'] == first


def test_bundle_merges_overlapping_source_spans_without_mutating_inputs():
    text = '안내 신청 기간과 방법 그리고 문의 부서 안내'
    records = {'r': {'text': text}}
    chunks = [make_chunk('a', 'r', text, 0, 18), make_chunk('b', 'r', text, 10)]
    original = deepcopy(chunks)
    engine = SelectedSearch(chunks, np.eye(2), records, Characters())
    result = engine.bundle([1, 0, 1], '문의')
    assert result['body'] == text
    assert len(result['source_spans']) == 1
    assert result['source_spans'][0]['start'] == 0
    assert result['source_spans'][0]['end'] == len(text)
    assert chunks == original


def test_bundle_does_not_invent_missing_source_gap():
    text = '안내 앞부분 생략된부분 안내 뒷부분'
    gap = text.index('생략된부분')
    end_gap = gap + len('생략된부분')
    records = {'r': {'text': text}}
    chunks = [make_chunk('a', 'r', text, 0, gap), make_chunk('b', 'r', text, end_gap)]
    engine = SelectedSearch(chunks, np.eye(2), records, Characters())
    result = engine.bundle([0, 1], '안내')
    assert len(result['source_spans']) == 2
    assert '생략된부분' not in result['body']
    assert result['body'] == text[:gap] + '\n' + text[end_gap:]


def test_bundle_preserves_record_appearance_order_not_lexicographic_ids():
    records = {'r2': {'text': '안내 앞부분'}, 'r10': {'text': '안내 뒷부분'}}
    chunk = make_chunk('a', 'r2', records['r2']['text'])
    chunk['source_spans'].append(dict(record_id='r10', start=0, end=6, locator={'page': 2}))
    chunk['text'] = '문서 제목: 안내\n\n안내 앞부분\n안내 뒷부분'
    other = make_chunk('b', 'r10', records['r10']['text'])
    engine = SelectedSearch([chunk, other], np.eye(2), records, Characters())
    result = engine.bundle([0], '안내')
    assert [ref['record_id'] for ref in result['source_spans']] == ['r2', 'r10']
    assert result['body'] == '안내 앞부분\n안내 뒷부분'


def test_bundle_preserves_header_with_internal_blank_lines():
    engine = engine_for(['안내 표 첫 행\n안내 표 다음 행', '안내 표 별도 행'])
    prefix = '문서 제목: 표 안내\n\n표 앞부분 참고:\n열A 열B'
    engine.chunks[0]['text'] = prefix + '\n\n' + engine.bodies[0]
    bundle = engine.bundle([0], '안내')
    assert bundle['text'] == prefix + '\n\n' + engine.bodies[0]
    assert bundle['body'] == engine.bodies[0]


def test_top_k_and_actual_final_text_token_budget():
    engine = engine_for(['안내 신청 기간', '안내 신청 장소', '안내 신청 방법'])
    unit = engine.count(engine.bundle([0], '안내')['text'])
    result = engine.search('안내', np.array([1., 0.]), 'vector', top_k=3,
                           max_bundle_tokens=unit, total_tokens=unit * 2)
    assert len(result) == 2
    assert sum(r['token_count'] for r in result) == unit * 2
    assert all(r['token_count'] == engine.count(r['text']) for r in result)
    assert len(engine.search('안내', np.array([1., 0.]), 'vector', top_k=1)) == 1


def test_oversized_candidates_are_never_returned():
    engine = engine_for(['안내 신청 기간', '안내 신청 장소'])
    result = engine.search('안내', np.array([1., 0.]), 'vector',
                           max_bundle_tokens=1, total_tokens=100)
    assert result == []


def test_expansion_cannot_discard_anchor_that_fits_remaining_budget():
    text = '안내 신청 기간\n안내 신청 설명이 이어짐'
    end = text.index('안내 신청 설명')
    records = {'r': {'text': text}}
    chunks = [make_chunk('a', 'r', text, 0, end), make_chunk('b', 'r', text, end)]
    engine = SelectedSearch(chunks, np.eye(2), records, Characters())
    anchor_budget = engine.count(engine.bundle([0], '안내')['text'])
    results = engine.search('안내', np.array([1., 0.]), 'vector_context', top_k=1,
                            max_bundle_tokens=512, total_tokens=anchor_budget)
    assert len(results) == 1
    assert results[0]['anchor_chunk_id'] == 'a'
    assert results[0]['chunk_ids'] == ['a']
    assert results[0]['token_count'] <= anchor_budget


def test_identical_substantial_body_dedup_is_notice_scoped():
    body = '안내 신청 절차와 필요 서류를 확인하고 기간 내 방문 제출합니다. ' * 5
    same = engine_for([body, body])
    results = same.search('안내', np.array([1., 0.]), 'vector_context')
    assert len(results) == 1
    assert results[0]['duplicate_candidate_ids'] == ['c1']
    duplicate = results[0]['duplicate_evidence'][0]
    assert duplicate['chunk_ids'] == ['c1']
    assert duplicate['source_spans'] == same.chunks[1]['source_spans']
    assert duplicate['source_aliases'] == same.chunks[1]['source_aliases']
    assert duplicate['source_url'] == same.chunks[1]['source_url']
    different = engine_for([body, body], same_notice=False)
    assert len(different.search('안내', np.array([1., 0.]), 'vector_context')) == 2


def test_short_identical_bodies_are_not_deduplicated_without_context():
    engine = engine_for(['안내 신청 불가', '안내 신청 불가'])
    assert len(engine.search('안내', np.array([1., 0.]), 'vector_context')) == 2


def test_hybrid_period_signal_can_prefer_matching_alias_but_vector_does_not():
    text = '안내 신청 절차를 확인하세요.'
    records = {'r0': {'text': text}, 'r1': {'text': text}}
    chunks = [make_chunk('a', 'r0', text, title='2025-1 안내'),
              make_chunk('b', 'r1', text, title='2026-2 안내')]
    vectors = np.array([[1., 0.], [.99, .01]])
    engine = SelectedSearch(chunks, vectors, records, Characters())
    query = '2026년 2학기 신청 안내'
    assert engine.rank(query, np.array([1., 0.]), hybrid=False)[0][0] == 0
    assert engine.rank(query, np.array([1., 0.]), hybrid=True)[0][0] == 1


@pytest.mark.parametrize('query,anchor_title,protected', [
    ('2026년 2학기 접수 마감 날짜 문의', '2026-2 안내', True),
    ('2026년 2학기 접수 마감 날짜 문의', '2025-1 안내', False),
    ('접수 마감 날짜 문의', '2025-1 안내', True),
])
def test_semantic_anchor_is_protected_except_explicit_period_mismatch(query, anchor_title, protected):
    texts = ['안내 달력상 금요일입니다.', '안내 접수 마감 날짜 문의',
             '안내 접수 마감 날짜 확인', '안내 접수 마감 날짜 정보']
    records = {f'r{i}': {'text': text} for i, text in enumerate(texts)}
    chunks = [make_chunk(f'c{i}', f'r{i}', text,
                         title=anchor_title if i == 0 else '2026-2 안내')
              for i, text in enumerate(texts)]
    vectors = np.array([[1. - i * .1, i * .1] for i in range(len(texts))])
    engine = SelectedSearch(chunks, vectors, records, Characters())
    ranked = engine.rank(query, np.array([1., 0.]), hybrid=True)
    anchor_position = next(i for i, (index, _) in enumerate(ranked) if index == 0)
    raw_score_order = sorted(ranked, key=lambda item: -item[1])
    assert next(i for i, (index, _) in enumerate(raw_score_order) if index == 0) > 1
    assert anchor_position == 1 if protected else anchor_position > 1


def mock_load_inputs(tmp_path, monkeypatch, *, ref=None, expected='normalized-sha'):
    source = tmp_path / 'selected.jsonl'
    normalized = tmp_path / 'records.jsonl'
    source.write_text('{}\n')
    normalized.write_text(json.dumps({'record_id': 'r0', 'text': '안내 신청'}) + '\n')
    (tmp_path / 'report.json').write_text(json.dumps({
        'input_sha256': {str(normalized.resolve()): expected},
    }))
    metadata = dict(source_chunks=str(source), source_sha256='source-sha', model='unused')
    chunks = [dict(chunk_id='c0', source_spans=[ref or dict(record_id='r0', start=0, end=5)])]
    monkeypatch.setattr(selected_search, 'load_index', lambda path: (np.eye(1), chunks, metadata))
    monkeypatch.setattr(selected_search, 'file_sha256',
                        lambda path: 'source-sha' if path == source else 'normalized-sha')
    return tmp_path / 'index.npz', normalized


def test_load_search_rejects_normalized_content_hash_mismatch(tmp_path, monkeypatch):
    index, normalized = mock_load_inputs(tmp_path, monkeypatch, expected='old-normalized-sha')
    with pytest.raises(ValueError, match='Normalized source differs'):
        selected_search.load_search(index, normalized)


@pytest.mark.parametrize('ref', [
    dict(record_id='missing-record', start=0, end=1),
    dict(record_id='r0', start=-1, end=1),
    dict(record_id='r0', start=2, end=2),
    dict(record_id='r0', start=0, end=999),
])
def test_load_search_rejects_missing_record_and_invalid_spans(tmp_path, monkeypatch, ref):
    index, normalized = mock_load_inputs(tmp_path, monkeypatch, ref=ref)
    with pytest.raises(ValueError, match='Invalid source span'):
        selected_search.load_search(index, normalized)


def test_context_expansion_respects_per_bundle_limit():
    text = '안내 신청 기간\n안내 신청 설명이 이어짐'
    end = text.index('안내 신청 설명')
    records = {'r': {'text': text}}
    chunks = [make_chunk('a', 'r', text, 0, end), make_chunk('b', 'r', text, end)]
    engine = SelectedSearch(chunks, np.eye(2), records, Characters())
    anchor_budget = engine.count(engine.bundle([0], '안내')['text'])
    results = engine.search('안내', np.array([1., 0.]), 'vector_context', top_k=1,
                            max_bundle_tokens=anchor_budget, total_tokens=1000)
    assert results[0]['chunk_ids'] == ['a']
    assert results[0]['token_count'] <= anchor_budget


@pytest.mark.parametrize('query,terms', [
    ('신청서를 어디로 가져가?', ('장소', '위치', '제출처', '방문')),
    ('언제 신청해?', ('기간', '시간', '일정')),
    ('얼마를 지원해?', ('금액', '지원', '한도')),
    ('문의 전화는?', ('문의', '연락처')),
    ('어떤 앱을 설치해?', ('설치', '앱', '다운로드')),
])
def test_intent_expansion_adds_generic_terms_not_answer_values(query, terms):
    expanded = expanded_query(query)
    assert expanded.startswith(query)
    assert all(term in expanded for term in terms)
    assert '학생지원과' not in expanded
    assert '104' not in expanded
    assert '오전 10시' not in expanded
    assert 'notice3-' not in expanded


def test_validation_of_strategy_budget_and_candidate_count():
    engine = engine_for(['안내 신청 기간', '안내 신청 장소'])
    qvector = np.array([1., 0.])
    with pytest.raises(ValueError):
        engine.search('안내', qvector, 'unknown')
    for kwargs in ({'top_k': 0}, {'max_bundle_tokens': 0}, {'total_tokens': 0}, {'candidate_k': 0}):
        with pytest.raises(ValueError):
            engine.search('안내', qvector, **kwargs)
