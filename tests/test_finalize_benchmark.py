from copy import deepcopy

import pytest

from sejong_rag.finalize_benchmark import aggregate


def fixture():
    row = dict(id='B01', kind='answerable', expected_status=['answered'],
        baseline={'first_relevant_rank': None}, current={'first_relevant_rank': 2},
        generation={'state': 'completed', 'result': {'status': 'answered', 'claims': [{'text': '3학점', 'citations': [2]}]}})
    judgment = dict(id='B01', answer_correct=True, behavior_correct=None, reason='원문과 일치', failure_types=[],
        claim_checks=[{'claim_index': 1, 'citations': [2], 'supported': True, 'reason': '2번 본문에 직접 명시'}])
    reviews = [dict(report_sha256='abc', reviewer_role='assistant', human_approved=False, results=[judgment])]
    protocol = dict(planned={'total': 1, 'answerable': 1}, reference_targets=dict(
        context_hit_at_5_min=.8, answer_accuracy_min=.8, claim_support_min=.95, behavior_correct_required=0))
    return {'results': [row]}, reviews, protocol


def test_metrics_use_full_question_denominator():
    report, reviews, protocol = fixture()
    second = deepcopy(report['results'][0])
    second.update(id='B02', current={'first_relevant_rank': None}, generation={'state': 'completed', 'result': {'status': 'insufficient_evidence', 'claims': []}})
    report['results'].append(second)
    protocol['planned'] = {'total': 2, 'answerable': 2}
    reviews[0]['results'].append(dict(id='B02', answer_correct=False, behavior_correct=None, reason='검색 근거 부족', failure_types=['retrieval_miss'], claim_checks=[]))
    result = aggregate(report, reviews, 'abc', protocol)
    assert result['metrics']['answer_accuracy'] == .5
    assert result['metrics']['answer_coverage'] == .5
    assert result['metrics']['selective_accuracy'] == 1
    assert result['strict_retrieval']['current']['mrr_at_5'] == .25
    assert not result['all_reference_targets_met']
    assert not result['deployment_ready']


@pytest.mark.parametrize('change', ['hash', 'duplicate', 'missing', 'claim_missing', 'citation', 'human', 'abstention', 'unfinished'])
def test_inconsistent_reviews_rejected(change):
    report, reviews, protocol = fixture()
    if change == 'hash': reviews[0]['report_sha256'] = 'different'
    elif change == 'duplicate': reviews[0]['results'] *= 2
    elif change == 'missing': reviews[0]['results'] = []
    elif change == 'claim_missing': reviews[0]['results'][0]['claim_checks'] = []
    elif change == 'citation': reviews[0]['results'][0]['claim_checks'][0]['citations'] = [9]
    elif change == 'human': reviews[0]['human_approved'] = True
    elif change == 'abstention': report['results'][0]['generation']['result']['status'] = 'needs_clarification'
    elif change == 'unfinished': report['results'][0]['generation']['state'] = 'not_run'
    with pytest.raises(ValueError): aggregate(report, reviews, 'abc', protocol)


def test_no_claims_is_not_perfect_faithfulness():
    report, reviews, protocol = fixture()
    report['results'][0]['generation'] = {'state': 'completed', 'result': {'status': 'insufficient_evidence', 'claims': []}}
    reviews[0]['results'][0].update(answer_correct=False, claim_checks=[])
    result = aggregate(report, reviews, 'abc', protocol)
    assert result['metrics']['claim_support'] is None
    assert result['metrics']['selective_accuracy'] is None
    assert not result['reference_targets']['claim_support']


def test_missing_planned_rows_cannot_shrink_denominator():
    report, reviews, protocol = fixture()
    protocol['planned'] = {'total': 2, 'answerable': 2}
    with pytest.raises(ValueError, match='distribution'):
        aggregate(report, reviews, 'abc', protocol)


def test_protocol_changed_after_benchmark_is_rejected(tmp_path):
    import json
    from sejong_rag.finalize_benchmark import finalize
    report, reviews, protocol = fixture()
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(protocol))
    report.update(live=True, integrity={'passed': True}, files_sha256={str(path.resolve()): 'old-hash'})
    report_path = tmp_path / 'report.json'
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Protocol'):
        finalize(report_path, [], tmp_path / 'new', path)
