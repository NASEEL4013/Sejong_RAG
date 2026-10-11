"""Validate separate claim-level reviews and aggregate a frozen benchmark report."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from sejong_rag.ask_selected import save_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def aggregate(report, reviews, report_hash, protocol):
    rows = report['results']
    by_id = {row['id']: row for row in rows}
    if not rows or len(by_id) != len(rows):
        raise ValueError('Report requires nonempty unique question IDs.')
    planned = protocol['planned']
    distribution = {key: value for key, value in planned.items() if key != 'total' and value}
    if len(rows) != planned['total'] or dict(Counter(row['kind'] for row in rows)) != distribution:
        raise ValueError('Report must preserve the complete predeclared question distribution.')
    reviewed = {}
    reviewer_roles = set()
    for review in reviews:
        if review.get('report_sha256') != report_hash:
            raise ValueError('Review belongs to a different report.')
        if review.get('reviewer_role') != 'assistant' or review.get('human_approved') is not False:
            raise ValueError('This report requires explicitly labeled assistant, non-human-approved reviews.')
        reviewer_roles.add(review['reviewer_role'])
        for item in review['results']:
            case_id = item['id']
            if case_id not in by_id or case_id in reviewed:
                raise ValueError('Unknown or duplicate reviewed question: ' + case_id)
            row = by_id[case_id]
            result = row['generation'].get('result') or {}
            is_answerable = row['kind'] == 'answerable'
            key = 'answer_correct' if is_answerable else 'behavior_correct'
            other = 'behavior_correct' if is_answerable else 'answer_correct'
            if type(item.get(key)) is not bool or item.get(other) is not None:
                raise ValueError('Review must distinguish answer accuracy and behavior: ' + case_id)
            if item[key] and row['generation']['state'] != 'completed':
                raise ValueError('Uncompleted generation cannot pass: ' + case_id)
            if is_answerable and item[key] and result.get('status') != 'answered':
                raise ValueError('Abstention cannot count as a correct complete factual answer: ' + case_id)
            if not is_answerable and item[key] and result.get('status') not in row['expected_status']:
                raise ValueError('Behavior pass conflicts with predeclared status: ' + case_id)
            if not isinstance(item.get('reason'), str) or not item['reason'].strip():
                raise ValueError('Each semantic judgment requires a reason: ' + case_id)
            claims = result.get('claims', [])
            checks = item.get('claim_checks', [])
            positions = [c.get('claim_index') for c in checks]
            if any(type(n) is not int for n in positions) or sorted(positions) != list(range(1, len(claims) + 1)):
                raise ValueError('Review must cover every actual claim exactly once: ' + case_id)
            for check in checks:
                claim = claims[check['claim_index'] - 1]
                if check.get('citations') != claim['citations']:
                    raise ValueError('Review citation mapping differs from actual response: ' + case_id)
                if type(check.get('supported')) is not bool or not isinstance(check.get('reason'), str) or not check['reason'].strip():
                    raise ValueError('Each claim requires a support verdict and reason: ' + case_id)
            reviewed[case_id] = item
    if set(reviewed) != set(by_id):
        raise ValueError('Missing reviews: ' + ', '.join(sorted(set(by_id) - set(reviewed))))

    answerable = [r for r in rows if r['kind'] == 'answerable']
    behavior = [r for r in rows if r['kind'] != 'answerable']
    answered = [r for r in answerable if (r['generation'].get('result') or {}).get('status') == 'answered']
    correct = sum(reviewed[r['id']]['answer_correct'] for r in answerable)
    behavior_correct = sum(reviewed[r['id']]['behavior_correct'] for r in behavior)
    claims = [c for item in reviewed.values() for c in item['claim_checks']]
    supported = sum(c['supported'] for c in claims)
    failed = [item['id'] for item in reviewed.values() if item.get('answer_correct') is False or item.get('behavior_correct') is False]
    retrieval = {}
    for strategy in ('baseline', 'current'):
        ranks = [r[strategy]['first_relevant_rank'] for r in answerable]
        retrieval[strategy] = {
            'questions': len(ranks),
            **{f'hit_at_{k}': rate(sum(rank is not None and rank <= k for rank in ranks), len(ranks)) for k in (1, 3, 5)},
            'mrr_at_5': rate(sum(1 / rank for rank in ranks if rank is not None and 1 <= rank <= 5), len(ranks)),
        }
    metrics = dict(answerable_questions=len(answerable), correct_answers=correct,
        answer_accuracy=rate(correct, len(answerable)), answered_questions=len(answered),
        answer_coverage=rate(len(answered), len(answerable)), selective_accuracy=rate(correct, len(answered)),
        behavior_questions=len(behavior), correct_behaviors=behavior_correct,
        behavior_accuracy=rate(behavior_correct, len(behavior)), generated_claims=len(claims),
        supported_claims=supported, claim_support=rate(supported, len(claims)),
        reviewed_questions=len(reviewed), failed_case_ids=failed)
    thresholds = protocol['reference_targets']
    at_least = lambda value, threshold: value is not None and value >= threshold
    current_hit, baseline_hit = retrieval['current']['hit_at_5'], retrieval['baseline']['hit_at_5']
    targets = dict(
        context_hit_at_5=at_least(current_hit, thresholds['context_hit_at_5_min']),
        retrieval_not_below_baseline=current_hit is not None and baseline_hit is not None and current_hit >= baseline_hit,
        answer_accuracy=at_least(metrics['answer_accuracy'], thresholds['answer_accuracy_min']),
        claim_support=at_least(metrics['claim_support'], thresholds['claim_support_min']),
        behavior=behavior_correct == thresholds['behavior_correct_required'] == len(behavior),
        complete_live_run=all(r['generation']['state'] == 'completed' for r in rows),
        planned_distribution=dict(Counter(r['kind'] for r in rows)) == {k: v for k, v in protocol['planned'].items() if k != 'total'},
    )
    return dict(report_sha256=report_hash, reviewer_roles=sorted(reviewer_roles), human_approved=False,
        review_scope='Assistant source comparison, not human approval or population accuracy.',
        metrics=metrics, strict_retrieval=retrieval, reference_targets=targets,
        all_reference_targets_met=all(targets.values()),
        deployment_ready=False,
        failure_types=dict(Counter(kind for item in reviewed.values() for kind in item.get('failure_types', []))),
        results=[reviewed[row['id']] for row in rows])


def finalize(report_path, review_paths, output_dir, protocol_path='eval/project_benchmark_protocol_v1.json'):
    report_path, output_dir, protocol_path = map(Path, (report_path, output_dir, protocol_path))
    if output_dir.exists():
        raise FileExistsError('Use a new review output directory.')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('live') is not True or report.get('integrity', {}).get('passed') is not True:
        raise ValueError('Semantic finalization requires a live report with verified frozen-input integrity.')
    frozen_protocol_hash = report.get('files_sha256', {}).get(str(protocol_path.resolve()))
    if frozen_protocol_hash != sha256(protocol_path):
        raise ValueError('Protocol must match the hash frozen before the benchmark.')
    reviews = [json.loads(Path(path).read_text(encoding='utf-8')) for path in review_paths]
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    result = aggregate(report, reviews, sha256(report_path), protocol)
    result['artifacts'] = dict(report=str(report_path), protocol=str(protocol_path),
        protocol_sha256=sha256(protocol_path), reviews={str(p): sha256(p) for p in review_paths})
    output_dir.mkdir(parents=True, exist_ok=False)
    save_json(output_dir / 'reviewed-report.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--reviews', type=Path, nargs='+', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=Path('eval/project_benchmark_protocol_v1.json'))
    args = p.parse_args()
    result = finalize(args.report, args.reviews, args.output_dir, args.protocol)
    print(json.dumps({k: result[k] for k in ('metrics', 'strict_retrieval', 'reference_targets', 'all_reference_targets_met')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
