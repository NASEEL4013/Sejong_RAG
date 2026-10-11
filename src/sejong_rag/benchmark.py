"""Frozen, budget-matched retrieval and bounded generation benchmark.

Automatic phrase/citation checks are proxies. Semantic accuracy and faithfulness
remain null until a separate, report-hash-bound review is performed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import html
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

from sejong_rag.ask_selected import (
    DEFAULT_INDEX, DEFAULT_NORMALIZED, create_client, load_runtime,
    prepare_request, save_json,
)
from sejong_rag.compare_selected import compact, is_evidence, notice_ids
from sejong_rag.embed_chunks import file_sha256
from sejong_rag.gemini_rag import DEFAULT_GENERATION_MODEL
from sejong_rag.selected_generation import generate_answer, validate_response

MAX_CASES = 32
MAX_API_CALLS = 64
USAGE_FIELDS = ('prompt_token_count', 'candidates_token_count', 'total_token_count')
KINDS = {'answerable', 'unanswerable', 'clarification', 'privacy', 'adversarial'}
STATUSES = {'answered', 'needs_clarification', 'insufficient_evidence'}
LIMITATIONS = [
    'Automatic concept matching can miss synonyms or accept negated statements; it is not semantic accuracy.',
    'A cited-gold proxy only checks whether at least one cited body contains a labeled answer, not every claim.',
    'Semantic accuracy and claim faithfulness require a separate review and are null in this report.',
    'Retrieval labels are an OR of notice-scoped phrase groups; each AND group must occur in one actual bundle body.',
    'The vector baseline uses the same 5/512/2200 context limits; only current vector_context is sent to Gemini.',
    'Current warm retrieval includes query encoding and production request assembly; baseline search excludes its separately recorded extra query encoding.',
    'Known SDK usage is summed across all observed attempts. Missing usage, including failed calls, remains explicitly unknown.',
    'Each model response is one sample; this small assistant-authored set cannot establish population-level service accuracy.',
    'Stored timestamps are UTC; the static report displays Asia/Seoul (KST).',
]


def load_cases(path):
    cases = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not cases or len(cases) > MAX_CASES:
        raise ValueError('벤치마크는 1~32개 문항만 실행할 수 있어.')
    seen = set()
    for case in cases:
        identifier = case.get('id')
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', identifier) or identifier in seen:
            raise ValueError('문항 ID는 안전한 고유 문자열이어야 해.')
        seen.add(identifier)
        if case.get('kind') not in KINDS or not isinstance(case.get('query'), str) or not case['query'].strip():
            raise ValueError('문항 종류 또는 질문이 올바르지 않아.')
        statuses = case.get('expected_status')
        if not isinstance(statuses, list) or not statuses or any(s not in STATUSES for s in statuses):
            raise ValueError('문항의 expected_status가 올바르지 않아.')
        concepts = case.get('answer_concepts', [])
        if not isinstance(concepts, list) or any(not isinstance(group, list) or not group or
                any(not isinstance(p, str) or not compact(p) for p in group) for group in concepts):
            raise ValueError('정답 개념은 비어 있지 않은 동의 표현 묶음이어야 해.')
        options = case.get('evidence_options', [])
        if not isinstance(options, list) or (case['kind'] == 'answerable' and not options):
            raise ValueError('답변 가능 문항에는 근거 라벨이 필요해.')
        for option in options:
            groups = option.get('evidence_any')
            if not isinstance(option.get('notice_id'), str) or not isinstance(groups, list) or not groups:
                raise ValueError('근거 라벨의 공지 범위나 문구가 잘못됐어.')
            if any(not isinstance(g, list) or not g or
                   any(not isinstance(p, str) or not compact(p) for p in g) for g in groups):
                raise ValueError('근거 라벨은 OR-of-AND 문구여야 해.')
        if not isinstance(case.get('gold_sources', []), list):
            raise ValueError('gold_sources는 배열이어야 해.')
    return cases


def retrieval_entry(bundles, options):
    top5 = []
    for rank, bundle in enumerate(bundles[:5], 1):
        item = deepcopy(bundle)
        item.update(rank=rank, is_evidence=is_evidence(item.get('body', ''), notice_ids(item), options))
        top5.append(item)
    return dict(top5=top5, first_relevant_rank=next((s['rank'] for s in top5 if s['is_evidence']), None))


def grade(case, request, result):
    """Label/string and citation-structure proxies, deliberately not entailment."""
    sources = request['sources']
    text = compact(str(result.get('answer') or ''))
    concepts = [any(compact(p) in text for p in group) for group in case.get('answer_concepts', [])]
    gold_ids = [source['id'] for source in sources if is_evidence(
        source.get('body', ''), notice_ids(source), case.get('evidence_options', []))]
    payload = {key: result.get(key) for key in ('status', 'claims', 'message')}
    claims = result.get('claims') if isinstance(result.get('claims'), list) else []
    valid_ids = {source['id'] for source in sources}
    cited, valid_claims, valid_citations = set(), 0, True
    for claim in claims:
        citations = claim.get('citations') if isinstance(claim, dict) else None
        good = isinstance(citations, list) and bool(citations) and all(
            type(n) is int and n in valid_ids for n in citations)
        if good and len(set(citations)) == len(citations):
            valid_claims += 1
            cited.update(citations)
        else:
            valid_citations = False
    coverage = valid_claims / len(claims) if claims else None
    checks = dict(
        structure_ok=bool(result.get('validation_passed')) and not validate_response(payload, sources),
        status_match=result.get('status') in case['expected_status'],
        content_concepts_match=all(concepts),
        citation_ids_valid=valid_citations,
        cited_gold_proxy=bool(cited.intersection(gold_ids)) if case['kind'] == 'answerable' else None,
    )
    passed = all(value for value in checks.values() if value is not None)
    if result.get('status') == 'answered':
        passed = passed and bool(claims) and coverage == 1.0
    return dict(checks=checks, automatic_proxy_pass=bool(passed), concept_results=concepts,
                retrieved_gold_ids=gold_ids, cited_ids=sorted(cited), claim_count=len(claims),
                claims_with_valid_citations=valid_claims, claim_citation_coverage=coverage)


def _usage(metadata):
    result = {}
    for field in USAGE_FIELDS:
        value = metadata.get(field) if isinstance(metadata, dict) else getattr(metadata, field, None)
        result[field] = value if type(value) is int and value >= 0 else None
    return result


class UsageLedger:
    """Observe responses before generation validation can raise; never log bodies."""
    def __init__(self, client, max_calls=MAX_API_CALLS):
        self.client, self.max_calls, self.records = client, max_calls, []
        self.case_id = None
        self.models = self

    def set_case(self, identifier):
        self.case_id = identifier

    def generate_content(self, **kwargs):
        if len(self.records) >= self.max_calls:
            raise RuntimeError('벤치마크 API 호출 상한에 도달했어.')
        row = dict(call_index=len(self.records) + 1, case_id=self.case_id,
                   status='error', usage=_usage(None), elapsed_ms=None)
        self.records.append(row)
        started = time.perf_counter()
        try:
            response = self.client.models.generate_content(**kwargs)
            row.update(status='returned', usage=_usage(getattr(response, 'usage_metadata', None)))
            return response
        except Exception as exc:
            row['error_type'] = type(exc).__name__
            raise
        finally:
            row['elapsed_ms'] = (time.perf_counter() - started) * 1000


def usage_summary(records):
    result = dict(api_call_count=len(records), response_count=sum(r.get('status') == 'returned' for r in records),
                  missing_count={}, reported_count={})
    for field in USAGE_FIELDS:
        values = [row.get('usage', {}).get(field) for row in records]
        known = [value for value in values if type(value) is int and value >= 0]
        result[field] = sum(known) if known else None
        result['missing_count'][field] = len(records) - len(known)
        result['reported_count'][field] = len(known)
    result['complete'] = bool(records) and all(count == 0 for count in result['missing_count'].values())
    return result


def percentiles(values):
    values = sorted(float(v) for v in values if v is not None)
    def percentile(p):
        if not values:
            return None
        offset = (len(values) - 1) * p
        lo, hi = math.floor(offset), math.ceil(offset)
        return values[lo] + (values[hi] - values[lo]) * (offset - lo)
    return dict(count=len(values), p50=percentile(.50), p95=percentile(.95))


def summarize(rows, cases, *, live, api_records, load_ms, warmup_ms):
    observed = {row['id']: row for row in rows}
    rows = [observed.get(case['id'], dict(case, generation={'state': 'not_run'})) for case in cases]
    answerable = [row for row in rows if row['kind'] == 'answerable']
    behavior = [row for row in rows if row['kind'] != 'answerable']
    completed = [row for row in rows if row['generation']['state'] == 'completed']
    def proxy_rate(key, selected):
        return (sum(bool(row['generation'].get('checks', {}).get(key)) for row in selected) / len(selected)
                if live and selected else None)
    retrieval = {}
    for strategy in ('baseline', 'current'):
        observed = [row for row in answerable if strategy in row]
        denominator = len(answerable)
        ready = bool(denominator) and len(observed) == denominator
        ranks = [row[strategy]['first_relevant_rank'] for row in observed]
        stats = dict(denominator=denominator, completed=len(observed))
        for k in (1, 3, 5):
            stats[f'hit_at_{k}'] = sum(r is not None and 1 <= r <= k for r in ranks) / denominator if ready else None
        stats['mrr_at_5'] = sum(1 / r for r in ranks if r is not None and 1 <= r <= 5) / denominator if ready else None
        retrieval[strategy] = stats
    claim_count = sum(row['generation'].get('claim_count', 0) for row in completed)
    covered = sum(row['generation'].get('claims_with_valid_citations', 0) for row in completed)
    passed = sum(bool(row['generation'].get('automatic_proxy_pass')) for row in rows)
    generation_proxy = dict(
        denominator=len(cases), completed=len(completed),
        pass_count=passed if live else None,
        pass_rate=passed / len(cases) if live and cases else None,
        status_match_rate=proxy_rate('status_match', rows),
        answerable_content_match_rate=proxy_rate('content_concepts_match', answerable),
        cited_gold_rate=proxy_rate('cited_gold_proxy', answerable),
        behavior_status_match_rate=proxy_rate('status_match', behavior),
        citation_ids_valid_rate=proxy_rate('citation_ids_valid', rows),
        claim_count=claim_count, claims_with_valid_citations=covered,
        claim_citation_coverage=covered / claim_count if claim_count else None,
        answered_coverage=(sum(row['generation'].get('result', {}).get('status') == 'answered'
                               for row in answerable) / len(answerable)) if live and answerable else None,
    )
    return dict(planned=len(cases), prepared=sum('current' in row for row in rows),
        generation_completed=len(completed), generation_errors=sum(row['generation']['state'] == 'error' for row in rows),
        answerable_count=len(answerable), behavior_count=len(behavior), retrieval=retrieval,
        generation_proxy=generation_proxy, usage=usage_summary(api_records),
        latency_ms=dict(initial_load=load_ms, warmup=warmup_ms,
            current_retrieval_with_embedding=percentiles(row.get('current', {}).get('retrieval_with_embedding_ms') for row in rows),
            baseline_search_only=percentiles(row.get('baseline', {}).get('search_ms') for row in rows),
            baseline_extra_query_embedding=percentiles(row.get('baseline', {}).get('extra_query_embedding_ms') for row in rows),
            generation_wall=percentiles(row['generation'].get('wall_ms') for row in rows),
            end_to_end_retrieval_plus_generation=percentiles(
                row['current']['retrieval_with_embedding_ms'] + row['generation']['wall_ms']
                for row in rows if row.get('current', {}).get('retrieval_with_embedding_ms') is not None
                and row['generation'].get('wall_ms') is not None)),
        semantic_accuracy=None, faithfulness=None)


def _environment():
    packages = {}
    for name in ('numpy', 'scikit-learn', 'sentence-transformers', 'torch', 'transformers', 'google-genai', 'beautifulsoup4'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=packages)


def _safe_error(exc, phase, identifier=None):
    return dict(type=type(exc).__name__, phase=phase, case_id=identifier,
                message='실행을 중단했어. 완료된 결과는 보존하며 원래 오류 본문·자격증명은 기록하지 않아.')


def check_hashes(hashes):
    changed = []
    for filename, expected in hashes.items():
        path = Path(filename)
        if not path.is_file() or file_sha256(path) != expected:
            changed.append(filename)
    return dict(passed=not changed, changed_paths=changed)


def _gold_paths(cases):
    """Validate literal gold quotes only; corrected/struck-through meaning is reviewed separately."""
    paths = set()
    for case in cases:
        for source in case.get('gold_sources', []):
            path = Path(source['path']).resolve()
            if not path.is_file():
                raise ValueError('정답 원문 파일을 찾을 수 없어.')
            paths.add(path)
            if path.suffix.lower() in {'.html', '.htm', '.txt', '.json', '.jsonl', '.md'}:
                text = path.read_text(encoding='utf-8')
                if path.suffix.lower() in {'.html', '.htm'}:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(text, 'html.parser')
                    text = (soup.select_one('.b-content-box') or soup).get_text(' ', strip=True)
                elif path.suffix.lower() == '.json':
                    text = json.dumps(json.loads(text), ensure_ascii=False)
                if any(compact(quote) not in compact(text) for quote in source.get('quotes', [])):
                    raise ValueError('정답 원문에서 라벨의 인용 문구를 찾을 수 없어.')
    return paths


def render_html(report):
    esc = lambda value: html.escape(str(value), quote=True)
    local_time = datetime.fromisoformat(report['completed_at_utc']).astimezone(timezone(timedelta(hours=9), 'KST')).isoformat()
    sections = []
    for row in report['results']:
        generation = row['generation']
        result = generation.get('result') or {}
        evidence = []
        for strategy in ('current', 'baseline'):
            cards = []
            for source in row.get(strategy, {}).get('top5', []):
                url = str(source.get('source_url') or '')
                link = f"<a href='{esc(url)}'>원문 공지</a>" if url.startswith(('http://', 'https://')) else esc(url)
                cards.append(f"<details><summary>{source['rank']}위 · 라벨 근거 {source['is_evidence']} · "
                             f"{esc(source.get('document_title', ''))}</summary>{link}"
                             f"<pre>{esc(source.get('text', ''))}</pre>"
                             f"<pre>{esc(json.dumps(source.get('source_spans', []), ensure_ascii=False))}</pre></details>")
            evidence.append(f"<details><summary>{esc(strategy)} 검색 근거</summary>{''.join(cards)}</details>")
        sections.append(f"<section><h2>{esc(row['id'])} · {esc(row['kind'])}</h2><p>{esc(row['query'])}</p>"
            f"<p>정답 요지: {esc(row.get('answer_note', ''))}</p><p>생성 상태: {esc(generation['state'])}</p>"
            f"<pre>{esc(result.get('answer') or '생성 미실행 또는 오류')}</pre>"
            f"<pre>{esc(json.dumps({key: value for key, value in generation.items() if key != 'result'}, ensure_ascii=False, indent=2))}</pre>"
            + ''.join(evidence) + '</section>')
    return ("<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>프로젝트 벤치마크</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px;line-height:1.6}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:12px}section{border-top:1px solid #ccd;margin-top:24px}"
        "details{margin:10px 0}summary{cursor:pointer}</style><h1>프로젝트 검색·답변 벤치마크</h1>"
        f"<p>완료 시각: {esc(local_time)} · Asia/Seoul (KST)</p>"
        "<p>자동 문구·인용 검사는 실제 의미적 정답률이 아니야. 의미 정확도와 충실성은 별도 검토 전에는 null이야.</p>"
        f"<pre>{esc(json.dumps(report['summary'], ensure_ascii=False, indent=2))}</pre>"
        f"<pre>{esc(json.dumps(report['limitations'], ensure_ascii=False, indent=2))}</pre>"
        + ''.join(sections) + '</html>')


def evaluate(cases_path, output_dir, *, index=DEFAULT_INDEX, normalized=DEFAULT_NORMALIZED,
             model=DEFAULT_GENERATION_MODEL, live=False, max_output_tokens=1600):
    cases_path, output_dir, index, normalized = map(Path, (cases_path, output_dir, index, normalized))
    if output_dir.exists():
        raise FileExistsError('기존 결과를 덮어쓸 수 없어. 새 출력 폴더를 지정해줘.')
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError('max_output_tokens는 양의 정수여야 해.')
    cases = load_cases(cases_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = [dict(deepcopy(case), generation=dict(state='not_run'), semantic_correct=None, faithfulness=None) for case in cases]
    source_hashes, code_hashes, artifact_hashes = {}, {}, {}
    load_ms = warmup_ms = None
    error = None
    client = ledger = None
    requests = []
    phase, identifier = 'initialization', None
    started_at = datetime.now(timezone.utc).isoformat()
    config = dict(model=model, live=live, max_output_tokens=max_output_tokens, max_attempts=2,
                  max_cases=MAX_CASES, max_api_calls=MAX_API_CALLS, top_k=5, candidate_k=100,
                  max_bundle_tokens=512, total_tokens=2200, baseline_strategy='vector', current_strategy='vector_context')
    def fingerprint(path, target):
        path = Path(path).resolve()
        target[str(path)] = file_sha256(path)
    try:
        for path in (cases_path, index, normalized, *_gold_paths(cases)):
            fingerprint(path, source_hashes)
        for filename in ('project_benchmark_protocol_v1.json', 'project_benchmark_dataset_v1.json'):
            extra = cases_path.parent / filename
            if cases_path.name == 'project_benchmark_v1.jsonl' and not extra.is_file():
                raise ValueError('정식 벤치마크의 사전 규칙과 데이터셋 메타데이터가 필요해.')
            if extra.is_file():
                fingerprint(extra, source_hashes)
        package = Path(__file__).resolve().parent
        for path in sorted(package.glob('*.py')):
            fingerprint(path, code_hashes)
        pyproject = package.parents[1] / 'pyproject.toml'
        if pyproject.is_file():
            fingerprint(pyproject, source_hashes)
        save_json(output_dir / 'code-snapshot.json', {path: Path(path).read_text(encoding='utf-8') for path in code_hashes})
        fingerprint(output_dir / 'code-snapshot.json', artifact_hashes)
        phase = 'load_runtime'
        started = time.perf_counter()
        runtime = load_runtime(index, normalized)
        load_ms = (time.perf_counter() - started) * 1000
        engine, metadata, embedding_model = runtime
        fingerprint(metadata['source_chunks'], source_hashes)
        selected_manifest = Path(metadata['source_chunks']).parent / 'report.json'
        if selected_manifest.is_file():
            fingerprint(selected_manifest, source_hashes)
        phase = 'warmup'
        started = time.perf_counter()
        prepare_request(cases[0]['query'], *runtime)
        warm_vector = embedding_model.encode(metadata['query_prefix'] + cases[0]['query'],
            normalize_embeddings=True, convert_to_numpy=True)
        engine.search(cases[0]['query'], warm_vector, strategy='vector', top_k=5,
                      candidate_k=100, max_bundle_tokens=512, total_tokens=2200)
        warmup_ms = (time.perf_counter() - started) * 1000
        phase = 'prepare_requests'
        for case, row in zip(cases, rows):
            identifier = case['id']
            started = time.perf_counter()
            request = prepare_request(case['query'], *runtime)
            latency = (time.perf_counter() - started) * 1000
            row['current'] = retrieval_entry(request['sources'], case.get('evidence_options', []))
            row['current']['retrieval_with_embedding_ms'] = latency
            request_path = output_dir / f'{identifier}-request.json'
            save_json(request_path, request)
            fingerprint(request_path, artifact_hashes)
            row['request_path'] = request_path.name
            requests.append(request)
            # Additional encoding is solely for the baseline and not charged to
            # the current production retrieval measurement or baseline search.
            started = time.perf_counter()
            vector = embedding_model.encode(metadata['query_prefix'] + case['query'],
                normalize_embeddings=True, convert_to_numpy=True)
            encoding_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            baseline = engine.search(case['query'], vector, strategy='vector', top_k=5,
                candidate_k=100, max_bundle_tokens=512, total_tokens=2200)
            search_ms = (time.perf_counter() - started) * 1000
            row['baseline'] = retrieval_entry(baseline, case.get('evidence_options', []))
            row['baseline'].update(search_ms=search_ms, extra_query_embedding_ms=encoding_ms)
            save_json(output_dir / f'{identifier}-retrieval.json',
                      dict(baseline=row['baseline'], current=row['current']))
            fingerprint(output_dir / f'{identifier}-retrieval.json', artifact_hashes)
        phase, identifier = 'freeze', None
        save_json(output_dir / 'frozen.json', dict(cases=cases, settings=config,
            files_sha256=source_hashes, code_sha256=code_hashes, artifacts_sha256=artifact_hashes,
            embedding_metadata=metadata, environment=_environment(),
            frozen_at_utc=datetime.now(timezone.utc).isoformat(),
            note='All labels, actual prompts, source/code hashes and metadata frozen before any Gemini call. Labels never enter the model request.'))
        fingerprint(output_dir / 'frozen.json', artifact_hashes)
        if not check_hashes({**source_hashes, **code_hashes, **artifact_hashes})['passed']:
            raise RuntimeError('동결 입력이 외부 호출 전에 바뀌었어.')
        print(f'프롬프트 {len(requests)}개 동결 완료; 실제 생성: {live}', flush=True)
        if live:
            phase = 'create_client'
            client = create_client()
            ledger = UsageLedger(client)
            for case, row, request in zip(cases, rows, requests):
                phase, identifier = 'generation', case['id']
                ledger.set_case(identifier)
                started = time.perf_counter()
                try:
                    result = generate_answer(ledger, request, model=model,
                                             max_output_tokens=max_output_tokens, max_attempts=2)
                    row['generation'] = dict(state='completed', result=result, **grade(case, request, result))
                except Exception:
                    row['generation'] = dict(state='error')
                    raise
                finally:
                    row['generation']['wall_ms'] = (time.perf_counter() - started) * 1000
                    row['generation']['usage'] = usage_summary([r for r in ledger.records if r['case_id'] == identifier])
                    save_json(output_dir / f'{identifier}-generation.json', row['generation'])
                print(identifier, row['generation']['state'], flush=True)
    except Exception as exc:
        error = _safe_error(exc, phase, identifier)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                if error is None:
                    error = _safe_error(exc, 'close_client')
    integrity = check_hashes({**source_hashes, **code_hashes, **artifact_hashes})
    if not integrity['passed'] and error is None:
        error = dict(type='FrozenInputChanged', phase='integrity_check', case_id=None,
                     message='동결한 입력 또는 코드가 변경되어 실행의 재현성을 확인할 수 없어.')
    api_records = ledger.records if ledger else []
    report = dict(settings=config, live=live, started_at_utc=started_at,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        completed=error is None and len(requests) == len(cases) and
            (not live or all(row['generation']['state'] == 'completed' for row in rows)),
        error=error, integrity=integrity, results=rows, api_calls=api_records,
        summary=summarize(rows, cases, live=live, api_records=api_records, load_ms=load_ms, warmup_ms=warmup_ms),
        files_sha256=source_hashes, code_sha256=code_hashes, artifacts_sha256=artifact_hashes,
        semantic_review=None, limitations=LIMITATIONS)
    save_json(output_dir / 'report.json', report)
    with (output_dir / 'index.html').open('x', encoding='utf-8') as output:
        output.write(render_html(report))
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=Path('eval/project_benchmark_v1.jsonl'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--normalized', type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument('--model', default=DEFAULT_GENERATION_MODEL)
    parser.add_argument('--max-output-tokens', type=int, default=1600)
    parser.add_argument('--live', action='store_true', help='실제 Gemini 생성; 없으면 검색·프롬프트 동결만 실행')
    args = parser.parse_args()
    report = evaluate(args.cases, args.output_dir, index=args.index, normalized=args.normalized,
                      model=args.model, live=args.live, max_output_tokens=args.max_output_tokens)
    if report['error']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
