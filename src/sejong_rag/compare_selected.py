"""Compare frozen source-grounded retrieval labels against actual evidence bundles.

No generation, re-embedding, label repair, or network model loading is performed.
Raw candidate ranking and the final budgeted context are intentionally scored apart.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import html
from importlib.metadata import version
import json
from pathlib import Path
import re
import time

import numpy as np

from sejong_rag.embed_chunks import file_sha256
from sejong_rag.selected_search import load_search


STRATEGIES = ('vector', 'hybrid', 'vector_context', 'improved')


def compact(text):
    """Keep the original evaluator's whitespace/case-only matching rule."""
    return re.sub(r'\s+', '', text).casefold()


def satisfies(text, groups):
    text = compact(text)
    return any(all(compact(phrase) in text for phrase in group) for group in groups)


def notice_ids(item):
    ids = {item.get('parent_document_id'), item.get('document_id')}
    for alias in item.get('source_aliases', []):
        ids.update((alias.get('parent_document_id'), alias.get('document_id')))
    return ids - {None}


def label_options(cases, audit=None):
    """An OR of (notice scope, phrase groups); do not mix different source labels."""
    result = {}
    for case in cases:
        groups = case['evidence_any']
        if (not groups or not all(isinstance(g, list) and g for g in groups)
                or not all(isinstance(p, str) and compact(p) for g in groups for p in g)):
            raise ValueError(f"Invalid evidence phrases: {case['id']}")
        scopes = [case['notice_id'], *case.get('additional_notice_ids', [])]
        result[case['id']] = [dict(notice_id=nid, evidence_any=groups,
                                      origin='case_file') for nid in dict.fromkeys(scopes)]
    for accepted in (audit or {}).get('accepted_alternatives', []):
        case_id = accepted['case_id']
        if case_id not in result:
            raise ValueError(f'Audit refers to an unknown case: {case_id}')
        phrase = accepted['required_phrase']
        if not isinstance(phrase, str) or not compact(phrase):
            raise ValueError(f'Invalid audit phrase: {case_id}')
        result[case_id].append(dict(notice_id=accepted['notice_id'],
            evidence_any=[[phrase]], origin='pre_existing_audit',
            reason=accepted.get('reason', '')))
    return result


def is_evidence(body, scopes, options):
    return any(option['notice_id'] in scopes and satisfies(body, option['evidence_any'])
               for option in options)


def first_rank(items, key='is_evidence'):
    return next((rank for rank, item in enumerate(items[:5], 1) if item[key]), None)


def summarize(rows):
    """MRR@5 assigns zero to every case with no complete evidence in top five."""
    if not rows:
        return {}
    result = {'questions': len(rows)}
    for prefix, key in (('raw_anchor', 'raw_anchor_first_rank'),
                        ('context', 'context_first_rank')):
        ranks = [row[key] for row in rows]
        result[prefix + '_hit_at_1'] = sum(r == 1 for r in ranks) / len(rows)
        result[prefix + '_hit_at_5'] = sum(r is not None for r in ranks) / len(rows)
        result[prefix + '_mrr_at_5'] = sum(1/r for r in ranks if r is not None) / len(rows)
    for key in ('context_tokens', 'serialized_context_tokens', 'warm_search_ms'):
        values = [row[key] for row in rows]
        result[key + '_mean'] = float(np.mean(values))
        result[key + '_max'] = float(np.max(values))
    result['bundle_count_mean'] = float(np.mean([len(row['top5']) for row in rows]))
    result['expanded_bundle_count'] = sum(
        bool(item.get('context_expanded')) for row in rows for item in row['top5'])
    return result


def serialized_context(items):
    return '\n\n'.join(f"[근거 {rank}]\n{item['text']}"
                        for rank, item in enumerate(items, 1))


def annotate(items, options):
    """Score exactly the assembled body, never its title or other metadata."""
    result = []
    for rank, item in enumerate(items, 1):
        row = dict(item)
        scopes = notice_ids(item)
        row.update(rank=rank, notice_ids=sorted(scopes),
                   is_evidence=is_evidence(item['body'], scopes, options))
        result.append(row)
    return result


def raw_item(engine, index, score):
    chunk = engine.chunks[index]
    return dict(chunk_id=chunk['chunk_id'], anchor_chunk_id=chunk['chunk_id'],
        chunk_ids=[chunk['chunk_id']], score=float(score),
        document_id=chunk.get('document_id'), parent_document_id=chunk.get('parent_document_id'),
        document_title=chunk.get('document_title', ''), source_url=chunk.get('source_url', ''),
        source_aliases=chunk.get('source_aliases', []), source_spans=chunk['source_spans'],
        body=engine.bodies[index], text=chunk['text'], token_count=engine.count(chunk['text']),
        context_expanded=False)


def result_row(engine, raw, contexts, latency):
    return dict(raw_anchor_first_rank=first_rank(raw), context_first_rank=first_rank(contexts),
        raw_top5=raw, top5=contexts, context_tokens=sum(c['token_count'] for c in contexts),
        serialized_context_tokens=engine.count(serialized_context(contexts)) if contexts else 0,
        warm_search_ms=float(latency))


def render_html(report):
    esc = html.escape
    names = {'baseline_rawvector': '기존 원본 벡터 top 5', 'vector': '벡터 + 공통 길이 제한',
             'hybrid': '벡터·키워드 결합', 'vector_context': '벡터 + 인접 문맥',
             'improved': '결합 검색 + 인접 문맥'}
    def summary_row(name, stats):
        return (f"<tr><td>{esc(names[name])}</td>"
                f"<td>{stats['raw_anchor_hit_at_1']:.0%} / {stats['raw_anchor_hit_at_5']:.0%}</td>"
                f"<td>{stats['context_hit_at_1']:.0%} / {stats['context_hit_at_5']:.0%}</td>"
                f"<td>{stats['context_mrr_at_5']:.3f}</td>"
                f"<td>{stats['context_tokens_mean']:.0f} / {stats['context_tokens_max']:.0f}</td>"
                f"<td>{stats['warm_search_ms_mean']:.1f}</td></tr>")
    summary = summary_row('baseline_rawvector', report['baseline_rawvector']['metrics'])
    summary += ''.join(summary_row(name, stats) for name, stats in report['strategy_metrics'].items())
    sections = []
    for row in report['results']:
        methods = []
        for name, entry in [('baseline_rawvector', row['baseline_rawvector']), *row['strategies'].items()]:
            cards = []
            for item in entry['top5']:
                marker = '정답 문구 포함' if item['is_evidence'] else '정답 문구 미포함'
                url = str(item.get('source_url', ''))
                link = (f"<a href='{esc(url, quote=True)}'>원문</a>"
                        if url.startswith(('https://', 'http://')) else esc(url))
                cards.append(f"<details><summary>{item['rank']}위 · {marker} · "
                    f"{esc(str(item.get('document_title') or ''))}</summary><p>{link} · "
                    f"{item['token_count']}토큰 · 청크 {len(item.get('chunk_ids', []))}개</p>"
                    f"<pre>{esc(item['body'])}</pre></details>")
            methods.append(f"<details><summary>{esc(names[name])}: 원래 청크 정답 순위 "
                f"{entry['raw_anchor_first_rank'] or '5위 밖'}, 전달 문맥 정답 순위 "
                f"{entry['context_first_rank'] or '5위 밖'}</summary>{''.join(cards)}</details>")
        sections.append(f"<section><h2>{esc(row['id'])}. {esc(row['query'])}</h2>"
            f"<p>정답 요지: {esc(row['answer_note'])}</p>{''.join(methods)}</section>")
    return ("<!doctype html><html lang='ko'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>검색 방식 비교 평가</title><style>body{font-family:system-ui,sans-serif;"
        "max-width:1200px;margin:24px auto;padding:16px;color:#17212e;line-height:1.6}"
        "table{border-collapse:collapse;width:100%;font-size:.9em}td,th{border:1px solid #ccd;padding:8px}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:12px}"
        "section{border-top:1px solid #ccd;margin-top:24px}summary{cursor:pointer;padding:8px}"
        "details details{margin-left:16px}h2{font-size:1.1em}</style>"
        f"<h1>검색 방식 비교: {len(report['results'])}문항</h1>"
        "<p>원래 청크 순위와 실제 전달 문맥을 별도로 채점했어. 제목은 정답 근거로 세지 않아. "
        "관련 공지의 본문 안에 사전에 지정한 답 문구가 함께 있어야 정답으로 인정해.</p>"
        "<table><thead><tr><th>방식</th><th>원래 청크 1위 / 5위 내</th>"
        "<th>전달 문맥 1위 / 5위 내</th><th>전달 문맥 MRR@5</th>"
        "<th>근거 토큰 평균 / 최대</th><th>검색 평균 ms</th></tr></thead>"
        f"<tbody>{summary}</tbody></table>"
        "<p>MRR@5는 첫 정답 순위의 역수 평균으로, 상위 5개에 없으면 0점이야. "
        "토큰은 답변 모델이 아닌 E5 기준이며, 개별 근거의 제목·본문을 포함해. "
        "검색 시간은 초기 준비와 질문 임베딩을 제외한 반복 실행 시간이고, 기존 원본 벡터 기준은 문맥 조립도 생략해.</p>"
        "<p>기존 원본 벡터 기준에는 문맥 길이 제한이 없어. 나머지는 같은 개수·토큰 한도를 적용했어. "
        "문항 수가 적고 정답 표현 라벨이 완전하지 않아 전체 서비스 정확도로 해석하면 안 돼.</p>"
        + ''.join(sections) + '</html>')


def evaluate(cases_path, index_path, normalized_path, output_dir, audit_path=None,
             strategies=STRATEGIES, data_dir=Path('data'), candidate_k=100,
             max_bundle_tokens=512, total_tokens=2200):
    cases_path, index_path, normalized_path, output_dir, data_dir = map(Path,
        (cases_path, index_path, normalized_path, output_dir, data_dir))
    strategies = tuple(strategies)
    if not strategies or len(set(strategies)) != len(strategies) or set(strategies) - set(STRATEGIES):
        raise ValueError('Specify distinct known strategies')
    if output_dir.exists():
        raise FileExistsError('Use a fresh evaluation directory')
    if min(candidate_k, max_bundle_tokens, total_tokens) <= 0:
        raise ValueError('Search budgets must be positive')
    cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Cases must be nonempty with unique IDs')
    audit = json.loads(Path(audit_path).read_text()) if audit_path else None
    options = label_options(cases, audit)
    input_hashes = {}
    def fingerprint(path):
        path = Path(path).resolve()
        input_hashes[str(path)] = file_sha256(path)
    for path in (cases_path, index_path, normalized_path):
        fingerprint(path)
    if audit_path:
        fingerprint(audit_path)
    code_hashes = {str(path.resolve()): file_sha256(path) for path in
        (Path(__file__), Path(__file__).with_name('selected_search.py'),
         Path(__file__).with_name('vector_search.py'), Path(__file__).with_name('embed_chunks.py'))}
    from bs4 import BeautifulSoup
    originals = {}
    for case in cases:
        for option in options[case['id']]:
            nid = option['notice_id']
            if not isinstance(nid, str) or not re.fullmatch(r'notice\d+-\d+', nid):
                raise ValueError(f'Unexpected notice ID: {nid}')
            if nid not in originals:
                raw = data_dir / 'raw' / 'notices' / f'{nid}.html'
                fingerprint(raw)
                originals[nid] = BeautifulSoup(raw.read_text(), 'html.parser').get_text(' ')
            if not satisfies(originals[nid], option['evidence_any']):
                raise ValueError(f"Unsupported gold label: {case['id']} / {nid}")
    load_start = time.perf_counter()
    engine, meta = load_search(index_path, normalized_path)
    search_setup_seconds = time.perf_counter() - load_start
    fingerprint(meta['source_chunks'])
    scopes = [notice_ids(chunk) for chunk in engine.chunks]
    gold = {case['id']: [chunk['chunk_id'] for chunk, body, scope in
            zip(engine.chunks, engine.bodies, scopes)
            if is_evidence(body, scope, options[case['id']])] for case in cases}
    settings = dict(strategies=list(strategies), top_k=5, candidate_k=candidate_k,
                    max_bundle_tokens=max_bundle_tokens, total_tokens=total_tokens,
                    model=meta['model'], query_prefix=meta['query_prefix'], device='cpu', cpu_threads=8)
    frozen = dict(cases=cases, label_options=options, source_sha256=input_hashes,
        code_sha256=code_hashes, relevant_original_chunk_ids=gold, settings=settings,
        frozen_at=datetime.now(timezone.utc).isoformat(),
        note='Labels, input/code hashes and settings frozen before query encoding or ranking; '
             'complete original chunks are not required for a context bundle to contain valid evidence.')
    output_dir.mkdir(parents=True)
    ground_truth = output_dir / 'ground_truth.json'
    ground_truth.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + '\n')
    frozen_hash = file_sha256(ground_truth)
    print(f'Frozen {len(cases)} questions before query encoding; '
          f'{sum(bool(v) for v in gold.values())} have complete original-chunk evidence', flush=True)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)
    model_start = time.perf_counter()
    model = SentenceTransformer(meta['model'], device='cpu', local_files_only=True)
    model_setup_seconds = time.perf_counter() - model_start
    queries = [meta['query_prefix'] + case['query'] for case in cases]
    if any(len(model.tokenizer.encode(query)) > model.max_seq_length for query in queries):
        raise ValueError('Question would be truncated')
    model.encode(queries[:1], normalize_embeddings=True, convert_to_numpy=True)
    encode_start = time.perf_counter()
    qvectors = model.encode(queries, batch_size=20, normalize_embeddings=True, convert_to_numpy=True)
    encoding_seconds = time.perf_counter() - encode_start
    kwargs = dict(top_k=5, candidate_k=candidate_k,
                  max_bundle_tokens=max_bundle_tokens, total_tokens=total_tokens)
    # Warm each path before recording query latency; no labels influence search.
    engine.rank(cases[0]['query'], qvectors[0], hybrid=False, candidate_k=5)
    for strategy in strategies:
        engine.search(cases[0]['query'], qvectors[0], strategy=strategy, **kwargs)
    results = []
    per_strategy = defaultdict(list)
    baselines = []
    for number, (case, qvector) in enumerate(zip(cases, qvectors), 1):
        case_options = options[case['id']]
        started = time.perf_counter()
        baseline_ranked = engine.rank(case['query'], qvector, hybrid=False, candidate_k=5)
        baseline_ms = (time.perf_counter() - started) * 1000
        raw = annotate([raw_item(engine, i, score) for i, score in baseline_ranked], case_options)
        baseline = result_row(engine, raw, raw, baseline_ms)
        baselines.append(baseline)
        row = dict(**case, indexed_original_evidence_count=len(gold[case['id']]),
                   baseline_rawvector=baseline, strategies={})
        # Rotate timing order to avoid always favoring a path through warm caches.
        ordered = strategies[number % len(strategies):] + strategies[:number % len(strategies)]
        temporary = {}
        for strategy in ordered:
            ranked = engine.rank(case['query'], qvector,
                                 hybrid=strategy in {'hybrid', 'improved'}, candidate_k=candidate_k)
            anchor_items = annotate([raw_item(engine, i, score) for i, score in ranked[:5]], case_options)
            started = time.perf_counter()
            bundles = engine.search(case['query'], qvector, strategy=strategy, **kwargs)
            latency = (time.perf_counter() - started) * 1000
            contexts = annotate(bundles, case_options)
            temporary[strategy] = result_row(engine, anchor_items, contexts, latency)
        for strategy in strategies:
            row['strategies'][strategy] = temporary[strategy]
            per_strategy[strategy].append(temporary[strategy])
        results.append(row)
        print(f"{case['id']}: baseline={baseline['context_first_rank']} " + ' '.join(
            f"{strategy}={row['strategies'][strategy]['context_first_rank']}" for strategy in strategies), flush=True)
    for path, fingerprint_value in {**input_hashes, **code_hashes}.items():
        if file_sha256(Path(path)) != fingerprint_value:
            raise ValueError(f'Frozen input/code changed during evaluation: {path}')
    if file_sha256(ground_truth) != frozen_hash:
        raise ValueError('Ground-truth file changed during evaluation')
    by_category = {}
    for category in sorted({case['category'] for case in cases}):
        indices = [i for i, case in enumerate(cases) if case['category'] == category]
        by_category[category] = {strategy: summarize([per_strategy[strategy][i] for i in indices])
                                for strategy in strategies}
    report = dict(settings=settings, cases=str(cases_path), index=str(index_path),
        chunk_count=len(engine.chunks), results=results,
        baseline_rawvector=dict(metrics=summarize(baselines),
            note='Original stored top-five vector chunks, no context assembly, deduplication or budget filter.'),
        strategy_metrics={strategy: summarize(rows) for strategy, rows in per_strategy.items()},
        by_category=by_category, input_sha256=input_hashes, code_sha256=code_hashes,
        ground_truth_sha256=frozen_hash, frozen_at=frozen['frozen_at'],
        completed_at=datetime.now(timezone.utc).isoformat(),
        timings=dict(search_setup_seconds=search_setup_seconds, model_setup_seconds=model_setup_seconds,
            question_encoding_batch_seconds=encoding_seconds,
            question_encoding_amortized_ms=encoding_seconds * 1000 / len(cases)),
        library_versions={name: version(name) for name in
                          ('numpy', 'scikit-learn', 'sentence-transformers', 'torch', 'transformers')},
        limitations=[
            'Small assistant-authored answerable question set; not representative service accuracy.',
            'Source-grounded phrase labels are not exhaustive semantic judgments.',
            'Only complete evidence within one actual returned body counts; headers never count.',
            'A raw anchor is the candidate before context assembly or budget/deduplication filtering.',
            'MRR@5 treats ranks beyond five as zero, unlike the older full-ranking MRR.',
            'Tokens use the local E5 tokenizer, not an answer-generation model tokenizer.',
            'context_tokens sums complete per-bundle tokens; serialized_context_tokens also counts numbered wrappers.',
            'Warm search timing excludes setup, question encoding, scoring, serialization, and generation.',
            'Raw-vector baseline timing excludes context assembly; other strategy timing includes assembly.',
            'Audit alternatives, when supplied, were known before this run and are not post-hoc repairs.',
            'No answer generation, abstention, ambiguous-query behavior, re-embedding, or service deployment tested.'])
    (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (output_dir / 'index.html').write_text(render_html(report))
    print(json.dumps(report['strategy_metrics'], ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--index', type=Path, default=Path('data/index/sejong-2025-2026-selected-e5-small.npz'))
    parser.add_argument('--normalized', type=Path, default=Path('data/processed/normalized-text/records.jsonl'))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--strategies', default=','.join(STRATEGIES))
    parser.add_argument('--candidate-k', type=int, default=100)
    parser.add_argument('--max-bundle-tokens', type=int, default=512)
    parser.add_argument('--total-tokens', type=int, default=2200)
    args = parser.parse_args()
    evaluate(args.cases, args.index, args.normalized, args.output_dir, args.audit,
             strategies=[s.strip() for s in args.strategies.split(',')], data_dir=args.data_dir,
             candidate_k=args.candidate_k, max_bundle_tokens=args.max_bundle_tokens, total_tokens=args.total_tokens)


if __name__ == '__main__':
    main()
