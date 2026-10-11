"""Small diagnostic generation evaluation; labels never enter the LLM prompt."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re

from sejong_rag.ask_selected import (DEFAULT_INDEX, DEFAULT_NORMALIZED,
    create_client, load_runtime, prepare_request, save_json)
from sejong_rag.compare_selected import compact, notice_ids, is_evidence
from sejong_rag.embed_chunks import file_sha256
from sejong_rag.gemini_rag import DEFAULT_GENERATION_MODEL
from sejong_rag.selected_generation import generate_answer


def grade(case, request, result):
    """Diagnostic phrase/citation checks, not an entailment or truth guarantee."""
    text = compact(result['answer'])
    concepts = [any(compact(phrase) in text for phrase in group)
                for group in case.get('answer_concepts', [])]
    options = case.get('evidence_options', [])
    sources = request['sources']
    cited = {i for claim in result.get('claims', []) for i in claim['citations']}
    gold_ids = [s['id'] for s in sources
                if is_evidence(s['body'], notice_ids(s), options)]
    privacy_request = bool(re.search(r'(?:학번|성명|주민등록번호).{0,30}(?:알려|입력|제공|보내)', result['answer']))
    expects_answer = 'answered' in case['expected_status']
    checks = dict(status_ok=result['status'] in case['expected_status'],
        answer_concepts_ok=all(concepts), structure_ok=result['validation_passed'],
        no_identity_request=not privacy_request,
        cited_gold_ok=bool(cited.intersection(gold_ids)) if expects_answer else not cited)
    return dict(checks=checks, automatic_checks_passed=all(checks.values()),
        concept_results=concepts, retrieved_gold_ids=gold_ids,
        citation_check_scope='At least one cited body contains the labeled answer; not every claim is semantically verified.',
        manual_review=None)


def summary(rows, planned):
    keys = ('prompt_token_count', 'candidates_token_count', 'total_token_count')
    return dict(planned=planned, completed=len(rows),
        automatic_pass_count=sum(r['automatic_checks_passed'] for r in rows),
        automatic_pass_rate=(sum(r['automatic_checks_passed'] for r in rows)/len(rows)) if rows else None,
        usage={key:sum(r['result'].get('usage', {}).get(key, 0) or 0 for r in rows) for key in keys},
        note='Diagnostic reused questions; automatic checks are not independent answer accuracy.')


def render_html(report):
    esc = html.escape
    sections = []
    for row in report['results']:
        citations = []
        for source in row['result']['cited_sources']:
            url = str(source.get('source_url') or '')
            link = f"<a href='{esc(url, quote=True)}'>공지</a>" if url.startswith(('https://', 'http://')) else esc(url)
            citations.append(f"<li>[근거 {source['id']}] {esc(source.get('document_title', ''))} {link}</li>")
        sections.append(f"<section><h2>{esc(row['id'])}. {esc(row['query'])}</h2>"
            f"<p>자동 검사: {'통과' if row['automatic_checks_passed'] else '검토 필요'} / "
            f"상태: {esc(row['result']['status'])}</p><pre>{esc(row['result']['answer'])}</pre>"
            f"<ul>{''.join(citations)}</ul><details><summary>자동 검사와 사용량</summary>"
            f"<pre>{esc(json.dumps(row['checks'], ensure_ascii=False, indent=2))}</pre>"
            f"<pre>{esc(json.dumps(row['result']['usage'], ensure_ascii=False, indent=2))}</pre></details></section>")
    return ("<!doctype html><html lang='ko'><meta charset='utf-8'><title>선별 인덱스 답변 평가</title>"
        "<style>body{font-family:system-ui;max-width:1000px;margin:32px auto;padding:20px;line-height:1.6}"
        "pre{white-space:pre-wrap;background:#f4f6f8;padding:14px}section{border-top:1px solid #ccd}</style>"
        "<h1>검색 → Gemini 답변 연결 진단</h1>"
        "<p>재사용 질문의 연결 점검이며 독립 정확도가 아니야. 자동 채점과 인용 번호 검사는 의미적 정답을 보장하지 않아.</p>"
        f"<pre>{esc(json.dumps(report['summary'], ensure_ascii=False, indent=2))}</pre>"
        + ''.join(sections) + '</html>')


def evaluate(cases_path, output_dir, *, index=DEFAULT_INDEX, normalized=DEFAULT_NORMALIZED,
             model=DEFAULT_GENERATION_MODEL, live=False, max_output_tokens=1600):
    cases_path, output_dir = Path(cases_path), Path(output_dir)
    cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('문항 ID는 비어 있거나 중복되면 안 돼.')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', c['id']) for c in cases):
        raise ValueError('문항 ID 형식이 올바르지 않아.')
    if max_output_tokens <= 0:
        raise ValueError('출력 토큰 한도는 양수여야 해.')
    output_dir.mkdir(parents=True, exist_ok=False)
    tracked = [cases_path, Path(index), Path(normalized), Path(__file__),
               Path(__file__).with_name('ask_selected.py'),
               Path(__file__).with_name('selected_generation.py'),
               Path(__file__).with_name('selected_search.py')]
    hashes = {str(p.resolve()): file_sha256(p) for p in tracked}
    runtime = load_runtime(index, normalized)
    requests = []
    for case in cases:
        request = prepare_request(case['query'], *runtime)
        requests.append(request)
        save_json(output_dir / (case['id'] + '-request.json'), request)
    save_json(output_dir / 'frozen.json', dict(cases=cases, input_code_sha256=hashes,
        model=model, max_output_tokens=max_output_tokens, max_attempts=2,
        frozen_at=datetime.now(timezone.utc).isoformat(),
        note='Cases and actual requests fixed before any Gemini call; expected answers never sent.'))
    save_json(output_dir / 'code-snapshot.json', {str(p): p.read_text()
        for p in tracked if p.suffix == '.py'})
    print(f'프롬프트 {len(requests)}개 고정. 외부 호출: {live}', flush=True)
    rows, error = [], None
    client = None
    if live:
        try:
            client = create_client()
            for case, request in zip(cases, requests):
                result = generate_answer(client, request, model=model,
                    max_output_tokens=max_output_tokens, max_attempts=2)
                row = dict(id=case['id'], query=case['query'], result=result,
                    **grade(case, request, result))
                rows.append(row)
                save_json(output_dir / (case['id'] + '-answer.json'), row)
                print(case['id'], result['status'], row['checks'], flush=True)
        except Exception as exc:
            error = dict(type=type(exc).__name__,
                message='API/생성 요청 실패로 평가를 중단했어. 원래 오류 본문과 자격증명은 기록하지 않아.')
            print(error['type'], error['message'], flush=True)
        finally:
            if client is not None:
                client.close()
    for path, expected in hashes.items():
        if file_sha256(Path(path)) != expected:
            raise ValueError('평가 도중 입력/코드가 변경됐어: ' + path)
    report = dict(model=model, live=live, summary=summary(rows, len(cases)),
        completed=live and len(rows)==len(cases), error=error, results=rows,
        input_code_sha256=hashes,
        limitations=['Small reused diagnostic set, not an untouched holdout.',
            'Phrase checks can miss synonyms or accept negated facts. Source review is separate.',
            'Citation validity is not proof that every generated assertion follows from its source.',
            'Prompt tokens estimated with E5 differ from actual Gemini API usage.',
            'No retrieval, extraction, chunking, embedding, or legacy pipeline changes.'])
    save_json(output_dir / 'report.json', report)
    with (output_dir / 'index.html').open('x', encoding='utf-8') as output:
        output.write(render_html(report))
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, default=Path('eval/selected_generation_v1.jsonl'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--index', type=Path, default=DEFAULT_INDEX)
    p.add_argument('--normalized', type=Path, default=DEFAULT_NORMALIZED)
    p.add_argument('--model', default=DEFAULT_GENERATION_MODEL)
    p.add_argument('--max-output-tokens', type=int, default=1600)
    p.add_argument('--live', action='store_true', help='실제 Gemini 호출(미지정 시 프롬프트만 저장)')
    a = p.parse_args()
    report = evaluate(a.cases, a.output_dir, index=a.index, normalized=a.normalized,
        model=a.model, live=a.live, max_output_tokens=a.max_output_tokens)
    if report['error']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
