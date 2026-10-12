"""Export reviewed Top5-union retrieval and generation comparison without API calls."""
from __future__ import annotations
import html
import json
from pathlib import Path
from collections import Counter
from sejong_rag.embed_chunks import file_sha256

ROOT=Path(__file__).resolve().parents[1]
def esc(x):return html.escape(str(x if x is not None else ''))
def frac(a,b):return f'{a}/{b} · {100*a/b:.1f}%' if b else '해당 없음'
def main():
    old_path=ROOT/'data/processed/project-benchmark-v1-resumed/report.json'
    new_path=ROOT/'data/processed/project-benchmark-v2-flash38-resumed/report.json'
    old=json.loads(old_path.read_text());new=json.loads(new_path.read_text())
    union=json.loads((ROOT/'eval/project_benchmark_retrieval_review_v2.json').read_text())
    review_path=ROOT/'eval/project_benchmark_generation_review_v2.json'
    review=json.loads(review_path.read_text())
    assert review['report_sha256']==file_sha256(new_path)
    assert union['source_report_sha256']==file_sha256(old_path)
    old_reviews={}
    for name in ['project_benchmark_review_a_v1.json','project_benchmark_review_b_v1.json']:
        old_reviews.update({r['id']:r for r in json.loads((ROOT/'eval'/name).read_text())['results']})
    amendments=json.loads((ROOT/'eval/project_benchmark_amendments_v1.json').read_text())
    for r in amendments['results']:old_reviews[r['id']].update({k:r[k] for k in ['answer_correct','failure_types','reason']})
    old_by_id={r['id']:r for r in old['results']}
    ur={r['id']:r for r in union['results']};nr={r['id']:r for r in review['results']}
    assert len(nr)==len(new['results'])==32
    answerable=[r for r in new['results'] if r['kind']=='answerable']
    behavior=[r for r in new['results'] if r['kind']!='answerable']
    correct=sum(nr[r['id']]['answer_correct'] is True for r in answerable)
    answered=sum(r['generation'].get('result',{}).get('status')=='answered' for r in answerable)
    errors=[r['id'] for r in new['results'] if r['generation']['state']!='completed']
    invalid=[r['id'] for r in new['results'] if r['generation'].get('result',{}).get('validation_passed') is False]
    claims=[c for r in nr.values() for c in r['claim_checks']]
    supported=sum(c['supported'] for c in claims)
    behaviors=sum(nr[r['id']]['behavior_correct'] is True for r in behavior)
    paired=[r for r in answerable if r['generation'].get('result',{}).get('validation_passed')]
    summary=dict(evaluation_status='incomplete_provider_errors',valid_response_cases=sum(bool(r['generation'].get('result',{}).get('validation_passed')) for r in new['results']),behavior_reviewed=sum(nr[r['id']]['behavior_correct'] is not None for r in behavior),model=new['settings']['model'],retrieval_complete=union['summary']['complete'],retrieval_total=24,
        correct_answers=correct,answerable=24,answered=answered,supported_claims=supported,claims=len(claims),
        correct_behaviors=behaviors,behavior_total=8,provider_failed_case_ids=errors,format_failed_case_ids=invalid,
        paired_valid_response_count=len(paired),paired_old_correct=sum(old_reviews[r['id']]['answer_correct'] for r in paired),
        paired_new_correct=sum(nr[r['id']]['answer_correct'] is True for r in paired),usage=new['usage'],
        provider_error_counts=dict(Counter(str(c.get('provider_status_code')) for c in new['api_calls'] if c['status']=='error')),
        failed_answer_case_ids=[r['id'] for r in answerable if nr[r['id']]['answer_correct'] is not True])
    metrics=[('Top5 통합 근거 확보',frac(19,24),frac(union['summary']['complete'],24)),
        ('완전한 답변 정확도',frac(19,24),frac(correct,24)),('실제 답변 제공',frac(20,24),frac(answered,24)),
        ('답변한 문항 중 정확도',frac(19,20),frac(correct,answered)),('주장별 인용 근거 지원',frac(46,47),frac(supported,len(claims))),
        ('행동 평가',frac(8,8),frac(behaviors,8) if all(nr[r['id']]['behavior_correct'] is not None for r in behavior) else '평가 불가 · API 오류로 8문항 미확보')]
    rows=[]
    for row in new['results']:
        ident=row['id'];o=old_by_id[ident];reviewed=nr[ident];u=ur.get(ident);g=row['generation'];result=g.get('result',{})
        old_ok=old_reviews[ident]['answer_correct'] if row['kind']=='answerable' else old_reviews[ident]['behavior_correct']
        new_ok=reviewed['answer_correct'] if row['kind']=='answerable' else reviewed['behavior_correct']
        verdict='정답' if new_ok else ('API 오류' if g['state']!='completed' else '형식 실패' if result.get('validation_passed') is False else '미통과')
        union_label=('성공' if u['complete'] else '미충족') if u else '행동 평가'
        checks=''.join(f'<li>{esc(c["text"])} <b>[근거 {esc(c["citations"])}]</b> — {"지원됨" if c["supported"] else "지원 안 됨"}<p>{esc(c["reason"])}</p></li>' for c in reviewed['claim_checks'])
        items=''.join(f'<li>{esc(i["name"])}: {"지원됨 — 근거 "+esc(i["supporting_ranks"]) if i["supported"] else "근거 부족"}<br>{esc(" / ".join(" + ".join(g) for o in i["expected_evidence"] for g in o["evidence_any"]))}</li>' for i in u['items']) if u else ''
        evidence=''.join(f'<details><summary>Rank {b["rank"]} · {esc(b["document_title"])}</summary><pre>{esc(b["body"])}</pre></details>' for b in row['current']['top5'])
        rows.append(f'''<article id="{ident}" data-failed="{str(not new_ok).lower()}" data-retrieval="{str(bool(u and not u['complete'])).lower()}"><h2>{ident} · {esc(row['query'])}</h2><p>통합 검색: <b>{union_label}</b> · 새 모델: <b>{verdict}</b></p><div class="columns"><section><h3>3.5 Flash-Lite · {'정답' if old_ok else '미통과'}</h3><pre class="oldanswer">{esc(o['generation']['result']['answer'])}</pre></section><section><h3>3.8 Flash · {verdict}</h3><pre class="newanswer">{esc(result.get('answer') or 'API 오류로 답변을 확보하지 못함')}</pre><p>{esc(reviewed['reason'])}</p></section></div><details><summary>정답 기준 및 인용 검토</summary><p>{esc(row['answer_note'])}</p><p>{esc(row['rubric'])}</p><ol>{checks}</ol></details><details><summary>Top5 통합 검색 판정</summary><p>{esc(u['reason']) if u else '행동 평가 문항'}</p><ol>{items}</ol></details><h3>두 모델에 동일하게 전달한 검색 Top5</h3>{evidence}</article>''')
    table=''.join(f'<tr><td>{esc(a)}</td><td>{b}</td><td>{c}</td></tr>' for a,b,c in metrics)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>세종 RAG 재평가 · Top5 통합 근거 / Gemini 3.8 Flash</title><style>body{font:15px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#193247;max-width:1200px;margin:auto;padding:24px}article,.summary{background:white;border:1px solid #dbe3eb;border-radius:12px;padding:24px;margin:20px 0}h1{font-size:28px}h2{font-size:19px}h3{font-size:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6fa;padding:15px;border-radius:8px;font:14px/1.7 system-ui,sans-serif}.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}summary{cursor:pointer;padding:10px 0}table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #dbe3eb}.notice{background:#fff1d8;padding:16px;border-radius:8px}input,select{padding:10px;font:inherit;border:1px solid #bbcbd7;border-radius:7px}input{width:55%}article[hidden]{display:none}@media(max-width:760px){.columns{grid-template-columns:1fr}body{padding:12px}article{padding:16px}} </style><h1>Top5 통합 근거 재채점 · Gemini 3.8 Flash 재평가</h1><p>2026-10-12 · 기존 32문항의 질문·프롬프트·검색 Top5를 그대로 재사용했어. 새로 검색하거나 근거를 추가하지 않았어.</p><div class="notice"><b>새 모델 평가 미완료:</b> 32문항 중 유효 응답 12개, JSON 검증 실패 2개, API 오류로 미확보 18개. 서버 오류(503)와 사용량 제한(429)이 발생했어. 아래 전체 답변 정확도는 실패를 포함한 실행 결과이며 모델 능력 비교 점수로 사용할 수 없어.</div><div class="summary"><table><tr><th>항목</th><th>기존 3.5 Flash-Lite</th><th>새 3.8 Flash</th></tr>'''+table+'''</table><p>검색 성공: 하나의 청크가 아니라 Top5 전체를 합쳐 모든 질문 항목을 지원하면 성공. B07·B10·B15·B17을 성공으로 정정했어. 기존의 15/24(62.5%)는 단일 묶음 기준 참고 지표야. B10의 질문하지 않은 1회 제한도 필수 정답 항목에서 제외했어.</p></div>'''
    page+=f'''<div class="notice">API 오류로 답변 미확보: {esc(', '.join(errors) or '없음')}<br>JSON 응답 검증 실패: {esc(', '.join(invalid) or '없음')}<br>전체 정확도는 이 실패들도 포함해. 서버 가용성·사용량 제한·응답 형식 문제가 섞여 있으므로 수치를 모델의 순수 추론 능력 차이로 해석하면 안 돼.<br>두 모델 모두 유효 응답을 확보한 답변 가능 {len(paired)}문항: 기존 {summary['paired_old_correct']}/{len(paired)}, 새 모델 {summary['paired_new_correct']}/{len(paired)}. 이것도 응답 성공 문항만 선택한 참고 비교야.</div>'''
    page+='''<p>검토: assistant의 질문·근거·주장 대조. 재사용된 작은 평가셋이며 새 독립 검증은 아니야. 새 모델은 기본 추론 설정과 기존 출력 상한 1,600토큰을 사용했어.</p><input id="search" placeholder="질문·답변·근거 검색"><select id="filter"><option value="all">전체</option><option value="failed">새 모델 미통과</option><option value="retrieval">통합 검색 미충족</option></select><span id="count"></span>'''+''.join(rows)
    page+=f'<details><summary>사용량·평가 집계 및 추적 정보</summary><pre>{esc(json.dumps(summary,ensure_ascii=False,indent=2))}</pre><p>새 보고서 SHA-256: {file_sha256(new_path)}<br>검토 기록: {esc(str(review_path.relative_to(ROOT)))}</p></details>'
    page+='''<script>const cards=[...document.querySelectorAll('article')],s=document.querySelector('#search'),f=document.querySelector('#filter');function apply(){let n=0;for(const c of cards){c.hidden=!(c.textContent.toLowerCase().includes(s.value.toLowerCase())&&(f.value==='all'||c.dataset[f.value]==='true'));if(!c.hidden)n++;}document.querySelector('#count').textContent=' '+n+' / 32문항';}s.oninput=apply;f.onchange=apply;apply();</script></html>'''
    out=ROOT/'reports/benchmark-comparison-2026-10-12.html';out.write_text(page)
    (ROOT/'eval/project_benchmark_release_v2.json').write_text(json.dumps(dict(summary=summary,source_report_sha256=file_sha256(new_path),review_sha256=file_sha256(review_path)),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2));print(out)

if __name__=='__main__':main()
