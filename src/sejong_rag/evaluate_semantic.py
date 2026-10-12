"""Frozen evaluation of the accepted semantic corpus, with union evidence review.

Does not alter the production index or retrieval/generation settings. Automated
coverage is a phrase proxy; publish semantic scores only after a separate review.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import time

from sejong_rag.ask_selected import load_runtime, prepare_request, create_client, save_json, DEFAULT_NORMALIZED
from sejong_rag.benchmark import UsageLedger, usage_summary, _gold_paths
from sejong_rag.compare_decomposed import evidence_coverage
from sejong_rag.embed_chunks import file_sha256
from sejong_rag.gemini_rag import DEFAULT_GENERATION_MODEL
from sejong_rag.selected_generation import generate_answer


def now():
    return datetime.now(timezone.utc).isoformat()


def run(cases_path, protocol_path, index, output, live=False):
    cases_path, protocol_path, index, output = map(Path,(cases_path,protocol_path,index,output))
    cases=[json.loads(l) for l in cases_path.read_text().splitlines() if l.strip()]
    protocol=json.loads(protocol_path.read_text())
    if len(cases)!=36 or len({c['id'] for c in cases})!=36:
        raise ValueError('Expected 36 unique frozen cases')
    if sum(c['kind']=='answerable' for c in cases)!=24:
        raise ValueError('Expected 24 answerable cases')
    gold=_gold_paths(cases)
    for p,sha in protocol['source_hashes'].items():
        if file_sha256(Path(p))!=sha:raise ValueError('Gold source changed')
    output.mkdir(parents=True,exist_ok=False)
    paths={cases_path,protocol_path,index,DEFAULT_NORMALIZED,*gold,*Path('src/sejong_rag').glob('*.py')}
    paths.add(Path('data/catalog/syllabus-originals.json'))
    frozen={str(p):file_sha256(p) for p in sorted(paths)}
    settings=dict(model=DEFAULT_GENERATION_MODEL,top_k=5,max_bundle_tokens=512,total_tokens=2200,
                  max_output_tokens=1600,max_attempts=2,max_api_calls=64,live=live)
    save_json(output/'frozen.json',dict(created_at_utc=now(),settings=settings,sha256=frozen,cases=cases,protocol=protocol))
    start=time.perf_counter()
    runtime=load_runtime(index,DEFAULT_NORMALIZED)
    load_ms=(time.perf_counter()-start)*1000
    source=Path(runtime[1]['source_chunks'])
    save_json(output/'corpus.json',dict(chunks=len(runtime[0].chunks),source=str(source),sha256=file_sha256(source),index=str(index)))
    # Save all model-visible requests before any answer API call. Gold is never
    # placed in the request. Full request bodies are retained for later review.
    rows=[]
    for c in cases:
        start=time.perf_counter();request=prepare_request(c['query'],*runtime)
        row=dict(case=c,request=request,retrieval_ms=(time.perf_counter()-start)*1000,
                 coverage_proxy=evidence_coverage(request['sources'],c),generation_state='not_run')
        save_json(output/(c['id']+'-request.json'),request);rows.append(row)
        print(c['id'],'prepared',row['coverage_proxy']['covered'],'/',row['coverage_proxy']['total'],flush=True)
    save_json(output/'requests-frozen.json',{r['case']['id']:file_sha256(output/(r['case']['id']+'-request.json')) for r in rows})
    ledger=None;blocked=False
    client=create_client() if live else None
    try:
        if client:ledger=UsageLedger(client,max_calls=64)
        for row in rows:
            if not live:continue
            c,request=row['case'],row['request']
            direct='original_document_response' in request
            if blocked and not direct:
                row['generation_state']='not_run_after_provider_error';continue
            start=time.perf_counter()
            try:
                ledger.set_case(c['id'])
                row['result']=generate_answer(ledger,request,model=settings['model'],max_output_tokens=1600,max_attempts=2)
                row['generation_state']='completed'
            except Exception as exc:
                row.update(generation_state='error',error_type=type(exc).__name__)
                # Avoid a repeated quota/network failure storm; preserve all
                # retrieval results and still execute deterministic routing.
                blocked=True
            row['generation_ms']=(time.perf_counter()-start)*1000
            save_json(output/(c['id']+'-answer.json'),{k:v for k,v in row.items() if k not in ('request','case')})
            print(c['id'],row['generation_state'],row.get('result',{}).get('status'),flush=True)
            if not direct and not blocked:time.sleep(2)
    finally:
        if client:client.close()
        report=dict(completed_at_utc=now(),settings=settings,load_ms=load_ms,rows=rows,
                    api_calls=ledger.records if ledger else [],usage=usage_summary(ledger.records if ledger else []),
                    integrity=all(file_sha256(Path(p))==s for p,s in frozen.items()))
        save_json(output/'report.json',report)
        (output/'review.html').write_text(render_html(report),encoding='utf-8')
    if not report['integrity']:raise RuntimeError('Frozen inputs changed')


def reviewed_summary(report,review):
    if review.get('report_sha256') is None:raise ValueError('Review must bind report hash')
    by_id={r['id']:r for r in review['cases']}
    if len(by_id)!=len(report['rows']):raise ValueError('Missing or duplicate reviews')
    groups={'answerable':[],'behavior':[],'original_document':[]};claims=[]
    for row in report['rows']:
        c=row['case'];r=by_id[c['id']]
        generated=row.get('result',{}).get('claims',[])
        checks=r['claims']
        if len(checks)!=len(generated):raise ValueError('Every claim must be reviewed')
        for i,(claim,check) in enumerate(zip(generated,checks),1):
            if check['claim_index']!=i or check['citations']!=claim['citations'] or not isinstance(check['supported'],bool) or not check['reason']:
                raise ValueError('Invalid claim review')
        if not isinstance(r['correct'],bool) or not r['reason']:raise ValueError('Invalid semantic review')
        if c['kind']=='answerable' and (len(r['retrieval_items'])!=len(c['items']) or any(not isinstance(i['supported'],bool) or not i['reason'] for i in r['retrieval_items'])):
            raise ValueError('Every evidence item must be reviewed')
        groups[c['kind'] if c['kind'] in groups else 'behavior'].append((row,r));claims+=checks
    a=groups['answerable'];answered=sum(row.get('result',{}).get('status')=='answered' for row,_ in a)
    correct=sum(r['correct'] for _,r in a)
    return dict(answerable=len(a),complete_evidence_at_5=sum(all(x['supported'] for x in r['retrieval_items']) for _,r in a),
        evidence_items=sum(len(r['retrieval_items']) for _,r in a),supported_evidence_items=sum(x['supported'] for _,r in a for x in r['retrieval_items']),
        correct_answers=correct,answered=answered,answer_accuracy=correct/len(a),selective_accuracy=correct/answered if answered else None,
        supported_claims=sum(c['supported'] for c in claims),claims=len(claims),
        behavior_correct=sum(r['correct'] for _,r in groups['behavior']),behavior_total=len(groups['behavior']),
        original_document_correct=sum(r['correct'] for _,r in groups['original_document']),original_document_total=len(groups['original_document']),
        reviewer_role=review['reviewer_role'],human_approved=review['human_approved'])


def render_html(report,review=None,summary=None):
    esc=lambda x:html.escape(str(x),quote=True)
    reviews={r['id']:r for r in review['cases']} if review else {}
    cards=[]
    for row in report['rows']:
        c=row['case'];r=reviews.get(c['id']);result=row.get('result',{})
        evidence=[]
        for i,s in enumerate(row['request']['sources'],1):
            evidence.append(f'<details><summary>Rank {i} · 근거 {s["id"]} · {esc(s.get("document_title"))}</summary><a href="{esc(s.get("source_url",""))}">원문 공지</a><pre>{esc(s.get("body",s.get("text","")))}</pre><small>{esc(s.get("anchor_chunk_id"))}</small></details>')
        checks=''
        if r:
            checks='<h3>항목별 검색·인용 검토</h3>'
            for item,checked in zip(c.get('items',[]),r['retrieval_items']):
                checks+=f'<p>{esc(item["name"])} · {"확보" if checked["supported"] else "미확보"} · {esc(checked["reason"])}</p>'
            for claim,checked in zip(result.get('claims',[]),r['claims']):
                checks+=f'<details><summary>주장 {checked["claim_index"]} · {"인용 지원" if checked["supported"] else "인용 미지원"}</summary><p>{esc(claim["text"])}</p><p>근거 {esc(checked["citations"])}: {esc(checked["reason"])}</p></details>'
        label=('정답' if r['correct'] else '실패') if r else '의미 채점 대기'
        cards.append(f'<section data-failed="{str(bool(r and not r["correct"])).lower()}"><h2>{c["id"]} · {esc(c["query"])}</h2><b>{label}</b><div class="grid"><div><h3>실제 모델 답변</h3><pre>{esc(result.get("answer",row["generation_state"]))}</pre></div><div><h3>정답·기대 행동</h3><p>{esc(c["answer_note"])}</p><p>{esc(r["reason"] if r else c["rubric"])}</p></div></div><h3>LLM에 제공한 Top 5 근거</h3>{"".join(evidence) or "본문 검색 없이 원문 안내"}{checks}<details><summary>정답 원문·문구</summary><pre>{esc(json.dumps(c["gold_sources"],ensure_ascii=False,indent=2))}</pre></details></section>')
    stats=summary or dict(semantic_review='pending',usage=report['usage'])
    stats_html='<pre>'+esc(json.dumps(stats,ensure_ascii=False,indent=2))+'</pre>'
    if summary:
        metrics=[('Top 5 전체 근거 확보',summary['complete_evidence_at_5'],summary['answerable']),
                 ('항목별 근거 확보',summary['supported_evidence_items'],summary['evidence_items']),
                 ('완전한 답변 정확도',summary['correct_answers'],summary['answerable']),
                 ('실제 답변 제공',summary['answered'],summary['answerable']),
                 ('답변한 질문 중 정확도',summary['correct_answers'],summary['answered']),
                 ('주장별 인용 지원',summary['supported_claims'],summary['claims']),
                 ('행동 평가',summary['behavior_correct'],summary['behavior_total']),
                 ('강의계획서 원문 안내',summary['original_document_correct'],summary['original_document_total'])]
        stats_html='<table><tr><th>항목</th><th>결과</th></tr>'+''.join(f'<tr><td>{label}</td><td>{n}/{d} ({n/d:.1%})</td></tr>' if d else f'<tr><td>{label}</td><td>해당 없음</td></tr>' for label,n,d in metrics)+'</table>'
    return '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>새 청킹 검색·답변 평가</title><style>body{font:16px system-ui;max-width:1250px;margin:auto;padding:24px;background:#f4f7fa;color:#15314a;line-height:1.65}section{background:white;border:1px solid #d4dce4;border-radius:12px;padding:24px;margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fa;padding:16px;font:inherit}.grid{display:grid;grid-template-columns:1fr 1fr;gap:28px}summary{cursor:pointer;padding:8px}td,th{padding:8px 24px;text-align:left;border-bottom:1px solid #ccd}a{color:#1468ac}@media(max-width:750px){.grid{grid-template-columns:1fr}}</style><h1>완료된 새 청크만 사용한 평가</h1><p>상위 5개 근거의 합집합으로 평가. 원문 안내는 별도 채점. 이전 평가와 문항·검색 범위가 달라 점수를 직접 비교할 수 없음. 보조 문구 점수와 별도로 assistant가 의미 검토했으며 사용자 검수는 아님.</p>'+stats_html+'<label><input type="checkbox" onchange="document.querySelectorAll(\'section\').forEach(s=>s.hidden=this.checked&&s.dataset.failed!==\'true\')"> 실패만 보기</label>'+''.join(cards)+'</html>'


def finalize(output, review_path):
    output,review_path=Path(output),Path(review_path)
    report_path=output/'report.json'
    review=json.loads(review_path.read_text())
    if review.get('report_sha256')!=file_sha256(report_path):raise ValueError('Review/report hash mismatch')
    if review.get('reviewer_role')!='assistant' or review.get('human_approved') is not False:
        raise ValueError('Assistant review must not claim human approval')
    report=json.loads(report_path.read_text())
    summary=reviewed_summary(report,review)
    save_json(output/'final.json',dict(report_sha256=file_sha256(report_path),review_sha256=file_sha256(review_path),summary=summary))
    (output/'review.html').write_text(render_html(report,review,summary),encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,default=Path('eval/semantic_v25_benchmark_v1.jsonl'))
    p.add_argument('--protocol',type=Path,default=Path('eval/semantic_v25_protocol_v1.json'))
    p.add_argument('--index',type=Path,default=Path('data/index/sejong-semantic-v25-e5-small.npz'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--live',action='store_true')
    a=p.parse_args();run(a.cases,a.protocol,a.index,a.output,a.live)

if __name__=='__main__':main()
