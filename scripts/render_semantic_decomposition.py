"""Render reviewed RAG-only decomposition results; preserve full evidence text."""
import argparse
import html
import json
from pathlib import Path
from bs4 import BeautifulSoup
from sejong_rag.embed_chunks import file_sha256


def render(output,review_path):
    output=Path(output);review_path=Path(review_path)
    r=json.loads((output/'report.json').read_text());v=json.loads(review_path.read_text())
    assert v['report_sha256']==file_sha256(output/'report.json')
    assert len(r['rows'])==len(v['cases'])==24
    reviews={x['id']:x for x in v['cases']};esc=lambda x:html.escape(str(x),quote=True)
    css=BeautifulSoup(Path('data/processed/semantic-v25-evaluation-v1-live/failed-retrieval-cases-full.html').read_text(),'html.parser').style.string
    def sources_html(sources,prefix):
        parts=[]
        for rank,s in enumerate(sources,1):
            parts.append(f'<article class="evidence"><div class="rank-header"><span class="rank">Rank {rank}</span></div><h3>{esc(s.get("document_title",""))}</h3><p>{esc(s.get("filename") or "공지 본문")}</p><p><a href="{esc(s.get("source_url",""))}" target="_blank" rel="noopener noreferrer">원문 공지 ↗</a></p><div class="evidence-text" data-arm="{prefix}">{esc(s["text"])}</div><details class="technical"><summary>청크와 출처 정보</summary><pre>{esc(json.dumps({k:s.get(k) for k in ["anchor_chunk_id","source_spans","source_aliases","score","token_count"]},ensure_ascii=False,indent=2))}</pre></details></article>')
        return ''.join(parts)
    cards=[]
    for row in r['rows']:
        c=row['case'];cid=c['id'];check=reviews[cid]
        before=all(i['before_supported'] for i in check['items']);after=all(i['supported'] for i in check['items'])
        label='개선' if after and not before else '퇴행' if before and not after else '성공 유지' if after else '미확보 유지'
        itemrows=''.join(f'<tr><td>{esc(i["name"])}</td><td>{"확보" if i["before_supported"] else "미확보"}</td><td>{"확보" if i["supported"] else "미확보"}</td><td>{esc(i["ranks"])}</td><td>{esc(i["reason"])}</td></tr>' for i in check['items'])
        plans=''.join('<li>'+esc(q)+'</li>' for q in row['plan']['questions'])
        candidates=''.join(f'<details><summary>분해 질문 {n}: {esc(q)}</summary>{sources_html(sources,"subquery")}</details>' for n,(q,sources) in enumerate(zip(row['plan']['questions'],row['subquery_candidates']),1))
        cards.append(f'<section class="case" id="{cid}" data-failed="{str(not after).lower()}"><div class="case-heading"><span class="case-id">{cid}</span><span class="badge">{label}</span></div><h2>{esc(c["query"])}</h2><div class="answer gold"><h3>정답 · 필요한 근거</h3><p>{esc(c["answer_note"])}</p></div><h3>실제 검색에 사용한 분해 질문</h3><ol>{plans}</ol><p class="muted">원질문 검색도 함께 사용 · 분해 대체 사유: {esc(row["plan"]["fallback"] or "없음")}<br>분해 검토: {esc(check["plan_review"])}</p><div class="table-wrap"><table><tr><th>항목</th><th>분해 전</th><th>분해 후</th><th>최종 근거 순위</th><th>판정 이유</th></tr>{itemrows}</table></div><h3>분해 후 최종 Top 5 전문</h3>{sources_html(row["b"]["request"]["sources"],"b")}<details><summary>분해 전 Top 5 전문</summary>{sources_html(row["a"]["request"]["sources"],"a")}</details><details><summary>항목별 검색 후보 전문 · 최종 5개 선정 전</summary>{candidates}</details></section>')
    s=v['summary'];stats=f'<table><tr><th>RAG 지표</th><th>분해 전</th><th>분해 후</th></tr><tr><td>전체 근거 확보</td><td>{s["before_complete"]}/24 ({s["before_complete"]/24:.1%})</td><td>{s["after_complete"]}/24 ({s["after_complete"]/24:.1%})</td></tr><tr><td>항목별 근거 확보</td><td>{s["before_items"]}/55 ({s["before_items"]/55:.1%})</td><td>{s["after_items"]}/55 ({s["after_items"]/55:.1%})</td></tr></table>'
    page=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>복합 질문 분해 적용 RAG 재평가</title><style>{css}</style></head><body><div class="toolbar"><label><input type="checkbox" onchange="document.body.classList.toggle('flow',this.checked)"> 줄바꿈을 접어 읽기</label><label><input type="checkbox" onchange="document.querySelectorAll('section.case').forEach(s=>s.hidden=this.checked&&s.dataset.failed!=='true')"> 분해 후 실패만 보기</label><button onclick="window.print()">인쇄 / PDF 저장</button></div><main><h1>복합 질문 분해 적용 RAG 재평가</h1><p>같은 24문항 · 같은 새 청크 44,766개 · 최종 Top 5 / 2,200 E5 토큰 제한 · S02 정답 인정 유지</p><div class="notice">질문만 Gemini에 보내 분해한 뒤 각 질문을 임베딩·검색하고 근거를 합쳤어. 정답·평가 항목·검색 근거는 분해기에 전달하지 않았어. 이번에는 답변을 새로 생성하지 않았으며 RAG 근거 확보만 채점했어.</div>{stats}<p>개선: {esc(s['improved'])} · 퇴행: {esc(s['regressed'])} · 분해 후 실패: {esc(s['failed'])}</p><p class="muted">같은 문항의 진단 비교이며 새로운 홀드아웃 평가는 아님. 분해 전 검색은 과거 평가의 실제 Top 5와 일치하는지 확인했어. 검색·분해 로직은 결과를 보고 수정하지 않았어. 의미 판정은 assistant 검토이며 S02 외 사용자 검수는 아님.</p>{''.join(cards)}<footer>원시 기록 report.json · 검토 {esc(review_path)} · Gemini 호출은 질문 분해용으로만 수행</footer></main></body></html>'''
    soup=BeautifulSoup(page,'html.parser')
    assert len(soup.select('section.case'))==24
    assert [x.get_text() for x in soup.select('.evidence-text[data-arm=b]')]==[s['text'] for row in r['rows'] for s in row['b']['request']['sources']]
    (output/'review.html').write_text(page,encoding='utf-8')
    (output/'final.json').write_text(json.dumps(dict(report_sha256=v['report_sha256'],review_sha256=file_sha256(review_path),summary=s),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(s,ensure_ascii=False,indent=2));print(output/'review.html')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--review',required=True);a=p.parse_args();render(a.output,a.review)
