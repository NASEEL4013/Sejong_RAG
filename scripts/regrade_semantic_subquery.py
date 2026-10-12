"""Regrade saved retrieval by per-subquery Top 5; no API calls or retrieval rerun."""
import hashlib
import html
import json
from pathlib import Path

BASE = Path('data/processed/semantic-v25-decomposition-recheck-v1')
OUT = Path('data/processed/semantic-v25-subquery-top5-review-v1')
REVIEW = Path('eval/semantic_v25_decomposition_review_v1.json')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    raw_path = BASE / 'report.json'
    raw = json.loads(raw_path.read_text())
    prior = json.loads(REVIEW.read_text())
    assert sha(raw_path) == prior['report_sha256']
    reviews = {c['id']: c for c in prior['cases']}
    rows = []
    for row in raw['rows']:
        cid = row['case']['id']
        groups = row['subquery_candidates'] or [row['a']['request']['sources']]
        questions = row['plan']['questions'] if row['subquery_candidates'] else [row['case']['query']]
        assert len(groups) == len(questions)
        assert all(len(g) <= 5 for g in groups)
        items = []
        for index, old in enumerate(reviews[cid]['items']):
            references = []
            if old['supported']:
                # Reuse reviewed evidence only when its complete text survives
                # in one subquery's own context. No cross-group item stitching.
                evidence = [row['b']['request']['sources'][i-1] for i in old['ranks']]
                for qindex, group in enumerate(groups, 1):
                    ranks = [next((i for i,s in enumerate(group,1)
                                   if s['anchor_chunk_id']==e['anchor_chunk_id'] and s['text']==e['text']), None)
                             for e in evidence]
                    if all(ranks):
                        references.append(dict(subquery=qindex, ranks=ranks))
            reason = old['reason']
            if cid == 'S08' and index == 0:
                # Either previously reviewed amount source independently gives
                # monthly limit. They need not appear together in a subgroup.
                references=[]
                for qindex,group in enumerate(groups,1):
                    ranks=[i for i,s in enumerate(group,1) if s['anchor_chunk_id'] in
                           {row['b']['request']['sources'][n-1]['anchor_chunk_id'] for n in old['ranks']}]
                    if ranks: references.append(dict(subquery=qindex,ranks=ranks))
            if cid == 'S10' and index == 0:
                group=groups[0]
                table='semantic-11cd27e7cdc6c0329591d26a'
                monthly='semantic-958ce48b9d6b71aac0604445'
                ranks=[next(i for i,s in enumerate(group,1) if s['anchor_chunk_id']==key) for key in [table,monthly]]
                assert ranks == [1,5]
                assert '월 최대 20 만원' in ' '.join(group[4]['text'].split())
                references=[dict(subquery=1,ranks=ranks)]
                reason='금액·지급시기 하위질문의 1위 지급표와 5위 월 최대20만원 근거를 함께 사용하면 지급 금액·월 단위·7월 초 지급을 모두 확보함.'
            if old['supported']:
                assert references, (cid,old['name'])
            items.append(dict(name=old['name'],supported=bool(references),evidence=references,
                              reason=reason,previous_merged_top5_supported=old['supported']))
        rows.append(dict(id=cid,query=row['case']['query'],answer_note=row['case']['answer_note'],
                         complete=all(i['supported'] for i in items),items=items,
                         subqueries=[dict(question=q,sources=g) for q,g in zip(questions,groups)]))
    summary=dict(questions=len(rows),complete=sum(x['complete'] for x in rows),
                 items=sum(len(x['items']) for x in rows),supported_items=sum(i['supported'] for x in rows for i in x['items']),
                 failed=[x['id'] for x in rows if not x['complete']])
    assert summary == dict(questions=24,complete=22,items=55,supported_items=52,failed=['S04','S13'])
    result=dict(protocol=dict(metric='Each required item must be supported within at least one subquery Top 5; all items required for question success. Multiple chunks within that Top 5 may jointly support the item.',
                              rule_change_authorized_by_user=True,semantic_reviewer='assistant',
                              unsplit_questions='Original query Top 5',merged_top5_not_primary=True,
                              retrieval_rerun=False,answer_generation_run=False,
                              limitation='Different total context budget from merged Top 5; this is rescoring saved retrieval, not a same-budget improvement.'),
                report_sha256=sha(raw_path),prior_review_sha256=sha(REVIEW),summary=summary,rows=rows)
    OUT.mkdir(exist_ok=True)
    (OUT/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    e=lambda x:html.escape(str(x))
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>하위질문별 Top 5 근거 확보 평가</title><style>body{font:17px/1.7 sans-serif;max-width:1150px;margin:40px auto;padding:20px;background:#f6f8fb}section,article{background:white;padding:24px;margin:20px 0;border:1px solid #ccd5df;border-radius:10px}pre{white-space:pre-wrap;font:inherit}table{border-collapse:collapse}td,th{border:1px solid #ccd5df;padding:10px}summary{cursor:pointer}a{color:#175b9a}</style><h1>하위질문별 Top 5 근거 확보 평가</h1>',
           '<p>사용자 변경 기준: 각 필수 항목의 근거가 어느 하위질문의 Top 5 안에 확보되면 인정. 모든 항목을 확보해야 해당 원질문 성공. 최종 5개로 다시 압축하지 않음. 분해하지 않은 질문은 원질문의 Top 5로 평가.</p>',
           '<p><b>22/24문항 (91.7%) · 52/55항목 (94.5%)</b> · 실패 S04, S13. S02 정답 인정 유지.</p>',
           '<p>기존 검색 기록을 새 기준으로 재채점했어. API 재호출·답변 생성은 하지 않았어. 하위질문마다 최대 5개이므로 기존 최종 5개 평가와 총 근거량 제한이 달라.</p>',
           '<p>S10은 금액·지급시기 하위질문 1위 지급표와 5위 월 최대20만원 근거로 성공. 아래 순위는 각 하위질문 안에서의 순위야.</p>',
           '<label><input type="checkbox" onchange="document.querySelectorAll(\'section\').forEach(x=>x.hidden=this.checked&&x.dataset.ok===\'true\')">실패 문항만 보기</label>']
    for row in rows:
        parts.append(f'<section id="{row["id"]}" data-ok="{str(row["complete"]).lower()}"><h2>{row["id"]} · {"성공" if row["complete"] else "미확보"}</h2><p>{e(row["query"])}</p><p><b>정답:</b> {e(row["answer_note"])}</p><table><tr><th>항목</th><th>판정</th><th>근거 위치</th></tr>')
        for item in row['items']:
            loc='; '.join(f'하위질문 {v["subquery"]} — Rank {v["ranks"]}' for v in item['evidence']) or '미확보'
            parts.append(f'<tr><td>{e(item["name"])}</td><td>{"확보" if item["supported"] else "미확보"}</td><td>{e(loc)}</td></tr>')
        parts.append('</table>')
        for n,q in enumerate(row['subqueries'],1):
            parts.append(f'<details open><summary><b>하위질문 {n}: {e(q["question"])}</b></summary>')
            for rank,s in enumerate(q['sources'],1):
                parts.append(f'<article><h3>Rank {rank} · {e(s["document_title"])}</h3><a href="{e(s["source_url"])}">원문</a><pre>{e(s["text"])}</pre></article>')
            parts.append('</details>')
        parts.append('</section>')
    parts.append('</html>')
    page=''.join(parts)
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(page,'html.parser')
    assert [p.get_text() for p in soup.select('article pre')]==[s['text'] for row in rows for q in row['subqueries'] for s in q['sources']]
    (OUT/'review.html').write_text(page)
    print(json.dumps(summary,ensure_ascii=False)); print(OUT/'review.html')

if __name__=='__main__':main()
