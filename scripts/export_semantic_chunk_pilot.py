"""Export the local semantic chunk pilot with exact before/after evidence."""
from __future__ import annotations
import html,json
from pathlib import Path
from sejong_rag.embed_chunks import file_sha256

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/processed/semantic-chunk-pilot-v1'
def esc(value):return html.escape(str(value))
def ratio(n,d):return f'{n}/{d} · {100*n/d:.1f}%'

NOTES={
'B01':'성공으로 개선. 환불 표를 반환금액 항목으로 분리하자 해당 근거의 벡터 순위가 7위→1위가 됐다. 변경 후 Top5 #1에서 12/28 취소 시 2/3 반환과 1/3 0시 이후 취소·환불 불가를 모두 확인했다.',
'B02':'미충족 유지. 신청 메뉴 경로는 원래도 69토큰의 독립 청크였고 이 구간은 본문·제목 그대로다. 벡터 순위 16위→18위로 Top5에 들지 못했다. 당일 대여 불가 조건은 #1에 있다. 다른 청크를 세분화한 것만으로 메뉴 검색 문제를 해결하지 못했다.',
'B05':'부분 개선이나 완전 성공은 아님. 공과대학 2학기 수강생도 레벨테스트를 봐야 한다는 2025년 조건이 변경 후 #1에 들어왔다. 일괄 수강신청은 #2에 있다. 그러나 「상기 기간」이 가리키는 2/14~2/19 근거가 빠졌다. #5는 추가합격자의 다른 응시기간이라 대신 쓸 수 없다. 단일 청크로 항목 전체를 지원하는 순위 없음은 상호참조 문장이 여러 청크에 나뉘어 있기 때문이다.',
'B12':'성공으로 개선. 전화·이메일이 있는 공지 본문 청크를 구비서류·우편주소에서 분리하자 연락처 청크의 벡터 순위가 6위→1위가 됐다. 변경 후 #1에서 2025년 전화와 이메일을 모두 확인했다. 기존 전화는 첨부 #3에도 있었으므로 기존 전화 항목은 의미 검토에서 지원됨으로 인정한다.',
'B21':'미충족 유지. 가구원 재동의 예외는 이미 서류제출 항목의 179토큰 청크에 있으며 이번 규칙으로 해당 구간은 더 나뉘지 않았다. 벡터 순위도 17위 그대로다. Top5 #1에 신청 마감은 있지만 예외는 없다. 추가 의미 분할 또는 항목별 검색은 이번 실험에서 시험하지 않았다.'}


def main():
    path=BASE/'evaluation.json';data=json.loads(path.read_text())
    changed=json.loads((BASE/'changed-chunks.json').read_text())
    old_review=json.loads((ROOT/'eval/project_benchmark_retrieval_review_v2.json').read_text())
    originals={r['id']:r for r in old_review['results']}
    reviews=[]
    for row in data['rows']:
        identifier=row['id'];before=originals[identifier]['complete'];after=row['after']['item_phrase_proxy']['complete']
        before_items=[i['supported'] for i in originals[identifier]['items']]
        after_items=row['after']['item_phrase_proxy']['items']
        reviews.append(dict(id=identifier,before_complete=before,after_complete=after,
            before_items=before_items,after_items=after_items,
            reason=NOTES.get(identifier,'原 성공 문항. 실제 전달 본문의 항목별 문구 일치를 확인했으며 완전 근거 확보 판정이 유지된다.').replace('原','기존')))
    summary=dict(before_complete=sum(r['before_complete'] for r in reviews),after_complete=sum(r['after_complete'] for r in reviews),total=24,
        target_before=0,target_after=2,target_total=5,regressed=[r['id'] for r in reviews if r['before_complete'] and not r['after_complete']],
        improved=[r['id'] for r in reviews if not r['before_complete'] and r['after_complete']],
        before_supported_items=sum(sum(r['before_items']) for r in reviews),after_supported_items=sum(sum(r['after_items']) for r in reviews),total_items=sum(len(r['after_items']) for r in reviews))
    review=dict(evaluation_sha256=file_sha256(path),reviewer='assistant',human_approved=False,summary=summary,results=reviews,
        criterion_note='Top5 union with matching year/entity/conditions. B05 explicitly requires February reference-period evidence; 2/14 phrase was added to both arms in cases.json before embeddings and comparison. No query/answer labels enter splitting.')
    (ROOT/'eval/semantic_chunk_pilot_review_v1.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    by_id={r['id']:r for r in reviews}
    table=''.join(f'<tr><td><a href="#{r["id"]}">{r["id"]}</a></td><td>{"성공" if r["before_complete"] else "미충족"}</td><td>{"성공" if r["after_complete"] else "미충족"}</td><td>{esc(r["reason"])}</td></tr>' for r in reviews if r['id'] in NOTES)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>의미 단위 청킹 실험</title><style>body{font:15px/1.7 system-ui,sans-serif;color:#193247;background:#f3f6fa;max-width:1250px;margin:auto;padding:24px}section,article{background:white;border:1px solid #dce4eb;padding:22px;border-radius:12px;margin:20px 0}h1{font-size:29px}h2{font-size:20px}h3{font-size:17px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.7 system-ui,sans-serif;background:#f3f7fb;padding:15px;border-radius:8px;max-height:580px;overflow:auto}.columns{display:grid;grid-template-columns:1fr 1fr;gap:22px}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid #dce4eb;padding:10px;vertical-align:top}summary{cursor:pointer;padding:8px 0}a{color:#075fa5}input{font:inherit;padding:10px;width:70%}.note{background:#fff1d9;padding:15px}article[hidden]{display:none}@media(max-width:760px){.columns{grid-template-columns:1fr}body{padding:10px}}</style><h1>의미 단위 청킹 · 다섯 공지 본문 실험</h1><p>2026-10-12 · E5 임베딩·vector_context·Top5/512/2,200토큰 검색 설정을 고정했어. 답변 생성 API는 호출하지 않았어.</p>'''
    page+=f'''<section><h2>결과</h2><p><strong>Top5 통합 근거 확보: {ratio(summary['before_complete'],24)} → {ratio(summary['after_complete'],24)}</strong><br>실패 5문항 중 2개 개선 · 기존 성공 19문항의 성공 판정 유지<br>공지 본문 청크 27개 → 64개, 나머지 {data['unchanged_chunks']:,}개 청크·벡터 보존</p><table><tr><th>문항</th><th>기존</th><th>변경</th><th>실제 근거 검토</th></tr>{table}</table></section>'''
    page+='''<p class="note">알려진 실패 문서로 진행한 소규모 개발 실험이야. 전체 데이터에 같은 향상을 보장하지 않아. 운영 인덱스는 바꾸지 않았어. 문항별 검색 성공을 재검토했지만 생성 답변 정확도는 측정하지 않았어.</p><details><summary>방법과 남은 한계</summary><p>번호·기호가 다음 줄의 항목 제목과 떨어진 경우도 경계로 인식하고, 표의 기간·금액·주의사항을 항목 안에 유지했어. 문서 제목 접두사와 청크 상한 448토큰은 그대로야. 본문은 정규화 원문의 정확한 구간이며 모든 비공백 문자의 보존을 검증했어. 실험용 새 청크의 passage 포함 최대 길이는 450 E5토큰으로 임베딩 잘림은 없어.</p><p>이 규칙은 LLM 의미 이해가 아니라 문서 구조를 활용한 분할이야. 일부 상위 제목만 남는 작은 청크, 상기 기간 같은 상호참조, 기존 인접 확장이 다른 항목까지 다시 합치는 문제가 남아 있어. B21의 서류제출 구간도 이번 규칙에서는 그대로야. 결과를 본 뒤 규칙을 추가 조정하지 않았어.</p><p>B05는 질문의 2월 조건을 검증하기 위해 두 비교군 모두 원래 응시기간(2/14) 근거를 요구하도록 실험 전에 고정했어. 본문에 있는 2학기 응시 의무만으로는 기간까지 확인한 것으로 채점하지 않았어.</p></details><p><a href="#chunks">청크 분할 전후 전체 보기 ↓</a></p><input id="search" placeholder="질문·근거 검색"><span id="count"></span>'''
    for row in data['rows']:
        identifier=row['id'];rr=by_id[identifier];arms=[]
        for arm,label in [('before','기존 청킹'),('after','의미 단위 청킹')]:
            entries=''.join(f'<details class="evidence"><summary>Rank {i} · {esc(b["document_title"])} · {b["token_count"]}토큰</summary><pre>{esc(b["body"])}</pre><small>청크: {esc(", ".join(b["chunk_ids"]))}</small></details>' for i,b in enumerate(row[arm]['top5'],1))
            ranks=', '.join(i['name']+': '+(str(i['first_single_chunk_dense_rank'])+'위' if i['first_single_chunk_dense_rank'] else '단일 청크 일치 없음') for i in row[arm]['item_ranks'])
            arms.append(f'<div><h3>{label} · {"성공" if rr[arm+"_complete"] else "미충족"}</h3><p>항목별 단일 청크 벡터 순위: {esc(ranks)}</p>{entries}</div>')
        page+=f'<article id="{identifier}"><h2>{identifier} · {esc(row["query"])}</h2><p>{esc(rr["reason"])}</p><p>정답 요지: {esc(row["answer_note"])}</p><div class="columns">'+''.join(arms)+'</div></article>'
    page+='<section id="chunks"><h2>공지 본문 청크 분할 전후</h2><p>검색된 Top5뿐 아니라 이번에 교체한 청크 전체를 볼 수 있어.</p>'
    for record in changed:
        page+=f'<h3>{esc(record["before"][0]["document_title"])} · {len(record["before"])}개 → {len(record["after"])}개</h3><div class="columns">'
        for arm,label in [('before','기존'),('after','변경')]:
            page+=f'<div><h3>{label}</h3>'
            for n,c in enumerate(record[arm],1):
                page+=f'<details class="chunk"><summary>청크 {n} · {c["token_count"]}토큰 · {esc(c["chunk_id"])}</summary><pre>{esc(c["text"])}</pre><small>{esc(json.dumps(c["source_spans"],ensure_ascii=False))}</small></details>'
            page+='</div>'
        page+='</div>'
    page+='</section><details><summary>검증·재현 정보</summary><pre>'+esc(json.dumps(dict(summary=summary,embedding_token_stats=data['embedding_token_stats'],integrity=data['integrity'],evaluation_sha256=review['evaluation_sha256']),ensure_ascii=False,indent=2))+'</pre><p>실험 인덱스: data/processed/semantic-chunk-pilot-v1/index.npz<br>청크: data/processed/semantic-chunk-pilot-v1/selected.jsonl<br>실행: scripts/run_semantic_chunk_pilot.py<br>평가 원본: data/processed/semantic-chunk-pilot-v1/evaluation.json</p></details>'
    page+='''<script>const s=document.querySelector('#search'),cards=[...document.querySelectorAll('article')];function apply(){let n=0;for(const c of cards){c.hidden=!c.textContent.toLowerCase().includes(s.value.toLowerCase());if(!c.hidden)n++;}document.querySelector('#count').textContent=' '+n+' / 24문항';}s.oninput=apply;apply();</script></html>'''
    output=ROOT/'reports/semantic-chunk-pilot-2026-10-12.html';output.write_text(page)
    print(json.dumps(summary,ensure_ascii=False));print(output)

if __name__=='__main__':main()
