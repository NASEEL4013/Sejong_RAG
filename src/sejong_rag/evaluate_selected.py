"""Frozen, source-grounded diagnostic evaluation of the selected vector index."""
from __future__ import annotations
import argparse
import html
from collections import defaultdict
import json
from pathlib import Path
import re
import numpy as np
from sejong_rag.embed_chunks import file_sha256
from sejong_rag.vector_search import load_index


def compact(text):
    return re.sub(r'\s+', '', text).casefold()


def satisfies(text, groups):
    text = compact(text)
    return any(all(compact(phrase) in text for phrase in group) for group in groups)


def notice_ids(chunk):
    ids = {chunk.get('parent_document_id'), chunk.get('document_id')}
    for alias in chunk.get('source_aliases', []):
        ids.update((alias.get('parent_document_id'), alias.get('document_id')))
    return ids - {None}


def evidence_body(chunk, records):
    return '\n'.join(records[ref['record_id']]['text'][ref['start']:ref['end']]
                     for ref in chunk['source_spans'])


def metrics(rows):
    if not rows: return {}
    ranks = [r['first_evidence_rank'] for r in rows]
    return dict(questions=len(rows), indexed_evidence_questions=sum(r is not None for r in ranks),
        hit_at_1=sum(r == 1 for r in ranks)/len(rows),
        hit_at_5=sum(r is not None and r <= 5 for r in ranks)/len(rows),
        mrr=sum(1/r for r in ranks if r is not None)/len(rows),
        notice_hit_at_5=sum(r['notice_hit_at_5'] for r in rows)/len(rows))


def audit_report(report_path, audit_path, data_dir):
    """Keep frozen automatic results and post-hoc source review visibly separate."""
    from bs4 import BeautifulSoup
    report_path,audit_path,data_dir=map(Path,(report_path,audit_path,data_dir))
    report=json.loads(report_path.read_text());audit=json.loads(audit_path.read_text())
    reviewed=json.loads(json.dumps(report['results']))
    reasons={r['case_id']:r['reason'] for r in audit['remaining_misses']}
    for accepted in audit['accepted_alternatives']:
        row=next(r for r in reviewed if r['id']==accepted['case_id'])
        candidate=next(r for r in row['top5'] if r['chunk_id']==accepted['chunk_id'])
        if accepted['notice_id'] not in {a.get('parent_document_id') for a in candidate['source_aliases']}:
            raise ValueError('Alternative source mismatch')
        raw=data_dir/'raw/notices'/f"{accepted['notice_id']}.html"
        original=BeautifulSoup(raw.read_text(),'html.parser').get_text(' ')
        if not satisfies(original,[[accepted['required_phrase']]]) or not satisfies(candidate['text'],[[accepted['required_phrase']]]):
            raise ValueError('Alternative evidence unsupported')
        row['automatic_first_evidence_rank']=row['first_evidence_rank']
        row['first_evidence_rank']=min(row['first_evidence_rank'] or float('inf'),candidate['rank'])
        row['notice_hit_at_5']=True
        reasons[row['id']]=accepted['reason']
    result=dict(automatic_metrics=report['metrics'],reviewed_metrics=metrics(reviewed),
        audit=audit,results=reviewed,original_report_sha256=file_sha256(report_path),audit_sha256=file_sha256(audit_path))
    out=report_path.parent
    (out/'reviewed-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    esc=html.escape
    sections=[]
    for original,row in zip(report['results'],reviewed):
        results=''.join(f"<details><summary>{c['rank']}위 · {esc(c['title'] or '')}</summary><p><a href='{esc(c['source_url'],quote=True)}'>원문 공지</a></p><pre>{esc(c['text'])}</pre></details>" for c in row['top5'])
        gold=original['first_evidence']
        evidence=f"<details><summary>사전 지정 정답 청크 ({original['first_evidence_rank']}위)</summary><pre>{esc(gold['text'])}</pre></details>" if gold else '<p>정답 청크 없음</p>'
        sections.append(f"<section><h2>{row['id']}. {esc(row['query'])}</h2><p>자동 순위: {original['first_evidence_rank']} · 원문 대조 후: {row['first_evidence_rank']} · 상위 5개 성공: {'예' if row['first_evidence_rank'] and row['first_evidence_rank']<=5 else '아니오'}</p><p>정답 요지: {esc(row['answer_note'])}</p><p>{esc(reasons.get(row['id'],''))}</p>{results}{evidence}</section>")
    auto,rev=report['metrics'],result['reviewed_metrics']
    intro=f"<h1>검색 품질 평가: 20문항</h1><p>순수 벡터 검색 · 인덱스 {report['chunk_count']:,}개 · 학습용 진단, 전체 서비스 정확도가 아님</p><p>자동 기준: 1위 {auto['hit_at_1']:.0%}, 상위 5개 {auto['hit_at_5']:.0%}. 원문 대조 보완 후: 1위 {rev['hit_at_1']:.0%}, 상위 5개 {rev['hit_at_5']:.0%}.</p><p>S08은 다른 공식 공지의 동등한 정답을 사후 인정했어. 자동 기준과 검색 결과는 바꾸지 않았어. S01은 이전 검색 시험에 사용한 질문이고, 나머지 19개는 이번 신규 질문이야. 답변 생성·자료에 없는 질문·모호한 질문은 이번 평가에 포함하지 않았어.</p>"
    document="<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>검색 품질 평가</title><style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:30px auto;padding:16px;line-height:1.6;color:#17212e}section{border-top:1px solid #ccd;padding:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:15px}summary{cursor:pointer;padding:8px}h2{font-size:1.1em}</style>"+intro+''.join(sections)+'</html>'
    (out/'index.html').write_text(document)
    return result


def evaluate(cases_path, index_path, normalized_path, data_dir, output_dir):
    cases_path,index_path,normalized_path,data_dir,output_dir=map(Path,
        (cases_path,index_path,normalized_path,data_dir,output_dir))
    if output_dir.exists():raise FileExistsError('Use a fresh evaluation directory')
    cases=[json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    if not cases or len({c['id'] for c in cases})!=len(cases):raise ValueError('Invalid case IDs')
    hashes={str(p):file_sha256(p) for p in (cases_path,index_path,normalized_path)}
    from bs4 import BeautifulSoup
    for c in cases:
        raw=data_dir/'raw/notices'/f"{c['notice_id']}.html"
        hashes[str(raw)]=file_sha256(raw)
        original=BeautifulSoup(raw.read_text(),'html.parser').get_text(' ')
        if not satisfies(original,c['evidence_any']):raise ValueError(f"Gold phrases absent in original: {c['id']}")
    vectors,chunks,meta=load_index(index_path)
    source=Path(meta['source_chunks'])
    if file_sha256(source)!=meta['source_sha256']:raise ValueError('Index source changed')
    records={r['record_id']:r for r in (json.loads(l) for l in normalized_path.open())}
    bodies=[compact(evidence_body(c,records)) for c in chunks]
    scopes=[notice_ids(c) for c in chunks]
    gold={}
    for c in cases:
        groups=[[compact(x) for x in g] for g in c['evidence_any']]
        gold[c['id']]=[i for i,(scope,body) in enumerate(zip(scopes,bodies))
            if c['notice_id'] in scope and any(all(x in body for x in g) for g in groups)]
    output_dir.mkdir(parents=True)
    frozen=dict(cases=cases,source_hashes=hashes,
        relevant_chunk_ids={c['id']:[chunks[i]['chunk_id'] for i in gold[c['id']]] for c in cases},
        note='Frozen before query encoding/ranking. Source scopes and whitespace-insensitive phrase rules, not human exhaustive labels.')
    (output_dir/'ground_truth.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n')
    print('Frozen gold:',len(cases),'questions;',sum(bool(v) for v in gold.values()),'with indexed evidence',flush=True)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)
    model=SentenceTransformer(meta['model'],device='cpu',local_files_only=True)
    queries=[meta['query_prefix']+c['query'] for c in cases]
    if any(len(model.tokenizer.encode(q))>model.max_seq_length for q in queries):raise ValueError('Query too long')
    qvectors=model.encode(queries,batch_size=20,normalize_embeddings=True,convert_to_numpy=True)
    rows=[]

    def result(i,score,rank):
        c=chunks[int(i)]
        return dict(rank=rank,chunk_id=c['chunk_id'],score=float(score),title=c.get('document_title'),
            source_url=c['source_url'],source_spans=c['source_spans'],text=c['text'],
            source_aliases=c.get('source_aliases',[]))

    for case,qvector in zip(cases,qvectors):
        scores=vectors@qvector;order=np.argsort(-scores,kind='stable');relevant=set(gold[case['id']])
        rank=next((rank for rank,i in enumerate(order,1) if int(i) in relevant),None)
        top=order[:5]
        row=dict(**case, relevant_chunk_count=len(relevant),first_evidence_rank=rank,
            notice_hit_at_5=any(case['notice_id'] in scopes[int(i)] for i in top),
            top5=[result(i,scores[i],j) for j,i in enumerate(top,1)],
            first_evidence=result(order[rank-1],scores[order[rank-1]],rank) if rank else None)
        rows.append(row)
        print(case['id'],'rank=',rank,'query=',case['query'],flush=True)
    categories=defaultdict(list)
    for row in rows:categories[row['category']].append(row)
    report=dict(index=str(index_path),model=meta['model'],chunk_count=len(chunks),
        metrics=metrics(rows),new_question_metrics=metrics([r for r in rows if not r.get('previously_smoke_tested')]),
        by_category={k:metrics(v) for k,v in categories.items()},results=rows,input_sha256=hashes,
        limitations=['20 assistant-authored diagnostic questions, not representative accuracy',
                    'Gold based on original HTML; phrase labels not exhaustive semantic judgments',
                    'Previously tested app query explicitly marked; other queries new to this evaluation',
                    'Answerable questions only; ambiguity/abstention/generation not evaluated',
                    'Development baseline, not an untouched holdout; no tuning during this run',
                    'Strict year/semester notice scope; headers alone never count as evidence'])
    for path,fingerprint in hashes.items():
        if file_sha256(Path(path))!=fingerprint:raise ValueError(f'Input changed: {path}')
    (output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['metrics'],ensure_ascii=False,indent=2))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',type=Path,default=Path('eval/selected_retrieval_v1.jsonl'))
    p.add_argument('--index',type=Path,default=Path('data/index/sejong-2025-2026-selected-e5-small.npz'))
    p.add_argument('--normalized',type=Path,default=Path('data/processed/normalized-text/records.jsonl'))
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--output-dir',type=Path,default=Path('data/processed/evaluation-selected-v1'))
    a=p.parse_args();evaluate(a.cases,a.index,a.normalized,a.data_dir,a.output_dir)


if __name__=='__main__':main()
