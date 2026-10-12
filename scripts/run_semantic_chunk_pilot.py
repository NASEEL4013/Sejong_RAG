"""Run a frozen, local-only A/B test of five notice-body chunk boundaries."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from sejong_rag.ask_selected import load_runtime, DEFAULT_INDEX, DEFAULT_NORMALIZED, save_json
from sejong_rag.semantic_chunk_pilot import split_record
from sejong_rag.selected_search import SelectedSearch
from sejong_rag.compare_decomposed import evidence_coverage
from sejong_rag.compare_selected import is_evidence,notice_ids
from sejong_rag.embed_chunks import save_index,file_sha256,token_length_stats,build_passages

OUT=Path('data/processed/semantic-chunk-pilot-v1')
TARGETS=['notice3-859241','notice3-855929','notice3-805913','notice7-810913','notice7-892486']


def diagnostics(engine,query,vector,case):
    bundles=engine.search(query,vector,strategy='vector_context',top_k=5,candidate_k=100,max_bundle_tokens=512,total_tokens=2200)
    order=np.argsort(-(engine.vectors@vector),kind='stable')
    items=[]
    for item in case.get('items',[]):
        first=next((rank for rank,i in enumerate(order,1) if is_evidence(engine.bodies[i],notice_ids(engine.chunks[i]),item['evidence_options'])),None)
        items.append(dict(name=item['name'],first_single_chunk_dense_rank=first))
    return dict(top5=bundles,item_phrase_proxy=evidence_coverage(bundles,case),item_ranks=items)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    cases=[json.loads(l) for l in Path('eval/project_benchmark_v2.jsonl').read_text().splitlines()]
    cases=[c for c in cases if c['kind']=='answerable']
    # Explicitly retain the February timing item when scoring the cross-reference.
    next(c for c in cases if c['id']=='B05')['items'][0]['evidence_options'][0]['evidence_any'][0].append('2/14')
    save_json(OUT/'cases.json',cases)
    frozen_paths=[Path(DEFAULT_INDEX),Path(DEFAULT_NORMALIZED),Path('data/processed/selected-chunks/selected.jsonl'),
        Path('eval/project_benchmark_v2.jsonl'),Path('src/sejong_rag/semantic_chunk_pilot.py'),Path(__file__)]
    frozen_paths+=list(Path('src/sejong_rag').glob('*.py'))
    hashes={str(p.resolve()):file_sha256(p) for p in set(frozen_paths)}
    protocol=dict(version='semantic_chunk_pilot_v1',targets=TARGETS,scope='Only the five gold notice-body records; all attachments and other notice records unchanged.',
        strategy='vector_context',top_k=5,candidate_k=100,max_bundle_tokens=512,total_tokens=2200,max_chunk_tokens=448,
        embedding='intfloat/multilingual-e5-small',generation_calls=0,
        metric='Union of all Top5 bodies supporting each requested item; phrase proxy followed by semantic review.',
        note='Development diagnostic on known failing sources; not a fresh holdout. Section rules use document structure, not query text or answer values. No post-result tuning.',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),files_sha256=hashes)
    save_json(OUT/'protocol.json',protocol)
    engine,meta,model=load_runtime();print('Loaded production index',len(engine.chunks),flush=True)
    wanted={nid+'-body' for nid in TARGETS}
    removed=[i for i,c in enumerate(engine.chunks) if any(s['record_id'] in wanted for s in c['source_spans'])]
    assert all(all(s['record_id'] in wanted for s in engine.chunks[i]['source_spans']) for i in removed)
    new=[];changes=[]
    for rid in sorted(wanted):
        old=[engine.chunks[i] for i in removed if engine.chunks[i]['source_spans'][0]['record_id']==rid]
        assert old
        chunks=split_record(engine.records[rid],old[0],engine.tokenizer)
        new.extend(chunks);changes.append(dict(record_id=rid,before=old,after=chunks))
    # Freeze the exact replacement chunks before any retrieval comparison.
    save_json(OUT/'changed-chunks.json',changes)
    protocol['replacement_chunks_sha256']=file_sha256(OUT/'changed-chunks.json')
    save_json(OUT/'split-frozen.json',protocol)
    stats=token_length_stats(model,build_passages(new));assert stats['truncated_count']==0
    new_vectors=model.encode(build_passages(new),batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    keep=[i for i in range(len(engine.chunks)) if i not in set(removed)]
    chunks=[engine.chunks[i] for i in keep]+new
    vectors=np.concatenate([engine.vectors[keep],new_vectors]).astype(np.float32)
    source=OUT/'selected.jsonl'
    with source.open('x') as stream:
        for chunk in chunks:stream.write(json.dumps(chunk,ensure_ascii=False)+'\n')
    save_json(OUT/'report.json',dict(input_sha256={str(Path(DEFAULT_NORMALIZED).resolve()):file_sha256(Path(DEFAULT_NORMALIZED))},note='Experimental selection manifest, production untouched.'))
    metadata=dict(meta,source_chunks=str(source.resolve()),source_sha256=file_sha256(source),chunk_count=len(chunks),pilot=True)
    save_index(OUT/'index.npz',vectors,chunks,metadata)
    qvectors=model.encode([meta['query_prefix']+c['query'] for c in cases],batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    # Batch dot products later reuse the exact query vectors for both arms.
    rows=[]
    for case,q in zip(cases,qvectors):
        rows.append(dict(id=case['id'],query=case['query'],answer_note=case['answer_note'],items=case['items'],before=diagnostics(engine,case['query'],q,case)))
    print('Baseline complete; replacements',len(removed),'->',len(new),flush=True)
    alternate=SelectedSearch(chunks,vectors,engine.records,engine.tokenizer)
    for row,case,q in zip(rows,cases,qvectors):
        row['after']=diagnostics(alternate,case['query'],q,case)
        print(row['id'],row['before']['item_phrase_proxy']['complete'],'->',row['after']['item_phrase_proxy']['complete'],flush=True)
    summary={arm:dict(complete=sum(r[arm]['item_phrase_proxy']['complete'] for r in rows),total=len(rows),
        target_complete=sum(r[arm]['item_phrase_proxy']['complete'] for r in rows if r['id'] in ['B01','B02','B05','B12','B21']),target_total=5) for arm in ['before','after']}
    result=dict(protocol=protocol,summary_proxy=summary,rows=rows,removed_chunks=len(removed),replacement_chunks=len(new),
        embedding_token_stats=stats,unchanged_chunks=len(keep),unchanged_vectors_verified=bool(np.array_equal(vectors[:len(keep)],engine.vectors[keep])),
        integrity=dict(passed=all(file_sha256(Path(p))==h for p,h in hashes.items())),production_index_unchanged=True)
    save_json(OUT/'evaluation.json',result)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
