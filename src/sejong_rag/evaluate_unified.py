"""학교 전체 통합 인덱스의 벡터 단독과 하이브리드 검색을 비교 평가한다."""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
from sejong_rag.evaluate_retrieval import metrics_from_ranks,normalize_space
from sejong_rag.unified_hybrid import build_lexical_index,hybrid_rank,load_notice_metadata
from sejong_rag.vector_search import load_index,rank_by_similarity
DEFAULT_EVAL=Path('eval/unified_questions.jsonl'); DEFAULT_INDEX=Path('data/index/sejong-2026-trusted-e5-small.npz')

def load_cases(path):
    cases=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]; ids=[x['id'] for x in cases]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate evaluation id')
    return cases

def relevant_ids(chunks,case):
    phrases=[normalize_space(str(x)) for x in case['required_phrases']]; expected=str(case['expected_document_id']); result=set()
    for chunk in chunks:
        parent=str(chunk.get('parent_document_id') or chunk.get('document_id')); text=normalize_space(str(chunk.get('text','')))
        if parent==expected and all(p in text for p in phrases): result.add(str(chunk['chunk_id']))
    return result

def rank_of(results,relevant,hybrid=False):
    for rank,item in enumerate(results,1):
        chunk=item['chunk'] if hybrid else item
        if str(chunk['chunk_id']) in relevant: return rank
    return None

def main():
    p=argparse.ArgumentParser(description='통합 검색의 벡터 단독/하이브리드 성능 비교'); p.add_argument('--eval',type=Path,default=DEFAULT_EVAL); p.add_argument('--index',type=Path,default=DEFAULT_INDEX); p.add_argument('--top-k',type=int,default=5); p.add_argument('--candidate-k',type=int,default=300)
    a=p.parse_args(); embeddings,chunks,index_meta=load_index(a.index); cases=load_cases(a.eval); truth={x['id']:relevant_ids(chunks,x) for x in cases}; missing=[x for x,v in truth.items() if not v]
    if missing: raise ValueError('unresolved ground truth: '+', '.join(missing))
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(str(index_meta['model']),device='cpu'); queries=[str(x['query']) for x in cases]; vectors=model.encode([f"{index_meta['query_prefix']}{q}" for q in queries],normalize_embeddings=True,convert_to_numpy=True,batch_size=32)
    metadata=load_notice_metadata(); lexical=build_lexical_index(chunks); vranks=[]; hranks=[]; by_category=defaultdict(lambda:[[],[]])
    for case,vector in zip(cases,vectors):
        vector_rows=rank_by_similarity(embeddings,vector,a.candidate_k); vector_results=[chunks[i] for i,_ in vector_rows]
        hybrid_results=hybrid_rank(chunks,embeddings,vector,str(case['query']),metadata,a.candidate_k,a.candidate_k,lexical)
        relevant=truth[case['id']]
        vrank=rank_of(vector_results,relevant); hrank=rank_of(hybrid_results,relevant,True); vranks.append(vrank); hranks.append(hrank); by_category[str(case['category'])][0].append(vrank); by_category[str(case['category'])][1].append(hrank)
        print(f"{case['id']}: vector={vrank or '-'} hybrid={hrank or '-'} delta={(vrank-hrank) if vrank and hrank else '-'}")
    print('\nOVERALL'); print('vector',json.dumps(metrics_from_ranks(vranks,a.top_k))); print('hybrid',json.dumps(metrics_from_ranks(hranks,a.top_k)))
    print('\nBY CATEGORY')
    for category,(v,h) in sorted(by_category.items()): print(category,'vector',round(metrics_from_ranks(v,a.top_k)[f'hit_rate@{a.top_k}'],3),'hybrid',round(metrics_from_ranks(h,a.top_k)[f'hit_rate@{a.top_k}'],3),'n',len(v))
