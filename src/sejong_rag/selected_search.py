"""Body-first hybrid retrieval and bounded source-preserving context assembly."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sejong_rag.embed_chunks import file_sha256
from sejong_rag.vector_search import load_index

VERSION = 'selected_context_v1'
DEFAULT_STRATEGY = 'vector_context'


def compact(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).casefold()


def period(text, query=False):
    if query and re.search(r'입학(?:년도|연도|자|생)|졸업년도', text): return None, None
    years=set(re.findall(r'(?<!\d)(20\d{2})(?!\d)', text))
    semesters=set(re.findall(r'(?<!\d)([12])\s*학기', text))
    semesters.update(re.findall(r'20\d{2}\s*-\s*([12])(?=\D|$)', text))
    return (int(next(iter(years))) if len(years)==1 else None,
            int(next(iter(semesters))) if len(semesters)==1 else None)


def period_score(query, aliases):
    wanted=period(query,True)
    if wanted==(None,None): return 0.0
    statuses=[]
    for alias in aliases:
        actual=period(str(alias.get('document_title') or ''))
        if any(w is not None and a is not None and w!=a for w,a in zip(wanted,actual)):
            statuses.append(-1)
        elif all(w is None or a==w for w,a in zip(wanted,actual)):
            statuses.append(1)
        else:statuses.append(0)
    best=max(statuses,default=0)
    return .008 if best==1 else -.018 if best==-1 else 0.0


def expanded_query(query):
    """Small general intent vocabulary; never inserts answer values or document IDs."""
    extra=[]
    if re.search(r'어디|어느\s*(?:곳|부서)|가져가',query):extra.append('장소 위치 제출처 방문')
    if re.search(r'언제|며칠|몇\s*(?:월|일|시)|시점',query):extra.append('기간 시간 일정')
    if re.search(r'얼마|금액',query):extra.append('금액 지원 한도')
    if re.search(r'전화|연락|문의',query):extra.append('문의 연락처')
    if re.search(r'설치|다운로드|깔아',query):extra.append('설치 앱 다운로드')
    return query+' '+ ' '.join(extra)


class SelectedSearch:
    def __init__(self,chunks,vectors,records,tokenizer):
        self.chunks,self.vectors,self.records,self.tokenizer=chunks,vectors,records,tokenizer
        self.by_id={c['chunk_id']:i for i,c in enumerate(chunks)}
        self.bodies=[self.span_text(c['source_spans']) for c in chunks]
        self.aliases=[c.get('source_aliases') or [c] for c in chunks]
        self.vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,3),min_df=2,
            max_features=150000,sublinear_tf=True,dtype=np.float32)
        self.lexical=self.vectorizer.fit_transform([compact(t) for t in self.bodies])
        self.by_record=defaultdict(list)
        for i,c in enumerate(chunks):
            refs=c['source_spans'];ids={r['record_id'] for r in refs}
            if len(ids)==1:
                self.by_record[next(iter(ids))].append((min(r['start'] for r in refs),max(r['end'] for r in refs),i))
        self.neighbors={}
        for entries in self.by_record.values():
            entries.sort()
            for j,(start,end,i) in enumerate(entries):
                neighbors=[]
                if j+1<len(entries) and 0<=entries[j+1][0]-end<=10:neighbors.append(entries[j+1][2])
                if j and 0<=start-entries[j-1][1]<=10:neighbors.append(entries[j-1][2])
                self.neighbors[i]=neighbors

    def span_text(self,spans):
        return '\n'.join(self.records[r['record_id']]['text'][r['start']:r['end']] for r in spans)

    def count(self,text):
        return len(self.tokenizer(text,add_special_tokens=True,truncation=False)['input_ids'])

    def clip(self,text,limit):
        lo,hi=0,len(text)
        while lo<hi:
            mid=(lo+hi+1)//2
            if self.count(text[:mid])<=limit:lo=mid
            else:hi=mid-1
        return text[:lo]

    def rank(self,query,qvector,hybrid=True,candidate_k=100):
        if candidate_k<=0:raise ValueError('candidate_k must be positive')
        scores=self.vectors@qvector
        vector=np.argsort(-scores,kind='stable')
        if not hybrid:return [(int(i),float(scores[i])) for i in vector[:candidate_k]]
        sparse=self.lexical@self.vectorizer.transform([compact(expanded_query(query))]).T
        lexical_scores=sparse.toarray().ravel()
        lex=np.argsort(-lexical_scores,kind='stable')[:candidate_k]
        fused=defaultdict(float)
        for order,weight in ((vector[:candidate_k],2.0),(lex,1.0)):
            for rank,i in enumerate(order,1):
                if weight==1 and lexical_scores[i]<=0:continue
                fused[int(i)]+=weight/(60+rank)
        for i in fused:fused[i]+=period_score(query,self.aliases[i])
        ranked=sorted(fused.items(),key=lambda x:(-x[1],self.chunks[x[0]]['chunk_id']))
        # Keep one semantic anchor: short direct answers can have little literal
        # overlap with the question. Do not protect an explicit period mismatch.
        anchor=int(vector[0])
        if period_score(query,self.aliases[anchor])>=0:
            position=next(j for j,(i,_) in enumerate(ranked) if i==anchor)
            if position>1:ranked.insert(1,ranked.pop(position))
        return ranked

    def bundle(self,indices,query):
        anchor=self.chunks[indices[0]];refs=[];seen=set()
        for i in indices:
            for ref in self.chunks[i]['source_spans']:
                key=(ref['record_id'],ref['start'],ref['end'])
                if key not in seen:refs.append(ref);seen.add(key)
        record_order={rid:j for j,rid in enumerate(dict.fromkeys(r['record_id'] for r in refs))}
        refs.sort(key=lambda r:(record_order[r['record_id']],r['start']))
        # Merge overlapping intervals to avoid overlap text duplication.
        merged=[]
        for r in refs:
            if merged and merged[-1]['record_id']==r['record_id'] and r['start']<=merged[-1]['end']:
                merged[-1]['end']=max(merged[-1]['end'],r['end'])
            else:merged.append(dict(r))
        aliases=anchor.get('source_aliases') or [anchor]
        best=max(aliases,key=lambda a:period_score(query,[a]))
        title=str(best.get('document_title') or anchor.get('document_title') or '')
        # Reuse the bounded indexed header unless an alias title is selected.
        original=anchor['text'];anchor_body=self.bodies[indices[0]]
        if not original.endswith(anchor_body):raise ValueError('Chunk/source body mismatch')
        prefix=original[:-len(anchor_body)].rstrip('\n')
        if best.get('document_title') and best.get('document_title')!=anchor.get('document_title'):
            lines=prefix.splitlines()
            if lines:
                lines[0]=self.clip('문서 제목: '+title,min(224,self.count(lines[0])))
                prefix='\n'.join(lines)
        body=self.span_text(merged)
        return dict(anchor_chunk_id=anchor['chunk_id'],chunk_ids=[self.chunks[i]['chunk_id'] for i in indices],
            source_spans=merged,source_aliases=aliases,document_title=title,
            source_url=best.get('source_url') or anchor['source_url'],
            attachment_url=best.get('attachment_url') or anchor.get('attachment_url'),body=body,
            text=(prefix+'\n\n'+body).strip())

    def search(self,query,qvector,strategy=DEFAULT_STRATEGY,top_k=5,candidate_k=100,max_bundle_tokens=512,total_tokens=2200):
        if strategy not in {'vector','hybrid','vector_context','improved'}:raise ValueError('Unknown strategy')
        if top_k<=0 or max_bundle_tokens<=0 or total_tokens<=0:raise ValueError('Invalid context budget')
        hybrid=strategy in {'hybrid','improved'};expand=strategy in {'vector_context','improved'}
        ranked=self.rank(query,qvector,hybrid,candidate_k)
        bundles=[];used=set();budget=0;duplicates={}
        for index,score in ranked:
            if index in used:continue
            components=[index];bundle=self.bundle(components,query)
            if expand and self.count(self.bodies[index])<=100:
                for neighbor in self.neighbors.get(index,[]):
                    if neighbor in used:continue
                    candidate=self.bundle(components+[neighbor],query)
                    candidate_cost=self.count(candidate['text'])
                    if candidate_cost<=max_bundle_tokens and budget+candidate_cost<=total_tokens:
                        components.append(neighbor);bundle=candidate
            cost=self.count(bundle['text'])
            if cost>max_bundle_tokens or budget+cost>total_tokens:continue
            # Only identical substantial, fully assembled evidence in the SAME notice scope.
            scope=tuple(sorted(str(a.get('parent_document_id') or a.get('document_id')) for a in bundle['source_aliases']))
            key=(scope,compact(bundle['body']))
            if expand and len(key[1])>=80 and key in duplicates:
                duplicates[key].setdefault('duplicate_candidate_ids',[]).append(self.chunks[index]['chunk_id'])
                duplicates[key].setdefault('duplicate_evidence',[]).append(dict(
                    chunk_ids=bundle['chunk_ids'],source_spans=bundle['source_spans'],
                    source_aliases=bundle['source_aliases'],source_url=bundle['source_url'],
                    attachment_url=bundle['attachment_url']))
                continue
            bundle.update(score=score,token_count=cost,strategy=strategy,context_expanded=len(components)>1,
                ranking_policy='rrf_with_dense_anchor' if hybrid else 'vector_similarity')
            bundles.append(bundle);duplicates[key]=bundle;used.update(components);budget+=cost
            if len(bundles)==top_k:break
        assert sum(b['token_count'] for b in bundles)<=total_tokens
        return bundles


def load_search(index,normalized):
    vectors,chunks,meta=load_index(Path(index))
    source=Path(meta['source_chunks']);normalized=Path(normalized)
    if file_sha256(source)!=meta['source_sha256']:raise ValueError('Index source mismatch')
    manifest=json.loads((source.parent/'report.json').read_text())
    expected=manifest.get('input_sha256',{}).get(str(normalized.resolve()))
    if not expected or file_sha256(normalized)!=expected:
        raise ValueError('Normalized source differs from the selected chunk manifest')
    records={r['record_id']:r for r in (json.loads(l) for l in normalized.open())}
    for chunk in chunks:
        for ref in chunk['source_spans']:
            if ref['record_id'] not in records or not 0<=ref['start']<ref['end']<=len(records[ref['record_id']]['text']):
                raise ValueError('Invalid source span in '+chunk['chunk_id'])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(meta['model'],local_files_only=True)
    return SelectedSearch(chunks,vectors,records,tokenizer),meta


def main():
    p=argparse.ArgumentParser()
    p.add_argument('query');p.add_argument('--strategy',default=DEFAULT_STRATEGY,choices=['vector','hybrid','vector_context','improved'])
    p.add_argument('--index',type=Path,default=Path('data/index/sejong-2025-2026-selected-e5-small.npz'))
    p.add_argument('--normalized',type=Path,default=Path('data/processed/normalized-text/records.jsonl'))
    a=p.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)
    engine,meta=load_search(a.index,a.normalized)
    model=SentenceTransformer(meta['model'],device='cpu',local_files_only=True)
    q=model.encode(meta['query_prefix']+a.query,normalize_embeddings=True,convert_to_numpy=True)
    print(json.dumps(engine.search(a.query,q,a.strategy),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
