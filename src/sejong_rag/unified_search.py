"""품질 계층별 통합 하이브리드 검색과 형식별 출처 표시."""
from __future__ import annotations
import argparse,re
from pathlib import Path
import numpy as np
from sejong_rag.embed_chunks import file_sha256
from sejong_rag.keyword_search import make_snippet
from sejong_rag.unified_hybrid import hybrid_rank,load_notice_metadata
from sejong_rag.vector_search import load_index
DEFAULT_TRUSTED=Path("data/index/sejong-2026-trusted-e5-small.npz"); DEFAULT_REVIEW=Path("data/index/sejong-2026-review-e5-small.npz")

def format_locations(chunk):
    labels=[]
    for locator in chunk.get("locators",[]):
        parts=[]
        for key,label in (("page","p."),("sheet","sheet="),("row","row="),("slide","slide="),("table","table="),("paragraph","paragraph="),("zip_member","zip="),("section","section=")):
            if key in locator: parts.append(f"{label}{locator[key]}")
        value=", ".join(parts) or "document"
        if value not in labels: labels.append(value)
    return "; ".join(labels[:4])+(f"; +{len(labels)-4} more" if len(labels)>4 else "")

def validate_index_source(metadata):
    source=Path(str(metadata["source_chunks"]))
    if source.is_file() and file_sha256(source)!=metadata["source_sha256"]: raise ValueError(f"chunk file changed after indexing: {source}")

def ambiguity_warning(query,results):
    if "장학금" not in query: return None
    specific=("국가장학금","성적향상장학금","주거안정","푸른등대","에델바이스","정수장학")
    if any(x in query for x in specific): return None
    titles={str(x["chunk"].get("document_title","")) for x in results}
    if len(titles)>=2: return "장학사업이 특정되지 않아 여러 장학금이 함께 검색됐습니다. 장학사업명과 학기를 알려주세요."
    return None

def search_indexes(query,trusted_index,review_index=None,top_k=5,candidate_k=300):
    embeddings,chunks,meta=load_index(trusted_index); validate_index_source(meta)
    if review_index:
        re,rc,rm=load_index(review_index); validate_index_source(rm)
        if rm["model"]!=meta["model"]: raise ValueError("trusted and review indexes use different models")
        embeddings=np.vstack([embeddings,re]); chunks=chunks+rc
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(str(meta["model"]),device="cpu"); q=model.encode(f"{meta['query_prefix']}{query}",normalize_embeddings=True,convert_to_numpy=True)
    rows=hybrid_rank(chunks,embeddings,q,query,load_notice_metadata(),top_k,candidate_k)
    for row in rows: row["index_tier"]=str(row["chunk"].get("quality_tier","trusted"))
    return rows

def main():
    p=argparse.ArgumentParser(description="세종대학교 2026 통합 하이브리드 검색"); p.add_argument("query"); p.add_argument("--trusted-index",type=Path,default=DEFAULT_TRUSTED); p.add_argument("--review-index",type=Path,default=DEFAULT_REVIEW); p.add_argument("--include-review",action="store_true"); p.add_argument("--top-k",type=int,default=5); p.add_argument("--candidate-k",type=int,default=300)
    a=p.parse_args(); results=search_indexes(a.query,a.trusted_index,a.review_index if a.include_review else None,a.top_k,a.candidate_k); warning=ambiguity_warning(a.query,results)
    if warning: print("주의:",warning)
    for rank,result in enumerate(results,1):
        c=result["chunk"]; print(f"\n[{rank}] hybrid={result['score']:.6f} quality={result['index_tier']} vector_rank={result.get('vector_rank')} keyword_rank={result.get('keyword_rank')}"); print(f"title={c['document_title']}"); print(f"file={c.get('filename') or '(공지 본문)'}"); print(f"location={format_locations(c)}"); print(f"source={c['source_url']}"); print("-"*72); print(make_snippet(str(c["text"]),[],limit=500))
if __name__=="__main__": main()
