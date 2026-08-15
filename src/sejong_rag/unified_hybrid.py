"""통합 벡터·키워드·제목·학기 메타데이터 순위를 결합한다."""
from __future__ import annotations
import json, math, re
from collections import Counter
from pathlib import Path
import numpy as np
from sejong_rag.document_policy import extract_academic_period
from sejong_rag.keyword_search import build_document_frequency, tokenize
from sejong_rag.vector_search import rank_by_similarity
RRF_K=60
CATEGORY_TERMS={"course_registration":("수강","강의"),"academic_status":("휴학","복학","재입학","전과","학적"),"graduation":("졸업","수료","영어인증"),"tuition":("등록금","납부"),"credit":("학점교류",),"scholarship":("장학금",)}

def load_notice_metadata(classified=Path('data/catalog/classified-notices-2026.json'),policy=Path('data/catalog/document-policy.json')):
    c=json.loads(classified.read_text(encoding='utf-8'))['items']; p=json.loads(policy.read_text(encoding='utf-8'))['items']; result={str(x['document_id']):dict(x) for x in c}
    for x in p: result.setdefault(str(x['document_id']),{}).update({k:x.get(k) for k in ('academic_year','semester','document_family','version_status','family_confidence')})
    return result

def expand_query(query):
    additions=[]
    admission_year = re.search(r"(20\d{2})(?:년|학년도)?\s*(?:입학(?:생|자)?|입학생)", query)
    if "단일전공" in query and re.search(r"졸업\s*요건|졸업\s*기준", query): additions.append("단일전공 이수 시 학과별 졸업 이수학점 전공필수 전공선택")
    if admission_year and re.search(r"졸업\s*요건|졸업\s*기준", query):
        year = admission_year.group(1)
        additions.append(f"{year}학년도 입학자 교과과정 졸업 기준")
        additions.append("영어졸업인증 이수 조건 면제 대상")
    if "신청" in query and re.search(r"언제|기간|마감", query): additions.append("신청기간")
    if "학점교류" in query and "평점" in query: additions.append("총 평점평균")
    if "지급" in query and re.search(r"언제|일정", query): additions.append("현금지급일정")
    if re.search(r"서류|가져", query): additions.append("제출서류")
    if "학위증" in query: additions.append("학위수여식 학위증서 배부")
    if "계절학기" in query and re.search(r"수업.*기간|기간.*수업", query): additions.append("계절학기 등록 및 수강신청 수업 기간")
    return query + (" " + " ".join(additions) if additions else "")

def infer_categories(query):
    return {category for category,terms in CATEGORY_TERMS.items() if any(term in query for term in terms)}

def build_lexical_index(chunks):
    body=[tokenize(str(x.get('text',''))) for x in chunks]; titles=[tokenize(str(x.get('document_title',''))) for x in chunks]
    return {"body":body,"titles":titles,"body_df":build_document_frequency(body),"title_df":build_document_frequency(titles)}

def lexical_ranks(chunks,query,index,title=False,limit=300):
    query_tokens=list(dict.fromkeys(tokenize(query))); docs=index['titles' if title else 'body']; df=index['title_df' if title else 'body_df']; n=len(chunks); scored=[]
    for i,tokens in enumerate(docs):
        tf=Counter(tokens); matched=[t for t in query_tokens if tf[t]]
        if not matched: continue
        coverage=len(matched)/max(len(query_tokens),1); score=sum((math.log((n+1)/(df[t]+1))+1)*(1+math.log(tf[t])) for t in matched)*(1+coverage)
        scored.append((i,score))
    scored.sort(key=lambda x:(-x[1],str(chunks[x[0]]['chunk_id'])))
    return scored[:limit]

def infer_admission_year_sections(chunks):
    """같은 파일에서 입학년도 제목부터 다음 제목 전까지의 소속 연도를 연결한다."""
    result = {}
    by_document = {}
    for index, chunk in enumerate(chunks):
        by_document.setdefault(str(chunk.get("document_id", "")), []).append((index, chunk))
    for rows in by_document.values():
        current_year = None
        for index, chunk in sorted(rows, key=lambda row: int(row[1].get("chunk_index", 0))):
            match = re.search(r"❚?\s*(20\d{2})학년도\s*입학자\s*교과과정", str(chunk.get("text", "")))
            if match:
                current_year = int(match.group(1))
            result[index] = current_year
    return result

def hybrid_rank(chunks,embeddings,query_embedding,query,metadata_by_document,top_k=10,candidate_k=300,lexical_index=None):
    lexical=lexical_index or build_lexical_index(chunks); vector=rank_by_similarity(embeddings,query_embedding,candidate_k); keyword=lexical_ranks(chunks,expand_query(query),lexical,False,candidate_k); section_years=infer_admission_year_sections(chunks)
    fused={}
    for name,rows,weight in (("vector_rank",vector,2.0),("keyword_rank",keyword,1.0)):
        for rank,(idx,raw) in enumerate(rows,1):
            item=fused.setdefault(idx,{"index":idx,"score":0.0,"vector_rank":None,"keyword_rank":None,"title_rank":None}); item[name]=rank; item['score']+=weight/(RRF_K+rank)
    query_tokens=set(tokenize(query))
    expanded_tokens=set(tokenize(expand_query(query)))
    added_tokens=expanded_tokens-query_tokens
    year,semester=extract_academic_period(query); categories=infer_categories(query); wants_latest=bool(re.search(r"최신|현재|이번",query))
    admission_year = re.search(r"(20\d{2})(?:년|학년도)?\s*(?:입학(?:생|자)?|입학생)", query)
    asks_requirements = bool(re.search(r"졸업\s*(?:요건|기준|학점)|졸업하려면|전공(?:필수|선택|영역|학점)", query))
    for idx,item in fused.items():
        chunk=chunks[idx]; meta=metadata_by_document.get(str(chunk.get('parent_document_id') or chunk.get('document_id')),{})
        chunk_tokens=set(tokenize(str(chunk.get("text", ""))))
        if added_tokens.intersection(chunk_tokens) and "official_html_body" in chunk.get("extraction_methods", []): item["score"] += 0.01
        title_tokens=set(tokenize(str(chunk.get("document_title", ""))))
        if "학위증" in query and "학위수여식" in str(chunk.get("document_title", "")): item["score"] += 0.08
        if "계절학기" in query and re.search(r"수업.*기간|기간.*수업", query) and "계절학기" in str(chunk.get("document_title", "")) and "수강신청" in str(chunk.get("document_title", "")): item["score"] += 0.03
        title_coverage=len(query_tokens.intersection(title_tokens))/max(len(query_tokens),1)
        item["score"]+=0.01*title_coverage
        asks_single_major = "단일전공" in query and asks_requirements
        department_terms = [term for term in re.findall(r"[가-힣A-Za-z]+(?:학과|학부|전공)", query) if term not in ("단일전공", "복수전공")]
        if asks_single_major and department_terms and any(term in str(chunk.get("text", "")) for term in department_terms):
            if admission_year and section_years.get(idx) == int(admission_year.group(1)):
                item["score"] += 0.10
            elif admission_year and section_years.get(idx) is not None:
                item["score"] -= 0.05
        roster_text = f"{chunk.get('document_title', '')} {chunk.get('filename', '')}"
        if asks_requirements and re.search(r"졸업(?:대상자|예정자|심사)" , roster_text) and "명단" in roster_text:
            item["score"] -= 0.20
        if admission_year := re.search(r"(20\d{2})(?:년|학년도)?\s*(?:입학(?:생|자)?|입학생)", query):
            exact_terms = (f"{admission_year.group(1)}학년도", "입학자", "졸업", "기준")
            if asks_requirements and all(term in str(chunk.get("text", "")) for term in exact_terms):
                item["score"] += 0.08
        if year and meta.get('academic_year')==year: item['score']+=0.004
        if semester and meta.get('semester')==semester: item['score']+=0.004
        if categories and categories.intersection(meta.get('categories',[])): item['score']+=0.003
        if wants_latest and meta.get("version_status") == "current_candidate": item["score"] += 0.003
        item['chunk']=chunk; item['metadata']=meta
    current_families={str(m.get('document_family')) for m in metadata_by_document.values() if m.get('version_status')=='current_candidate' and (not year or m.get('academic_year')==year) and (not semester or m.get('semester')==semester)} if (year or semester) else set()
    for item in fused.values():
        meta=item['metadata']
        if meta.get('version_status')=='historical' and str(meta.get('document_family')) in current_families: item['score']-=0.02
    initial=sorted(fused.values(),key=lambda x:(-x['score'],str(x['chunk']['chunk_id'])))
    top_parents=[]
    for item in initial[:20]:
        parent=str(item['chunk'].get('parent_document_id') or item['chunk'].get('document_id'))
        if parent not in top_parents: top_parents.append(parent)
        if len(top_parents)>=5: break
    for item in fused.values():
        chunk=item['chunk']; parent=str(chunk.get('parent_document_id') or chunk.get('document_id'))
        if parent in top_parents and 'official_html_body' in chunk.get('extraction_methods',[]): item['score']+=0.012
    results=sorted(fused.values(),key=lambda x:(-x['score'],str(x['chunk']['chunk_id'])))
    return results[:top_k]
