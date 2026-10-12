"""Metadata-only syllabus lookup and deterministic original-document guidance."""
import re
from collections import defaultdict
from pathlib import Path
from sejong_rag.extract_attachments import load_attachment_metadata

SYLLABUS_NAME=re.compile(r'(?:수업|강의)\s*계획서')


def compact(text):return re.sub(r'[^0-9a-z가-힣]','',text.casefold())


def academic_period(text):
    year=re.search(r'20\d{2}',text)
    term=re.search(r'(?:20\d{2}\s*-\s*|(?:학년도|년도)\s*)([12])(?:\s*학기|\b)',text)
    if not term:term=re.search(r'([12])\s*학기',text)
    season=next((s for s in ('봄','여름','가을','겨울','하계','동계') if s in text),None)
    semester=int(term[1]) if term else {'봄':1,'가을':2}.get(season)
    return (int(year[0]) if year else None),semester,season


def build_catalog(records,data_dir=Path('data')):
    metadata={m['document_id']:m for m in load_attachment_metadata(Path(data_dir))}
    groups=defaultdict(list)
    for r in records:
        member=r.get('locator',{}).get('zip_member','')
        if SYLLABUS_NAME.search(member or r.get('filename') or ''):
            groups[r['document_id'],member].append(r)
    entries=[]
    for (doc,member),rs in groups.items():
        # A standalone PDF can contain several course syllabi.
        starts=[i for i,r in enumerate(rs) if re.search(r'(?m)^교\s*과\s*목\s*명',r['text'])]
        if not starts:starts=[0]
        for i,j in zip(starts,starts[1:]+[len(rs)]):
            r=rs[i];text=r['text'];m=metadata.get(doc,{})
            spaced='교 과 목 명' in text
            if spaced:text='\n'.join(line.replace(' ','') for line in text.splitlines())
            name=re.search(r'교과목명\s*(?:Course Title\s*)?\(국문\)\s*([^\n]+)',text)
            if not name:name=re.search(r'교과목명[ \t]*([^\n]+)',text)
            course=name[1].split('(영문)')[0].strip() if name else None
            period=re.search(r'개설학기\s*([^\n]+)',text)
            source_term=period[1].strip() if period else next((l.strip() for l in text.splitlines()[:4] if re.search(r'20\d{2}',l)),None)
            notice=r.get('document_title','');year,semester,season=academic_period(source_term or notice)
            notice_year,notice_semester,_=academic_period(notice)
            conflict=bool(source_term and ((notice_year and year and notice_year!=year) or
                          (notice_semester and semester and notice_semester!=semester)))
            teacher=re.search(r'담당교수[^\n]*',text)
            teacher=teacher[0].split('학수번호')[0].strip() if teacher else None
            if teacher and re.fullmatch(r'담당교수(?:\(소속\))?',teacher):
                following=text[text.index(teacher)+len(teacher):].lstrip().split('\n')[0]
                teacher=following.removeprefix('Lecturer ').split('학수번호')[0].strip()
            teacher_names=re.findall(r'[가-힣]{2,4}',re.sub(r'\([^)]*\)|담당교수|제작교수|운영교수|소속|Lecturer','',teacher or ''))
            entries.append(dict(teacher_names=teacher_names,catalog_id=f'{doc}:{member}:{r["record_id"]}',document_id=doc,
                course=course,teacher=teacher,academic_period=source_term or notice,
                year=year,semester=semester,season=season,period_conflict=conflict,
                notice_year=notice_year,notice_title=notice,source_url=r.get('source_url') or m.get('source_url'),
                attachment_url=m.get('attachment_url') or m.get('download_url') or r.get('attachment_url'),
                filename=r.get('filename'),zip_member=member or None,
                pages=[x['locator']['page'] for x in rs[i:j] if 'page' in x['locator']],
                retrieval_mode='original_document_only'))
    return entries


def lookup(query,catalog):
    """Return only original links. No lesson/evaluation content enters a prompt."""
    q=compact(query)
    names=[e for e in catalog if e['course'] and compact(e['course']) in q]
    explicit=bool(SYLLABUS_NAME.search(query))
    intent=bool(re.search(r'주차|강의\s*내용|수업\s*내용|강의\s*일정|수업\s*일정|강의\s*목표',query))
    details=bool(re.search(r'중간|기말|시험|평가|성적|출석|교재|과제|교수|알려|설명',query))
    if not (explicit or intent or (names and details)):return None
    candidates=names
    if not candidates and explicit:
        candidates=[e for e in catalog if any(compact(n) in q for n in e.get('teacher_names',[]))]
    year,semester,season=academic_period(query)
    if year:candidates=[e for e in candidates if (e['notice_year'] if e['period_conflict'] else e['year'])==year]
    if semester:candidates=[e for e in candidates if e['semester']==semester and not e['period_conflict']]
    if season:
        candidates=[e for e in candidates if {'하계':'여름','동계':'겨울'}.get(e['season'],e['season'])=={'하계':'여름','동계':'겨울'}.get(season,season)]
    if not candidates:
        message=('요청한 과목·학기의 강의계획서 원문을 찾지 못했어. 과목명과 학기를 확인해줘.' if names else
                 '어떤 과목의 강의계획서가 필요해? 과목명과 학기를 알려줘.')
        return dict(status='needs_clarification',message=message,documents=[])
    periods={(e['year'],e['semester'],e['season']) for e in candidates}
    ambiguous=len(periods)>1 or len({e['course'] for e in candidates})>1
    if len(candidates)>8:
        return dict(status='needs_clarification',message='여러 강의계획서가 있어. 과목명과 학기를 더 구체적으로 알려줘.',documents=[])
    lines=['강의계획서의 세부 내용은 원문에서 확인해줘.']
    if ambiguous:lines.append('여러 학기 자료가 있어. 필요한 학기의 원문을 선택해줘.')
    for e in candidates:
        label=f"{e['course'] or e['filename']} — {e['academic_period']}"
        label=label.replace('[','(').replace(']',')')
        url=e['attachment_url'] or e['source_url']
        lines.append(f'- [{label}]({url})' if url else '- '+label)
        if e['zip_member']:lines.append('  ZIP 내부 파일: '+e['zip_member'])
        if e['period_conflict']:lines.append('  공지와 PDF 본문의 학기 표기가 달라. 원문을 확인해줘.')
    return dict(status='needs_clarification' if ambiguous else 'answered',message='\n'.join(lines),documents=candidates)


def guidance_request(query,catalog):
    response=lookup(query,catalog)
    if response is None:return None
    return dict(query=query,sources=[],messages=[],original_document_response=response,
                retrieval=dict(strategy='syllabus_original_document_lookup',generation_api_calls=0,body_evidence_used=False))


def main():
    import argparse,json,hashlib
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--normalized',type=Path,default=Path('data/processed/normalized-text/records.jsonl'))
    p.add_argument('--output',type=Path,default=Path('data/catalog/syllabus-originals.json'))
    a=p.parse_args()
    records=[json.loads(l) for l in a.normalized.open()]
    entries=build_catalog(records,a.normalized.parents[2])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(entries=len(entries),documents=len({e['document_id'] for e in entries}),
                         missing_course_names=sum(not e['course'] for e in entries),
                         source_sha256=hashlib.sha256(a.normalized.read_bytes()).hexdigest()),ensure_ascii=False))

if __name__=='__main__':main()
