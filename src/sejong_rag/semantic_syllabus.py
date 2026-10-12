"""P: user-approved syllabus topics; preserve full topics, never size-split them."""
from collections import defaultdict
import re
from sejong_rag.semantic_rechunk import NeedsReview, make_chunk

TOPICS = ('과목 소개·목표·난이도·역량', '교재·학습 안내', '평가 기준·예외', '전체 주차별 일정')
PATTERNS = (r'^교과소개[ \t]*$', r'^교과교재[ \t]*$', r'^교과평가[ \t]*$',
            r'^주별\s*(?:수업내용|강의\s*내용)[ \t]*$')


def syllabus_groups(records):
    """Never concatenate different ZIP members, even if their course names match."""
    groups = defaultdict(list)
    for r in records:
        groups[r.get('locator', {}).get('zip_member', '')].append(r)
    return [(member,rs) for member,rs in groups.items()
            if re.search(r'수업계획서|강의계획서', member or rs[0].get('filename') or '')
            or any('교과소개' in r['text'] and '교과목명' in r['text'] for r in rs)]


def split_syllabus(records, template, helper):
    from sejong_rag.semantic_bulk import ordered_text, refs_for, issue
    identities={(r['document_id'],r.get('locator',{}).get('zip_member','')) for r in records}
    if len(identities)!=1:raise NeedsReview('syllabus_mixed_sources')
    pages=[r.get('locator',{}).get('page') for r in records]
    if not pages or not all(isinstance(p,int) for p in pages) or pages!=list(range(1,len(pages)+1)):
        raise NeedsReview('syllabus_missing_or_unordered_pages')
    text,ranges=ordered_text(records)
    headings=[list(re.finditer(p,text,re.M)) for p in PATTERNS]
    if any(len(ms)!=1 for ms in headings):raise NeedsReview('syllabus_other_heading_layout')
    starts=[ms[0].start() for ms in headings]
    if starts!=sorted(starts):raise NeedsReview('syllabus_topic_order_unknown')
    if len(re.findall(r'(?m)^교과목명',text))!=1:raise NeedsReview('syllabus_multiple_courses')
    prefix=text[:starts[0]]
    if not all(x in re.sub(r'\s+','',prefix) for x in ('교과목명','개설학기','담당교수','학점')):
        raise NeedsReview('syllabus_missing_course_context')
    context_spans=refs_for(ranges,0,starts[0])
    course_match=re.search(r'교과목명\s*\(국문\)\s*(.+)',prefix)
    if not course_match:raise NeedsReview('syllabus_missing_course_name')
    course=course_match[1].split('(영문)')[0].strip()
    # Retrieval text keeps full topic content plus course/term identity. The
    # full original course header remains in the LLM context of every chunk.
    search_prefix='\n'.join(line for line in prefix.splitlines()
                            if '교과목명' in line or '개설학기' in line)
    chunks=[];pending=[]
    for i,(a,b) in enumerate(zip(starts,starts[1:]+[len(text)])):
        value=text[a:b]
        if '<표>' in value or '<그림>' in value:
            pending.append(issue('syllabus_extraction_placeholder',records,refs_for(ranges,a,b)));continue
        # Keep the source's entire evaluation block, including regulations,
        # zero-weight rows and caveats. Do not guess cell/header mappings.
        body=prefix+'\n'+value
        try:
            c=make_chunk(template,records[0],refs_for(ranges,a,b),search_prefix+'\n'+value,
                         context_spans+refs_for(ranges,a,b),body,helper,'P_syllabus_topic_'+str(i+1))
        except NeedsReview as e:
            pending.append(issue('syllabus_'+str(e),records,refs_for(ranges,a,b)));continue
        c.update(syllabus_course=course,syllabus_member=next(iter(identities))[1],
                 syllabus_topic=TOPICS[i],syllabus_topic_index=i+1)
        if i==3:c['syllabus_schedule_scope']='whole_course'
        chunks.append(c)
    return chunks,pending
