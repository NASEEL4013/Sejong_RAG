"""Q: three-week syllabus blocks with PDF-verified row boundaries."""
import re
import unicodedata
from sejong_rag.semantic_rechunk import NeedsReview, make_chunk
from sejong_rag.semantic_syllabus import PATTERNS


def compact(text):return ''.join(unicodedata.normalize('NFC',text).split())


def pdf_row_starts(page,record):
    """Map source offsets to table-row tops, including vertically centered labels.

    pypdf's layout groups are checked against the entire stored page text. A
    version/layout mismatch is held for review instead of guessing offsets.
    """
    from pypdf._text_extraction import _layout_mode
    from pypdf.generic import ContentStream
    bt=_layout_mode.text_show_operations(iter(ContentStream(page['/Contents'].get_object(),page.pdf,'bytes').operations),page._layout_mode_fonts(),True,None)
    groups=_layout_mode.y_coordinate_groups(bt,None)
    width=_layout_mode.fixed_char_width(bt,1.25)
    lines=[];joined=''
    for y,items in groups.items():
        value=compact(_layout_mode.fixed_width_page({y:items},width,False,1))
        if value:
            lines.append((len(joined),len(joined)+len(value),min(i['ty'] for i in items),max(i['ty'] for i in items)))
            joined+=value
    source=record['text']
    if joined!=compact(source):raise NeedsReview('schedule_pdf_text_mapping')
    offsets=[i for i,c in enumerate(source) if not c.isspace()]
    if len(offsets)!=len(joined):raise NeedsReview('schedule_unicode_mapping')
    borders=[];last=None
    def xy(x,y,cm):return (x*cm[0]+y*cm[2]+cm[4],x*cm[1]+y*cm[3]+cm[5])
    def operand(op,args,cm,tm):
        nonlocal last
        if op==b're':
            x,y,w,h=map(float,args)
            if w>100 and 0<h<2:
                p=xy(x,y,cm);q=xy(x+w,y,cm)
                if abs(p[1]-q[1])<.1:borders.append(p[1])
        elif op==b'm':last=xy(float(args[0]),float(args[1]),cm)
        elif op==b'l':
            point=xy(float(args[0]),float(args[1]),cm)
            if last and abs(last[1]-point[1])<.1 and abs(last[0]-point[0])>100:borders.append(point[1])
            last=point
    page.extract_text(visitor_operand_before=operand)
    def row_start(offset):
        index=next((i for i,(a,b,lo,hi) in enumerate(lines) if offsets[a]<=offset<=offsets[b-1]),None)
        if index is None:raise NeedsReview('schedule_week_label_mapping')
        _,_,lo,hi=lines[index]
        above=[y for y in borders if y>hi+.1]
        if not above:raise NeedsReview('schedule_pdf_row_border_missing')
        top=min(above)
        first=index
        while first>0 and lines[first-1][3]<top-.1:first-=1
        # No line may contain text from opposite sides of this table boundary.
        if any(low<top<high for _,_,low,high in lines):raise NeedsReview('schedule_pdf_crossed_row')
        return offsets[lines[first][0]]
    return row_start


def split_schedule(records,template,helper,pdf):
    from sejong_rag.semantic_bulk import ordered_text, refs_for, issue
    identities={(r['document_id'],r['locator'].get('zip_member','')) for r in records}
    if len(identities)!=1:raise NeedsReview('schedule_mixed_courses')
    text,ranges=ordered_text(records)
    intro=list(re.finditer(PATTERNS[0],text,re.M));schedule=list(re.finditer(PATTERNS[-1],text,re.M))
    if len(intro)!=1 or len(schedule)!=1:raise NeedsReview('schedule_heading_layout')
    prefix=text[:intro[0].start()];a=schedule[0].start()
    course_match=re.search(r'교과목명\s*\(국문\)\s*(.+)',prefix)
    if not course_match:raise NeedsReview('schedule_course_context_missing')
    course=course_match[1].split('(영문)')[0].strip()
    search_prefix='\n'.join(l for l in prefix.splitlines() if '교과목명' in l or '개설학기' in l)
    if '개설학기' not in search_prefix:raise NeedsReview('schedule_term_missing')
    labels=list(re.finditer(r'(?m)^[ \t]*(\d{1,2})\s*주',text[a:]))
    numbers=[int(m[1]) for m in labels]
    unique=list(dict.fromkeys(numbers))
    if numbers!=sorted(numbers) or unique!=list(range(1,16)):
        raise NeedsReview('schedule_incomplete_or_unordered_weeks')
    positions={};mappers={}
    for m in labels:
        week=int(m[1])
        if week in positions:continue
        offset=a+m.start(1)
        x,y,r=next(v for v in ranges if v[0]<=offset<v[1])
        if r['record_id'] not in mappers:mappers[r['record_id']]=pdf_row_starts(pdf.pages[r['locator']['page']-1],r)
        positions[week]=x+mappers[r['record_id']](offset-x)
    if not (a<positions[1] and list(positions.values())==sorted(set(positions.values()))):
        raise NeedsReview('schedule_row_boundaries_overlap')
    table_header=text[a:positions[1]]
    chunks=[];pending=[]
    for first,last in [(1,3),(4,6),(7,9),(10,12),(13,15)]:
        start=positions[first];end=positions[last+1] if last<15 else len(text)
        body=text[start:end]
        relation=f'강의내용: {course} / 전체 15주 중 {first}~{last}주차\n일정 순서: 1~3주 → 4~6주 → 7~9주 → 10~12주 → 13~15주'
        source=refs_for(ranges,start,end)
        context=refs_for(ranges,0,intro[0].start())+refs_for(ranges,a,positions[1])+source
        try:
            c=make_chunk(template,records[0],source,search_prefix+'\n'+relation+'\n'+table_header+body,
                         context,prefix+'\n'+relation+'\n'+table_header+body,helper,'Q_syllabus_three_week_schedule')
        except NeedsReview as e:
            pending.append(issue('schedule_'+str(e),records,source));continue
        c.update(syllabus_course=course,syllabus_member=next(iter(identities))[1],
                 syllabus_topic='주차별 강의내용',syllabus_topic_index=4,syllabus_schedule_scope='three_weeks',
                 syllabus_week_start=first,syllabus_week_end=last,syllabus_total_weeks=15,
                 syllabus_row_geometry_verified=True)
        chunks.append(c)
    return chunks,pending
