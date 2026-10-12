"""Apply approved semantic rules to every remaining selected document.

Incremental staging: accepted chunks are immutable; uncovered source ranges are
explicitly queued. New/ambiguous structures are never split just to hit a budget.
"""
from __future__ import annotations
from sejong_rag.corpus_policy import allowed_chunk, EXCLUDED_DOCUMENTS

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import html
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
from sejong_rag.semantic_rechunk import (NeedsReview, read_jsonl, write_jsonl, digest, ref,
    make_chunk, cells, LABELS, TABLE_LABELS, document_progress)
from sejong_rag.semantic_reference import uncovered_spans
from sejong_rag.semantic_matrix import source_grid, MatrixReview
from sejong_rag.structure_chunk import Chunker, MODEL

VERSION='approved_rules_bulk_v6'
ENGLISH_COLUMNS={'no.','department/major','departmentofopening','courseno.','class','coursetitle',
    'classification','year','credit','classhour','venue','lecturer','e-learning','classlanguage',
    'fornon-koreansonly','koreansonly'}
MORE_LABELS={'접수기간','환불기간','환불방법','기타','문의','신청기간','신청방법','신청대상',
    '등록안내','등록기간','납부기간','납부방법','제출서류','제출방법','제출처','관련문의','지원내용',
    '유의사항','주의사항','선발기준','지원자격','신청자격','지원대상','신청절차','신청서류',
    '수강신청기간','수강신청방법','신청및접수','신청및접수기간','개요','선발개요','사업목적','사업내용','신청안내','지원안내','등록금납부기간','학점등록금액산출기준','수강신청안내'}
ALL_LABELS=LABELS|MORE_LABELS
REFERENCE=re.compile(r'상기|위의|아래의|하단|상단|전술|후술|동일기간|상동|다음페이지|앞페이지')
PLACEHOLDER=re.compile(r'<(?:표|그림)>|\[\s*\][^\n]*(?:>|→)[^\n]*\[\s*\]')


def compact(text):return re.sub(r'\s+','',text).casefold()


class CachedHelper:
    def __init__(self,tokenizer):
        helper=Chunker(tokenizer)
        self.count=lru_cache(maxsize=4096)(helper.count)


def issue(reason,records,spans=None):
    out=dict(reason=reason,record_ids=[r['record_id'] for r in records])
    if spans is not None:out['source_spans']=spans
    return out


def ordered_text(records):
    """Preserve normalized extraction order; offsets are local to this view."""
    parts=[];ranges=[];cursor=0
    for r in records:
        if parts:cursor+=1
        parts.append(r['text']);ranges.append((cursor,cursor+len(r['text']),r))
        cursor+=len(r['text'])
    return '\n'.join(parts),ranges


def refs_for(ranges,a,b):
    return [ref(r,max(a,start)-start,min(b,end)-start) for start,end,r in ranges
            if max(a,start)<min(b,end)]


def prose_sections(text):
    # First choose one visible outer numbering level. This keeps nested AND
    # conditions and exceptions together (B/C), rather than splitting sentences.
    roman=list(re.finditer(r'(?m)^[ \t]*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ][.\s]+(?=\S)',text))
    numbers=list(re.finditer(r'(?m)^[ \t]*\d{1,2}\s*[.)][ \t]*(?:\n[ \t]*)?(?=[가-힣A-Za-z<「])',text))
    korean=list(re.finditer(r'(?m)^[ \t]*[가-하]\s*[.)][ \t]*(?:\n[ \t]*)?',text))
    bullets=list(re.finditer(r'(?m)^[ \t]*[■□◆●○ㅁ][ \t]*(?:\n[ \t]*)?',text))
    def named(m):
        first=compact(text[m.end():m.end()+100])
        return any(first.startswith(label) for label in ALL_LABELS)
    roman=[m for m in roman if named(m)]
    levels=[[m for m in level if named(m)] for level in (numbers,korean,bullets)]
    levels=[level for level in levels if len(level)>=2]
    if roman: starts=[m.start() for m in roman]
    elif levels: starts=[m.start() for m in min(levels,key=lambda level:level[0].start())]
    else:
        starts=[]
        for m in re.finditer(r'(?m)^.*\S.*$',text):
            line=m.group().strip()
            # Strip presentation markers, not content or suffix conditions.
            stripped=re.sub(r'^(?:[■□◆●○ㅁ-]|[가-하][.)]|\d{1,2}[.)])\s*','',line)
            label=compact(re.split(r'[:：]',stripped,maxsplit=1)[0])
            if label in ALL_LABELS:starts.append(m.start())
    if not starts:return [(0,len(text))],[]
    starts=sorted(set(starts));prefix=[(0,starts[0])] if starts[0] and text[:starts[0]].strip() else []
    return list(zip(starts,starts[1:]+[len(text)])),prefix


def looks_flat_table(text):
    if ' | ' in text or re.search(r'(?m)\S[ \t]{3,}\S',text):return True
    # PDF timetable/syllabus/roster extraction is not trustworthy cell geometry.
    joined=compact(text)
    return any(all(word in joined for word in terms) for terms in (
        ('교과목명','학수번호'),('교과목표','평가방법'),('주별','교수내용'),
        ('선발대상','선발인원','심사기준'),('학번','성명','학과','명단')))


def make_from_ranges(template,primary,ranges,spans,context,body,context_body,helper,rule):
    return make_chunk(template,primary,[s for a,b in spans for s in refs_for(ranges,a,b)],body,
        [s for a,b in context for s in refs_for(ranges,a,b)],context_body,helper,rule)


def prose(records,template,helper,*,is_notice=False,html_text=None):
    text,ranges=ordered_text(records);primary=records[0]
    # J's approved example itself contains picture markers. Accept them only
    # when the complete, sequential textual procedure has explicit boundaries.
    steps=list(re.finditer(r'(?m)^\s*(\d{1,2})[.)]\s*',text))
    if (len(records)==1 and len(steps)>=2 and [int(m[1]) for m in steps]==list(range(1,len(steps)+1))
            and re.search(r'로그인|접속',text[steps[0].end():steps[1].start()])
            and '완료' in text[steps[-1].end():] and '<표>' not in text
            and not re.search(r'\[\s*\][^\n]*(?:>|→)[^\n]*\[\s*\]',text)):
        try:
            spans=[(0,len(text))]
            c=make_from_ranges(template,primary,ranges,spans,spans,text,text,helper,'J_complete_short_application_procedure')
            c['procedure_step_count']=len(steps)
            return [c],[]
        except NeedsReview:pass
    if PLACEHOLDER.search(text):raise NeedsReview('extraction_missing_table_image_or_button')
    if re.search(r'(?:차\s*례|목\s*차|\bIndex\b|\bContents\b)',text[:250],re.I):
        raise NeedsReview('new_layout_contents_or_multipage_outline')
    if not is_notice and looks_flat_table(text):raise NeedsReview('new_layout_flattened_attachment_table')
    table_ranges=[];table_nodes={}
    if html_text:
        soup=BeautifulSoup(html_text,'html.parser')
        for table in soup.find_all('table'):
            if table.find('table'):continue
            try:
                cs,grid,span,*_=source_grid(text,table)
                table_ranges.append(span);table_nodes[span]=table
            except MatrixReview:
                # Ignore navigation tables not represented in the source body;
                # any body-looking table that cannot be mapped blocks this record.
                value=compact(table.get_text())
                if value and value[:min(40,len(value))] in compact(text):
                    raise NeedsReview('new_layout_html_table_mapping')
    # A compact complete procedure/form/topic retains all conditions. Tables need
    # mapped cells, not a generic short-text escape hatch.
    whole_spans=[(0,len(text))]
    if not table_ranges:
        method=None
        joined=compact(text)
        form_like=bool(re.search(r'신청서|사유서|지원서',text[:180]) and re.search(r'성\s*명\s*:',text))
        if form_like and re.search(r'(?m)^[•□* \t]*성\s*명\s*:[_\t ]*$',text) and re.search(r'서명|\(인\)|날인',text):
            method='M_complete_blank_application_form'
        elif form_like:
            raise NeedsReview('new_layout_form_fields_or_filled_example')
        elif re.search(r'신청|매뉴얼|방법',text[:180]) and re.search(r'클릭|선택|저장',text) and re.search(r'완료|승인|제출|확인',text):
            method='J_complete_short_application_procedure'
        elif is_notice or (len(records)==1 and re.search(r'합니다|바랍니다|됩니다|안내|신청|유의',text)):
            method='BC_complete_short_topic_with_conditions'
        if method:
            body=('문서 용도: 작성용 신청서 (빈 서식)\n' if method.startswith('M_') else '')+text
            try:
                c=make_from_ranges(template,primary,ranges,whole_spans,whole_spans,body,body,helper,method)
                if method.startswith('M_'):c['document_role']='blank_application_form'
                return [c],[]
            except NeedsReview:pass
    if REFERENCE.search(text):raise NeedsReview('existing_rule_unresolved_cross_reference')
    sections,prefix=prose_sections(text)
    if len(sections)==1 and not table_ranges:raise NeedsReview('existing_rule_indivisible_topic_or_unknown_heading')
    if any(max(x,a)<min(y,b) for x,y in prefix for a,b in table_ranges):
        raise NeedsReview('new_layout_table_in_shared_preamble')
    chunks=[];pending=[]
    for a,b in sections:
        # A table crossing the outer topic boundary needs its own validated rule.
        overlaps=[span for span in table_ranges if max(a,span[0])<min(b,span[1])]
        if overlaps:
            if len(overlaps)==1 and a<=overlaps[0][0] and overlaps[0][1]<=b:
                try:
                    cs,ps=html_table_section(text,ranges,a,b,prefix,table_nodes[overlaps[0]],template,primary,helper)
                    chunks.extend(cs);pending.extend(ps)
                except (NeedsReview,MatrixReview) as e:
                    pending.append(issue(str(e),records,refs_for(ranges,a,b)))
            else:pending.append(issue('new_layout_table_with_topic_context',records,refs_for(ranges,a,b)))
            continue
        value=text[a:b]
        if not is_notice and looks_flat_table(value):
            pending.append(issue('new_layout_flattened_attachment_table',records,refs_for(ranges,a,b)));continue
        if not is_notice and len(records)>1:
            # Accept only sections with an explicit descriptive heading, not a
            # page boundary or arbitrary paragraph from a longer manual.
            first=compact(re.sub(r'^[\s\dⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ.)■□◆-]+','',value).split('\n',1)[0])
            if not any(first.startswith(label) for label in ALL_LABELS):
                pending.append(issue('new_layout_attachment_section_scope',records,refs_for(ranges,a,b)));continue
        ctx=prefix+[(a,b)];context='\n'.join(text[x:y] for x,y in ctx)
        try:
            chunks.append(make_from_ranges(template,primary,ranges,[(a,b)],ctx,value,context,helper,'BC_topic_with_conditions_bulk'))
        except NeedsReview as e:pending.append(issue('existing_rule_'+str(e),records,refs_for(ranges,a,b)))
    return chunks,pending


def html_table_section(text,ranges,a,b,prefix,table,template,primary,helper):
    parsed,grid,span,_,_,_,nr,nc=source_grid(text,table)
    if nr<2 or nc<2 or any(c['rows']!=1 or c['cols']!=1 for c in parsed):
        raise NeedsReview('new_layout_html_merged_or_transposed_table')
    labels=[parsed[grid[0,col]]['text'] for col in range(nc)]
    known={compact(x) for x in TABLE_LABELS}|ENGLISH_COLUMNS|{'구분','과목','중간고사','기말고사','일자','시간','연락처','교과목','등록기간','납부금액','기간','장학금액','선발인원'}
    if (len(set(map(compact,labels)))!=nc or any(not v.strip() or len(v)>60 for v in labels)
            or sum(compact(v) in known for v in labels)<2):
        raise NeedsReview('new_layout_html_header_requires_review')
    # All notes before/after this table in the same outer topic are common
    # context, including dates, exceptions, and instructions.
    common=prefix+[(a,span[0]),(span[1],b)]
    common=[(x,y) for x,y in common if x<y and text[x:y].strip()]
    common_text='\n'.join(text[x:y] for x,y in common)
    head_spans=[parsed[grid[0,col]]['span'] for col in range(nc) if parsed[grid[0,col]]['span']]
    chunks=[];pending=[]
    for row in range(1,nr):
        values=[parsed[grid[row,col]]['text'] for col in range(nc)]
        if list(map(compact,values))==list(map(compact,labels)):
            raise NeedsReview('new_layout_html_repeated_header')
        if not values[0].strip():raise NeedsReview('new_layout_html_missing_row_key')
        row_spans=[parsed[grid[row,col]]['span'] for col in range(nc) if parsed[grid[row,col]]['span']]
        body=('공통 문맥:\n'+common_text+'\n\n' if common_text else '')+'행 내용:\n'
        body+='\n'.join(f'{label}: {value if value.strip() else "[원문 빈칸]"}' for label,value in zip(labels,values))
        all_spans=common+head_spans+row_spans
        try:
            c=make_from_ranges(template,primary,ranges,head_spans+row_spans,all_spans,body,body,helper,'D_row_with_table_context_html')
            c['table_geometry']=dict(normalized_table_span=list(span),row=row,columns=nc)
            chunks.append(c)
        except NeedsReview as e:pending.append(issue('existing_rule_'+str(e),[primary],refs_for(ranges,*span)))
    return chunks,pending


def simple_table(records,template,helper):
    records=sorted(records,key=lambda r:r['locator']['row']);parsed=[cells(r['text']) for r in records]
    labels_known={compact(x) for x in TABLE_LABELS}|ENGLISH_COLUMNS|{'실습조','학과(부)','이름학번'}
    hits=[i for i,values in enumerate(parsed) if len(values)>=3 and sum(compact(v) in labels_known for v in values.values())>=3]
    if len(hits)!=1:raise NeedsReview('new_layout_missing_or_repeated_table_header')
    hi=hits[0];labels=parsed[hi]
    if hi>6 or any(len(v)>2 for v in parsed[:hi]):raise NeedsReview('new_layout_multilevel_table_context')
    if len(set(compact(v) for v in labels.values()))!=len(labels):raise NeedsReview('new_layout_duplicate_column_labels')
    if not records[hi+1:]:raise NeedsReview('new_layout_header_only_or_blank_form')
    # Unknown footer/second data block may qualify all rows; never assume locality.
    if any(len(v)<3 or min(labels) not in v or not set(v)<=set(labels) for v in parsed[hi+1:]):
        raise NeedsReview('new_layout_sparse_or_side_by_side_table')
    common=records[:hi+1];prefix='표 공통 문맥:\n'+'\n'.join(r['text'] for r in common)+'\n\n행 내용:\n'
    chunks=[];pending=[]
    for r,values in zip(records[hi+1:],parsed[hi+1:]):
        if any(re.search(r'변경\s*전|변경\s*후',v) for v in values.values()):
            raise NeedsReview('new_layout_before_after_table')
        body=prefix+'\n'.join(f'{labels[c]}: {v}' for c,v in values.items())
        spans=[ref(x) for x in common]+[ref(r)]
        try:chunks.append(make_chunk(template,r,spans,body,spans,body,helper,'D_row_with_table_context_bulk'))
        except NeedsReview as e:pending.append(issue('existing_rule_'+str(e),[r]))
    return chunks,pending


def family(reason,record):
    if 'extraction' in reason or PLACEHOLDER.search(record['text']):return '추출 누락·이미지 의존'
    if any(t in reason for t in ('budget','indivisible','long_shared')):return '기존 규칙: 길이·공통 문맥 검토'
    if any(t in reason for t in ('reference','remaining','partial')):return '기존 규칙: 참조·남은 구간 검토'
    if any(t in reason for t in ('sparse','side_by_side','header','table','hwpx')):return '새 구조: 표·병합·서식'
    if any(t in reason for t in ('contents','outline','multipage')):return '새 구조: 목차·다단계 문서'
    if 'scope' in reason:return '새 구조: 첨부 문서의 항목 경계'
    return '새 구조: 기타 문맥 확인'


REVIEW_EXPLANATIONS={
    'extraction_missing_table_image_or_button':'표·그림·버튼 글자가 텍스트에 빠져 있어 원문 확인이나 재추출이 필요해.',
    'existing_rule_indivisible_topic_or_unknown_heading':'독립된 주제 경계를 확인하지 못했거나, 조건을 함께 보존하면 청크 한도를 넘어.',
    'existing_rule_semantic_unit_exceeds_embedding_budget':'조건을 유지한 한 항목이 검색 청크의 토큰 한도를 넘어.',
    'existing_rule_required_context_exceeds_bundle_budget':'필수 공통 문맥을 함께 넣으면 전달 한도를 넘어.',
    'existing_rule_unresolved_cross_reference':'상기·위의·아래의 등 다른 구간을 가리키는 문구의 연결 범위를 확인해야 해.',
    'existing_rule_partial_reference_context':'이미 처리한 참조형 청크 외의 남은 구간이야. 공통 조건의 적용 범위를 확인해야 해.',
    'new_layout_partial_attachment_context':'첨부 일부만 현재 검색 대상으로 선택되어 있어 전체 문맥을 임의로 연결하지 않았어.',
    'new_layout_archive_member_context':'압축파일 안에 여러 문서가 있어 내부 파일별 문맥과 순서를 확인해야 해.',
    'new_layout_contents_or_multipage_outline':'목차와 여러 장·절이 함께 있는 문서야. 장별 공통 문맥을 정해야 해.',
    'new_layout_table_in_shared_preamble':'주제 앞 공통 안내에 표가 있어 어느 항목에 붙일지 확인해야 해.',
    'new_layout_missing_or_repeated_table_header':'열 이름이 없거나 여러 번 등장해 표의 범위를 확정하지 못했어.',
    'new_layout_header_only_or_blank_form':'자료 행 없이 제목·열 이름만 있거나 빈 표 서식이야.',
    'new_layout_table_with_topic_context':'한 주제에 여러 표가 있거나 표가 주제 경계에 걸쳐 있어.',
    'new_layout_html_merged_or_transposed_table':'병합 셀·가로형 비교 구조로 단순한 행 단위 규칙을 적용하기 어려워.',
    'new_layout_sparse_or_side_by_side_table':'빈 행·별도 목록·옆으로 나란한 표가 섞여 있을 수 있어.',
    'new_layout_flattened_attachment_table':'추출 텍스트에서 표의 행·열 관계가 펼쳐져 원본 배치를 확인해야 해.',
    'new_layout_html_header_requires_review':'표 첫 행을 열 이름으로 확정하기 어려워.',
    'new_layout_attachment_table_geometry':'첨부파일의 병합·표 구조에 맞는 원본 위치 연결이 필요해.',
    'new_layout_multilevel_table_context':'표 위 제목·주석·다단 열 이름의 적용 범위를 확인해야 해.',
    'new_layout_html_table_mapping':'원본 표와 추출된 문장을 유일하게 연결하지 못했어.',
    'table_cell_structure_unknown':'열 번호 형식을 해석하지 못했어.',
    'existing_rule_uncovered_context':'다른 항목과 함께 보존해야 하는 공통 부분이 남아 있어.',
}


def build_review(output,pending,records,progress):
    types=defaultdict(list)
    for item in pending:
        for rid in item['record_ids']:
            r=records[rid];key=(family(item['reason'],r),r.get('file_type','notice'),item['reason'])
            types[key].append((r,item))
    rows=[]
    for i,(key,items) in enumerate(sorted(types.items(),key=lambda kv:(kv[0][0],-len({r['document_id'] for r,_ in kv[1]}))),1):
        bydoc=defaultdict(list)
        for r,item in items:bydoc[r['document_id']].append((r,item))
        docs=[]
        for doc,entries in sorted(bydoc.items()):
            r=entries[0][0];examples=[]
            for rr,item in entries[:3]:
                spans=[s for s in item.get('source_spans',[]) if s['record_id']==rr['record_id']]
                sample='\n'.join(rr['text'][s['start']:s['end']] for s in spans) if spans else rr['text']
                examples.append(dict(record_id=rr['record_id'],locator=rr.get('locator'),text=sample[:2500],excerpt_truncated=len(sample)>2500))
            docs.append(dict(document_id=doc,title=r.get('document_title'),filename=r.get('filename'),source_url=r.get('source_url'),
                             records=len({rr['record_id'] for rr,_ in entries}),examples=examples))
        rows.append(dict(type_id=f'R{i:03}',category=key[0],file_type=key[1],reason=key[2],explanation=REVIEW_EXPLANATIONS.get(key[2],'원문 구조와 공통 문맥을 함께 확인할 항목이야.'),documents=len(docs),items=docs))
    (output/'review-types.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    esc=html.escape
    cards=[]
    for row in rows:
        docs=[]
        for doc in row['items']:
            excerpts=''.join('<p>'+esc(x['record_id'])+' · '+esc(json.dumps(x['locator'],ensure_ascii=False))+'</p><pre>'+esc(x['text'])+('\n[발췌: 전체 원문은 normalized records에 보존]' if x['excerpt_truncated'] else '')+'</pre>' for x in doc['examples'])
            docs.append('<details class="doc"><summary>'+esc(str(doc['title']))+' — '+esc(str(doc['filename'] or '공지 본문'))+'</summary><p>'+esc(doc['document_id'])+'</p>'+excerpts+'</details>')
        cards.append('<section><h2>'+row['type_id']+' · '+esc(row['category'])+' · '+str(row['documents'])+'개 문서</h2><p>'+esc(row['file_type']+' · '+row['explanation'])+'</p>'+''.join(docs)+'</section>')
    summary=f"대상 {progress['selected_documents']}개 · 완료 {progress['complete']}개 · 일부 {progress['partial']}개 · 대기 {progress['not_chunked']}개"
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>세종 RAG 청킹 검토 모음</title><style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 20px;color:#183047;background:#f5f7fa}section{background:white;border:1px solid #dae2e9;border-radius:10px;padding:20px;margin:20px 0}summary{cursor:pointer;padding:10px;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6;background:#f5f7fa;padding:15px}input{padding:12px;width:90%;font-size:16px}.doc{border-top:1px solid #eee}h2{font-size:20px}</style><h1>청킹 검토 모음</h1><p>'''+esc(summary)+'''</p><p>새 구조와 기존 규칙의 길이·참조 문제, 추출 누락을 구분했어. 처리하지 않은 원문은 그대로 보존돼 있어. 한 문서가 여러 검토 유형에 포함될 수 있어. 유형별 문서를 펼쳐서 예시를 확인할 수 있어.</p><input id="q" placeholder="제목·파일명·유형·원문 검색">'''+''.join(cards)+'''<script>document.querySelector('#q').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('section').forEach(s=>{let show=false;s.querySelectorAll('.doc').forEach(d=>{const hit=(s.querySelector('h2').textContent+' '+s.querySelector('p').textContent+' '+d.textContent).toLowerCase().includes(q);d.hidden=!hit;show ||=hit});s.hidden=!show})})</script></html>'''
    (output/'review.html').write_text(page)
    return [dict(type_id=r['type_id'],category=r['category'],file_type=r['file_type'],reason=r['reason'],explanation=r['explanation'],documents=r['documents']) for r in rows]


def build(baseline,output,tokenizer,normalized=Path('data/processed/normalized-text/records.jsonl'),source=Path('data/processed/selected-chunks/selected.jsonl'),data_dir=Path('data'), *, alternatives_only=False, policy_only=False, syllabus_only=False, schedule_only=False):
    baseline,output=Path(baseline),Path(output)
    if output.exists():raise FileExistsError('Use a fresh staging directory')
    normalized,source=Path(normalized),Path(source)
    hashes={str(p):digest(p) for p in (normalized,source,baseline/'chunks.jsonl',baseline/'needs-review.jsonl')}
    records={r['record_id']:r for r in read_jsonl(normalized)}
    old=read_jsonl(source);accepted=read_jsonl(baseline/'chunks.jsonl');previous_pending=read_jsonl(baseline/'needs-review.jsonl')
    baseline_total=len(accepted)
    policy_removed=[c for c in accepted if not allowed_chunk(c)]
    old=[c for c in old if allowed_chunk(c)]
    accepted=[c for c in accepted if allowed_chunk(c)]
    previous_pending=[p for p in previous_pending if all(allowed_chunk(records[rid]) for rid in p['record_ids'])]
    templates={}
    for c in old:
        for s in c['source_spans']:templates.setdefault(s['record_id'],c)
    all_docs=defaultdict(list)
    for r in records.values():all_docs[r['document_id']].append(r)
    pending_ids={rid for p in previous_pending for rid in p['record_ids']}
    base_coverage=defaultdict(list)
    for c in accepted:
        for s in c['source_spans']+c.get('context_source_spans',[]):base_coverage[s['record_id']].append((s['start'],s['end']))
    selected_docs=defaultdict(list)
    for rid in templates:selected_docs[records[rid]['document_id']].append(records[rid])
    helper=CachedHelper(tokenizer);new=[];attempts=[];processed=set();reasons=defaultdict(list)
    schedule_audit=[];syllabus_audit=[];alternative_audit=[];superseded=[];baseline_count=baseline_total
    if schedule_only:
        import io, zipfile
        from pypdf import PdfReader
        from sejong_rag.extract_attachments import load_attachment_metadata, _zip_display_name
        from sejong_rag.semantic_syllabus import syllabus_groups
        from sejong_rag.syllabus_schedule import split_schedule
        raw_paths={m['document_id']:Path(m['resolved_raw_path']) for m in load_attachment_metadata(Path(data_dir))}
        known={(c['document_id'],c.get('syllabus_member','')) for c in accepted if c.get('syllabus_course')}
        complete={(c['document_id'],c.get('syllabus_member','')) for c in accepted if c.get('syllabus_schedule_scope')=='whole_course'}
        existing_ids={c['chunk_id'] for c in accepted}
        for doc,selected in selected_docs.items():
            for member,rs in syllabus_groups(all_docs[doc]):
                if (doc,member) not in known or (doc,member) in complete:continue
                try:
                    path=raw_paths[doc];hashes[str(path)]=digest(path)
                    if member:
                        with zipfile.ZipFile(path) as z:
                            name=next(n for n in z.namelist() if _zip_display_name(n)==member)
                            pdf=PdfReader(io.BytesIO(z.read(name)))
                    else:pdf=PdfReader(path)
                    cs,wait=split_schedule(rs,templates[selected[0]['record_id']],helper,pdf)
                    cs=[c for c in cs if c['chunk_id'] not in existing_ids]
                    new.extend(cs);attempts.extend(wait)
                    schedule_audit.append(dict(document_id=doc,member=member,added_chunks=len(cs),pending_blocks=wait,
                                               course=cs[0]['syllabus_course'] if cs else None))
                except NeedsReview as e:
                    attempts.append(issue(str(e),rs))
                    schedule_audit.append(dict(document_id=doc,member=member,added_chunks=0,reason=str(e)))
        attempts.extend(previous_pending);processed.update(pending_ids)
    if syllabus_only:
        from sejong_rag.semantic_syllabus import syllabus_groups, split_syllabus
        for doc, selected in selected_docs.items():
            for member,rs in syllabus_groups(all_docs[doc]):
                # Previously accepted units remain immutable; current syllabus
                # candidates were all queued. Avoid duplicating an accepted topic.
                try:
                    cs,wait=split_syllabus(rs,templates[selected[0]['record_id']],helper)
                    cs=[c for c in cs if c['chunk_id'] not in {x['chunk_id'] for x in accepted}]
                    new.extend(cs);attempts.extend(wait)
                    syllabus_audit.append(dict(document_id=doc,member=member,added_chunks=len(cs),
                                               pending_topics=wait,course=cs[0]['syllabus_course'] if cs else None))
                except NeedsReview as e:
                    attempts.append(issue(str(e),rs))
                    syllabus_audit.append(dict(document_id=doc,member=member,added_chunks=0,reason=str(e)))
        attempts.extend(previous_pending);processed.update(pending_ids)
    if policy_only:
        attempts.extend(previous_pending);processed.update(pending_ids)
    if alternatives_only:
        from sejong_rag.semantic_alternatives import split_alternatives, CHOICE
        # Scan all selected documents, including already complete short notices.
        for doc, selected in selected_docs.items():
            rs=all_docs[doc]
            if not CHOICE.search(compact('\n'.join(r['text'] for r in rs))):continue
            try:
                cs,reason=split_alternatives(rs,templates[selected[0]['record_id']],helper)
                new.extend(cs)
                alternative_audit.append(dict(document_id=doc,added_chunks=len(cs),reason=reason))
            except NeedsReview as e:
                alternative_audit.append(dict(document_id=doc,added_chunks=0,reason=str(e)))
        # Replace only baseline units wholly covered by the new source/context.
        # Their exact previous representation remains in baseline + audit file.
        new_coverage=defaultdict(list)
        for c in new:
            for s in c['source_spans']+c['context_source_spans']:
                new_coverage[s['record_id']].append((s['start'],s['end']))
        for c in accepted:
            spans=c['source_spans']+c.get('context_source_spans',[])
            if spans and all(not uncovered_spans(records[s['record_id']]['text'][s['start']:s['end']],
                    [(max(a,s['start'])-s['start'],min(b,s['end'])-s['start'])
                     for a,b in new_coverage[s['record_id']] if max(a,s['start'])<min(b,s['end'])]) for s in spans):
                superseded.append(c)
        removed={c['chunk_id'] for c in superseded}
        accepted=[c for c in accepted if c['chunk_id'] not in removed]
        attempts.extend(previous_pending);processed.update(pending_ids)
    # D groups first (including English equivalents), with per-row budget handling.
    for item in ([] if (alternatives_only or policy_only or syllabus_only or schedule_only) else previous_pending):
        rs=[records[rid] for rid in item['record_ids']]
        if not rs or not all('sheet' in r.get('locator',{}) for r in rs):continue
        processed.update(r['record_id'] for r in rs)
        try:
            chunks,wait=simple_table(rs,templates[rs[0]['record_id']],helper)
            new.extend(chunks);attempts.extend(wait)
        except NeedsReview as e:attempts.append(issue(str(e),rs))
    for doc,selected in ([] if (alternatives_only or policy_only or syllabus_only or schedule_only) else selected_docs.items()):
        remaining=[r for r in selected if r['record_id'] in pending_ids and r['record_id'] not in processed]
        if not remaining:continue
        whole=all_docs[doc]
        # Existing partial F/G/H/I outputs retain their original exact gap queue;
        # do not infer the scope of a remaining fragment from a different chunk.
        if any(r['record_id'] in base_coverage for r in remaining):
            for r in remaining:
                attempts.append(issue('existing_rule_partial_reference_context',[r]));processed.add(r['record_id'])
            continue
        if len(remaining)!=len(whole):
            attempts.append(issue('new_layout_partial_attachment_context',remaining));processed.update(r['record_id'] for r in remaining);continue
        rs=whole;processed.update(r['record_id'] for r in rs)
        first=rs[0];is_notice=first['source_kind']=='notice_body';html_text=None
        if is_notice:
            path=Path(data_dir)/'raw'/'notices'/(doc+'.html')
            if path.is_file():hashes[str(path)]=digest(path);html_text=path.read_text()
            else:attempts.append(issue('missing_original_html',rs));continue
        elif any('table' in r.get('locator',{}) or 'sheet' in r.get('locator',{}) for r in rs):
            attempts.append(issue('new_layout_attachment_table_geometry',rs));continue
        elif any(r.get('locator',{}).get('zip_member') for r in rs):
            attempts.append(issue('new_layout_archive_member_context',rs));continue
        try:
            chunks,wait=prose(rs,templates[first['record_id']],helper,is_notice=is_notice,html_text=html_text)
            new.extend(chunks);attempts.extend(wait)
        except NeedsReview as e:attempts.append(issue(str(e),rs))
    assert processed==pending_ids,(len(processed),len(pending_ids))
    coverage=defaultdict(list)
    for c in accepted+new:
        for s in c['source_spans']+c.get('context_source_spans',[]):coverage[s['record_id']].append((s['start'],s['end']))
    for p in attempts:
        for rid in p['record_ids']:reasons[rid].append(p['reason'])
    pending=[];done=set()
    for rid in templates:
        r=records[rid];gaps=uncovered_spans(r['text'],coverage[rid])
        if not gaps:done.add(rid);continue
        reason=next(iter(reasons[rid]),'existing_rule_uncovered_context')
        pending.append(dict(reason=reason,record_ids=[rid],document_title=r.get('document_title'),filename=r.get('filename'),
                            source_spans=[ref(r,a,b) for a,b in gaps],locator=r.get('locator'),
                            sample=[dict(record_id=rid,text=r['text'][a:min(b,a+1600)]) for a,b in gaps]))
    waiting={rid for p in pending for rid in p['record_ids']}
    assert done|waiting==set(templates) and not done&waiting
    all_chunks=accepted+new
    assert len({c['chunk_id'] for c in all_chunks})==len(all_chunks)
    for p,h in hashes.items():assert digest(p)==h
    output.mkdir(parents=True)
    write_jsonl(output/'superseded-chunks.jsonl',superseded)
    write_jsonl(output/'chunks.jsonl',all_chunks);write_jsonl(output/'needs-review.jsonl',pending)
    write_jsonl(output/'pending-original-chunks.jsonl',[c for c in old if any(s['record_id'] in waiting for s in c['source_spans'])])
    progress,doc_rows=document_progress(records,set(templates),waiting,{rid for rid,v in coverage.items() if v})
    write_jsonl(output/'document-progress.jsonl',doc_rows)
    excluded=[dict(document_id=d,title=rs[0].get('document_title'),reason=EXCLUDED_DOCUMENTS.get(d,'not_in_original_selected_search_corpus')) for d,rs in all_docs.items() if d not in selected_docs]
    write_jsonl(output/'out-of-scope-documents.jsonl',excluded)
    review_types=build_review(output,pending,records,progress)
    (output/'README.txt').write_text(f"현재 검색 대상 {progress['selected_documents']}개 문서에 대한 기존 규칙 적용 결과."+'\nreview.html: 브라우저에서 열어 유형별 원문 예시 확인.\nreview-types.json: 보류 이유·유형·모든 해당 문서 목록.\nneeds-review.jsonl: 처리하지 않은 정확한 원문 구간.\ndocument-progress.jsonl: 문서별 완료·일부 처리·대기 상태.\nout-of-scope-documents.jsonl: 수집했지만 원래 검색 대상에 없었던 문서.\n기존 청킹과 운영 인덱스는 변경하지 않았음. 새 결과는 별도 인덱스 구축·검증 후 적용해야 함.\n')
    report=dict(version=VERSION,baseline=str(baseline),status='needs_user_review' if pending else 'complete',production_applied=False,
        embedding_calls=0,generation_api_calls=0,document_progress=progress,all_selected_documents_audited=True,
        original_selected_chunks=len(old),policy_excluded_documents=EXCLUDED_DOCUMENTS,policy_removed_chunks=len(policy_removed),baseline_chunks=baseline_count,superseded_chunks=len(superseded),alternative_audit=alternative_audit,syllabus_audit=syllabus_audit,schedule_audit=schedule_audit,added_chunks=len(new),new_chunks=len(all_chunks),
        completed_records=len(done),pending_records=len(waiting),rules=dict(Counter(c['semantic_rule'] for c in all_chunks)),
        review_types=review_types,input_sha256=hashes,code_sha256=digest(Path(__file__)),
        supporting_code_sha256={name:digest(Path(__file__).with_name(name)) for name in ('semantic_rechunk.py','semantic_reference.py','semantic_matrix.py','semantic_alternatives.py','corpus_policy.py','semantic_syllabus.py','syllabus_schedule.py')},
        source_coverage_verified=True,baseline_chunks_preserved=not (superseded or policy_removed),unaffected_baseline_chunks_preserved=True,
        note='All selected documents audited; new formats, extraction defects, and unresolved existing-rule cases remain explicitly queued. Staging only.')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,default=Path('data/processed/semantic-rechunk-v15'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--alternatives-only',action='store_true');p.add_argument('--policy-only',action='store_true');p.add_argument('--syllabus-only',action='store_true');p.add_argument('--schedule-only',action='store_true');a=p.parse_args()
    from transformers import AutoTokenizer
    result=build(a.baseline,a.output,AutoTokenizer.from_pretrained(MODEL,local_files_only=True),alternatives_only=a.alternatives_only,policy_only=a.policy_only,syllabus_only=a.syllabus_only,schedule_only=a.schedule_only)
    print(json.dumps({k:result[k] for k in ('document_progress','added_chunks','new_chunks','completed_records','pending_records')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
