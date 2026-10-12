"""Conservative A/B/C/D/E/F/G/H/I/J/K/L/M/N rechunking with explicit review queues, never silent fallback.

Creates a staging dataset, not a production index. Original spans are retained for
all text (including shared table context). Unknown structures remain pending.
"""
from __future__ import annotations
from sejong_rag.corpus_policy import allowed_chunk, EXCLUDED_DOCUMENTS

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from sejong_rag.structure_chunk import Chunker, MODEL
from sejong_rag.semantic_chunk_pilot import PLAIN_HEADINGS
from sejong_rag.semantic_hwpx import LayoutReview, read_layout, row_plans
from sejong_rag.semantic_matrix import MatrixReview, candidate as matrix_candidate, matrix_plans
from sejong_rag.semantic_score_table import score_plans
from sejong_rag.semantic_workflow import WorkflowReview, candidate as workflow_candidate, workflow_plans, reference_list_candidate, reference_list_plans
from sejong_rag.semantic_reference import ReferenceReview, REFERENCE, reference_plans, cohort_reference_plans, uncovered_spans

from sejong_rag.semantic_docx_form import FormLayoutReview, candidate as table_form_candidate, topic_plans

VERSION = 'user_guided_semantic_v11'
LABELS = {re.sub(r'\s+', '', s) for s in PLAIN_HEADINGS} | {
    '지원내용', '지원대상', '자격요건', '선발인원', '신청절차', '관련문의',
    '교부일자', '교부처', '신청기간', '신청방법', '제출서류', '제출처',
    '문의처', '지원자격', '선발기준', '신청대상', '신청자격', '접수기간',
}
TABLE_LABELS = {'순번','수험번호','이름','모집단위','수강과목','분반','실습시간','강의실',
                '비고','학번','학과','학과명','성명','번호','교과목명','학수번호','이수구분',
                '학점','학년','개설학과','개설대학','선택영역','전공','연번','No.'}


class NeedsReview(ValueError):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open('x') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')


def ref(record, start=0, end=None):
    return dict(record_id=record['record_id'], start=start,
                end=len(record['text']) if end is None else end, locator=record.get('locator', {}))


def compact(text):
    return re.sub(r'\s+', '', text)


def section_spans(text):
    """Use the outer heading level; inner conditions and exceptions stay together."""
    marked = list(re.finditer(r'(?m)^[■□◆](?:[ \t]*\n|[ \t]+)', text))
    numbered = list(re.finditer(r'(?m)^\d{1,2}[.)](?:[ \t]*\n|[ \t]+)(?!\d)', text))
    starts = [m.start() for m in (marked or numbered)]
    if not starts:
        starts = [m.start() for m in re.finditer(r'(?m)^.+$', text)
                  if compact(m.group()) in LABELS]
    if not starts:
        raise NeedsReview('no_recognized_heading')
    starts = sorted(set([0]+starts))
    return [(a,b) for a,b in zip(starts, starts[1:]+[len(text)]) if text[a:b].strip()]


def header(record):
    result = '문서 제목: '+str(record.get('document_title') or '')+'\n'
    if record.get('filename'): result += '첨부파일: '+record['filename']+'\n'
    if 'sheet' in record.get('locator', {}): result += '시트: '+str(record['locator']['sheet'])+'\n'
    return result+'\n'


def make_chunk(template, record, spans, body, context_spans, context_body, helper, method):
    text = header(record)+body
    context_text = header(record)+context_body
    if helper.count('passage: '+text) > 448:
        raise NeedsReview('semantic_unit_exceeds_embedding_budget')
    if helper.count(context_text) > 512:
        raise NeedsReview('required_context_exceeds_bundle_budget')
    identity = json.dumps([VERSION, method, spans, context_spans], sort_keys=True, ensure_ascii=False)
    chunk = deepcopy(template)
    chunk.update(chunk_id='semantic-'+hashlib.sha256(identity.encode()).hexdigest()[:24],
        text=text, text_length=len(text), token_count=helper.count(text),
        source_spans=spans, record_ids=list(dict.fromkeys(s['record_id'] for s in spans)),
        context_source_spans=context_spans, context_text=context_text,
        context_record_ids=list(dict.fromkeys(s['record_id'] for s in context_spans)),
        context_note='Explicit required context; consume context_text when integrating retrieval',
        chunk_method=VERSION, semantic_rule=method, chunk_max_tokens=448, chunk_overlap_tokens=0)
    return chunk


def split_notice(record, template, helper):
    text = record['text']; chunks = []
    sections=section_spans(text)
    preamble=[]; preamble_text=''
    first=text[sections[0][0]:sections[0][1]]
    first_label=compact(first.splitlines()[0])
    if len(sections)>1 and not re.match(r'^(?:[■□◆]|\d{1,2}[.)])',first) and first_label not in LABELS:
        a,b=sections.pop(0)
        preamble=[ref(record,a,b)];preamble_text=text[a:b]+'\n'
    for a,b in sections:
        body=text[a:b]; spans=[ref(record,a,b)]
        # A: a pickup location and its visiting/identification notes are searchable
        # separately, but always carry the complete location block as context.
        label=compact(body[:70])
        location=bool(re.match(r'^(?:\d{1,2}[.)])?(?:교부처|수령장소|수령처)[:：]', label))
        notes=list(re.finditer(r'(?m)^[※*](?:[ \t]*\n|[ \t]+)',body)) if location else []
        starts=[0]+[m.start() for m in notes]
        for x,y in zip(starts,starts[1:]+[len(body)]):
            if not body[x:y].strip(): continue
            chunks.append(make_chunk(template,record,[ref(record,a+x,a+y)],body[x:y],
                preamble+spans,preamble_text+body,helper,'A_location_with_notes' if notes else 'BC_topic_with_conditions'))
    return chunks


def cells(text):
    matches=list(re.finditer(r'(?:^| \| )(\d+)열: ',text))
    if not matches or matches[0].start()!=0: raise NeedsReview('table_cell_structure_unknown')
    result={}
    for i,m in enumerate(matches):
        col=int(m.group(1))
        if col in result: raise NeedsReview('duplicate_column')
        result[col]=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)]
    return result


def split_table(records, template, helper):
    """D: one explicit single-header table row per chunk; no fill-down inference."""
    records=sorted(records,key=lambda r:r['locator']['row'])
    parsed=[cells(r['text']) for r in records]
    candidates=[i for i,c in enumerate(parsed)
                if len(c)>=3 and sum(compact(v) in TABLE_LABELS for v in c.values())>=3]
    if len(candidates)!=1: raise NeedsReview('table_header_missing_or_multiple')
    hi=candidates[0]; labels=parsed[hi]
    if any(not v.strip() for v in labels.values()) or len(set(labels.values()))!=len(labels):
        raise NeedsReview('table_header_empty_or_duplicate')
    # Only a short title/date/note prefix is covered by the approved D example.
    if hi>6 or any(len(c)>2 for c in parsed[:hi]):
        raise NeedsReview('table_prefix_or_multilevel_header')
    if not records[hi+1:]: raise NeedsReview('table_has_no_data_rows')
    common=records[:hi+1]
    prefix='표 공통 문맥:\n'+'\n'.join(r['text'] for r in common)+'\n\n행 내용:\n'
    chunks=[]
    for r,values in zip(records[hi+1:],parsed[hi+1:]):
        if len(values)<3 or min(labels) not in values or not set(values)<=set(labels):
            raise NeedsReview('table_sparse_row_or_footer_or_merged_cells')
        if any(re.search(r'변경전|변경후|변경 전|변경 후',v) for v in values.values()):
            raise NeedsReview('table_before_after_structure')
        body=prefix+'\n'.join(f'{labels[c]}: {v}' for c,v in values.items())
        spans=[ref(x) for x in common]+[ref(r)]
        chunks.append(make_chunk(template,r,spans,body,spans,body,helper,'D_row_with_table_context'))
    return chunks


def split_hwpx_table(table, records, template, helper, eligible, raw_path, raw_sha):
    plans = row_plans(table, records)
    chunks = []
    for plan in plans:
        if not set(plan['record_ids']) <= eligible:
            raise NeedsReview('hwpx_required_context_not_selected')
        common = '\n'.join(records[rid]['text'] for rid in table['context_ids'])
        body = ('공통 문맥:\n'+common+'\n\n' if common else '')
        body += '\n'.join(f'{label}: {value if value.strip() else "[원문 빈칸]"}' for label,value in plan['values'])
        spans = [ref(records[rid]) for rid in plan['record_ids']]
        primary = records[plan['record_ids'][-1]]
        chunk = make_chunk(template, primary, spans, body, spans, body, helper,
                           'E_before_after_row_with_verified_merges')
        chunk['table_geometry'] = dict(source_path=str(raw_path), source_sha256=raw_sha,
            section=table['section'], table=table['number'], row=plan['row'],
            physical_rows=plan['physical_rows'], cells=plan['geometry'])
        chunks.append(chunk)
    return chunks


def split_reference(record, template, helper):
    text=record['text'];chunks=[]
    try: plans=reference_plans(text)
    except ReferenceReview as e:
        if str(e)!='F_multiple_cohort_periods': raise
        plans=cohort_reference_plans(text)
    for plan in plans:
        spans=[ref(record,a,b) for a,b in plan['spans']]
        body='\n'.join(text[a:b] for a,b in plan['spans'])
        chunk=make_chunk(template,record,spans,body,spans,body,helper,plan['kind'])
        chunk['reference_period_core']=ref(record,*plan['period_core'])
        if 'cohort' in plan: chunk['reference_cohort']=plan['cohort']
        chunks.append(chunk)
    # Retain independently valid old sections, without duplicating the two F units.
    if record['source_kind']=='notice_body':
        try: other=split_notice(record,template,helper)
        except NeedsReview: other=[]
        occupied=[span for plan in plans for span in plan['spans']]
        for chunk in other:
            if not any(max(s['start'],a)<min(s['end'],b) for s in chunk['source_spans'] for a,b in occupied):
                chunks.append(chunk)
    return chunks


def split_matrix(record,template,helper,raw_path,raw_sha):
    html=raw_path.read_text()
    plans=matrix_plans(record['text'],html)
    score_review=None
    try: plans+=score_plans(record['text'],html)
    except MatrixReview as e:
        # Retain H when an unrecognized I table needs separate review.
        if str(e).startswith('I_'): score_review=str(e)
        else: raise
    chunks=[]
    for plan in plans:
        spans=[ref(record,a,b) for a,b in plan['spans']]
        chunk=make_chunk(template,record,spans,plan['body'],spans,plan['body'],helper,plan.get('kind','H_exam_column_with_labelled_criteria'))
        if score_review:chunk['deferred_score_table_reason']=score_review
        chunk['exam_name']=plan['exam']
        if 'cohort' in plan:chunk['score_cohort']=plan['cohort']
        chunk['table_geometry']=dict(source_path=str(raw_path),source_sha256=raw_sha,
            source_span=ref(record,*plan['table_span']),cells=plan['cells'],
            resolved_same_as_above=plan['resolved_same_as_above'])
        chunks.append(chunk)
    try: other=split_notice(record,template,helper)
    except NeedsReview: other=[]
    occupied=[span for plan in plans for span in plan['spans']]+[plan['table_span'] for plan in plans]
    for chunk in other:
        if not any(max(s['start'],a)<min(s['end'],b) for s in chunk['source_spans'] for a,b in occupied):
            chunks.append(chunk)
    return chunks


def procedure_candidate(record):
    title = record['text'].strip().splitlines()[0] if record['text'].strip() else ''
    return bool(re.search(r'신청.*(?:방법|매뉴얼)|(?:방법|매뉴얼).*신청', title)
                and re.search(r'(?m)^1[.)][ \t]+', record['text'])
                and '클릭' in record['text'])


def split_procedure(record, template, helper, document_record_count):
    """J: retain a complete short application workflow, including all cautions."""
    text = record['text']
    # A page or paragraph of a longer manual is not a complete workflow.
    if document_record_count != 1:
        raise NeedsReview('J_multi_record_manual_requires_review')
    steps = list(re.finditer(r'(?m)^(\d{1,2})[.)][ \t]+', text))
    if len(steps) < 2 or [int(m[1]) for m in steps] != list(range(1, len(steps)+1)):
        raise NeedsReview('J_nonsequential_steps_require_review')
    first = text[steps[0].end():steps[1].start()]
    last = text[steps[-1].end():]
    if '로그인' not in first or not re.search(r'신청.*완료|완료.*신청', last):
        raise NeedsReview('J_workflow_boundaries_require_review')
    if '<표>' in text:
        raise NeedsReview('J_unresolved_table_requires_review')
    spans = [ref(record)]
    chunk = make_chunk(template, record, spans, text, spans, text, helper,
                       'J_complete_short_application_procedure')
    chunk['procedure_step_count'] = len(steps)
    return [chunk]


def split_workflow(records, templates, helper):
    chunks = []
    for plan in workflow_plans(records):
        spans = [ref(r) for r in plan['records']]
        context_spans = [ref(r) for r in plan['common']] + spans
        primary = plan['records'][0]
        chunk = make_chunk(templates[primary['record_id']], primary, spans, plan['body'],
                           context_spans, plan['body'], helper, 'K_ordered_workflow_stage')
        chunk.update({k: v for k, v in plan.items() if k.startswith('procedure_')})
        # Covers and index pages are provenance/context, not standalone search hits.
        chunk['procedure_document_context'] = [dict(record_id=r['record_id'], text=r['text'])
                                                for r in plan['common']]
        chunks.append(chunk)
    return chunks


def split_reference_list(record, template, helper, document_record_count):
    if document_record_count != 1:
        raise WorkflowReview('L_multi_record_manual_requires_review')
    chunks=[]
    for plan in reference_list_plans(record):
        spans=[ref(record,a,b) for a,b in plan['spans']]
        context=[ref(record,a,b) for a,b in plan['common']]+spans
        chunk=make_chunk(template,record,spans,plan['body'],context,plan['body'],helper,
                         'L_workflow_with_referenced_list')
        chunk.update({k:v for k,v in plan.items() if k.startswith('procedure_')})
        chunks.append(chunk)
    return chunks


def form_candidate(records):
    return (all(r.get('file_type')=='docx' and 'paragraph' in r.get('locator',{}) for r in records)
            and any(r['text'].strip().endswith('신청서') for r in records))


def split_form(records, templates, helper, document_record_count):
    """M: complete short blank paragraph form, including choices and signature."""
    if len(records)!=document_record_count:
        raise NeedsReview('M_required_form_records_not_selected')
    records=sorted(records,key=lambda r:r['locator']['paragraph'])
    positions=[r['locator']['paragraph'] for r in records]
    if len(set(positions))!=len(positions) or not records[0]['text'].strip().endswith('신청서'):
        raise NeedsReview('M_form_order_requires_review')
    text='\n'.join(r['text'] for r in records)
    if '<표>' in text or '<그림>' in text or ' | ' in text:
        raise NeedsReview('M_form_layout_requires_review')
    # A completed example or a missing signature is not the approved blank form.
    names=[r['text'] for r in records if re.search(r'성\s*명\s*:',r['text'])]
    if len(names)!=1 or not re.fullmatch(r'[•□*\s]*성\s*명\s*:[_\s]*',names[0]):
        raise NeedsReview('M_blank_identity_field_requires_review')
    if not re.search(r'본인.*(?:\(인\)|서명)',records[-1]['text']):
        raise NeedsReview('M_signature_missing')
    body='문서 용도: 작성용 신청서 (빈 서식)\n'+text
    spans=[ref(r) for r in records]
    chunk=make_chunk(templates[records[0]['record_id']],records[0],spans,body,spans,body,
                     helper,'M_complete_blank_application_form')
    chunk['document_role']='blank_application_form'
    return [chunk]


def split_table_form(records, templates, helper, raw_path, raw_sha):
    chunks=[]
    for plan in topic_plans(raw_path,records):
        primary=plan['records'][0]
        spans=[ref(r) for r in plan['records']]
        context=[ref(r) for r in plan['common']]+spans
        chunk=make_chunk(templates[primary['record_id']],primary,spans,plan['body'],context,plan['body'],helper,
                         'N_topic_group_in_application_form')
        chunk['document_role']='blank_application_form'
        chunk['form_topic']=plan['title']
        chunk['table_geometry']=dict(source_path=str(raw_path),source_sha256=raw_sha,table=1,rows=plan['geometry'])
        chunks.append(chunk)
    return chunks


def document_progress(records,selected_ids,waiting,covered_ids):
    by_doc=defaultdict(set)
    for rid in selected_ids:by_doc[records[rid]['document_id']].add(rid)
    rows=[]
    for doc,rids in sorted(by_doc.items()):
        pending=rids & waiting
        status='complete' if not pending else 'partial' if rids & covered_ids else 'not_chunked'
        rows.append(dict(document_id=doc,title=records[next(iter(sorted(rids)))].get('document_title'),
            status=status,selected_records=len(rids),pending_records=len(pending)))
    counts=Counter(row['status'] for row in rows)
    total=len(rows)
    summary=dict(basis='Unique documents currently represented in selected-chunks; complete requires all selected source text accounted for without pending review.',
        selected_documents=total,complete=counts['complete'],partial=counts['partial'],not_chunked=counts['not_chunked'],
        complete_percent=round(100*counts['complete']/total,2) if total else 0,
        touched_percent=round(100*(counts['complete']+counts['partial'])/total,2) if total else 0,
        all_normalized_documents=len({r['document_id'] for r in records.values()}))
    return summary,rows


def build(source, normalized, output, tokenizer, data_dir=Path('data')):
    source,normalized,output=map(Path,(source,normalized,output))
    if output.exists(): raise FileExistsError('Use a fresh staging directory')
    hashes={str(p):digest(p) for p in (source,normalized)}
    old=[c for c in read_jsonl(source) if allowed_chunk(c)]; all_records={r['record_id']:r for r in read_jsonl(normalized)}
    document_record_counts=Counter(r['document_id'] for r in all_records.values())
    templates={}
    for c in old:
        for s in c['source_spans']: templates.setdefault(s['record_id'],c)
    selected_documents=defaultdict(list)
    for rid in templates: selected_documents[all_records[rid]['document_id']].append(all_records[rid])
    workflow_docs={doc for doc,rs in selected_documents.items()
                   if all('page' in r.get('locator',{}) for r in rs) and workflow_candidate(rs)}
    table_form_docs={doc for doc,rs in selected_documents.items() if table_form_candidate(rs)}
    form_docs={doc for doc,rs in selected_documents.items() if form_candidate(rs)}
    groups=defaultdict(list)
    hwpx_records=defaultdict(list)
    for r in all_records.values():
        if r.get('file_type')=='hwpx': hwpx_records[r['document_id']].append(r)
    for rid in templates:
        r=all_records[rid]; loc=r.get('locator',{})
        if r['document_id'] in workflow_docs:
            key=('workflow',r['document_id'])
        elif r['document_id'] in table_form_docs:
            key=('table_form',r['document_id'])
        elif r['document_id'] in form_docs:
            key=('form',r['document_id'])
        elif 'sheet' in loc:
            key=('table',r['document_id'],loc.get('zip_member'),loc.get('sheet'))
        elif r.get('file_type')=='hwpx' and 'table' in loc:
            key=('hwpx_table',r['document_id'],loc['section'],loc['table'])
        else: key=('record',rid)
        groups[key].append(r)
    helper=Chunker(tokenizer); new=[]; pending=[]; done=set()
    raw_paths={}; layouts={}; partial_allowed=set()
    if any(key[0] in {'hwpx_table','table_form'} for key in groups):
        from sejong_rag.extract_attachments import load_attachment_metadata
        raw_paths={m['document_id']:Path(m['resolved_raw_path']) for m in load_attachment_metadata(Path(data_dir))}
    for key,records in groups.items():
        r=records[0]
        try:
            if key[0]=='workflow':
                if len(records)!=document_record_counts[r['document_id']]:
                    raise WorkflowReview('K_required_pages_not_selected')
                chunks=split_workflow(records,templates,helper)
            elif key[0]=='table_form':
                if len(records)!=document_record_counts[r['document_id']]:
                    raise FormLayoutReview('N_required_records_not_selected')
                raw_path=raw_paths.get(r['document_id'])
                if raw_path is None or not raw_path.is_file():
                    raise FormLayoutReview('N_original_file_missing')
                hashes[str(raw_path)]=digest(raw_path)
                chunks=split_table_form(records,templates,helper,raw_path,hashes[str(raw_path)])
            elif key[0]=='form':
                chunks=split_form(records,templates,helper,document_record_counts[r['document_id']])
            elif key[0]=='table': chunks=split_table(records,templates[r['record_id']],helper)
            elif key[0]=='hwpx_table':
                doc=r['document_id']
                if doc not in raw_paths or not raw_paths[doc].is_file():
                    raise NeedsReview('hwpx_original_file_missing')
                raw_path=raw_paths[doc]
                if doc not in layouts:
                    hashes[str(raw_path)]=digest(raw_path)
                    try: layouts[doc]=read_layout(raw_path,hwpx_records[doc])
                    except LayoutReview as e: layouts[doc]=e
                if isinstance(layouts[doc],LayoutReview): raise layouts[doc]
                table=layouts[doc].get((key[2],key[3]))
                if table is None: raise NeedsReview('hwpx_table_mapping_missing')
                chunks=split_hwpx_table(table,all_records,templates[r['record_id']],helper,
                    set(templates),raw_path,hashes[str(raw_path)])
            elif reference_list_candidate(r):
                chunks=split_reference_list(r,templates[r['record_id']],helper,document_record_counts[r['document_id']])
            elif procedure_candidate(r):
                chunks=split_procedure(r,templates[r['record_id']],helper,document_record_counts[r['document_id']])
            elif REFERENCE.search(r['text']):
                chunks=split_reference(r,templates[r['record_id']],helper)
                partial_allowed.add(r['record_id'])
            elif r['source_kind']=='notice_body' and matrix_candidate(r['text']):
                raw_path=Path(data_dir)/'raw'/'notices'/(r['document_id']+'.html')
                if not raw_path.is_file(): raise NeedsReview('H_original_html_missing')
                hashes[str(raw_path)]=digest(raw_path)
                chunks=split_matrix(r,templates[r['record_id']],helper,raw_path,hashes[str(raw_path)])
                partial_allowed.add(r['record_id'])
            elif r['source_kind']=='notice_body': chunks=split_notice(r,templates[r['record_id']],helper)
            else: raise NeedsReview('attachment_layout_requires_review')
        except (NeedsReview,LayoutReview,ReferenceReview,MatrixReview,WorkflowReview,FormLayoutReview) as e:
            pending.append(dict(reason=str(e),record_ids=[r['record_id'] for r in records],
                document_title=r.get('document_title'),filename=r.get('filename'),
                locator=r.get('locator'),sample=[dict(record_id=x['record_id'],text=x['text'][:1600]) for x in records[:8]]))
            continue
        new.extend(chunks);done.update(r['record_id'] for r in records)
    # Prove every selected original character is preserved in accepted units,
    # or its record is explicitly pending. Shared context counts as preserved text.
    coverage=defaultdict(list)
    for c in new:
        for s in c['source_spans']+c['context_source_spans']: coverage[s['record_id']].append((s['start'],s['end']))
    # F may finish reference units while leaving other source ranges pending.
    done=set(); partial=set()
    for rid,spans in coverage.items():
        gaps=uncovered_spans(all_records[rid]['text'],spans)
        if gaps:
            if rid not in partial_allowed: raise ValueError('Coverage missing: '+rid)
            partial.add(rid);r=all_records[rid]
            rule_prefix=next((prefix for prefix in ('I','H','G','F') if any(c['semantic_rule'].startswith(prefix+'_') and rid in c['record_ids'] for c in new)),'F')
            pending.append(dict(reason=rule_prefix+'_remaining_sections_require_review',record_ids=[rid],
                document_title=r.get('document_title'),filename=r.get('filename'),locator=r.get('locator'),
                source_spans=[ref(r,a,b) for a,b in gaps],
                sample=[dict(record_id=rid,text=r['text'][a:min(b,a+1600)]) for a,b in gaps]))
        else: done.add(rid)
    # Shared E context may finish records otherwise queued as standalone prose.
    for item in pending:
        item['record_ids']=[rid for rid in item['record_ids'] if rid not in done]
        item['sample']=[x for x in item['sample'] if x['record_id'] in item['record_ids']]
    pending=[item for item in pending if item['record_ids']]
    waiting={rid for p in pending for rid in p['record_ids']}
    assert not done & waiting and done | waiting == set(templates)
    assert len({c['chunk_id'] for c in new})==len(new)
    for p,h in hashes.items(): assert digest(p)==h
    output.mkdir(parents=True)
    write_jsonl(output/'chunks.jsonl',new)
    write_jsonl(output/'needs-review.jsonl',pending)
    # These are preserved originals, never presented as newly split chunks.
    write_jsonl(output/'pending-original-chunks.jsonl',[c for c in old if any(s['record_id'] in waiting for s in c['source_spans'])])
    progress,progress_rows=document_progress(all_records,set(templates),waiting,set(coverage))
    write_jsonl(output/'document-progress.jsonl',progress_rows)
    report=dict(document_progress=progress,version=VERSION,status='needs_user_review' if pending else 'complete',
        production_applied=False,embedding_calls=0,input_sha256=hashes,code_sha256=digest(Path(__file__)),
        supporting_code_sha256={name:digest(Path(__file__).with_name(name)) for name in ('semantic_hwpx.py','semantic_reference.py','semantic_matrix.py','semantic_score_table.py','semantic_workflow.py','semantic_docx_form.py')},original_chunks=len(old),
        new_chunks=len(new),completed_records=len(done),pending_records=len(waiting),partially_chunked_records=len(partial),
        pending_groups=len(pending),reasons=dict(Counter(p['reason'] for p in pending)),
        rules=dict(Counter(c['semantic_rule'] for c in new)),source_coverage_verified=True,
        note='Staging only. Explicit context fields require retrieval integration before promotion.')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('data/processed/selected-chunks/selected.jsonl'))
    parser.add_argument('--normalized',type=Path,default=Path('data/processed/normalized-text/records.jsonl'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    args=parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    print(json.dumps(build(args.source,args.normalized,args.output,tokenizer,args.data_dir),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
