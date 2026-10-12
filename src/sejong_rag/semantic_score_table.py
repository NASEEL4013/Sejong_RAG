"""I: both score bands and incremental-payment rules per major category."""
from __future__ import annotations
import re
from bs4 import BeautifulSoup
from sejong_rag.semantic_matrix import MatrixReview,compact,source_grid


def score_plans(text,html):
    soup=BeautifulSoup(html,'html.parser');tables=[]
    for table in soup.find_all('table'):
        if table.find('table'):continue
        first=table.find('tr')
        labels=[compact(c.get_text()) for c in first.find_all(['td','th'],recursive=False)] if first else []
        if len(labels)==3 and labels[1:]==['점수','장학금액'] and '구분' in labels[0]:tables.append(table)
    if len(tables)!=1:raise MatrixReview('I_score_table_missing_or_multiple')
    cells,grid,table_span,positions,source,start,nr,nc=source_grid(text,tables[0])
    if nr!=5 or nc!=3:raise MatrixReview('I_unapproved_score_band_structure')
    captions=list(re.finditer(r'\[([^\]]+)\]',source[:start]))
    if not captions or captions[-1].group(1)!='TOEIC':raise MatrixReview('I_exam_caption_unclear')
    caption=captions[-1];caption_span=(positions[caption.start()],positions[caption.end()-1]+1)
    # Keep all TOEIC-specific notes (test cutoff and incremental payment), bounded
    # by the next exam-category caption, without borrowing the adjacent matrix.
    end=start+len(''.join(compact(c.get_text()) for c in tables[0].find_all(['td','th'])))
    next_caption=re.search(r'\[[^\]]+\]',source[end:])
    if next_caption is None:raise MatrixReview('I_notes_boundary_unclear')
    note_end=end+next_caption.start()
    if note_end==end:raise MatrixReview('I_increment_rule_missing')
    note_span=(positions[end],positions[note_end-1]+1)
    notes=text[slice(*note_span)]
    if not all(word in compact(notes) for word in ('하위','상위','차액지급')):
        raise MatrixReview('I_increment_rule_missing')
    limits=list(re.finditer(r'장학금은언어별각\d+회에한하여수혜가능\(총\d+회\)',source[:caption.start()]))
    if len(limits)!=1:raise MatrixReview('I_shared_limit_missing_or_multiple')
    limit=limits[0];limit_span=(positions[limit.start()],positions[limit.end()-1]+1)
    groups={}
    for row in range(1,nr):groups.setdefault(grid[row,0],[]).append(row)
    if len(groups)!=2 or any(len(rows)!=2 for rows in groups.values()):
        raise MatrixReview('I_cohort_spans_not_two_bands')
    plans=[];seen=set()
    for cohort_index,rows in groups.items():
        cohort=cells[cohort_index];name=compact(cohort['text'])
        category='nonmajor' if name.startswith('비전공자') else 'major' if name.startswith('전공자') else None
        if category is None or category in seen:raise MatrixReview('I_cohort_label_unclear')
        seen.add(category)
        used=[grid[0,c] for c in range(nc)]+[cohort_index]
        lines=['시험 종류: TOEIC','대상: '+cohort['text']]
        for row in rows:
            score=grid[row,1];amount=grid[row,2]
            if not cells[score]['text'] or not cells[amount]['text']:
                raise MatrixReview('I_score_or_amount_missing')
            if cells[score]['rows']!=1 or cells[amount]['rows']!=1:
                raise MatrixReview('I_band_value_merge_unclear')
            used.extend([score,amount])
            lines.append('점수: '+cells[score]['text']+' / 장학금액: '+cells[amount]['text'])
        lines.append('공통 수혜 제한: '+re.sub(r'\s+',' ',text[slice(*limit_span)]).strip())
        lines.append('TOEIC 접수·차액 지급 안내:\n'+notes)
        used=list(dict.fromkeys(used))
        spans=list(dict.fromkeys([caption_span]+[cells[i]['span'] for i in used if cells[i]['span']]+[limit_span,note_span]))
        plans.append(dict(exam='TOEIC',cohort=category,body='\n'.join(lines),spans=spans,
            table_span=table_span,resolved_same_as_above=[],
            cells=[{k:cells[i][k] for k in ('row','col','rows','cols','span')} for i in used],
            kind='I_score_bands_with_increment_rule'))
    return plans
