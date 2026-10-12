"""H: exam-column scholarship chunks with verified HTML cell/source mapping."""
from __future__ import annotations
import re
from bs4 import BeautifulSoup


class MatrixReview(ValueError):
    pass


def compact(text):
    return re.sub(r'\s+', '', text)


def candidate(text):
    return all(label in compact(text) for label in ('TOEFL','비전공자','전공자','장학금액'))


def matrix_plans(text, html):
    soup=BeautifulSoup(html,'html.parser')
    candidates=[]
    for table in soup.find_all('table'):
        if table.find('table'): continue
        rows=table.find_all('tr')
        labels=[]
        for row in rows:
            cells=row.find_all(['td','th'],recursive=False)
            if cells: labels.append(compact(cells[0].get_text()))
        if all(label in labels for label in ('비전공자','전공자','장학금액')):
            candidates.append(table)
    if len(candidates)!=1: raise MatrixReview('H_matrix_missing_or_multiple')
    table=candidates[0]
    cells,grid,table_span,positions,source,start,nr,nc=source_grid(text,table)
    label_rows={}
    for row in range(nr):
        label=compact(cells[grid[row,0]]['text'])
        if label in ('비전공자','전공자','장학금액'):
            if label in label_rows: raise MatrixReview('H_duplicate_row_label')
            label_rows[label]=row
    if set(label_rows)!=set(('비전공자','전공자','장학금액')):
        raise MatrixReview('H_required_rows_missing')
    header_end=min(label_rows.values())
    if not header_end or len(label_rows)+header_end!=nr:
        raise MatrixReview('H_unapproved_matrix_rows')
    # Preserve the full language-level award limit, not a fixed numerical value.
    restriction=list(re.finditer(r'장학금은언어별각\d+회에한하여수혜가능\(총\d+회\)',source[:start]))
    if len(restriction)!=1: raise MatrixReview('H_shared_award_limit_missing_or_multiple')
    m=restriction[0];limit_span=(positions[m.start()],positions[m.end()-1]+1)
    limit=text[limit_span[0]:limit_span[1]]
    plans=[];names=set()
    for col in range(1,nc):
        headers=list(dict.fromkeys(grid[row,col] for row in range(header_end)))
        exam=cells[headers[-1]]['text']
        if not exam or exam in names: raise MatrixReview('H_exam_header_not_unique')
        names.add(exam)
        language=' / '.join(cells[i]['text'] for i in headers[:-1])
        used=list(headers);fields=[];resolutions=[]
        for label in ('비전공자','전공자','장학금액'):
            row=label_rows[label];i=grid[row,col];cell=cells[i]
            used.extend([grid[row,0],i]);value=cell['text']
            if not value.strip(): raise MatrixReview('H_empty_required_value')
            if compact(value)=='상동':
                if label!='전공자': raise MatrixReview('H_same_as_above_target_unclear')
                target=grid[label_rows['비전공자'],col]
                if not cells[target]['text'] or compact(cells[target]['text'])=='상동':
                    raise MatrixReview('H_same_as_above_chain_unclear')
                used.append(target)
                value=cells[target]['text']+' (원문: '+cell['text']+')'
                resolutions.append(dict(source_span=cell['span'],target_span=cells[target]['span']))
            fields.append((label+' 기준' if label!='장학금액' else label,value))
        used=list(dict.fromkeys(used))
        spans=list(dict.fromkeys([cells[i]['span'] for i in used if cells[i]['span']]+[limit_span]))
        body='시험 종류: '+exam+'\n'
        if language: body+='언어: '+language+'\n'
        body+='\n'.join(label+': '+value for label,value in fields)
        body+='\n공통 수혜 제한: '+re.sub(r'\s+',' ',limit).strip()
        plans.append(dict(exam=exam,body=body,spans=spans,table_span=table_span,
            resolved_same_as_above=resolutions,
            cells=[{k:cells[i][k] for k in ('row','col','rows','cols','span')} for i in used]))
    return plans


def source_grid(text,table):
    """Map a non-nested HTML table to verified normalized-text spans."""
    positions=[i for i,ch in enumerate(text) if not ch.isspace()]
    source=''.join(text[i] for i in positions)
    raw_cells=table.find_all(['td','th'])
    table_text=''.join(compact(cell.get_text()) for cell in raw_cells)
    start=source.find(table_text)
    if not table_text or start<0 or source.find(table_text,start+1)>=0:
        raise MatrixReview('H_table_source_mapping_not_unique')
    table_span=(positions[start],positions[start+len(table_text)-1]+1)
    grid={};cells=[];cursor=start
    rows=table.find_all('tr')
    for row_index,row in enumerate(rows):
        col=0
        for node in row.find_all(['td','th'],recursive=False):
            while (row_index,col) in grid: col+=1
            try: height=int(node.get('rowspan',1));width=int(node.get('colspan',1))
            except ValueError: raise MatrixReview('H_invalid_cell_span')
            if not 1<=height<=len(rows) or not 1<=width<=100:
                raise MatrixReview('H_invalid_cell_span')
            value=node.get_text(' ',strip=True);key=compact(node.get_text())
            if compact(value)!=key: raise MatrixReview('H_cell_text_mismatch')
            span=None
            if key:
                if source[cursor:cursor+len(key)]!=key: raise MatrixReview('H_cell_source_mismatch')
                span=(positions[cursor],positions[cursor+len(key)-1]+1)
            cursor+=len(key)
            cell=dict(row=row_index,col=col,rows=height,cols=width,text=value,span=span)
            index=len(cells);cells.append(cell)
            for rr in range(row_index,row_index+height):
                for cc in range(col,col+width):
                    if (rr,cc) in grid: raise MatrixReview('H_overlapping_cells')
                    grid[rr,cc]=index
            col+=width
    nr=len(rows);nc=max(c for r,c in grid)+1
    if len(grid)!=nr*nc or any(r>=nr for r,c in grid): raise MatrixReview('H_incomplete_grid')
    return cells,grid,table_span,positions,source,start,nr,nc
