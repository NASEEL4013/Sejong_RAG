"""Read verified HWPX cell geometry for E before/after table chunks.

Normalized paragraph IDs remain the source of all text. XML contributes only
cell coordinates/spans and outline order; no guessed fill-down or column offsets.
"""
from __future__ import annotations
from collections import Counter
import re
import zipfile
from xml.etree import ElementTree as ET
from sejong_rag.extract_attachments import clean_text, _safe_zip_members


class LayoutReview(ValueError):
    pass


def tag(node):
    return node.tag.rsplit('}', 1)[-1]


def paragraph_text(node):
    if tag(node) == 't': return node.text or ''
    return ''.join(paragraph_text(c) for c in node if tag(c) not in {'p', 'tbl'})


def compact(text):
    return re.sub(r'\s+', '', text)


def unique(items):
    return list(dict.fromkeys(items))


def read_layout(path, records):
    """Reproduce extractor paragraph numbering; verify every nonempty paragraph."""
    lookup = {(r['locator']['section'], r['locator']['paragraph']): r for r in records}
    tables, outside = {}, []
    with zipfile.ZipFile(path) as archive:
        sections = sorted((m for m in _safe_zip_members(archive)
                           if re.fullmatch(r'Contents/section\d+\.xml', m.filename)),
                          key=lambda m: int(re.search(r'section(\d+)', m.filename).group(1)))
        for section in sections:
            data = archive.read(section)
            if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
                raise LayoutReview('hwpx_entity_declaration')
            root = ET.fromstring(data)
            counts = Counter()

            def walk(node, table=None, cell=None):
                kind = tag(node)
                if kind == 'tbl':
                    counts['table'] += 1
                    if table is not None: table['nested'] = True
                    table = dict(section=section.filename, number=counts['table'], cells=[],
                                 order=(int(re.search(r'section(\d+)', section.filename).group(1)), counts['paragraph'], 1),
                                 nested=False)
                    tables[(section.filename, counts['table'])] = table
                    cell = None
                if kind == 'tc' and table is not None:
                    address = next((c for c in node if tag(c) == 'cellAddr'), None)
                    span = next((c for c in node if tag(c) == 'cellSpan'), None)
                    if address is None or span is None:
                        table['invalid'] = 'hwpx_cell_geometry_missing'
                        cell = dict(record_ids=[])
                    else:
                        try:
                            cell = dict(row=int(address.attrib['rowAddr']), col=int(address.attrib['colAddr']),
                                        rows=int(span.attrib['rowSpan']), cols=int(span.attrib['colSpan']), record_ids=[])
                        except (KeyError, ValueError):
                            table['invalid'] = 'hwpx_cell_geometry_invalid'
                            cell = dict(record_ids=[])
                    table['cells'].append(cell)
                if kind == 'p':
                    counts['paragraph'] += 1
                    text = clean_text(paragraph_text(node))
                    if text:
                        r = lookup.get((section.filename, counts['paragraph']))
                        if r is None or compact(r['text']) != compact(text):
                            raise LayoutReview('hwpx_normalized_paragraph_mismatch')
                        if table is not None:
                            if cell is None: table['invalid'] = 'hwpx_text_outside_cell'
                            else: cell['record_ids'].append(r['record_id'])
                        else:
                            outside.append(dict(record_id=r['record_id'], text=r['text'],
                                order=(int(re.search(r'section(\d+)', section.filename).group(1)), counts['paragraph'], 0)))
                for child in node: walk(child, table, cell)
            walk(root)
    if not tables: raise LayoutReview('hwpx_no_tables')
    ordered = sorted(tables.values(), key=lambda t: t['order'])
    # A numbered local heading ends the document-wide preface. Subsequent local
    # headings apply only until the next heading/table, never across departments.
    initial = [r for r in outside if r['order'] < ordered[0]['order']]
    global_ids = []
    for r in initial:
        if re.match(r'^\s*\d+[.)]\s', r['text']): break
        global_ids.append(r['record_id'])
    previous = None
    for table in ordered:
        local = [r for r in outside if (previous is None or previous < r['order']) and r['order'] < table['order']]
        table['context_ids'] = unique(global_ids + [r['record_id'] for r in local])
        # Without a local heading do not guess that a preceding department applies.
        table['has_local_context'] = bool([r for r in local if r['record_id'] not in global_ids])
        previous = table['order']
    return tables


def row_plans(table, records):
    """Return header-labelled rows and only XML-proven shared cells."""
    if table.get('invalid'): raise LayoutReview(table['invalid'])
    if table['nested']: raise LayoutReview('hwpx_nested_table')
    grid = {}
    for index, cell in enumerate(table['cells']):
        if min(cell['row'], cell['col']) < 0 or min(cell['rows'], cell['cols']) < 1:
            raise LayoutReview('hwpx_cell_geometry_invalid')
        if cell['rows'] * cell['cols'] > 10000:
            raise LayoutReview('hwpx_cell_geometry_too_large')
        cell['text'] = ' '.join(records[rid]['text'] for rid in cell['record_ids'])
        for row in range(cell['row'], cell['row'] + cell['rows']):
            for col in range(cell['col'], cell['col'] + cell['cols']):
                if (row, col) in grid: raise LayoutReview('hwpx_cell_overlap')
                grid[(row, col)] = index
    if not grid: raise LayoutReview('hwpx_empty_table')
    nr, nc = max(r for r,c in grid)+1, max(c for r,c in grid)+1
    if nr*nc > 100000 or len(grid) != nr*nc: raise LayoutReview('hwpx_grid_incomplete')
    before = [c for c in table['cells'] if compact(c['text']) == '변경전']
    after = [c for c in table['cells'] if compact(c['text']) == '변경후']
    if len(before)!=1 or len(after)!=1 or before[0]['row']!=after[0]['row']:
        raise LayoutReview('hwpx_not_single_before_after_header')
    header_end = max(c['row']+c['rows'] for c in before+after)
    if header_end >= nr: raise LayoutReview('hwpx_table_has_no_data')
    header_indices = unique(grid[(r,c)] for r in range(header_end) for c in range(nc))
    if any(table['cells'][i]['row']+table['cells'][i]['rows']>header_end for i in header_indices):
        raise LayoutReview('hwpx_header_crosses_data')
    labels = []
    for col in range(nc):
        parts = [table['cells'][i]['text'] for i in unique(grid[(r,col)] for r in range(header_end))]
        label = ' / '.join(p for p in parts if p.strip())
        if not label: raise LayoutReview('hwpx_unlabelled_column')
        labels.append(label)
    if len(set(labels)) != len(labels): raise LayoutReview('hwpx_duplicate_labels')
    header_ids = unique(rid for i in header_indices for rid in table['cells'][i]['record_ids'])
    plans = []
    for row in range(header_end, nr):
        indices = unique(grid[(row,c)] for c in range(nc))
        if any(table['cells'][i]['cols'] != 1 for i in indices):
            raise LayoutReview('hwpx_data_spans_multiple_columns')
        values = [(labels[col], table['cells'][grid[(row,col)]]['text']) for col in range(nc)]
        row_ids = unique(rid for i in indices for rid in table['cells'][i]['record_ids'])
        if not row_ids: raise LayoutReview('hwpx_empty_data_row')
        # Fully merged physical rows are one logical row, not duplicate chunks.
        if plans and plans[-1]['values']==values and plans[-1]['row_record_ids']==row_ids:
            plans[-1]['physical_rows'].append(row+1)
            continue
        plans.append(dict(row=row+1, physical_rows=[row+1], row_record_ids=row_ids, values=values,
            record_ids=unique(table['context_ids'] + header_ids + row_ids),
            geometry=[{k:table['cells'][i][k] for k in ('row','col','rows','cols','record_ids')} for i in indices]))
    return plans
