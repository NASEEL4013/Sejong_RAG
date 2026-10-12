import json
import zipfile
from pathlib import Path
import pytest
from sejong_rag.extract_attachments import extract_hwpx
from sejong_rag.semantic_hwpx import LayoutReview, read_layout, row_plans
from sejong_rag.semantic_rechunk import build, split_hwpx_table


def cell(row, col, text, rows=1, cols=1):
    return (f'<tc><cellAddr rowAddr="{row}" colAddr="{col}"/>'
            f'<cellSpan rowSpan="{rows}" colSpan="{cols}"/>'
            f'<p><run><t>{text}</t></run></p></tc>')


def fixture(tmp_path):
    path=tmp_path/'example.hwpx'
    xml='<root><p><run><t>교과목 변경 안내</t></run></p><p><run><t>2017학년도 입학자부터</t></run></p><p><run><t>1. 국어국문학과</t></run></p><tbl>'
    xml+='<tr>'+cell(0,0,'학과',rows=2)+cell(0,1,'기본이수영역',rows=2)+cell(0,2,'지정 교과목',cols=2)+'</tr>'
    xml+='<tr>'+cell(1,2,'변경전')+cell(1,3,'변경후')+'</tr>'
    xml+='<tr>'+cell(2,0,'국어국문학과',rows=2)+cell(2,1,'국어교육론')+cell(2,2,'한국어교육론')+cell(2,3,'변경없음')+'</tr>'
    xml+='<tr>'+cell(3,1,'국어학개론')+cell(3,2,'한국어의이해')+cell(3,3,'신규과목')+'</tr></tbl></root>'
    with zipfile.ZipFile(path,'w') as z:z.writestr('Contents/section0.xml',xml)
    extracted,_=extract_hwpx(path)
    rs=[]
    for i,item in enumerate(extracted):
        rs.append(dict(item,record_id=f'r{i}',document_id='doc',parent_document_id='notice',
            document_title='교직과정 교과목 변경',filename='example.hwpx',file_type='hwpx',source_kind='attachment'))
    return path,rs


class Helper:
    def count(self,text):return len(text)//4+1


def test_original_coordinates_restore_merged_context_and_before_after(tmp_path):
    path,rs=fixture(tmp_path);records={r['record_id']:r for r in rs}
    table=read_layout(path,rs)[('Contents/section0.xml',1)]
    plans=row_plans(table,records)
    assert len(plans)==2
    assert dict(plans[1]['values'])=={'학과':'국어국문학과','기본이수영역':'국어학개론',
        '지정 교과목 / 변경전':'한국어의이해','지정 교과목 / 변경후':'신규과목'}
    # Old extraction numbers this physical first cell as column 1; XML says col 2.
    assert next(r for r in rs if r['text']=='국어학개론')['locator']['column']==1
    chunks=split_hwpx_table(table,records,{},Helper(),set(records),path,'hash')
    assert len(chunks)==2
    assert all('2017학년도 입학자부터' in c['text'] and '학과: 국어국문학과' in c['text'] for c in chunks)
    assert '변경후: 변경없음' in chunks[0]['text']
    assert '한국어의이해' not in chunks[0]['text']
    assert chunks[1]['table_geometry']['cells'][0]['rows']==2
    used={s['record_id'] for c in chunks for s in c['source_spans']}
    assert used==set(records)


def test_mismatched_source_paragraph_is_rejected(tmp_path):
    path,rs=fixture(tmp_path);rs[-1]['text']='다른 내용'
    with pytest.raises(LayoutReview,match='paragraph_mismatch'):read_layout(path,rs)


def test_merge_does_not_fill_beyond_original_span(tmp_path):
    path,rs=fixture(tmp_path);table=read_layout(path,rs)[('Contents/section0.xml',1)]
    next(c for c in table['cells'] if c['row']==2 and c['col']==0)['rows']=1
    with pytest.raises(LayoutReview,match='grid_incomplete'):row_plans(table,{r['record_id']:r for r in rs})


def test_cell_overlap_is_rejected(tmp_path):
    path,rs=fixture(tmp_path);table=read_layout(path,rs)[('Contents/section0.xml',1)]
    next(c for c in table['cells'] if c['row']==2 and c['col']==1)['col']=0
    with pytest.raises(LayoutReview,match='overlap'):row_plans(table,{r['record_id']:r for r in rs})


def test_build_accounts_for_context_records_and_preserves_originals(tmp_path):
    path,rs=fixture(tmp_path)
    meta=tmp_path/'data/metadata/downloaded-attachments';meta.mkdir(parents=True)
    (meta/'doc.json').write_text(json.dumps(dict(document_id='doc',raw_path=str(path))))
    source=tmp_path/'selected.jsonl';normalized=tmp_path/'records.jsonl';out=tmp_path/'out'
    source.write_text(''.join(json.dumps(dict(chunk_id=r['record_id'],source_spans=[dict(record_id=r['record_id'],start=0,end=len(r['text']))]))+'\n' for r in rs))
    normalized.write_text(''.join(json.dumps(r)+'\n' for r in rs))
    before=source.read_bytes()
    class Tokenizer:
        def __call__(self,text,**kw):return {'input_ids':list(range(len(text)//4+1))}
    report=build(source,normalized,out,Tokenizer(),tmp_path/'data')
    assert report['status']=='complete' and report['new_chunks']==2
    assert report['completed_records']==len(rs) and report['pending_records']==0
    assert str(path) in report['input_sha256']
    assert source.read_bytes()==before


def test_empty_after_cell_is_marked_not_assumed_unchanged(tmp_path):
    path,rs=fixture(tmp_path);records={r['record_id']:r for r in rs}
    table=read_layout(path,rs)[('Contents/section0.xml',1)]
    next(c for c in table['cells'] if c['row']==3 and c['col']==3)['record_ids']=[]
    chunks=split_hwpx_table(table,records,{},Helper(),set(records),path,'hash')
    assert '변경후: [원문 빈칸]' in chunks[1]['text']
    assert '변경없음' not in chunks[1]['text']
