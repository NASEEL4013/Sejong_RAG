import re
import pytest
from sejong_rag.syllabus_schedule import split_schedule, pdf_row_starts
from sejong_rag.semantic_rechunk import NeedsReview
from test_semantic_syllabus import Helper,record,PREFIX


def data(extra=''):
    text=PREFIX+'교과소개\n설명\n교과교재\n책\n교과평가\n규정\n주별수업내용\n주차 유형 강의명\n'
    for week in range(1,16):
        text+=f'{week}주차 1강 수업 {week}\n'
        if week==3:text+='3주차 2강 두 번째 강의\n'
        if week==2:text+=extra
    return [record(text)]

class PDF:
    pages=[None]

@pytest.fixture
def simple_rows(monkeypatch):
    monkeypatch.setattr('sejong_rag.syllabus_schedule.pdf_row_starts',
                        lambda page,r: lambda offset:r['text'].rfind('\n',0,offset)+1)


def test_every_block_has_own_course_term_and_range(simple_rows):
    cs,ps=split_schedule(data(),{},Helper(),PDF())
    assert len(cs)==5 and not ps
    assert [(c['syllabus_week_start'],c['syllabus_week_end']) for c in cs]==[(1,3),(4,6),(7,9),(10,12),(13,15)]
    for c in cs:
        for field in ('text','context_text'):
            assert '강의내용: 코딩' in c[field]
            assert '2025학년도 봄학기' in c[field]
            assert f"전체 15주 중 {c['syllabus_week_start']}~{c['syllabus_week_end']}주차" in c[field]
        assert PREFIX in c['context_text']
    assert '3주차 2강' in cs[0]['text'] and '4주차 1강' not in cs[0]['text']
    assert '3주차 2강' not in cs[1]['text']


def test_missing_week_and_oversize_block_are_never_silently_fixed(simple_rows):
    rs=data();rs[0]['text']=rs[0]['text'].replace('4주차 1강 수업 4\n','')
    with pytest.raises(NeedsReview,match='incomplete'):split_schedule(rs,{},Helper(),PDF())
    cs,ps=split_schedule(data('긴 설명 '*2000+'\n'),{},Helper(),PDF())
    assert len(cs)==4 and len(ps)==1
    assert all(c['syllabus_week_start']!=1 for c in cs)
    assert 'budget' in ps[0]['reason']


def test_pdf_row_top_includes_description_above_centered_week_number():
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject,NameObject,DecodedStreamObject
    w=PdfWriter();p=w.add_blank_page(width=400,height=400)
    font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
    p[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):w._add_object(font)})})
    stream=DecodedStreamObject();stream.set_data(b'''0 350 300 0.5 re f
0 270 300 0.5 re f
0 180 300 0.5 re f
BT /F1 10 Tf 1 0 0 1 120 330 Tm (Previous row) Tj ET
BT /F1 10 Tf 1 0 0 1 120 250 Tm (Next row activity begins here) Tj ET
BT /F1 10 Tf 1 0 0 1 20 220 Tm (Week4) Tj ET
BT /F1 10 Tf 1 0 0 1 120 205 Tm (Next row activity ends here) Tj ET
''')
    p[NameObject('/Contents')]=w._add_object(stream)
    import io
    from pypdf import PdfReader
    buffer=io.BytesIO();w.write(buffer);buffer.seek(0)
    p=PdfReader(buffer).pages[0]
    text='\n'.join(' '.join(line.split()) for line in p.extract_text(extraction_mode='layout').splitlines() if line.strip())
    r=record(text)
    start=pdf_row_starts(p,r)(text.index('Week4'))
    assert text[start:].startswith('Next row activity begins here')
    assert 'Previous row' not in text[start:]
    r['text']+='\nUnmapped source'
    with pytest.raises(NeedsReview,match='text_mapping'):pdf_row_starts(p,r)
