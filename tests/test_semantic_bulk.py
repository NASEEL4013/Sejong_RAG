import json
from pathlib import Path
import pytest
from sejong_rag.semantic_bulk import (prose,simple_table,prose_sections,build,build_review)
from sejong_rag.semantic_rechunk import NeedsReview

class Tokens:
    def __call__(self,text,**kw):return {'input_ids':list(range(len(text)//2+2))}
class Helper:
    def count(self,text):return len(Tokens()(text)['input_ids'])
def record(text,rid='r',**kw):
    return dict(record_id=rid,document_id='d',document_title='2026-2 신청 안내',text=text,source_kind='notice_body',locator={},**kw)
def test_whole_short_topic_preserves_conditions_and_inline_labels():
    r=record('- 접수 기간: 내일\n- 환불 기간: 모레\n부분 환불 불가\n- 환불 방법: 메일')
    cs,ps=prose([r],r,Helper(),is_notice=True)
    assert len(cs)==1 and not ps and r['text'] in cs[0]['text']
def test_oversized_section_does_not_discard_independent_section():
    text='2026-2 공통 안내\n1. 신청대상\n조건 '*1+'내용 '*1600+'\n2. 문의\n담당 부서 전화 1234'
    r=record(text);cs,ps=prose([r],r,Helper(),is_notice=True)
    assert len(cs)==1 and '담당 부서' in cs[0]['text'] and '2026-2 공통 안내' in cs[0]['context_text']
    assert ps and 'budget' in ps[0]['reason']
def test_numbered_eligibility_conditions_not_split_as_topics():
    text='1. 대한민국 국적을 가진 학생\n2. 성적 3.0 이상인 학생\n3. 등록한 학생'
    sections,_=prose_sections(text)
    assert sections==[(0,len(text))]
def test_unresolved_reference_and_missing_table_are_not_guessed():
    for text,reason in [('신청 방법\n<표>','extraction'),('1. 신청대상\n'+'조건 '*900+'\n2. 신청방법\n상기 기준','reference')]:
        r=record(text)
        with pytest.raises(NeedsReview,match=reason):prose([r],r,Helper(),is_notice=True)
def rows(texts):
    rs=[record(t,str(i)) for i,t in enumerate(texts)]
    for i,r in enumerate(rs):r['locator']={'sheet':'Class Schedule','row':i+1}
    return rs
def test_english_table_header_and_row_budget_are_handled_independently():
    rs=rows(['1열: No. | 2열: Course Title | 3열: Credit',
             '1열: 1 | 2열: Mathematics | 3열: 3',
             '1열: 2 | 2열: '+'Very long description '*1000+' | 3열: 3'])
    cs,ps=simple_table(rs,rs[0],Helper())
    assert len(cs)==1 and 'Course Title: Mathematics' in cs[0]['text'] and 'Credit: 3' in cs[0]['text']
    assert len(ps)==1 and ps[0]['record_ids']==['2']
def test_sparse_side_by_side_table_is_not_partially_misinterpreted():
    rs=rows(['1열: No. | 2열: Course Title | 3열: Credit','1열: 1 | 2열: Math | 3열: 3','6열: Another table'])
    with pytest.raises(NeedsReview,match='sparse'):simple_table(rs,rs[0],Helper())
def test_partial_html_table_record_keeps_table_out_of_prose():
    text='1. 신청대상\n학생\n2. 지원내용\n구분금액A100\n3. 문의\n전화'+(' 확인'*700)
    html='<table><tr><td>구분</td><td>금액</td></tr><tr><td>A</td><td>100</td></tr></table>'
    r=record(text);cs,ps=prose([r],r,Helper(),is_notice=True,html_text=html)
    assert all('구분금액A100' not in c['text'] for c in cs)
    assert any('header_requires_review' in p['reason'] for p in ps)
def test_bulk_audit_and_review_are_complete_and_escape_untrusted_text(tmp_path):
    r=record('짧은 안내입니다.');bad=record('<표> <script>alert(1)</script>','bad');bad['document_id']='other'
    norm=tmp_path/'norm.jsonl';src=tmp_path/'selected.jsonl';base=tmp_path/'base';base.mkdir()
    norm.write_text('\n'.join(json.dumps(x) for x in [r,bad])+'\n')
    original=[dict(chunk_id=x['record_id'],source_spans=[dict(record_id=x['record_id'],start=0,end=len(x['text']))]) for x in [r,bad]]
    src.write_text('\n'.join(json.dumps(x) for x in original)+'\n');before=src.read_bytes()
    (base/'chunks.jsonl').write_text('');(base/'needs-review.jsonl').write_text('\n'.join(json.dumps(dict(record_ids=[x['record_id']],reason='no_recognized_heading')) for x in [r,bad])+'\n')
    data=tmp_path/'data';(data/'raw/notices').mkdir(parents=True)
    for x in [r,bad]:(data/'raw/notices'/f"{x['document_id']}.html").write_text('<p>source</p>')
    report=build(base,tmp_path/'out',Tokens(),norm,src,data)
    assert report['all_selected_documents_audited'] and report['document_progress']['complete']==1
    assert report['document_progress']['not_chunked']==1 and src.read_bytes()==before
    page=(tmp_path/'out/review.html').read_text()
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in page
    assert '<script>alert(1)</script>' not in page


def test_html_simple_rows_keep_labels_and_shared_notes():
    text='1. 지원내용\n대상: 신입생\n구분학점과목A3수학B2영어\n※ 변경 불가\n2. 문의\n전화'
    html='<table><tr><td>구분</td><td>학점</td><td>과목</td></tr><tr><td>A</td><td>3</td><td>수학</td></tr><tr><td>B</td><td>2</td><td>영어</td></tr></table>'
    r=record(text);cs,ps=prose([r],r,Helper(),is_notice=True,html_text=html)
    table=[c for c in cs if c['semantic_rule']=='D_row_with_table_context_html']
    assert len(table)==2 and not ps
    assert all('대상: 신입생' in c['context_text'] and '변경 불가' in c['context_text'] for c in table)
    assert '과목: 수학' in table[0]['text'] and '과목: 영어' in table[1]['text']


def test_multiline_korean_headings_preserve_inner_numbered_conditions():
    text='안내\n가\n.\n지원 대상\n1. 대한민국 국적\n2. 성적 충족\n나\n.\n제출 서류\n신청서'
    sections,prefix=prose_sections(text)
    assert len(sections)==2 and prefix
    assert '대한민국 국적' in text[slice(*sections[0])] and '성적 충족' in text[slice(*sections[0])]


def test_j_picture_markers_and_spacing_variants_keep_complete_textual_steps():
    text='복학신청 방법\n1. 학사정보서비스 접속\n<그림>\n2.종류 선택\n<그림>\n3. 저장 클릭. 신청절차 완료. 최종 승인은 개강일\n<그림>'
    r=record(text);cs,ps=prose([r],r,Helper())
    assert len(cs)==1 and cs[0]['semantic_rule']=='J_complete_short_application_procedure'
    assert text in cs[0]['text'] and cs[0]['procedure_step_count']==3
    with pytest.raises(NeedsReview,match='extraction'):prose([record(text.replace('2.','4.'))],r,Helper())


def test_form_like_filled_example_not_tagged_as_blank():
    text='신청서\n성명: 홍길동\n본인 (서명)'
    with pytest.raises(NeedsReview,match='filled_example'):prose([record(text)],record(text),Helper())


def test_new_table_context_reaches_generation_prompt_without_raw_suffix_assumption():
    import numpy as np
    from sejong_rag.selected_search import SelectedSearch
    from sejong_rag.selected_generation import build_request
    rs=rows(['1열: No. | 2열: Course Title | 3열: Credit',
             '1열: 1 | 2열: Mathematics | 3열: 3','1열: 2 | 2열: English | 3열: 2'])
    cs,_=simple_table(rs,rs[0],Helper())
    e=SelectedSearch(cs,np.eye(2),{r['record_id']:r for r in rs},Tokens())
    out=e.search('Mathematics',np.array([1.,0.]),top_k=1)
    request=build_request('Mathematics',out)
    assert 'Course Title: Mathematics' in request['messages'][1]['content']
    assert request['sources'][0]['source_spans']==cs[0]['context_source_spans']
