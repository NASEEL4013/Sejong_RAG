import json
import pytest
from sejong_rag.semantic_rechunk import split_form,NeedsReview,build

class Tokens:
    def __call__(self,text,**kw):return {'input_ids':list(range(len(text)//2+2))}
class Helper:
    def count(self,text):return len(Tokens()(text)['input_ids'])
def records():
    texts=['대학영어 학점인정 신청서','• 학부(과): / 학번:', '• 성 명 :',
           '• 응시시험: TOEIC ( ) TOEFL/IBT ( ) TEPS ( )',
           '• 응시일자: 년 월 일','• 취득점수: 점/급','대학영어 ( )\n* 첨부서류: 성적표(사진포함)',
           '본인은 위와 같이 신청합니다.','년 월 일','위 본인 (인)']
    return [dict(record_id=f'r{i}',document_id='d',file_type='docx',locator={'paragraph':i*2+1},
                 text=t,document_title='학점 인정 안내',filename='신청서.docx',source_kind='attachment') for i,t in enumerate(texts)]
def split(rs,count=None):return split_form(rs,{r['record_id']:r for r in rs},Helper(),len(rs) if count is None else count)
def test_whole_form_preserves_order_choices_and_required_attachment():
    rs=records();c,=split(list(reversed(rs)))
    assert '\n'.join(r['text'] for r in rs) in c['text']
    assert '작성용 신청서 (빈 서식)' in c['text']
    assert c['context_text']==c['text'] and len(c['source_spans'])==len(rs)
    assert c['document_role']=='blank_application_form'
@pytest.mark.parametrize('change,reason',[
    (lambda rs:rs[2].update(text='• 성명: 홍길동'),'blank_identity'),
    (lambda rs:rs[-1].update(text='감사합니다'),'signature'),
    (lambda rs:rs[3].update(text='<표>'),'layout'),
    (lambda rs:rs[3].update(text='항목 '*2000),'embedding_budget'),
])
def test_incomplete_filled_or_oversized_forms_remain_pending(change,reason):
    rs=records();change(rs)
    with pytest.raises(NeedsReview,match=reason):split(rs)
def test_missing_selected_records_are_not_treated_as_complete():
    with pytest.raises(NeedsReview,match='not_selected'):split(records()[:-1],10)
def test_document_is_grouped_and_all_paragraphs_accounted_for(tmp_path):
    rs=records();src=tmp_path/'selected.jsonl';norm=tmp_path/'records.jsonl';out=tmp_path/'out'
    old=[dict(chunk_id=r['record_id'],source_spans=[dict(record_id=r['record_id'],start=0,end=len(r['text']))]) for r in rs]
    src.write_text(''.join(json.dumps(c)+'\n' for c in old));before=src.read_bytes()
    norm.write_text(''.join(json.dumps(r)+'\n' for r in rs))
    result=build(src,norm,out,Tokens())
    assert result['new_chunks']==1 and result['completed_records']==len(rs) and result['pending_records']==0
    assert src.read_bytes()==before
