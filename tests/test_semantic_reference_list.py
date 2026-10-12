import numpy as np
import pytest
from sejong_rag.semantic_rechunk import split_reference_list
from sejong_rag.semantic_workflow import WorkflowReview, reference_list_plans
from sejong_rag.selected_search import SelectedSearch
from sejong_rag.selected_generation import build_request

TEXT=('교직이수 신청 방법\n1. 학사정보시스템 로그인 > 교직신청\n'
      '2. 교직이수 신청\n표시과목(맨 아래 참고) 확인\n'
      '※ 전공이 2개 이상이면 전공 배정 신청 먼저\n저장(신청) 클릭\n'
      '3. 신청서 출력 및 제출\n출력 후 학과장 확인 및 날인\n학생 본인 직접 제출: 학생회관 205호\n'
      '※ 학과별 표시과목 현황\n국어국문학과: 국어, 영어영문학전공: 영어')
class Tokens:
    def __call__(self,text,**kw):return {'input_ids':list(range(len(text)//2+2))}
class Helper:
    def count(self,text):return len(Tokens()(text)['input_ids'])
def record(text=TEXT):
    return dict(record_id='r',document_id='d',text=text,document_title='2025학년도 교직이수 신청',
                filename='신청 방법.hwp',source_url='https://example.org/notice',source_kind='attachment')
def make(text=TEXT):
    r=record(text)
    return split_reference_list(r,r,Helper(),1)

def test_reference_list_belongs_to_online_step_and_other_conditions_remain():
    a,b=make()
    assert '영어영문학전공: 영어' in a['text'] and '전공 배정 신청 먼저' in a['text']
    assert '영어영문학전공: 영어' not in b['text']
    assert '학생회관 205호' in b['text'] and '학생 본인 직접' in b['text']
    assert all('전체 순서:' in c['text'] for c in [a,b])
    from sejong_rag.semantic_reference import uncovered_spans
    assert uncovered_spans(TEXT,[(s['start'],s['end']) for c in [a,b] for s in c['context_source_spans']])==[]
    changed=make(TEXT.replace('205호','310호').replace('영어영문학전공','영어데이터융합전공'))
    assert '310호' in changed[1]['text'] and '영어데이터융합전공' in changed[0]['text']
    assert changed[0]['procedure_id']!=a['procedure_id']

@pytest.mark.parametrize('text',[TEXT.replace('3.','4.'),TEXT.replace('맨 아래 참고','확인'),TEXT.replace('날인','검토')])
def test_unapproved_reference_or_structure_is_reviewed(text):
    with pytest.raises(WorkflowReview):reference_list_plans(record(text))

def test_whole_and_specific_questions_reach_prompt_with_correct_scope():
    cs=make();e=SelectedSearch(cs,np.array([[0.,0.],[1.,0.]]),{'r':record()},Tokens())
    out=e.search('2025학년도 교직이수 신청 방법',np.array([1.,0.]),candidate_k=1)
    assert [b['procedure_step'] for b in out]==[1,2]
    assert '영어영문학전공' in build_request('신청 방법',out)['messages'][1]['content']
    out=e.search('교직이수 신청서 출력하고 어디에 제출해?',np.array([1.,0.]))
    assert len(out)==1 and out[0]['procedure_step']==2
    assert '205호' in out[0]['text']
