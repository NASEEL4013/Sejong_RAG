import pytest
from sejong_rag.semantic_rechunk import NeedsReview, procedure_candidate, split_procedure

TEXT = ('◎ 휴학신청 방법안내\n1. 시스템 로그인 후 학적 클릭\n<그림>\n'
        '2. 휴학종류 선택\n단, 활성화되지 않으면 새로작성 클릭\n'
        '3. 휴학기간 선택 (1학기 또는 2학기)\n'
        '4. 저장(신청) 클릭\n접수前 표시되면 신청절차 완료.\n최종 승인은 개강일 처리.')

class Helper:
    def count(self, text): return len(text)

def record(text=TEXT):
    return dict(record_id='r', document_id='d', text=text,
                document_title='2025-1학기 휴학신청 안내', filename='휴학신청매뉴얼.hwp')

def test_complete_workflow_keeps_all_conditions_and_original_span():
    r=record()
    assert procedure_candidate(r)
    c,=split_procedure(r,{},Helper(),1)
    assert TEXT in c['text'] and c['text']==c['context_text']
    assert c['source_spans'][0]['start']==0 and c['source_spans'][0]['end']==len(TEXT)
    assert c['procedure_step_count']==4

@pytest.mark.parametrize('text,count,reason',[
    (TEXT,2,'multi_record'),
    (TEXT.replace('3.','5.'),1,'nonsequential'),
    (TEXT.replace('신청절차 완료','저장됨'),1,'boundaries'),
    (TEXT+'\n<표>',1,'unresolved_table'),
    (TEXT+' 주의사항'*1000,1,'embedding_budget'),
])
def test_incomplete_or_oversized_manual_is_not_silently_accepted(text,count,reason):
    with pytest.raises(NeedsReview,match=reason):
        split_procedure(record(text),{},Helper(),count)
