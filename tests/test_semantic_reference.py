import json
from pathlib import Path
import pytest
from sejong_rag.semantic_reference import ReferenceReview,reference_plans,uncovered_spans
from sejong_rag.semantic_rechunk import split_reference,build

TEXT=('안내문\n1. 응시 대상\n신입생\n2. 응시 기간:\n2/14 오후 6시 ~ 2/19 오후 1시\n'
      '※ 마감 직전 시작 가능, 재접속 불가\n3. 응시 방법\n홈페이지\n'
      '4. IT계열 수강은 2학기입니다.\n단, 2학기 학생들도 상기 기간 중 레벨테스트를 실시하여야 합니다.\n'
      '2025학년도 수업시간표\n행1\n5. 문의\n전화')


class Helper:
    def count(self,text):return len(text)//4+1


def record(text=TEXT):
    return dict(record_id='r',document_id='doc',source_kind='notice_body',
                document_title='2025학년도 신입생 레벨테스트',text=text,locator={})


def test_f_12_and_134_without_copying_deadline_caution():
    plans=reference_plans(TEXT)
    assert len(plans)==2
    units=['\n'.join(TEXT[a:b] for a,b in p['spans']) for p in plans]
    assert '재접속 불가' in units[0]
    assert '2/14 오후 6시' in units[1] and '2학기' in units[1]
    assert '재접속' not in units[1] and '수업시간표' not in units[1]
    out=split_reference(record(),{},Helper())
    f=[c for c in out if c['semantic_rule'].startswith('F_')]
    assert len(f)==2 and all(c['text'].startswith('문서 제목: 2025') for c in f)
    assert all(c['text']==c['context_text'] for c in f)


def test_period_is_read_from_each_document_not_fixed_answer():
    other=TEXT.replace('2/14','2/21').replace('2/19','2/23')
    plans=reference_plans(other)
    body='\n'.join(other[a:b] for a,b in plans[1]['spans'])
    assert '2/21' in body and '2/23' in body and '2/14' not in body


def test_multiple_cohort_schedules_require_review():
    text=TEXT.replace('2/14 오후 6시 ~ 2/19 오후 1시',
                      '가. 신입생: 2/7~2/13\n나. 추가합격자: 2/14~2/19')
    with pytest.raises(ReferenceReview,match='multiple_cohort'):
        reference_plans(text)


def test_multiple_period_headers_are_not_guessed():
    text=TEXT.replace('3. 응시 방법','3. 시험 기간:')
    with pytest.raises(ReferenceReview,match='missing_or_multiple'):
        reference_plans(text)


def test_korean_bullet_ends_before_next_topic():
    text='1. 응시 일정:\n2/20~2/22\n2. 유의사항\n마. 2학기 학생들도 상기 기간 중 레벨테스트를 실시해야 함\n바. 성적 반영 10%'
    plans=reference_plans(text)
    body='\n'.join(text[a:b] for a,b in plans[1]['spans'])
    assert '2/20~2/22' in body and '반영' not in body


def test_partial_output_keeps_unprocessed_ranges_and_old_chunks(tmp_path):
    # Oversized unrelated section prevents the older atomic notice splitter.
    text=TEXT.replace('홈페이지','홈페이지 '+'안내 '*2000)
    r=record(text);src=tmp_path/'old.jsonl';norm=tmp_path/'records.jsonl';out=tmp_path/'out'
    src.write_text(json.dumps({'chunk_id':'old','source_spans':[{'record_id':'r','start':0,'end':len(text)}]})+'\n')
    norm.write_text(json.dumps(r)+'\n')
    class Tokenizer:
        def __call__(self,text,**kw):return {'input_ids':list(range(len(text)//4+1))}
    before=src.read_bytes()
    report=build(src,norm,out,Tokenizer())
    assert report['new_chunks']==2 and report['partially_chunked_records']==1
    assert report['completed_records']==0 and report['pending_records']==1
    pending=json.loads((out/'needs-review.jsonl').read_text())
    assert pending['reason']=='F_remaining_sections_require_review'
    spans=[(s['start'],s['end']) for s in pending['source_spans']]
    for line in (out/'chunks.jsonl').open():
        c=json.loads(line);spans.extend((s['start'],s['end']) for s in c['source_spans'])
    assert uncovered_spans(text,spans)==[]
    assert json.loads((out/'pending-original-chunks.jsonl').read_text())['chunk_id']=='old'
    assert src.read_bytes()==before


G_TEXT=('1. 응시 일정:\n'
        '가. 신입생(2/5까지 등록) 레벨테스트 기간: 2/7~2/13\n'
        '나. 추가합격자(2/6~2/12 합격) 레벨테스트 기간: 2/14~2/19\n'
        '2. 유의사항\n'
        '마. 2학기 학생들도 상기 기간 중 레벨테스트를 실시해야 함\n'
        '바. 성적 반영\n'
        '아. 2/13 이후 등록한 추가합격자는 응시가 불가능합니다. 해당 1학기 대상자는 재수강반에 개별수강하시기 바랍니다.\n'
        '3. 문의\n전화')


def test_g_cohort_conditions_dates_and_exclusion_stay_scoped():
    from sejong_rag.semantic_reference import cohort_reference_plans
    plans=cohort_reference_plans(G_TEXT)
    assert len(plans)==2
    general,extra=['\n'.join(G_TEXT[a:b] for a,b in p['spans']) for p in plans]
    assert '2/5까지 등록' in general and '2/7~2/13' in general
    assert '2/14~2/19' not in general and '불가능' not in general
    assert '2/6~2/12 합격' in extra and '2/14~2/19' in extra
    assert '2/13 이후 등록' in extra and '재수강반' in extra
    assert '2/7~2/13' not in extra
    assert all('상기 기간' in t for t in [general,extra])
    out=split_reference(record(G_TEXT),{},Helper())
    assert len([c for c in out if c['semantic_rule'].startswith('G_')])==2


def test_g_does_not_borrow_dates_or_eligibility_words_from_other_document():
    from sejong_rag.semantic_reference import cohort_reference_plans
    text=G_TEXT.replace('2/6~2/12 합격','2/12까지 등록').replace('2/14~2/19','3/1~3/5')
    extra=cohort_reference_plans(text)[1]
    body='\n'.join(text[a:b] for a,b in extra['spans'])
    assert '2/12까지 등록' in body and '3/1~3/5' in body
    assert '합격)' not in body and '2/14~2/19' not in body


def test_g_rejects_unapproved_third_cohort_or_missing_exclusion():
    from sejong_rag.semantic_reference import cohort_reference_plans
    with pytest.raises(ReferenceReview,match='cohort_structure'):
        cohort_reference_plans(G_TEXT.replace('2. 유의사항','다. 편입생 시험 기간: 3/1~3/5\n2. 유의사항'))
    with pytest.raises(ReferenceReview,match='exclusion_missing'):
        cohort_reference_plans(G_TEXT.replace('응시가 불가능','참여 가능'))


def test_g_keeps_repeated_reference_statements_without_cross_cohort_dates():
    from sejong_rag.semantic_reference import cohort_reference_plans
    text=G_TEXT.replace('3. 문의','3. 추가 안내\n상기 기간 중 레벨테스트를 실시해야 합니다.\n4. 문의')
    plans=cohort_reference_plans(text)
    assert len(plans)==2
    assert all(sum('상기' in text[a:b] for a,b in p['spans'])==2 for p in plans)
