import pytest
from sejong_rag.semantic_syllabus import split_syllabus, syllabus_groups
from sejong_rag.semantic_rechunk import NeedsReview

class Helper:
    def count(self,text):return len(text)//4

def record(text,page=1,member='강의계획서_코딩.pdf'):
    return dict(record_id=f'r{page}',document_id='d',document_title='2025-1 학점교류',filename='수업계획서.zip',
                locator={'zip_member':member,'page':page},text=text)

PREFIX='수업계획서\n교과목명 (국문)코딩\n학과 교양 개설학기 2025학년도 봄학기\n담당교수 김교수\n학점 3\n교과목코드 123\n'
P1=PREFIX+'교과소개\n입문자를 위한 강좌\n교과목표\n원리 이해\n난이도 기초\n핵심역량 사고력 100%\n교과교재\n주교재 책 A\n'
P2='부교재 책 B\n학습 Tip\n메모하기\n교과평가\n중간고사 25% 기말고사 35% 참여도 20% 출석 20%\n관련규정\n75% 미만 수강 시 F. 중간·기말·대체과제 모두 미응시 시 F 가능.\n주별수업내용\n1주차 소개\n2주차 코딩\n15주차 기말고사'

def test_four_topics_keep_context_exceptions_and_complete_schedule():
    rs=[record(P1),record(P2,2)]
    cs,ps=split_syllabus(rs,{},Helper())
    assert len(cs)==4 and not ps
    for c in cs:
        assert PREFIX in c['context_text']
        assert '코딩' in c['text'] and '2025학년도 봄학기' in c['text']
        assert c['syllabus_member']=='강의계획서_코딩.pdf'
    assert '핵심역량' in cs[0]['context_text']
    assert '책 A' in cs[1]['text'] and '책 B' in cs[1]['text']
    assert {s['record_id'] for s in cs[1]['source_spans']}=={'r1','r2'}
    assert '25%' in cs[2]['text'] and '75%' in cs[2]['text'] and '미응시' in cs[2]['text']
    assert '1주차' in cs[3]['text'] and '15주차' in cs[3]['text']
    assert '75%' not in cs[3]['text']

def test_zip_members_and_nonconsecutive_pages_never_mix():
    rs=[record(P1),record(P2,2,member='수업계획서_다른과목.pdf')]
    assert len(syllabus_groups(rs))==2
    with pytest.raises(NeedsReview,match='mixed_sources'):split_syllabus(rs,{},Helper())
    with pytest.raises(NeedsReview,match='pages'):split_syllabus([record(P1),record(P2,3)],{},Helper())

def test_oversized_topic_is_exactly_queued_not_truncated_or_split():
    rs=[record(P1),record(P2+'\n'+'긴 일정 '*1800,2)]
    cs,ps=split_syllabus(rs,{},Helper())
    assert len(cs)==3 and len(ps)==1
    assert all(c['syllabus_topic_index']!=4 for c in cs)
    span=ps[0]['source_spans'][0]
    assert span['end']==len(rs[1]['text'])
    assert rs[1]['text'][span['start']:].startswith('주별수업내용')

def test_other_syllabus_layout_and_multiple_courses_stay_pending():
    with pytest.raises(NeedsReview,match='heading_layout'):
        split_syllabus([record(P1.replace('교과교재','교 재')),record(P2,2)],{},Helper())
    with pytest.raises(NeedsReview,match='multiple_courses'):
        split_syllabus([record(P1+'\n교과목명 다른과목\n'),record(P2,2)],{},Helper())
