from types import SimpleNamespace
from pathlib import Path
import pytest
from sejong_rag.syllabus_originals import build_catalog,lookup,guidance_request
from sejong_rag.ask_selected import prepare_request
from sejong_rag.selected_generation import generate_answer
from sejong_rag.corpus_policy import allowed_chunk,SYLLABUS_DOCUMENTS


def entry(year=2026,semester=2):
    return dict(course='현대인의건강라이프',teacher='조영미',teacher_names=['조영미'],academic_period=f'{year}학년도 {semester}학기',
                year=year,semester=semester,season=None,period_conflict=False,notice_year=year,
                attachment_url='https://www.sejong.ac.kr/download',source_url='https://www.sejong.ac.kr/notice',
                zip_member='강의계획서_현대인의건강라이프.pdf',filename='강의계획서.zip')


def test_details_return_links_without_search_embedding_or_generation():
    def forbidden(*a,**kw):raise AssertionError('body retrieval or generation called')
    engine=SimpleNamespace(syllabus_catalog=[entry()],count=forbidden,search=forbidden)
    request=prepare_request('2026-2 현대인의 건강라이프 중간고사 몇 주차야?',engine,{},SimpleNamespace(encode=forbidden))
    assert request['sources']==[] and request['messages']==[]
    result=generate_answer(SimpleNamespace(models=SimpleNamespace(generate_content=forbidden)),request,'unused')
    assert result['status']=='answered' and result['attempts']==[]
    assert 'https://www.sejong.ac.kr/download' in result['answer']
    assert 'ZIP 내부 파일' in result['answer']
    assert '7주차' not in result['answer']


def test_unknown_or_ambiguous_semester_is_not_replaced_by_another():
    cs=[entry(2025,2),entry(2026,2)]
    assert lookup('2027-2 현대인의건강라이프 교재 알려줘',cs)['documents']==[]
    r=lookup('현대인의건강라이프 강의내용',cs)
    assert r['status']=='needs_clarification' and len(r['documents'])==2
    assert lookup('2026-2 현대인의건강라이프 교재 알려줘',cs)['documents']==[cs[1]]
    assert lookup('조영미 교수 강의계획서',cs)['documents']==cs
    assert lookup('국가장학금 신청기간',cs) is None
    assert lookup('수업계획서 보여줘',cs)['status']=='needs_clarification'


def test_multiple_courses_and_spaced_headers_become_metadata_only(monkeypatch):
    monkeypatch.setattr('sejong_rag.syllabus_originals.load_attachment_metadata',lambda p:[dict(document_id='d',attachment_url='https://www.sejong.ac.kr/d')])
    rs=[]
    for page,name in [(1,'SW기초코딩'),(3,'고급프로그래밍활용')]:
        rs.append(dict(record_id=f'r{page}',document_id='d',document_title='2026학년도 예비대학 안내',
                       filename='교양코딩 수업계획서.pdf',source_url='https://www.sejong.ac.kr/n',locator={'page':page},
                       text='2026학년도 예비대학 수업계획서\n교 과 목 명 '+' '.join(name)+'\n담 당 교 수 ( 소 속 ) 이 종 원 학 수 번 호 123\n평가기준 비공개검증문자열'))
    cs=build_catalog(rs)
    assert [c['course'] for c in cs]==['SW기초코딩','고급프로그래밍활용']
    assert all(c['teacher_names']==['이종원'] for c in cs)
    assert '비공개검증문자열' not in str(cs)
    assert [c['pages'] for c in cs]==[[1],[3]]


def test_historical_chunks_and_future_syllabus_filenames_are_excluded():
    for doc in SYLLABUS_DOCUMENTS:
        assert not allowed_chunk({'document_id':doc})
        assert not allowed_chunk({'source_spans':[{'record_id':doc+'-record-1'}]})
    assert not allowed_chunk({'document_id':'future','filename':'새 강의계획서.pdf'})
    assert allowed_chunk({'document_id':'notice','document_title':'강의계획서 및 수강신청 안내'})


def test_cli_original_route_does_not_create_api_client(monkeypatch,capsys):
    from sejong_rag import ask_selected
    monkeypatch.setattr(ask_selected,'load_runtime',lambda *args:())
    monkeypatch.setattr(ask_selected,'prepare_request',lambda *args:guidance_request('현대인의건강라이프 강의내용',[entry()]))
    monkeypatch.setattr(ask_selected,'create_client',lambda:pytest.fail('external API client created'))
    monkeypatch.setattr('sys.argv',['ask-selected','현대인의건강라이프 강의내용'])
    ask_selected.main()
    assert 'https://www.sejong.ac.kr/download' in capsys.readouterr().out
