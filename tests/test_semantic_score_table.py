from bs4 import BeautifulSoup
import pytest
from sejong_rag.semantic_matrix import MatrixReview
from sejong_rag.semantic_score_table import score_plans
from sejong_rag.semantic_rechunk import document_progress


def fixture(extra_major='영어영문학과',amount='150',exam_round='551'):
    html=('<p>장학금은 언어별 각 1회에 한하여 수혜가능 (총 3회)</p><p>[TOEIC]</p>'
        '<table><tr><td>구분 (영어)</td><td>점수</td><td>장학금액</td></tr>'
        '<tr><td rowspan="2">비전공자</td><td>900 이상 – 950 미만</td><td>100만원</td></tr>'
        f'<tr><td>950 이상</td><td>{amount}만원</td></tr>'
        f'<tr><td rowspan="2">전공자 ({extra_major})</td><td>950 이상 – 980 미만</td><td>100만원</td></tr>'
        f'<tr><td>980 이상</td><td>{amount}만원</td></tr></table>'
        f'<p>※ TOEIC 시험: 제 {exam_round}회까지 접수가능</p>'
        '<p>※ 하위점수구간 취득 후 상위점수구간 취득시 차액지급 (1회에 한하며, 2016학년도 수혜자부터 적용)</p>'
        '<p>[TOEIC 외 기타]</p><p>다른 시험 내용</p>')
    return html,BeautifulSoup(html,'html.parser').get_text('\n',strip=True)


def test_i_keeps_both_bands_and_increment_rule_for_each_cohort():
    html,text=fixture();plans=score_plans(text,html)
    assert len(plans)==2
    general,major=plans
    assert general['cohort']=='nonmajor' and major['cohort']=='major'
    assert all(s in general['body'] for s in ['900 이상 – 950 미만','950 이상','100만원','150만원'])
    assert '980 이상' not in general['body']
    assert '950 이상 – 980 미만' in major['body'] and '980 이상' in major['body']
    assert '900 이상' not in major['body']
    for p in plans:
        assert '차액지급' in p['body'] and '1회에 한하며' in p['body'] and '2016학년도' in p['body']
        assert '언어별 각 1회' in p['body'] and '제 551회' in p['body']
        assert '다른 시험 내용' not in p['body']
        assert all(0<=a<b<=len(text) for a,b in p['spans'])


def test_i_uses_each_document_own_major_definition_amount_and_cutoff():
    html,text=fixture(extra_major='영어데이터융합전공',amount='170',exam_round='579')
    plans=score_plans(text,html)
    assert '영어데이터융합전공' in plans[1]['body']
    assert all('170만원' in p['body'] and '제 579회' in p['body'] for p in plans)
    assert all('150만원' not in p['body'] and '551' not in p['body'] for p in plans)


def test_i_unknown_notes_or_exam_caption_need_review():
    html,text=fixture()
    with pytest.raises(MatrixReview,match='increment_rule'):
        score_plans(text.replace('차액지급','별도 안내'),html.replace('차액지급','별도 안내'))
    with pytest.raises(MatrixReview,match='caption'):
        score_plans(text.replace('[TOEIC]','[다른 시험]'),html.replace('[TOEIC]','[다른 시험]'))


def test_progress_counts_documents_not_records_or_chunks():
    rs={rid:dict(document_id=doc,document_title=doc) for rid,doc in [('a1','a'),('a2','a'),('b1','b'),('b2','b'),('c','c'),('excluded','outside')]}
    summary,rows=document_progress(rs,{'a1','a2','b1','b2','c'},{'b2','c'},{'a1','a2','b1'})
    assert summary['selected_documents']==3 and summary['all_normalized_documents']==4
    assert summary['complete']==1 and summary['partial']==1 and summary['not_chunked']==1
    assert summary['complete_percent']==33.33
    assert next(r for r in rows if r['document_id']=='b')['status']=='partial'
