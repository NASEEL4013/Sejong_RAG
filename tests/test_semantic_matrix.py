import json
from pathlib import Path
import pytest
from bs4 import BeautifulSoup
from sejong_rag.semantic_matrix import MatrixReview,matrix_plans
from sejong_rag.semantic_rechunk import build
from sejong_rag.semantic_reference import uncovered_spans


def fixture(merged=False,nonmajor='100',major='428',amount='100'):
    row2=(f'<tr><td>비전공자</td><td rowspan="2">IBT {nonmajor}점</td><td>370점</td></tr>' if merged else
          f'<tr><td>비전공자</td><td>IBT {nonmajor}점</td><td>370점</td></tr>')
    row3=(f'<tr><td>전공자</td><td>{major}점</td></tr>' if merged else
          f'<tr><td>전공자</td><td>상 동</td><td>{major}점</td></tr>')
    html=('<p>장학금은 언어별 각 1회에 한하여 수혜가능 (총 3회)</p>'
          '<table><tr><td rowspan="2">구분</td><td colspan="2">영어</td></tr>'
          '<tr><td>TOEFL</td><td>New TEPS</td></tr>'+row2+row3+
          f'<tr><td>장학금액</td><td colspan="2">{amount}만원</td></tr></table><p>별도 메모</p>')
    return html,BeautifulSoup(html,'html.parser').get_text('\n',strip=True)


def test_h_each_exam_has_labels_values_and_shared_award_limit():
    html,text=fixture();plans=matrix_plans(text,html)
    assert len(plans)==2
    toefl,teps=plans
    assert all(label in p['body'] for p in plans for label in ['비전공자 기준:','전공자 기준:','장학금액:','공통 수혜 제한:'])
    assert '전공자 기준: IBT 100점 (원문: 상 동)' in toefl['body']
    assert '370점' not in toefl['body'] and '428점' not in toefl['body']
    assert '비전공자 기준: 370점' in teps['body'] and '전공자 기준: 428점' in teps['body']
    assert 'TOEFL' not in teps['body'] and 'IBT' not in teps['body']
    resolution=toefl['resolved_same_as_above'][0]
    assert text[slice(*resolution['source_span'])]=='상 동'
    assert text[slice(*resolution['target_span'])]=='IBT 100점'


def test_h_merged_cells_supply_major_and_shared_amount_without_fake_same_above():
    html,text=fixture(merged=True,nonmajor='110',major='450',amount='120')
    plans=matrix_plans(text,html)
    assert '전공자 기준: IBT 110점' in plans[0]['body']
    assert '원문:' not in plans[0]['body']
    assert '전공자 기준: 450점' in plans[1]['body']
    assert all('장학금액: 120만원' in p['body'] for p in plans)
    assert any(c['rows']==2 for c in plans[0]['cells'])


def test_h_does_not_guess_when_source_or_shared_limit_differs():
    html,text=fixture()
    with pytest.raises(MatrixReview,match='source_mapping'):
        matrix_plans(text.replace('428점','500점'),html)
    with pytest.raises(MatrixReview,match='award_limit'):
        matrix_plans(text.replace('언어별','시험별'),html.replace('언어별','시험별'))


def test_h_duplicate_matrix_requires_review():
    html,text=fixture()
    with pytest.raises(MatrixReview,match='multiple'):
        matrix_plans(text+'\n'+text,html+html)


def test_h_build_keeps_source_ranges_and_original_chunks(tmp_path):
    html,text=fixture();data=tmp_path/'data';raw=data/'raw/notices';raw.mkdir(parents=True)
    (raw/'doc.html').write_text(html)
    r=dict(record_id='r',document_id='doc',source_kind='notice_body',document_title='2025-2 어학우수장학금',text=text,locator={})
    source=tmp_path/'selected.jsonl';normalized=tmp_path/'records.jsonl';out=tmp_path/'out'
    source.write_text(json.dumps({'chunk_id':'old','source_spans':[{'record_id':'r','start':0,'end':len(text)}]})+'\n')
    normalized.write_text(json.dumps(r)+'\n');original=source.read_bytes()
    class Tokenizer:
        def __call__(self,text,**kw):return {'input_ids':list(range(len(text)//4+1))}
    report=build(source,normalized,out,Tokenizer(),data)
    assert report['partially_chunked_records']==1
    chunks=[json.loads(l) for l in (out/'chunks.jsonl').open()]
    assert len([c for c in chunks if c['semantic_rule'].startswith('H_')])==2
    assert all('문서 제목: 2025-2 어학우수장학금' in c['text'] for c in chunks)
    p=json.loads((out/'needs-review.jsonl').read_text());assert p['reason']=='H_remaining_sections_require_review'
    spans=[(s['start'],s['end']) for s in p['source_spans']]
    spans.extend((s['start'],s['end']) for c in chunks for s in c['source_spans'])
    assert uncovered_spans(text,spans)==[]
    assert source.read_bytes()==original
    assert (out/'pending-original-chunks.jsonl').read_text()
