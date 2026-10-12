import json
from pathlib import Path
import pytest
from sejong_rag.semantic_rechunk import (NeedsReview, section_spans, split_notice, split_table, build)

class Helper:
    def count(self, text): return len(text)//4+1


def record(text, rid='r', **kw):
    return dict(record_id=rid,text=text,document_title='테스트 장학금',document_id='doc',
                source_kind='notice_body',locator={},**kw)


def test_outer_sections_preserve_inner_conditions_and_exceptions():
    text='■\n지원내용\n1. 등록금\n2. 계속지원\n단, 졸업 직전 예외\n■\n자격요건\n성적\n동점자 우선순위'
    parts=[text[a:b] for a,b in section_spans(text)]
    assert len(parts)==2
    assert '단, 졸업 직전 예외' in parts[0]
    assert '동점자 우선순위' in parts[1]


def test_a_separate_retrieval_shared_context():
    r=record('1. 대상\n학생\n2. 교부일자\n내일\n3. 교부처\n: 학생회관\n※ 방문시간\n10시\n* 신분증 지참')
    out=split_notice(r,{},Helper())
    assert len(out)==5
    for c in out[2:]:
        assert all(s in c['context_text'] for s in ['학생회관','방문시간','신분증'])
        assert c['text'].startswith('문서 제목: 테스트 장학금')


def rows(texts):
    out=[]
    for i,t in enumerate(texts):
        r=record(t,str(i));r['locator']={'sheet':'안내','row':i+1};out.append(r)
    return out


def test_d_rows_carry_notes_and_map_columns():
    rs=rows(['1열: 분반표','9열: 변경 불가','1열: 순번 | 2열: 이름 | 3열: 분반',
             '1열: 1 | 2열: 김*영 | 3열: 001','1열: 2 | 2열: 이*영 | 3열: 002'])
    out=split_table(rs,{},Helper())
    assert len(out)==2
    assert '이름: 김*영' in out[0]['text'] and '분반: 001' in out[0]['text']
    assert '김*영' not in out[1]['text']
    assert all('변경 불가' in c['context_text'] for c in out)
    assert {s['record_id'] for c in out for s in c['source_spans']}=={r['record_id'] for r in rs}


def test_ambiguous_tables_are_not_guessed():
    rs=rows(['1열: 순번 | 2열: 이름 | 3열: 분반','1열: 변경전 | 2열: 변경후'])
    with pytest.raises(NeedsReview): split_table(rs,{},Helper())


def test_large_unit_not_silently_truncated():
    with pytest.raises(NeedsReview,match='exceeds'):
        split_notice(record('대상\n'+'조건 '*1000),{},Helper())


def test_build_preserves_pending_and_refuses_overwrite(tmp_path):
    class Tokenizer:
        def __call__(self,text,**kwargs): return {'input_ids':list(range(len(text)//4+1))}
    rs=[record('대상\n재학생'),record('알 수 없는 형식','pending')]
    source=tmp_path/'old.jsonl';normalized=tmp_path/'records.jsonl';out=tmp_path/'new'
    old=[{'chunk_id':r['record_id'],'source_spans':[{'record_id':r['record_id'],'start':0,'end':len(r['text'])}]} for r in rs]
    source.write_text(''.join(json.dumps(c)+'\n' for c in old))
    normalized.write_text(''.join(json.dumps(r)+'\n' for r in rs))
    before=source.read_bytes()
    result=build(source,normalized,out,Tokenizer())
    assert result['completed_records']==1 and result['pending_records']==1
    assert result['status']=='needs_user_review'
    assert source.read_bytes()==before
    assert json.loads((out/'pending-original-chunks.jsonl').read_text())['chunk_id']=='pending'
    with pytest.raises(FileExistsError): build(source,normalized,out,Tokenizer())


def test_preamble_is_shared_context_not_standalone_retrieval_chunk():
    r=record('장학금 안내문\n1. 대상\n학생\n2. 신청기간\n내일까지')
    out=split_notice(r,{},Helper())
    assert len(out)==2
    assert all('장학금 안내문' not in c['text'] for c in out)
    assert all('장학금 안내문' in c['context_text'] for c in out)


def test_missing_leading_context_cell_requires_review():
    rs=rows(['1열: 학과 | 2열: 이름 | 3열: 학번 | 4열: 비고',
             '2열: 김*영 | 3열: 123 | 4열: 확인'])
    with pytest.raises(NeedsReview,match='merged_cells'):
        split_table(rs,{},Helper())
