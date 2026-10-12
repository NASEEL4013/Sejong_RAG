import pytest
from sejong_rag.semantic_alternatives import split_alternatives
from sejong_rag.semantic_rechunk import NeedsReview

class Helper:
    def count(self, text): return len(text)//3

def record(text, page=4):
    return dict(record_id=f'r{page}',document_id='d',document_title='2026-2 등록 안내',
                text=text,locator={'page':page})

PREFIX='Ⅱ 전액 장학생의 등록 절차 및 방법\n납부금액이 “0원”인 학생은 3가지 방법 중 1가지를 선택하여 등록.\n'
FIRST=PREFIX+'납부방법 1. 자율경비 납부 국민은행 가상계좌 이용\n납부방법 2. 국민은행 공과금 납부\n온라인은 16시 30분까지\n학번 입력 후 조회\n- 4 -'
SECOND='총 납부금액 0원 확인 후 납부 버튼 클릭\n납부방법 3. 고지서 지참 후 국민은행 지점 방문하여 등록 요청\n※ 유의사항\n자율경비 추가 납부 불가\n- 5 -'

def test_alternatives_join_pages_and_keep_branch_conditions_local():
    rs=[record(FIRST),record(SECOND,5),record('Ⅲ 다음 장\n다른 규칙',6)]
    cs,reason=split_alternatives(rs,{},Helper())
    assert reason is None and len(cs)==3
    for c in cs:
        assert c['alternative_relation']=='choose_one'
        assert '자율경비 추가 납부 불가' in c['context_text']
        assert len(c['alternative_labels'])==3
        assert '다른 규칙' not in c['context_text']
        assert '- 4 -' not in c['text'] and '- 5 -' not in c['text']
        assert 'procedure_step_index' not in c
    assert '조회' in cs[1]['text'] and '클릭' in cs[1]['text']
    assert {s['record_id'] for s in cs[1]['source_spans']}=={'r4','r5'}
    assert ['16시 30분' in c['text'] for c in cs]==[False,True,False]
    assert all(s['record_id']!='r6' for c in cs for s in c['context_source_spans'])

def test_fragmented_html_labels_and_source_notes_are_preserved():
    text=(FIRST+'\n'+SECOND).replace('납부방법 1.', '납부방법\n1\n.').replace('납부방법 2.', '납부방법\n2.').replace('납부방법 3.','방법\n3.')
    cs,_=split_alternatives([record(text)],{},Helper())
    assert len(cs)==3
    assert all('이중 납부' not in c['text'] for c in cs)
    cs,_=split_alternatives([record(text+'\n이중 납부 금지')],{},Helper())
    assert all('이중 납부 금지' in c['text'] for c in cs)

def test_missing_page_or_completion_or_choice_is_not_inferred():
    with pytest.raises(NeedsReview,match='nonconsecutive'):
        split_alternatives([record(FIRST),record(SECOND,6)],{},Helper())
    with pytest.raises(NeedsReview,match='incomplete'):
        split_alternatives([record(FIRST),record(SECOND.replace('클릭',''),5)],{},Helper())
    cs,_=split_alternatives([record((FIRST+'\n'+SECOND).replace('3가지 방법 중 1가지를 선택하여','순서대로'))],{},Helper())
    assert not cs

def test_inconsistent_method_count_and_unknown_profile_are_held():
    with pytest.raises(NeedsReview,match='numbers'):
        split_alternatives([record((FIRST+'\n'+SECOND).replace('납부방법 3.','납부방법 4.'))],{},Helper())
    cs,reason=split_alternatives([record('두 가지 방법 중 하나 선택: 가. 성적표 제출 나. 교과목 이수')],{},Helper())
    assert not cs and reason=='alternative_profile_not_verified'

def test_incremental_build_replaces_only_covered_units_and_preserves_pending(tmp_path):
    import json
    from sejong_rag.semantic_bulk import build
    def write(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    r=record(FIRST+'\n'+SECOND);r['source_kind']='attachment'
    other=record('확인이 필요한 다른 장',7);other['source_kind']='attachment'
    def chunk(r):return dict(chunk_id=r['record_id'],document_id='d',source_spans=[dict(record_id=r['record_id'],start=0,end=len(r['text']))])
    norm=tmp_path/'norm';source=tmp_path/'source';base=tmp_path/'base';base.mkdir()
    write(norm,[r,other]);write(source,[chunk(r),chunk(other)])
    write(base/'chunks.jsonl',[chunk(r)])
    write(base/'needs-review.jsonl',[dict(reason='new_layout_other',record_ids=['r7'])])
    # Put the unrelated page in a new chapter, so its scope cannot leak.
    other['text']='Ⅲ 다른 장\n'+other['text'];write(norm,[r,other]);write(source,[chunk(r),chunk(other)])
    class Tokens:
        def __call__(self,text,**kw):return {'input_ids':list(range(len(text)//3))}
    original=(base/'chunks.jsonl').read_bytes()
    out=tmp_path/'out'
    report=build(base,out,Tokens(),norm,source,alternatives_only=True)
    assert report['added_chunks']==3 and report['superseded_chunks']==1
    assert (base/'chunks.jsonl').read_bytes()==original
    pending=[json.loads(l) for l in (out/'needs-review.jsonl').read_text().splitlines()]
    assert len(pending)==1 and pending[0]['record_ids']==['r7']
    assert pending[0]['reason']=='new_layout_other'
    assert report['document_progress']['partial']==1
