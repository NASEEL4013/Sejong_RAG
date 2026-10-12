import json
from copy import deepcopy
import numpy as np
import pytest
from sejong_rag.semantic_workflow import WorkflowReview, workflow_plans
from sejong_rag.semantic_rechunk import split_workflow, build
from sejong_rag.selected_search import SelectedSearch
from sejong_rag.selected_generation import build_request


class Tokenizer:
    def __call__(self,text,**kw): return {'input_ids':list(range(len(text)//2+2))}

class Helper:
    def count(self,text): return len(Tokenizer()(text)['input_ids'])


def records(doc='doc-2025',year=2025):
    texts=['타대학 학점인정 신청 방법\nSejong-MOOC센터',
           'Index\n1. 소속기관 등록\n2. 이수증 기관등록(발송)\n3. 발송된 이수정보 발송이력 확인 방법',
           '1. 소속기관 등록 신청 방법\n① 마이페이지 개인정보관리 클릭',
           '1. 소속기관 등록 신청 방법\n② 소속기관과 학과 학번 이름 저장\n※ 수정 불가',
           '2. 이수증 기관등록 (발송) 신청 방법\n① 이수증 기관등록 클릭\n※ 설문 참여 및 이수증 발급 필요',
           f'2. 이수증 기관등록 (발송) 신청 방법\n② 연계코드 YEAR{year} 검증 후 발송\n※ 정정 신청서는 10월 15일까지 709호 제출',
           '3. 발송된 이수정보 발송이력 확인 방법\n① 마이페이지 발송이력 확인',
           '감사합니다.\nSejong-MOOC센터\n02-3408-3854']
    return [dict(record_id=f'{doc}-r{i}',document_id=doc,parent_document_id=f'notice-{year}',
                 document_title=f'{year}-2 타대학 K-MOOC 학점인정 신청 안내',
                 text=t,locator={'page':i},filename='학점인정 신청방법.pdf',source_kind='attachment',
                 file_type='pdf',source_url=f'https://example.org/{year}',attachment_url=f'https://example.org/{year}.pdf')
            for i,t in enumerate(texts,1)]


def chunks(rs):
    templates={r['record_id']:dict(r) for r in rs}
    return split_workflow(rs,templates,Helper())


def engine(rs=None):
    rs=rs or records()+records('doc-2026',2026)
    by_doc={}
    for r in rs:by_doc.setdefault(r['document_id'],[]).append(r)
    cs=[c for group in by_doc.values() for c in chunks(group)]
    vectors=np.array([[.5,0],[1,0],[.1,0],[.05,0],[.03,0],[.02,0]][:len(cs)])
    return SelectedSearch(cs,vectors,{r['record_id']:r for r in rs},Tokenizer())


def test_page_groups_outline_and_full_conditions():
    rs=records();cs=chunks(rs)
    assert len(cs)==3
    assert [[s['locator']['page'] for s in c['source_spans']] for c in cs]==[[3,4],[5,6],[7]]
    assert [c['procedure_step'] for c in cs]==[1,2,3]
    assert len({c['procedure_id'] for c in cs})==1
    assert all('1. 소속기관 등록 → 2. 이수증 기관등록(발송) → 3.' in c['text'] for c in cs)
    assert all('02-3408-3854' in c['text'] for c in cs)
    assert '설문 참여' in cs[1]['text'] and '10월 15일까지 709호' in cs[1]['text']
    assert {s['record_id'] for c in cs for s in c['context_source_spans']}=={r['record_id'] for r in rs}
    assert cs[0]['procedure_id']!=chunks(records('doc-2026',2026))[0]['procedure_id']
    changed=deepcopy(rs);changed[5]['text']+=' 새 원문'
    assert cs[0]['procedure_id']!=chunks(changed)[0]['procedure_id']

@pytest.mark.parametrize('mutation,reason',[
    (lambda rs:rs.pop(3),'missing_or_duplicate_pages'),
    (lambda rs:rs[3].update(text=rs[3]['text'].replace('②','[ ]')),'missing_extracted_buttons'),
    (lambda rs:rs[5].update(text=rs[5]['text'].replace('2.','3.')),'section_order'),
])
def test_unknown_or_incomplete_structure_requires_review(mutation,reason):
    rs=records();mutation(rs)
    with pytest.raises(WorkflowReview,match=reason):workflow_plans(rs)


def test_build_processes_whole_document_and_preserves_original(tmp_path):
    rs=records();src=tmp_path/'selected.jsonl';norm=tmp_path/'records.jsonl';out=tmp_path/'new'
    originals=[dict(chunk_id=r['record_id'],source_spans=[dict(record_id=r['record_id'],start=0,end=len(r['text']))]) for r in rs]
    src.write_text(''.join(json.dumps(r)+'\n' for r in originals));before=src.read_bytes()
    norm.write_text(''.join(json.dumps(r)+'\n' for r in rs))
    report=build(src,norm,out,Tokenizer())
    assert report['new_chunks']==3 and report['pending_records']==0
    assert report['document_progress']['complete']==1 and src.read_bytes()==before

@pytest.mark.parametrize('query',[
    '2025-2 타대학 K-MOOC 학점인정 신청 방법',
    '2025-2 타대학 K-MOOC 학점인정 신청',
    '2025-2 처음부터 끝까지 이수증 발송 절차 알려줘',
])
def test_whole_question_expands_outside_candidate_pool_in_source_order(query):
    e=engine();out=e.search(query,np.array([1.,0.]),candidate_k=1)
    assert [b['procedure_step'] for b in out]==[1,2,3]
    assert all('2025' in b['document_title'] for b in out)
    assert out[0]['procedure_expansion_anchor']==e.chunks[1]['chunk_id']
    request=build_request(query,out)
    evidence=json.loads(request['messages'][1]['content'])['evidence']
    assert len(evidence)==3 and '10월 15일까지' in evidence[1]['text']
    assert evidence[0]['locations'] and evidence[1]['locations']


def test_specific_step_keeps_outline_without_forcing_other_details():
    e=engine(records());e.vectors=np.array([[0.,0.],[0.,0.],[1.,0.]])
    out=e.search('2025-2 발송이력 어디서 확인해?',np.array([1.,0.]))
    assert len(out)==1 and out[0]['procedure_step']==3
    assert '전체 순서' in out[0]['text'] and '10월 15일까지' not in out[0]['text']


def test_explicit_period_never_borrows_other_semester():
    e=engine();e.vectors=np.array([[.1,0],[.2,0],[.05,0],[1,0],[.8,0],[.9,0]])
    out=e.search('2025-2 신청 방법',np.array([1.,0.]))
    assert len(out)==3 and all('2025' in b['document_title'] for b in out)
    assert e.last_procedure_decisions[0]['status']=='period_mismatch'

@pytest.mark.parametrize('limits',[{'top_k':2},{'total_tokens':100},{'max_bundle_tokens':100}])
def test_budget_never_silently_returns_partial_whole_workflow(limits):
    e=engine(records())
    assert e.search('전체 신청 방법',np.array([1.,0.]),**limits)==[]
    assert e.last_procedure_decisions[0]['status']=='insufficient_budget'


def test_incomplete_group_not_presented_as_complete():
    e=engine(records());e.procedures[next(iter(e.procedures))].pop()
    out=e.search('전체 신청 방법',np.array([1.,0.]),candidate_k=1)
    assert out==[] and e.last_procedure_decisions[0]['status']=='incomplete_group'
