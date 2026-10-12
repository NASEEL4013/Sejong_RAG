"""Behavioral guards for the experimental decomposition path."""
from copy import deepcopy
from types import SimpleNamespace
import json
import numpy as np
import pytest
from sejong_rag.decomposed_search import validate_plan, decompose, fuse, retrieve
from sejong_rag.compare_decomposed import evidence_coverage


def bundle(name, body='본문', cost=100, notice='n'):
    return dict(anchor_chunk_id=name, body=body, text=body, token_count=cost,
        source_spans=[dict(record_id=name,start=0,end=len(body))],
        source_aliases=[dict(document_id=notice,parent_document_id=notice)],
        parent_document_id=notice,document_id=notice,source_url='https://example.org/'+notice)


def test_planner_rejects_new_facts_and_duplicate_or_excess_queries():
    query='2026년 신청 기간과 장소는?'
    for p in ({'questions':['2027년 신청은?']}, {'questions':['어디?','어디?']},
              {'questions':['a','b','c','d','e']}, {'questions':[]}, {'questions':[' ']}):
        with pytest.raises(ValueError):validate_plan(p,query)


def test_planner_only_sends_question_and_falls_back():
    class Client:
        models=None
        def generate_content(self,**kwargs):
            assert json.loads(kwargs['contents'])=={'question':'2026년 신청은?'}
            return SimpleNamespace(text='{"questions":["2027년 신청은?"]}')
    client=Client();client.models=client
    result=decompose('2026년 신청은?',client,'test')
    assert result['questions']==['2026년 신청은?']
    assert result['fallback']=='invalid_plan'


def test_fusion_preserves_source_and_budget_and_subquery_representation():
    baseline=[bundle('original'),bundle('old2')]
    lists=[[bundle('menu','신청 메뉴')],[bundle('same-day','당일 불가')]]
    original=deepcopy((baseline,lists))
    result=fuse(baseline,lists,top_k=3,total_tokens=300)
    assert [b['anchor_chunk_id'] for b in result]==['original','menu','same-day']
    assert (baseline,lists)==original
    assert sum(b['token_count'] for b in result)==300


def test_fusion_skips_duplicate_and_oversized_candidate_and_continues():
    first=bundle('first','동일 본문')
    lists=[[deepcopy(first),bundle('large','큰 근거',cost=600),bundle('next','새 근거')]]
    result=fuse([first],lists,total_tokens=200)
    assert [b['anchor_chunk_id'] for b in result]==['first','next']
    assert fuse([bundle('huge',cost=600)],[],max_bundle_tokens=512)==[]


def test_different_notice_identical_body_is_not_lost():
    result=fuse([bundle('a','기간 안내',notice='n1')],[[bundle('b','기간 안내',notice='n2')]])
    assert len(result)==2


def test_union_coverage_does_not_mix_notice_versions_or_use_title():
    case={'items':[{'evidence_options':[{'notice_id':'n1','evidence_any':[['메뉴','당일 불가']]}]}]}
    assert evidence_coverage([bundle('a','메뉴'),bundle('b','당일 불가')],case)['covered']==0
    assert evidence_coverage([bundle('a','메뉴',notice='n1'),bundle('b','당일 불가',notice='n2')],case)['covered']==0
    assert evidence_coverage([bundle('a','메뉴',notice='n1'),bundle('b','당일 불가',notice='n1')],case)['covered']==1
    b=bundle('a','다른 내용',notice='n1');b['document_title']='메뉴 당일 불가'
    assert evidence_coverage([b],case)['covered']==0


def test_single_query_exact_baseline_and_original_question_in_generation():
    class Engine:
        def count(self,text):return len(text)
        def search(self,query,vector,**kwargs):return [bundle('a','본문1'),bundle('b','본문2')]
    class Model:
        max_seq_length=512
        def encode(self,texts,**kwargs):return np.ones((len(texts),2))
    result=retrieve('신청은?',['신청은?'],Engine(),{'query_prefix':'query: '},Model())
    assert result['a']==result['b']
    assert json.loads(result['b']['messages'][1]['content'])['question']=='신청은?'
