import pytest
from sejong_rag.evaluate_selected import evidence_body,metrics,notice_ids,satisfies


def test_phrase_groups_allow_whitespace_not_missing_conditions():
    assert satisfies('총\n6학기. 군입대휴학 제외',[['총 6 학기','군입대휴학 제외']])
    assert not satisfies('총 6학기',[['총 6학기','군입대휴학 제외']])
    assert satisfies('다른 정확한 표현',[['없는 문장'],['다른 정확한 표현']])


def test_body_excludes_title():
    c={'text':'정답 제목\n다른 본문','source_spans':[{'record_id':'r','start':0,'end':5}]}
    assert evidence_body(c,{'r':{'text':'다른 본문'}})=='다른 본문'


def test_alias_notice_scope():
    assert 'notice2' in notice_ids({'document_id':'attachment','parent_document_id':'notice1','source_aliases':[{'parent_document_id':'notice2'}]})


def test_metrics_count_coverage_failure_as_miss():
    rows=[{'first_evidence_rank':r,'notice_hit_at_5':True} for r in (1,5,10,None)]
    m=metrics(rows)
    assert m['hit_at_1']==.25 and m['hit_at_5']==.5
    assert m['mrr']==pytest.approx((1+.2+.1)/4)
    assert m['indexed_evidence_questions']==3
