from copy import deepcopy
from sejong_rag.evaluate_semantic_decomposition import cases_with_amendment
from sejong_rag.compare_decomposed import evidence_coverage

def test_s02_user_amendment_accepts_confirmation_without_mutating_original():
    case={'id':'S02','kind':'answerable','answer_note':'old','rubric':'old','gold_sources':[],
          'items':[{'evidence_options':[{'notice_id':'notice3-805895','evidence_any':[['출력']]}]},
                   {'evidence_options':[{'notice_id':'notice3-805895','evidence_any':[['제출']]}]},
                   {'evidence_options':[{'notice_id':'notice3-805895','evidence_any':[['조기졸업신청서','성적증명서']]}]}]}
    baseline={'rows':[{'case':case}]};before=deepcopy(baseline)
    result=cases_with_amendment(baseline,{'case_id':'S02','answer_note':'new','rubric':'both','additional_gold_source':{}})
    sources=[{'parent_document_id':'notice3-805922','body':'출력 제출 조기졸업확인신청서 성적증명서'}]
    assert evidence_coverage(sources,result[0])['complete']
    assert not evidence_coverage(sources,case)['complete']
    assert baseline==before
