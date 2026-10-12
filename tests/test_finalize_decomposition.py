from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('finalize_decomposition',Path(__file__).parents[1]/'scripts/finalize_decomposition.py')
finalize=importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalize)


def fixture(tmp_path):
    case_path=tmp_path/'cases.jsonl';case_path.write_text(json.dumps({'id':'Q','kind':'answerable'})+'\n')
    arm={'result':{'claims':[{'text':'fact','citations':[1]}]}}
    report={'rows':[{'id':'Q','a':arm,'b':arm,'plan':{'fallback':None}}],
        'summary':{'arms':{}},'api_calls':[],'integrity':True}
    report_path=tmp_path/'report.json';report_path.write_text(json.dumps(report))
    judgment={'completed':True,'full_correct':True,'behavior_correct':None,
        'claim_checks':[{'claim_index':1,'citations':[1],'supported':True}], 'message_facts':[]}
    return {'report_path':str(report_path),'report_sha256':finalize.file_sha256(report_path),
        'cases_path':str(case_path),'cases_sha256':finalize.file_sha256(case_path),
        'results':[{'id':'Q','arms':{'a':deepcopy(judgment),'b':deepcopy(judgment)},'identical_generation_input':True}]}


def test_summary_distinguishes_fact_accuracy_from_citation_support(tmp_path):
    review=fixture(tmp_path)
    review['results'][0]['arms']['b']['claim_checks'][0]['supported']=False
    result=finalize.summarize_review(review)
    assert result['arms']['b']['full_correct']==1
    assert result['arms']['b']['grounded_full_correct']==0
    assert result['arms']['a']['grounded_full_correct']==1


def test_review_cannot_be_applied_to_changed_report(tmp_path):
    review=fixture(tmp_path)
    Path(review['report_path']).write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):finalize.summarize_review(review)


def test_review_cannot_omit_claims_or_change_citations(tmp_path):
    review=fixture(tmp_path)
    review['results'][0]['arms']['a']['claim_checks'][0]['citations']=[2]
    with pytest.raises(ValueError,match='Claim mapping mismatch'):finalize.summarize_review(review)
    review['results'][0]['arms']['a']['claim_checks']=[]
    with pytest.raises(ValueError,match='Missing claim checks'):finalize.summarize_review(review)
