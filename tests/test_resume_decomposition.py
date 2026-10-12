from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec=importlib.util.spec_from_file_location('resume_decomposition',Path(__file__).parents[1]/'scripts/resume_decomposition.py')
resume=importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


def test_resume_preserves_completed_generation_and_retries_only_remaining_budget(tmp_path,monkeypatch):
    request={'sources':[],'evidence_tokens_e5':0}
    coverage={'items':[],'total':0,'covered':0,'complete':None}
    report={'settings':{'model':'test','max_api_calls':5},'api_calls':[{'case_id':'Q:b','status':'error','usage':{}}],
        'rows':[{'id':'Q','query':'질문','kind':'unanswerable','plan':{'questions':['질문']},
        'a':{'request':request,'evidence_coverage':coverage,'result':{'answer':'already generated'}},
        'b':{'request':request,'evidence_coverage':coverage}}]}
    original=deepcopy(report)
    def forbidden(*a,**k):raise AssertionError('completed planning/retrieval must be reused')
    monkeypatch.setattr(resume,'decompose',forbidden)
    monkeypatch.setattr(resume,'retrieve',forbidden)
    def generate(client,request,**kwargs):
        assert client.case_id=='Q:b'
        assert kwargs['max_attempts']==1
        return {'answer':'new'}
    monkeypatch.setattr(resume,'generate_answer',generate)
    result=resume.execute([{'id':'Q','query':'질문','kind':'unanswerable'}],report,None,SimpleNamespace(),tmp_path)
    assert result['rows'][0]['a']['result']['answer']=='already generated'
    assert result['rows'][0]['b']['result']['answer']=='new'
    assert report==original
