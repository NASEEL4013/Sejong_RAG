from sejong_rag.compare_decomposed import evidence_coverage
from sejong_rag.evaluate_semantic import render_html


def test_union_can_cover_one_item_across_two_bodies_but_not_wrong_notice():
    case={'items':[{'name':'기간과 장소','evidence_options':[{'notice_id':'notice3-1','evidence_any':[['1월 9일','집현관']]}]}]}
    a={'parent_document_id':'notice3-1','body':'1월 9일'}
    b={'parent_document_id':'notice3-1','body':'집현관'}
    assert evidence_coverage([a,b],case)['complete']
    assert not evidence_coverage([a,dict(b,parent_document_id='notice3-2')],case)['complete']


def test_html_escapes_question_answer_and_source():
    c=dict(id='T01',query='<script>alert(1)</script>',answer_note='gold',rubric='rule',gold_sources=[])
    row=dict(case=c,result={'answer':'<img src=x>'},generation_state='completed',request={'sources':[]})
    text=render_html({'rows':[row],'usage':{}})
    assert '<script>alert' not in text and '&lt;script&gt;' in text
    assert '<img src=x>' not in text


def test_finalize_rejects_unbound_review(tmp_path):
    import json
    import pytest
    from sejong_rag.evaluate_semantic import finalize
    (tmp_path/'report.json').write_text('{}')
    review=tmp_path/'review.json'
    review.write_text(json.dumps({'report_sha256':'different'}))
    with pytest.raises(ValueError,match='hash mismatch'):
        finalize(tmp_path,review)


def test_summary_requires_every_generated_claim_reviewed():
    import pytest
    from sejong_rag.evaluate_semantic import reviewed_summary
    report={'rows':[{'case':{'id':'A','kind':'answerable','items':[]},'result':{'claims':[{'text':'fact','citations':[1]}]}}]}
    review={'report_sha256':'hash','cases':[{'id':'A','claims':[]}]}
    with pytest.raises(ValueError,match='Every claim'):
        reviewed_summary(report,review)
