from sejong_rag.evaluate_unified_generation import evaluate_unified_answer, summary

def test_evaluate_unified_answer_checks_values_source_and_citation():
    context=[{"chunk_id":"a","parent_document_id":"notice-a"},{"chunk_id":"b","parent_document_id":"notice-b"}]
    result=evaluate_unified_answer("최대 300만 원이다 [근거 2]",[["300만원","300만 원"]],context,"notice-b")
    assert all(result[k] for k in ("answer_values_ok","expected_source_in_context","expected_source_cited","citation_numbers_valid"))

def test_evaluate_unified_answer_rejects_wrong_source_citation():
    context=[{"chunk_id":"a","parent_document_id":"notice-a"},{"chunk_id":"b","parent_document_id":"notice-b"}]
    result=evaluate_unified_answer("300만원 [근거 1]",[["300만원"]],context,"notice-b")
    assert result["answer_values_ok"] and not result["expected_source_cited"]

def test_summary_counts_checks():
    row={"answer_values_ok":True,"expected_source_in_context":True,"expected_source_cited":True,"citation_numbers_valid":True}
    assert summary([row])["all_checks"]==1.0

def test_evaluate_unified_answer_parses_composite_citations():
    context=[{"chunk_id":"a","parent_document_id":"notice-a"},{"chunk_id":"b","parent_document_id":"notice-b"}]
    result=evaluate_unified_answer("300만원 [근거 1, 근거 2]",[["300만원"]],context,"notice-b")
    assert result["cited_numbers"] == [1,2]
    assert result["expected_source_cited"]

def test_composite_citation_with_expected_source_is_valid():
    context=[{"chunk_id":"a","parent_document_id":"notice-a"},{"chunk_id":"b","parent_document_id":"notice-b"}]
    result=evaluate_unified_answer("답 [근거 1, 근거 2]",[["답"]],context,"notice-b")
    assert result["expected_source_cited"] and result["citation_numbers_valid"]
