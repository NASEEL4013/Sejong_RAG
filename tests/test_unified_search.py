from sejong_rag.unified_search import format_locations

def test_format_locations_supports_pages_and_sheets():
    chunk={"locators":[{"page":3},{"sheet":"졸업요건","row":8}]}
    value=format_locations(chunk)
    assert "p.3" in value and "sheet=졸업요건" in value and "row=8" in value

def test_format_locations_deduplicates():
    assert format_locations({"locators":[{"page":1},{"page":1}]})=="p.1"

def test_ambiguous_scholarship_question_warns():
    from sejong_rag.unified_search import ambiguity_warning
    results=[{"chunk":{"document_title":"국가장학금"}},{"chunk":{"document_title":"성적향상장학금"}}]
    assert ambiguity_warning("장학금 서류가 뭐야?",results)
    assert ambiguity_warning("국가장학금 서류가 뭐야?",results) is None
