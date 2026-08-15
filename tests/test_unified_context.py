import pytest
from sejong_rag.unified_context import select_unified_context, source_records

def chunk(number, parent="notice-1", document="attachment-1"):
    return {"chunk_id": f"chunk-{number}-{parent}", "chunk_index": number, "document_id": document, "parent_document_id": parent, "document_title": f"공지 {parent}", "filename": "guide.xlsx", "quality_tier": "trusted", "locators": [{"sheet": "Sheet1", "row": number}], "source_url": f"https://example.test/{parent}", "attachment_url": "https://example.test/file", "text": f"내용 {number}"}

def test_selects_retrieved_first_then_same_file_neighbors():
    chunks = [chunk(1), chunk(2), chunk(3)]
    selected = select_unified_context([{"chunk": chunks[1]}], chunks)
    assert [item["chunk_id"] for item in selected] == ["chunk-2-notice-1", "chunk-1-notice-1", "chunk-3-notice-1"]
    assert selected[0]["context_relation"] == "retrieved"
    assert selected[1]["context_relation"] == "neighbor"

def test_limits_direct_results_from_one_notice():
    chunks = [chunk(i) for i in range(1, 5)]
    other = chunk(1, parent="notice-2", document="attachment-2")
    selected = select_unified_context([{"chunk": item} for item in chunks + [other]], chunks + [other], neighbor_window=0, max_retrieved_per_notice=2)
    assert [item["parent_document_id"] for item in selected] == ["notice-1", "notice-1", "notice-2"]

def test_source_records_preserve_non_pdf_location_and_urls():
    records = source_records([chunk(7), chunk(7)])
    assert len(records) == 1
    assert records[0]["location"] == "sheet=Sheet1, row=7"
    assert records[0]["attachment_url"].endswith("/file")

def test_rejects_invalid_limits():
    with pytest.raises(ValueError):
        select_unified_context([], [], max_chars=0)

def test_same_page_sibling_is_added_before_adjacent_chunk():
    base = chunk(1)
    sibling = chunk(2)
    neighbor = chunk(3)
    base["locators"] = [{"page": 10}]
    sibling["locators"] = [{"page": 10}]
    neighbor["locators"] = [{"page": 11}]
    selected = select_unified_context([{"chunk": base}], [base, sibling, neighbor], max_chunks=2)
    assert [item["chunk_id"] for item in selected] == [base["chunk_id"], sibling["chunk_id"]]
    assert selected[1]["context_relation"] == "same_page"
