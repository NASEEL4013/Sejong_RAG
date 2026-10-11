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


def body_chunk(number=1, parent="notice-1", text="첨부의 마감일을 8월 20일로 정정합니다."):
    return {**chunk(number, parent=parent, document=parent), "chunk_id": f"body-{parent}-{number}",
            "filename": None, "attachment_url": None, "file_type": "html",
            "locators": [{"section": "notice_body"}], "text": text}


def test_parent_correction_precedes_file_neighbor():
    attachment, neighbor, body = chunk(1), chunk(2), body_chunk()
    selected = select_unified_context([{"chunk": attachment}],
                                     [attachment, neighbor, body], max_chunks=2)
    assert [item["chunk_id"] for item in selected] == [attachment["chunk_id"], body["chunk_id"]]
    assert selected[1]["context_relation"] == "parent_notice"


def test_long_parent_does_not_displace_direct_answer_evidence():
    attachment = chunk(1)
    other = chunk(1, parent="notice-2", document="attachment-2")
    bodies = [body_chunk(i) for i in range(1, 10)]
    second_body = body_chunk(parent="notice-2")
    selected = select_unified_context([{"chunk": attachment}, {"chunk": other}],
        [attachment, other, *bodies, second_body], max_chunks=4)
    assert [c["chunk_id"] for c in selected[:2]] == [attachment["chunk_id"], other["chunk_id"]]
    assert second_body["chunk_id"] in {c["chunk_id"] for c in selected}


def test_shared_attachment_includes_both_parent_notices():
    attachment = chunk(1)
    attachment["source_aliases"] = [{"parent_document_id": "notice-2"}]
    bodies = [body_chunk(), body_chunk(parent="notice-2")]
    selected = select_unified_context([{"chunk": attachment}], [attachment, *bodies])
    assert {item["document_id"] for item in selected} == {"attachment-1", "notice-1", "notice-2"}


def test_budget_omission_and_missing_parent_are_explicit():
    attachment, body = chunk(1), body_chunk()
    selected = select_unified_context([{"chunk": attachment}], [attachment, body], max_chunks=1)
    assert selected[0]["evidence_warning"]
    selected = select_unified_context([{"chunk": attachment}], [attachment])
    assert selected[0]["evidence_warning"]
    assert select_unified_context([{"chunk": body}], [body], max_chars=1) == []


def test_correction_at_end_of_long_notice_is_prioritized_and_warns_if_partial():
    attachment = chunk(1)
    opening = body_chunk(text="일반 안내")
    correction = body_chunk(9)
    selected = select_unified_context([{"chunk": attachment}], [attachment, opening, correction], max_chunks=2)
    assert selected[1]["chunk_id"] == correction["chunk_id"]
    assert selected[0]["evidence_warning"]


def test_parent_warnings_and_relationship_reach_prompt():
    from sejong_rag.prompt_builder import build_messages
    attachment, body = chunk(1), body_chunk()
    body["evidence_warning"] = "본문 이미지 미검증"
    selected = select_unified_context([{"chunk": attachment}], [attachment, body])
    messages = build_messages("신청 마감일?", selected)
    assert "본문 이미지 미검증" in messages[1]["content"]
    assert "context_relation: parent_notice" in messages[1]["content"]
    assert "명시적으로 정정" in messages[0]["content"]
