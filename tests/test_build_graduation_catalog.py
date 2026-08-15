from sejong_rag.build_graduation_catalog import extract_catalog

def test_extract_catalog_labels_six_values_and_deduplicates():
    chunks = [
        {"chunk_id":"h", "document_id":"g", "chunk_index":1, "filename":"guide.pdf", "text":"❚ 2021학년도 입학자 교과과정"},
        {"chunk_id":"r1", "document_id":"g", "chunk_index":2, "filename":"guide.pdf", "locators":[{"page":61}], "text":"단일전공 이수 시 학과별 졸업 이수학점 전공학점계 소프트웨어학과 14 9 72 36 36 130"},
        {"chunk_id":"r2", "document_id":"g", "chunk_index":3, "filename":"guide.pdf", "locators":[{"page":61}], "text":"전공학점계 소프트웨어학과 14 9 72 36 36 130"},
    ]
    result = extract_catalog(chunks, "guide.pdf")
    assert not result["conflicts"] and len(result["items"]) == 1
    item = result["items"][0]
    assert item["major_total"] == 72 and item["major_required"] == 36
    assert item["source_chunk_ids"] == ["r1", "r2"]

def test_extract_catalog_reports_conflicting_duplicate_rows():
    chunks = [
        {"chunk_id":"h", "document_id":"g", "chunk_index":1, "filename":"guide.pdf", "text":"2021학년도 입학자 교과과정"},
        {"chunk_id":"a", "document_id":"g", "chunk_index":2, "filename":"guide.pdf", "text":"전공학점계 소프트웨어학과 14 9 72 36 36 130"},
        {"chunk_id":"b", "document_id":"g", "chunk_index":3, "filename":"guide.pdf", "text":"전공학점계 소프트웨어학과 14 9 73 36 37 130"},
    ]
    assert extract_catalog(chunks, "guide.pdf")["conflicts"]

def test_extract_catalog_preserves_seven_column_schema_without_shifting():
    chunks = [
        {"chunk_id":"h", "document_id":"g", "chunk_index":1, "filename":"guide.pdf", "text":"2022학년도 입학자 교과과정"},
        {"chunk_id":"r", "document_id":"g", "chunk_index":2, "filename":"guide.pdf", "locators":[{"page":56}], "text":"전공학점계 소프트웨어학과 13 6 15 72 36 36 130"},
    ]
    item = extract_catalog(chunks, "guide.pdf")["items"][0]
    assert (item["common_required"], item["balanced_required"], item["foundational"]) == (13, 6, 15)
    assert (item["major_total"], item["major_required"], item["graduation_total"]) == (72, 36, 130)

def test_extract_catalog_preserves_nine_column_schema_and_totals():
    chunks = [
        {"chunk_id":"h", "document_id":"g", "chunk_index":1, "filename":"guide.pdf", "text":"2024학년도 입학자 교과과정"},
        {"chunk_id":"r", "document_id":"g", "chunk_index":2, "filename":"guide.pdf", "text":"전공영역 총계 소프트웨어학과 13 9 9 15 36 36 72 87 130"},
    ]
    item = extract_catalog(chunks, "guide.pdf")["items"][0]
    assert item["major_foundational"] == 15
    assert item["major_core_total"] == 72
    assert item["major_area_total"] == 87
    assert item["graduation_total"] == 130

def test_extract_catalog_rejects_truncated_row_for_year_schema():
    chunks = [
        {"chunk_id":"h", "document_id":"g", "chunk_index":1, "filename":"guide.pdf", "text":"2023학년도 입학자 교과과정"},
        {"chunk_id":"r", "document_id":"g", "chunk_index":2, "filename":"guide.pdf", "text":"전공학점계 정보보호학과 13 6 15 72 36 36"},
    ]
    result = extract_catalog(chunks, "guide.pdf")
    assert not result["items"]
    assert result["rejected_rows"][0]["expected_columns"] == 7

def test_extract_catalog_reads_continuation_chunk_on_same_table_page():
    chunks = [
        {"chunk_id":"h", "document_id":"g", "chunk_index":1, "filename":"guide.pdf", "text":"2023학년도 입학자 교과과정"},
        {"chunk_id":"title", "document_id":"g", "chunk_index":2, "filename":"guide.pdf", "locators":[{"page":51}], "text":"단일전공 이수 시 학과별 졸업 이수학점 전공학점계"},
        {"chunk_id":"continued", "document_id":"g", "chunk_index":3, "filename":"guide.pdf", "locators":[{"page":51}], "text":"소프트웨어학과 13 6 15 72 36 36 130"},
    ]
    result = extract_catalog(chunks, "guide.pdf")
    assert result["items"][0]["department"] == "소프트웨어학과"
    assert result["items"][0]["graduation_total"] == 130
