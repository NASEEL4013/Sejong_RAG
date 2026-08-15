from sejong_rag.classify_notices import classify_catalog, classify_item


def item(title, board="academic_notice"):
    return {
        "document_id": "notice3-1",
        "title": title,
        "board": board,
        "department": "학사지원과",
    }


def test_classify_item_supports_multiple_categories():
    result = classify_item(item("2026-2학기 학점교류 수강 안내"))

    assert "course_registration" in result["categories"]
    assert "credit" in result["categories"]
    assert "academic_general" in result["categories"]
    assert result["priority"] == "high"
    assert "수강" in result["matched_keywords"]


def test_scholarship_board_is_preserved_without_keyword():
    result = classify_item(item("재학생 신청 안내", "scholarship_notice"))

    assert result["categories"] == ["scholarship"]
    assert result["priority"] == "normal"


def test_unknown_title_is_not_discarded():
    result = classify_item(item("학생 대상 안내"))

    assert result["categories"] == ["academic_general"]
    assert result["matched_keywords"] == []
    assert result["priority"] == "normal"


def test_classify_catalog_counts_multi_labels():
    result = classify_catalog(
        {"collected_at": "now", "items": [item("졸업 학점 안내"), item("휴학 안내")]}
    )

    assert result["item_count"] == 2
    assert result["category_counts"]["academic_general"] == 2
    assert result["category_counts"]["graduation"] == 1
    assert result["category_counts"]["credit"] == 1
    assert result["category_counts"]["academic_status"] == 1
    assert result["priority_counts"] == {"high": 2}
