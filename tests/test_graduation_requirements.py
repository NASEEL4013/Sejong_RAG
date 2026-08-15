from sejong_rag.graduation_requirements import find_requirement, query_profile, structured_chunk

def test_query_profile_extracts_year_department_and_single_major():
    assert query_profile("2021년 입학생 소프트웨어학과 단일전공 졸업요건") == (2021, "소프트웨어학과", "single")

def test_verified_requirement_has_expected_courses_and_totals():
    item = find_requirement("2021년 입학생 소프트웨어학과 단일전공 졸업요건")
    assert item is not None
    assert sum(row["credits"] for row in item["foundational_liberal_arts"]) == 9
    assert [row["course"] for row in item["foundational_liberal_arts"]] == ["기초미적분학", "공업수학1", "통계학개론"]

def test_structured_chunk_preserves_pages_and_official_urls():
    chunk = structured_chunk("2021년 입학생 소프트웨어학과 단일전공 졸업요건")
    assert chunk is not None
    assert chunk["locators"] == [{"page": 58}, {"page": 60}]
    assert "Technical Writing 기초" in chunk["text"]
    assert chunk["source_url"].startswith("https://www.sejong.ac.kr/")

def test_credit_structured_chunk_selects_exact_year_department(tmp_path):
    from sejong_rag.graduation_requirements import credit_structured_chunk
    import json
    path = tmp_path / "credits.json"
    path.write_text(json.dumps({"items": [{"admission_year": 2022, "department": "소프트웨어학과", "column_count": 7, "common_required": 13, "balanced_required": 6, "foundational": 15, "major_total": 72, "major_required": 36, "major_elective": 36, "graduation_total": 130, "page": 56}]}, ensure_ascii=False), encoding="utf-8")
    chunk = credit_structured_chunk("2022년 입학생 소프트웨어학과 졸업요건", path)
    assert "공통교양필수=13학점" in chunk["text"]
    assert "전공합계=72학점" in chunk["text"]
    assert chunk["locators"] == [{"page": 56}]

def test_credit_structured_chunk_does_not_apply_single_table_to_multiple_major(tmp_path):
    from sejong_rag.graduation_requirements import credit_structured_chunk
    import json
    path = tmp_path / "credits.json"
    path.write_text(json.dumps({"items": [{"admission_year": 2022, "department": "소프트웨어학과"}]}, ensure_ascii=False), encoding="utf-8")
    assert credit_structured_chunk("2022년 입학생 소프트웨어학과 복수전공 졸업요건", path) is None

def test_software_department_alias_uses_new_name_from_2025():
    from sejong_rag.graduation_requirements import find_credit_requirement
    item = find_credit_requirement("2025년 입학생 소프트웨어학과 단일전공 졸업요건")
    assert item["department"] == "콘텐츠소프트웨어학과"
    assert item["query_department"] == "소프트웨어학과"
    assert item["major_area_total"] == 76

def test_content_software_alias_uses_historical_name_before_2025():
    from sejong_rag.graduation_requirements import find_credit_requirement
    item = find_credit_requirement("2024년 입학생 콘텐츠소프트웨어학과 단일전공 졸업요건")
    assert item["department"] == "소프트웨어학과"
    assert item["major_area_total"] == 75

def test_alias_is_explained_in_structured_evidence():
    from sejong_rag.graduation_requirements import credit_structured_chunk
    chunk = credit_structured_chunk("2026년 입학생 소프트웨어학과 단일전공 졸업요건")
    assert "학과 명칭 연결" in chunk["text"]
    assert "콘텐츠소프트웨어학과" in chunk["text"]
    assert "전공영역총계(A+B+C)=76학점" in chunk["text"]
