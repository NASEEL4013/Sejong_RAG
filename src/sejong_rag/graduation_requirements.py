"""사람이 좌표 대조한 입학년도·학과별 졸업 과목 요건을 검색 근거로 제공한다."""
from __future__ import annotations
import json, re
from pathlib import Path
DEFAULT_REQUIREMENTS = Path("data/structured/graduation-requirements.json")

def query_profile(query: str) -> tuple[int | None, str | None, str | None]:
    year = re.search(r"(20\d{2})(?:년|학년도)?\s*(?:입학(?:생|자)?|입학생)", query)
    department = re.search(r"([가-힣A-Za-z]+학과)", query)
    major_type = "single" if "단일전공" in query else "multiple" if "복수전공" in query else None
    return (int(year.group(1)) if year else None, department.group(1) if department else None, major_type)

def find_requirement(query: str, path: Path = DEFAULT_REQUIREMENTS) -> dict[str, object] | None:
    year, department, major_type = query_profile(query)
    if not year or not department or not major_type or not path.is_file(): return None
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    return next((item for item in items if item["admission_year"] == year and item["department"] == department and item["major_type"] == major_type), None)

def structured_chunk(query: str, path: Path = DEFAULT_REQUIREMENTS) -> dict[str, object] | None:
    item = find_requirement(query, path)
    if not item: return None
    lines = [f"검증된 과목별 졸업요건: {item['admission_year']}학년도 {item['department']} 단일전공", "지정 교양선택:"]
    for row in item["designated_general_electives"]: lines.append(f"- {row['course']}: {row['credits']}학점, {row['recommended_term']}")
    lines.append("학문기초교양:")
    for row in item["foundational_liberal_arts"]: lines.append(f"- {row['course']}: {row['credits']}학점, {row['recommended_term']}")
    lines.extend(f"주의: {note}" for note in item["notes"])
    return {"chunk_id": f"verified-graduation-{item['admission_year']}-{item['department']}-single", "document_id": item["source_document_id"], "parent_document_id": "notice3-891086", "document_title": item["source_title"], "filename": item["filename"], "file_type": "structured_verified", "quality_tier": "trusted", "locators": [{"page": 58}, {"page": 60}], "source_url": item["source_url"], "attachment_url": item["attachment_url"], "text": "\n".join(lines), "extraction_methods": [item["extraction_method"]], "context_relation": "verified_structured_table", "retrieval_rank": 0}

DEFAULT_CREDIT_CATALOG = Path("data/structured/single-major-credit-requirements.json")
GUIDE_SOURCE_URL = "https://www.sejong.ac.kr/kor/intro/notice3.do?mode=view&articleNo=891086&article.offset=0&articleLimit=10"
GUIDE_ATTACHMENT_URL = "https://www.sejong.ac.kr/kor/intro/notice3.do?mode=download&articleNo=891086&attachNo=236497"

def official_credit_department(year: int, department: str) -> str:
    """명칭 변경 연도를 기준으로 수강편람 학점표의 공식 학과명을 반환한다."""
    if department not in {"소프트웨어학과", "콘텐츠소프트웨어학과"}:
        return department
    return "콘텐츠소프트웨어학과" if year >= 2025 else "소프트웨어학과"

def find_credit_requirement(query: str, path: Path = DEFAULT_CREDIT_CATALOG) -> dict[str, object] | None:
    """질문의 입학년도·학과와 일치하는 검증된 단일전공 학점 행을 찾는다."""
    year, department, major_type = query_profile(query)
    if not year or not department or major_type == "multiple" or not path.is_file():
        return None
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    official_department = official_credit_department(year, department)
    matched = next((item for item in items if item["admission_year"] == year and item["department"] == official_department), None)
    if matched is None:
        return None
    result = dict(matched)
    result["query_department"] = department
    return result

def credit_structured_chunk(query: str, path: Path = DEFAULT_CREDIT_CATALOG) -> dict[str, object] | None:
    """검증된 단일전공 학점 행을 검색 문맥과 같은 형태로 만든다."""
    item = find_credit_requirement(query, path)
    if not item:
        return None
    query_department = str(item.get("query_department", item["department"]))
    lines = [f"검증된 단일전공 졸업 이수학점표: {item['admission_year']}학년도 {query_department}"]
    if query_department != item["department"]:
        lines.append(f"학과 명칭 연결: 질문의 {query_department}는 해당 연도 공식 표의 {item['department']}로 조회함")
    if item["column_count"] == 6:
        lines.append(f"교양필수={item['general_required']}학점")
        lines.append(f"학문기초교양={item['foundational']}학점")
        lines.append(f"전공합계={item['major_total']}학점, 전공필수={item['major_required']}학점, 전공선택={item['major_elective']}학점")
    elif item["column_count"] == 7:
        lines.append(f"공통교양필수={item['common_required']}학점, 균형교양필수={item['balanced_required']}학점, 학문기초교양={item['foundational']}학점")
        lines.append(f"전공합계={item['major_total']}학점, 전공필수={item['major_required']}학점, 전공선택={item['major_elective']}학점")
    else:
        lines.append(f"공통교양필수={item['common_required']}학점, 균형교양필수={item['balanced_required']}학점, 학문기초교양={item['academic_foundational']}학점")
        lines.append(f"전공기초(A)={item['major_foundational']}학점, 전공필수(B)={item['major_required']}학점, 전공선택(C)={item['major_elective']}학점")
        lines.append(f"전공소계(B+C)={item['major_core_total']}학점, 전공영역총계(A+B+C)={item['major_area_total']}학점")
    lines.append(f"졸업학점={item['graduation_total']}학점")
    lines.append("주의: 이 행은 단일전공 이수 기준이며, 과목별 필수 이수 여부와 별도 인증 요건은 다른 근거도 함께 확인해야 함.")
    return {
        "chunk_id": f"verified-credit-{item['admission_year']}-{item['department']}-single",
        "document_id": "notice3-891086-attachment-236497",
        "parent_document_id": "notice3-891086",
        "document_title": "2026-2학기 수강편람 및 강의시간표 안내",
        "filename": "2026-2학기 수강편람_20260730.pdf",
        "file_type": "structured_verified",
        "quality_tier": "trusted",
        "locators": [{"page": item["page"]}],
        "source_url": GUIDE_SOURCE_URL,
        "attachment_url": GUIDE_ATTACHMENT_URL,
        "text": "\n".join(lines),
        "extraction_methods": ["pdftotext_layout_schema_validated"],
        "context_relation": "verified_single_major_credit_table",
        "retrieval_rank": 0,
    }
