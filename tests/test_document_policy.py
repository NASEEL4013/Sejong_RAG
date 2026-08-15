from sejong_rag.document_policy import (
    apply_document_policy,
    extract_academic_period,
    infer_document_family,
)


def notice(title, article, date, attachments=None):
    return {
        "document_id": f"notice3-{article}",
        "article_no": str(article),
        "title": title,
        "published_at": date,
        "attachments": attachments or [],
    }


def test_extract_academic_period_supports_long_and_short_years():
    assert extract_academic_period("2026학년도 2학기 수강편람") == (2026, 2)
    assert extract_academic_period("26-1학기 수강편람") == (2026, 1)
    assert extract_academic_period("2026-2 국가장학금 안내") == (2026, 2)


def test_document_family_uses_explainable_rules():
    assert infer_document_family("2026-2학기 수강편람")[0] == "course_guide"
    assert infer_document_family("알 수 없는 행사")[1] == "low"


def test_newer_semester_is_current_but_history_is_preserved():
    items = apply_document_policy(
        [
            notice("2026-1학기 수강편람", 1, "2026.01.01"),
            notice("2026-2학기 수강편람", 2, "2026.07.30"),
        ]
    )

    assert len(items) == 2
    assert items[0]["version_status"] == "historical"
    assert items[1]["version_status"] == "current_candidate"


def test_any_attachment_is_required_until_content_review():
    items = apply_document_policy(
        [notice("2026-2학기 수강편람", 2, "2026.07.30", [{"filename": "a.pdf"}])]
    )

    assert items[0]["attachment_requirement"] == "required"


def test_low_confidence_family_requires_review_not_latest_guess():
    item = apply_document_policy([notice("학생 대상 특별 안내", 3, "2026.08.01")])[0]
    assert item["version_status"] == "review_required"


def test_related_but_different_purpose_documents_are_separate_families():
    general, _ = infer_document_family("2026-2학기 수강신청 안내")
    english, _ = infer_document_family("2026년도 2학기 글로벌잉글리쉬1 수강신청 안내")
    outgoing, _ = infer_document_family("2026-2학기 학점교류 안내(본교 → 타 대학교)")
    incoming, _ = infer_document_family("2026-2학기 학점교류생(타교생) 학번 안내")
    assert general != english
    assert outgoing != incoming


def test_scholarship_application_result_and_payment_are_not_versions():
    application, _ = infer_document_family("2026-1 성적향상장학금 신청안내")
    result, _ = infer_document_family("2026-1 성적향상장학금 선발결과 안내")
    payment, _ = infer_document_family("2026-1 국가장학금 지급안내")
    national_application, _ = infer_document_family("2026-2 국가장학금 신청안내")
    assert application != result
    assert payment != national_application


def test_extract_period_supports_year_wording():
    assert extract_academic_period("2026년도 2학기 글로벌잉글리쉬") == (2026, 2)
