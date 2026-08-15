"""공지 목록을 버리지 않고 정보 분야 메타데이터로 분류한다."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sejong_rag.list_notices import DEFAULT_CATALOG

DEFAULT_OUTPUT = Path("data/catalog/classified-notices.json")

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "scholarship": ("장학", "학자금", "근로학생"),
    "course_registration": (
        "수강", "강의", "수업", "교과목", "e-러닝", "이러닝", "k-mooc"
    ),
    "graduation": ("졸업", "학위", "수료", "조기졸업"),
    "credit": ("학점", "학점교류", "이수학점", "재수강"),
    "academic_status": ("휴학", "복학", "재입학", "제적", "자퇴", "전과", "학적"),
    "curriculum": ("교과과정", "교육과정", "전공", "복수전공", "부전공"),
    "tuition": ("등록금", "등록", "납부"),
    "certification": ("인증", "영어졸업", "공학인증"),
    "department": ("학과", "학부", "전공", "소프트웨어", "컴퓨터"),
}

BOARD_FALLBACK_CATEGORY = {
    "academic_notice": "academic_general",
    "scholarship_notice": "scholarship",
    "general_notice": "campus_general",
}


def classify_item(item: dict[str, object]) -> dict[str, object]:
    """제목과 게시판을 사용해 복수 카테고리와 근거 키워드를 붙인다."""
    title = str(item.get("title", ""))
    title_lower = title.lower()
    categories: list[str] = []
    matched_keywords: list[str] = []

    for category, keywords in CATEGORY_KEYWORDS.items():
        matched = [keyword for keyword in keywords if keyword.lower() in title_lower]
        if matched:
            categories.append(category)
            matched_keywords.extend(matched)

    fallback = BOARD_FALLBACK_CATEGORY.get(str(item.get("board")), "other")
    if fallback not in categories:
        categories.append(fallback)

    # 키워드가 하나라도 있으면 먼저 상세 내용을 확인할 고우선 후보로 본다.
    priority = "high" if matched_keywords else "normal"
    return {
        **item,
        "categories": categories,
        "matched_keywords": list(dict.fromkeys(matched_keywords)),
        "priority": priority,
        "classification_method": "title_keyword_v1",
    }


def classify_catalog(payload: dict[str, object]) -> dict[str, object]:
    items = [classify_item(item) for item in payload.get("items", [])]
    category_counts = Counter(
        category for item in items for category in item["categories"]
    )
    priority_counts = Counter(str(item["priority"]) for item in items)
    return {
        "source_catalog_collected_at": payload.get("collected_at"),
        "classification_method": "title_keyword_v1",
        "item_count": len(items),
        "category_counts": dict(sorted(category_counts.items())),
        "priority_counts": dict(sorted(priority_counts.items())),
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="공지 목록 전체를 보존하면서 정보 분야 메타데이터를 붙입니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = classify_catalog(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "item_count": result["item_count"],
                "category_counts": result["category_counts"],
                "priority_counts": result["priority_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
