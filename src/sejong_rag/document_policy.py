"""문서 시점·문서군과 보수적인 첨부 필요성을 판정한다."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DEFAULT_DETAILS_DIR = Path("data/metadata/details")
DEFAULT_OUTPUT = Path("data/catalog/document-policy.json")

FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("course_guide", ("수강편람",)),
    ("course_registration", ("수강신청",)),
    ("tuition_payment", ("등록금", "납부")),
    ("early_graduation", ("조기졸업",)),
    ("graduation_review", ("졸업", "심사")),
    ("leave_return", ("휴학", "복학")),
    ("credit_exchange", ("학점교류",)),
    ("national_scholarship", ("국가장학금",)),
    ("grade_improvement_scholarship", ("성적향상장학금",)),
    ("housing_scholarship", ("주거", "장학금")),
)


def extract_academic_period(title: str) -> tuple[int | None, int | None]:
    """제목에서 학년도와 학기를 보수적으로 추출한다."""
    year_match = re.search(r"(?<!\d)(20\d{2})(?!\d)", title)
    if year_match:
        year = int(year_match.group(1))
    else:
        short = re.search(r"(?<!\d)(\d{2})\s*-\s*[12](?:\s*학기)?", title)
        year = 2000 + int(short.group(1)) if short else None

    semester = None
    patterns = (
        r"(?:20)?\d{2}\s*-\s*([12])\s*학기",
        r"20\d{2}(?:학?년도)?\s*([12])\s*학기",
        r"20\d{2}\s*-\s*([12])(?=\s|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            semester = int(match.group(1))
            break
    return year, semester


def infer_document_family(title: str) -> tuple[str, str]:
    """설명 가능한 핵심어 규칙으로 같은 종류의 문서를 묶는다."""
    lowered = title.lower()

    if "수강신청" in lowered:
        if "글로벌잉글리쉬" in lowered:
            return "course_registration:global_english", "high"
        if "군" in lowered and ("e-러닝" in lowered or "원격강좌" in lowered):
            return "course_registration:military_elearning", "high"
        return "course_registration:general", "high"

    if "학점교류" in lowered:
        if "타교생" in lowered:
            return "credit_exchange:incoming_student_id", "high"
        if "본교" in lowered and "타" in lowered:
            return "credit_exchange:outgoing", "high"
        return "credit_exchange:general", "high"

    scholarship_bases = (
        ("national_scholarship", "국가장학금"),
        ("grade_improvement_scholarship", "성적향상장학금"),
        ("housing_scholarship", "주거안정지원장학금"),
    )
    for base, keyword in scholarship_bases:
        if keyword in lowered:
            action = (
                "result" if ("결과" in lowered or "선발결과" in lowered)
                else "payment" if "지급" in lowered
                else "application" if ("신청" in lowered or "선발" in lowered)
                else "general"
            )
            return f"{base}:{action}", "high"

    for family, keywords in FAMILY_RULES:
        if all(keyword.lower() in lowered for keyword in keywords):
            return family, "high"

    # 자동 최신 선택에 쓰지 않을 저신뢰 fallback이다.
    korean = re.sub(r"[^가-힣]+", " ", title)
    tokens = [token for token in korean.split() if len(token) >= 2]
    fallback = "other:" + "-".join(tokens[:5]) if tokens else "other:unknown"
    return fallback, "low"


def version_key(item: dict[str, object]) -> tuple[int, int, str, int]:
    year = int(item["academic_year"] or 0)
    semester = int(item["semester"] or 0)
    published = str(item.get("published_at") or "")
    article = str(item.get("article_no") or "0")
    return year, semester, published, int(article) if article.isdigit() else 0


def apply_document_policy(notices: list[dict[str, object]]) -> list[dict[str, object]]:
    enriched = []
    for notice in notices:
        year, semester = extract_academic_period(str(notice.get("title", "")))
        family, confidence = infer_document_family(str(notice.get("title", "")))
        attachments = list(notice.get("attachments", []))
        enriched.append(
            {
                **notice,
                "academic_year": year,
                "semester": semester,
                "document_family": family,
                "family_confidence": confidence,
                "attachment_requirement": "required" if attachments else "none",
                "attachment_reason": (
                    "정확도 우선 정책: 첨부가 있으면 내용 확인 전까지 모두 필요"
                    if attachments
                    else "상세 공지에 첨부가 없음"
                ),
                "version_status": "review_required" if confidence == "low" else "unresolved",
            }
        )

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in enriched:
        if item["family_confidence"] == "high":
            groups[str(item["document_family"])].append(item)
    for items in groups.values():
        newest = max(items, key=version_key)
        for item in items:
            item["version_status"] = (
                "current_candidate" if item is newest else "historical"
            )
    return enriched


def load_details(details_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(details_dir.glob("*.json"))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="공지의 문서군·학년도·학기·첨부 필요성과 현재 후보를 판정합니다."
    )
    parser.add_argument("--details-dir", type=Path, default=DEFAULT_DETAILS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    items = apply_document_policy(load_details(args.details_dir))
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "policy": "accuracy_first_v1",
        "item_count": len(items),
        "required_attachment_notices": sum(
            item["attachment_requirement"] == "required" for item in items
        ),
        "review_required": sum(item["version_status"] == "review_required" for item in items),
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: payload[key] for key in (
                "output", "item_count", "required_attachment_notices", "review_required"
            ) if key in payload} | {"output": str(args.output)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
