"""공식 첨부파일의 수동 다운로드 체크리스트를 생성한다."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sejong_rag.import_attachments import (
    DEFAULT_DETAILS_DIR,
    DEFAULT_INPUT_DIR,
    load_expected_attachments,
)

DEFAULT_POLICY = Path("data/catalog/document-policy.json")
DEFAULT_OUTPUT = Path("docs/ATTACHMENT_DOWNLOAD_CHECKLIST.md")


def imported_attachment_ids(output_dir: Path) -> set[str]:
    result = {
        path.stem
        for directory in ("imported-attachments", "downloaded-attachments")
        for path in (output_dir / "metadata" / directory).glob("*.json")
    }
    for path in (output_dir / "metadata").glob("*.json"):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for item in metadata.get("processed_attachments", []):
            if item.get("document_id"):
                result.add(str(item["document_id"]))
    return result


def build_checklist(
    expected: list[dict[str, object]],
    policies: dict[str, dict[str, object]],
    completed: set[str],
    input_dir: Path,
) -> str:
    by_parent: dict[str, list[dict[str, object]]] = {}
    for item in expected:
        by_parent.setdefault(str(item["parent_document_id"]), []).append(item)
    type_counts = Counter(str(item.get("file_type", "unknown")) for item in expected)

    lines = [
        "# 공식 첨부파일 수동 다운로드 체크리스트",
        "",
        "> 생성 기준: 상세 HTML에서 발견한 모든 첨부를 포함한다. 파일명만으로 중복 제외하지 않는다.",
        "",
        f"- 전체: {len(expected)}개",
        f"- 확보 완료: {sum(item['attachment_id'] in completed for item in expected)}개",
        f"- 남음: {sum(item['attachment_id'] not in completed for item in expected)}개",
        "- 형식: " + ", ".join(f"{name.upper()} {count}개" for name, count in sorted(type_counts.items())),
        "",
        "다운로드한 파일은 아래 `저장 위치` 그대로 둔다. 같은 파일명이 여러 공지에 있으면 공지 ID 폴더가 충돌을 방지한다.",
        "",
    ]
    for parent_id, items in by_parent.items():
        policy = policies.get(parent_id, {})
        title = str(items[0]["parent_title"])
        lines.extend(
            [
                f"## {title}",
                "",
                f"- 문서 ID: `{parent_id}`",
                f"- 시점: {policy.get('academic_year') or '미확정'}-{policy.get('semester') or '미확정'}",
                f"- 버전 상태: `{policy.get('version_status', '미판정')}`",
                f"- [공식 공지 페이지]({items[0]['source_url']})",
                "",
            ]
        )
        for item in items:
            done = str(item["attachment_id"]) in completed
            suggested = input_dir / parent_id / str(item["filename"])
            lines.extend(
                [
                    f"- [{'x' if done else ' '}] **{item['filename']}**",
                    f"  - 첨부 ID: `{item['attachment_id']}`",
                    f"  - [직접 다운로드 링크]({item['download_url']})",
                    f"  - 저장 위치: `{suggested}`",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="수동 첨부 다운로드 체크리스트 생성")
    parser.add_argument("--details-dir", type=Path, default=DEFAULT_DETAILS_DIR)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    expected = load_expected_attachments(args.details_dir)
    policy_payload = json.loads(args.policy.read_text(encoding="utf-8"))
    policies = {str(item["document_id"]): item for item in policy_payload["items"]}
    completed = imported_attachment_ids(args.output_dir)
    content = build_checklist(expected, policies, completed, args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "total": len(expected),
                "completed": sum(item["attachment_id"] in completed for item in expected),
                "remaining": sum(item["attachment_id"] not in completed for item in expected),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
