"""사용자가 직접 받은 공식 첨부파일을 부모 공지와 연결해 로컬 처리한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from sejong_rag.ingest import extract_pdf_pages

DEFAULT_INPUT_DIR = Path("data/manual-input")
DEFAULT_DETAILS_DIR = Path("data/metadata/details")
DEFAULT_OUTPUT_DIR = Path("data")


def normalize_filename(filename: str) -> str:
    """운영체제의 유니코드 표현 차이와 연속 공백만 정규화한다."""
    normalized = unicodedata.normalize("NFC", filename).strip()
    return " ".join(normalized.split()).casefold()


def load_expected_attachments(details_dir: Path) -> list[dict[str, object]]:
    """상세 메타데이터에서 공식 첨부 목록과 부모 공지 연결을 읽는다."""
    expected: list[dict[str, object]] = []
    for path in sorted(details_dir.glob("*.json")):
        notice = json.loads(path.read_text(encoding="utf-8"))
        for attachment in notice.get("attachments", []):
            attach_no = str(attachment.get("attach_no") or "unknown")
            attachment_id = f"{notice['document_id']}-attachment-{attach_no}"
            normalized = normalize_filename(str(attachment["filename"]))
            expected.append(
                {
                    **attachment,
                    "attachment_id": attachment_id,
                    "parent_document_id": notice["document_id"],
                    "parent_title": notice["title"],
                    "source_url": notice["source_url"],
                    "normalized_filename": normalized,
                }
            )
    return expected


def top_level_files(input_dir: Path) -> dict[str, Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        normalized = normalize_filename(path.name)
        if normalized in result:
            raise ValueError(f"정규화 후 같은 이름인 입력 파일이 둘 이상입니다: {path.name}")
        result[normalized] = path
    return result


def build_manifest(
    expected: list[dict[str, object]], input_dir: Path
) -> list[dict[str, object]]:
    available = top_level_files(input_dir)
    name_counts: dict[str, int] = {}
    for item in expected:
        name = str(item["normalized_filename"])
        name_counts[name] = name_counts.get(name, 0) + 1
    rows = []
    for item in expected:
        suggested = input_dir / str(item["parent_document_id"]) / str(item["filename"])
        local = suggested if suggested.is_file() else None
        normalized = str(item["normalized_filename"])
        if local is None and name_counts[normalized] == 1:
            local = available.get(normalized)
        supported = str(item.get("file_type", "")).lower() == "pdf"
        rows.append(
            {
                **item,
                "local_path": str(local) if local else None,
                "suggested_path": str(suggested),
                "filename_is_duplicated": name_counts[normalized] > 1,
                "status": "ready" if local and supported else "unsupported" if local else "missing",
            }
        )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_pdf(item: dict[str, object], output_dir: Path) -> dict[str, object]:
    source_path = Path(str(item["local_path"]))
    if source_path.suffix.lower() != ".pdf":
        raise ValueError(f"현재 PDF만 처리할 수 있습니다: {source_path.name}")
    if source_path.read_bytes()[:5] != b"%PDF-":
        raise ValueError(f"확장자는 PDF지만 PDF 서명이 없습니다: {source_path.name}")

    attachment_id = str(item["attachment_id"])
    raw_path = output_dir / "raw" / "manual-attachments" / f"{attachment_id}.pdf"
    processed_path = output_dir / "processed" / "pdf" / f"{attachment_id}.jsonl"
    metadata_path = output_dir / "metadata" / "imported-attachments" / f"{attachment_id}.json"
    content_hash = sha256_file(source_path)

    old_hash = None
    if metadata_path.exists():
        old_hash = json.loads(metadata_path.read_text(encoding="utf-8")).get("content_hash")
    change_status = (
        "new" if old_hash is None else "unchanged" if old_hash == content_hash else "updated"
    )

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, raw_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    page_count = 0
    empty_pages = 0
    with processed_path.open("w", encoding="utf-8") as output:
        for page in extract_pdf_pages(raw_path):
            page_count += 1
            empty_pages += int(not page["extraction_ok"])
            output.write(
                json.dumps(
                    {
                        "document_id": attachment_id,
                        "parent_document_id": item["parent_document_id"],
                        "source_url": item["source_url"],
                        "attachment_url": item["download_url"],
                        "filename": item["filename"],
                        **page,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metadata = {
        "document_id": attachment_id,
        "parent_document_id": item["parent_document_id"],
        "parent_title": item["parent_title"],
        "source_url": item["source_url"],
        "attachment_url": item["download_url"],
        "attach_no": item["attach_no"],
        "filename": item["filename"],
        "file_type": item["file_type"],
        "manual_input_path": str(source_path),
        "raw_path": str(raw_path),
        "processed_path": str(processed_path),
        "content_hash": content_hash,
        "change_status": change_status,
        "page_count": page_count,
        "empty_text_pages": empty_pages,
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "acquisition_method": "user_manual_download",
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="직접 다운로드한 공식 PDF를 부모 공지와 연결하고 텍스트를 추출합니다."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--details-dir", type=Path, default=DEFAULT_DETAILS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--list", action="store_true", help="필요한 파일과 공식 링크만 표시")
    args = parser.parse_args()

    expected = load_expected_attachments(args.details_dir)
    manifest = build_manifest(expected, args.input_dir)
    if args.list:
        print(
            json.dumps(
                [
                    {
                        "status": item["status"],
                        "filename": item["filename"],
                        "parent_title": item["parent_title"],
                        "notice_url": item["source_url"],
                        "attachment_url": item["download_url"],
                        "suggested_path": item["suggested_path"],
                        "filename_is_duplicated": item["filename_is_duplicated"],
                    }
                    for item in manifest
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    imported = [
        import_pdf(item, args.output_dir)
        for item in manifest
        if item["status"] == "ready"
    ]
    print(
        json.dumps(
            {
                "expected": len(manifest),
                "ready": sum(item["status"] == "ready" for item in manifest),
                "missing": sum(item["status"] == "missing" for item in manifest),
                "unsupported": sum(item["status"] == "unsupported" for item in manifest),
                "imported": [
                    {
                        "document_id": item["document_id"],
                        "status": item["change_status"],
                        "page_count": item["page_count"],
                        "empty_text_pages": item["empty_text_pages"],
                    }
                    for item in imported
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
