"""최신 수강편람에서 입학년도·학과별 단일전공 학점표를 구조화한다."""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path
from sejong_rag.unified_hybrid import infer_admission_year_sections
from sejong_rag.vector_search import load_index

DEFAULT_INDEX = Path("data/index/sejong-2026-trusted-e5-small.npz")
DEFAULT_OUTPUT = Path("data/structured/single-major-credit-requirements.json")
DEFAULT_GUIDE = "2026-2학기 수강편람_20260730.pdf"
ROW = re.compile(r"([가-힣A-Za-z·‧()]+(?:학과|전공|학부))((?:\s+\d+){6,9})(?!\s+\d)")
SCHEMAS = {
    6: ("general_required", "foundational", "major_total", "major_required", "major_elective", "graduation_total"),
    7: ("common_required", "balanced_required", "foundational", "major_total", "major_required", "major_elective", "graduation_total"),
    9: ("common_required", "balanced_required", "academic_foundational", "major_foundational", "major_required", "major_elective", "major_core_total", "major_area_total", "graduation_total"),
}
VARIANTS = {6: "six_columns", 7: "seven_columns", 9: "nine_columns"}

def expected_columns(year: int) -> int:
    if year <= 2021:
        return 6
    if year <= 2023:
        return 7
    return 9

def _values(record: dict[str, object]) -> tuple[object, ...]:
    return tuple(record[label] for label in SCHEMAS[int(record["column_count"])])

def _valid_sums(record: dict[str, object]) -> bool:
    if record["column_count"] in (6, 7):
        return record["major_total"] == record["major_required"] + record["major_elective"]
    return (
        record["major_core_total"] == record["major_required"] + record["major_elective"]
        and record["major_area_total"] == record["major_foundational"] + record["major_core_total"]
    )

def extract_catalog(chunks: list[dict[str, object]], filename: str = DEFAULT_GUIDE) -> dict[str, object]:
    section_years = infer_admission_year_sections(chunks)
    table_pages: set[tuple[int, int]] = set()
    for index, chunk in enumerate(chunks):
        if chunk.get("filename") != filename:
            continue
        year = section_years.get(index)
        body = str(chunk.get("text", ""))
        if year is None or not ("단일전공 이수 시 학과별 졸업 이수학점" in body or "전공학점계" in body or "전공영역 총계" in body):
            continue
        table_pages.update((year, int(locator["page"])) for locator in chunk.get("locators", []) if "page" in locator)
    found: defaultdict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    rejected_rows: list[dict[str, object]] = []
    for index, chunk in enumerate(chunks):
        if chunk.get("filename") != filename:
            continue
        year = section_years.get(index)
        body = str(chunk.get("text", ""))
        pages = sorted({int(x["page"]) for x in chunk.get("locators", []) if "page" in x})
        has_table_marker = "단일전공 이수 시 학과별 졸업 이수학점" in body or "전공학점계" in body or "전공영역 총계" in body
        if year is None or not (any((year, page) in table_pages for page in pages) or (not pages and has_table_marker)):
            continue
        for match in ROW.finditer(body):
            department = match.group(1)
            values = tuple(map(int, match.group(2).split()))
            wanted = expected_columns(year)
            if len(values) != wanted:
                rejected_rows.append({"admission_year": year, "department": department, "expected_columns": wanted, "found_values": list(values), "page": pages[0] if pages else None, "chunk_id": chunk["chunk_id"]})
                continue
            labels = SCHEMAS[len(values)]
            record = {"admission_year": year, "department": department, "schema_variant": VARIANTS[len(values)], "column_count": len(values), **dict(zip(labels, values)), "page": pages[0] if pages else None, "chunk_id": chunk["chunk_id"]}
            if not _valid_sums(record):
                rejected_rows.append({**record, "reason": "major_sum_mismatch"})
                continue
            found[(year, department)].append(record)
    items, conflicts = [], []
    for key, records in sorted(found.items()):
        value_sets = {(record["schema_variant"], *_values(record)) for record in records}
        if len(value_sets) != 1:
            conflicts.append({"admission_year": key[0], "department": key[1], "records": records})
            continue
        canonical = dict(records[0])
        canonical["source_chunk_ids"] = sorted({str(row["chunk_id"]) for row in records})
        canonical.pop("chunk_id", None)
        items.append(canonical)
    return {"schema_version": 3, "source_filename": filename, "schema_variants": {VARIANTS[n]: list(labels) for n, labels in SCHEMAS.items()}, "items": items, "conflicts": conflicts, "rejected_rows": rejected_rows}

def main() -> None:
    parser = argparse.ArgumentParser(description="입학년도·학과별 단일전공 학점표 구조화")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    _, chunks, _ = load_index(args.index)
    catalog = extract_catalog(chunks)
    if catalog["conflicts"]:
        raise RuntimeError(f"학과별 학점 충돌 {len(catalog['conflicts'])}건: 자동 저장을 중단합니다.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"저장: {args.output}")
    print(f"구조화 행: {len(catalog['items'])}, 검증 제외: {len(catalog['rejected_rows'])}")
    print(f"입학년도: {sorted({item['admission_year'] for item in catalog['items']})}")
if __name__ == "__main__":
    main()
