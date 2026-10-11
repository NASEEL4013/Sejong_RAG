"""구형 DOC와 이미지 PDF 페이지를 로컬 변환/OCR로 보완한다."""
from __future__ import annotations
import argparse, csv, io, json, re, subprocess, tempfile
from collections import Counter
from pathlib import Path
from sejong_rag.extract_attachments import clean_text, extract_docx, load_attachment_metadata
DEFAULT_DATA_DIR = Path("data")
EMPTY_PAGE_PATTERN = re.compile(r"page (\d+): no embedded text")

def run_command(command: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        detail = clean_text(result.stderr) or clean_text(result.stdout)
        raise RuntimeError(f"command failed ({result.returncode}): {detail}")
    return result

def convert_doc_to_docx(source: Path, work_dir: Path) -> Path:
    output_dir, profile = work_dir / "output", work_dir / "profile"
    output_dir.mkdir(parents=True, exist_ok=True); profile.mkdir(parents=True, exist_ok=True)
    run_command(["libreoffice", f"-env:UserInstallation={profile.resolve().as_uri()}", "--headless", "--norestore", "--nodefault", "--nolockcheck", "--convert-to", "docx", "--outdir", str(output_dir), str(source.resolve())])
    converted = output_dir / f"{source.stem}.docx"
    if not converted.is_file() or converted.stat().st_size == 0:
        raise RuntimeError("LibreOffice did not create a non-empty DOCX")
    return converted

def parse_tesseract_tsv(tsv: str) -> tuple[str, float | None]:
    lines, current_key, current_words, confidences = [], None, [], []
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
        word = clean_text(row.get("text", ""))
        if not word: continue
        key = (row.get("page_num"), row.get("block_num"), row.get("par_num"), row.get("line_num"))
        if current_key is not None and key != current_key and current_words:
            lines.append(" ".join(current_words)); current_words = []
        current_key = key; current_words.append(word)
        try:
            confidence = float(row.get("conf", "-1"))
            if confidence >= 0: confidences.append(confidence)
        except ValueError: pass
    if current_words: lines.append(" ".join(current_words))
    text = clean_text("\n".join(lines))
    mean = sum(confidences) / len(confidences) if confidences else None
    return text, mean

def ocr_pdf_page(pdf_path: Path, page: int, work_dir: Path, dpi: int = 300) -> tuple[str, float | None]:
    prefix = work_dir / f"page-{page}"
    run_command(["pdftoppm", "-f", str(page), "-l", str(page), "-singlefile", "-r", str(dpi), "-png", str(pdf_path.resolve()), str(prefix)])
    image = prefix.with_suffix(".png")
    if not image.is_file(): raise RuntimeError(f"page renderer produced no image for page {page}")
    result = run_command(["tesseract", str(image), "stdout", "-l", "kor+eng", "--psm", "6", "tsv"])
    return parse_tesseract_tsv(result.stdout)

def _read_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def _write_records(path: Path, document_id: str, records: list[dict[str, object]]) -> None:
    records.sort(key=lambda item: (int(item.get("locator", {}).get("page", 0)), str(item.get("record_id", ""))))
    with path.open("w", encoding="utf-8") as output:
        for number, record in enumerate(records, 1):
            record["record_id"] = f"{document_id}-record-{number}"
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

def _base_record(metadata: dict[str, object], text: str, locator: dict[str, object], method: str) -> dict[str, object]:
    return {"record_id": "", "document_id": metadata["document_id"], "parent_document_id": metadata.get("parent_document_id"), "filename": metadata.get("filename"), "file_type": metadata.get("file_type"), "source_url": metadata.get("source_url"), "attachment_url": metadata.get("attachment_url"), "text": text, "text_length": len(text), "locator": locator, "extraction_method": method}

def recover_doc(metadata: dict[str, object], output_dir: Path) -> dict[str, object]:
    document_id, source = str(metadata["document_id"]), Path(str(metadata["resolved_raw_path"]))
    records_path, report_path = output_dir / "records" / f"{document_id}.jsonl", output_dir / "reports" / f"{document_id}.json"
    with tempfile.TemporaryDirectory(prefix="sejong-doc-") as temporary:
        converted = convert_doc_to_docx(source, Path(temporary)); extracted, warnings = extract_docx(converted)
    records = [_base_record(metadata, str(item["text"]), {"converted_from": "doc", **item["locator"]}, "libreoffice_doc_to_docx+" + str(item["extraction_method"])) for item in extracted]
    _write_records(records_path, document_id, records)
    report = json.loads(report_path.read_text(encoding="utf-8")); report.update({"status": "success" if records and not warnings else "review_required", "record_count": len(records), "text_length": sum(int(x["text_length"]) for x in records), "warnings": warnings, "error": None, "recovery": {"method": "libreoffice_doc_to_docx", "converted_format": "docx", "source_preserved": True}})
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

def missing_text_pages(report: dict[str, object]) -> list[int]:
    return sorted({int(match.group(1)) for warning in report.get("warnings", []) if (match := EMPTY_PAGE_PATTERN.search(str(warning)))})

def recover_pdf(metadata: dict[str, object], output_dir: Path, dpi: int = 300, max_pages: int = 8) -> dict[str, object]:
    document_id, source = str(metadata["document_id"]), Path(str(metadata["resolved_raw_path"]))
    records_path, report_path = output_dir / "records" / f"{document_id}.jsonl", output_dir / "reports" / f"{document_id}.json"
    report = json.loads(report_path.read_text(encoding="utf-8")); pages = missing_text_pages(report)[:max_pages]
    previous_recovery = report.get("recovery", {})
    previous_results = list(previous_recovery.get("ocr_pages", []))
    previous_failures = list(previous_recovery.get("failures", []))
    existing = [r for r in _read_records(records_path) if int(r.get("locator", {}).get("page", -1)) not in pages]
    results, failures = [], []
    with tempfile.TemporaryDirectory(prefix="sejong-ocr-") as temporary:
        for page in pages:
            try:
                text, confidence = ocr_pdf_page(source, page, Path(temporary), dpi)
                if not text: raise RuntimeError("OCR returned no text")
                existing.append(_base_record(metadata, text, {"page": page, "ocr": True, "ocr_mean_confidence": confidence}, "pdftoppm+tesseract_kor_eng"))
                results.append({"page": page, "text_length": len(text), "mean_confidence": confidence})
            except Exception as exc: failures.append({"page": page, "error": f"{type(exc).__name__}: {exc}"})
    _write_records(records_path, document_id, existing)
    completed_pages = {x["page"] for x in results}
    other_warnings = [w for w in report.get("warnings", []) if not (m := EMPTY_PAGE_PATTERN.search(str(w))) or int(m.group(1)) not in completed_pages]
    other_warnings.extend(f"page {x['page']}: OCR text requires source verification (mean confidence {x['mean_confidence']})" for x in results)
    report.update({"status": "review_required" if other_warnings or failures else "success", "record_count": len(existing), "text_length": sum(int(x["text_length"]) for x in existing), "warnings": other_warnings, "error": None if not failures else f"OCR failed on {len(failures)} page(s)", "recovery": {"method": "pdftoppm_300dpi+tesseract_kor_eng", "ocr_pages": previous_results + results, "failures": previous_failures + failures, "source_preserved": True}})
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

def repair_ocr_history(report: dict[str, object], output_dir: Path) -> dict[str, object]:
    records_path = Path(str(report["records_path"]))
    if not records_path.is_absolute():
        records_path = output_dir.parents[2] / records_path
    ocr_records = [item for item in _read_records(records_path) if item.get("locator", {}).get("ocr")]
    if not ocr_records:
        return report
    recovery = dict(report.get("recovery", {}))
    known = {int(item["page"]): item for item in recovery.get("ocr_pages", [])}
    for item in ocr_records:
        locator = item["locator"]; page = int(locator["page"])
        known[page] = {"page": page, "text_length": int(item["text_length"]), "mean_confidence": locator.get("ocr_mean_confidence")}
    recovery.update({"method": "pdftoppm_300dpi+tesseract_kor_eng", "ocr_pages": [known[p] for p in sorted(known)], "failures": recovery.get("failures", []), "source_preserved": True})
    report["recovery"] = recovery
    report_path = output_dir / "reports" / (str(report["document_id"]) + ".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

def recover_all(data_dir: Path = DEFAULT_DATA_DIR, output_dir: Path | None = None, dpi: int = 300, max_ocr_pages: int = 8) -> dict[str, object]:
    output_dir = output_dir or data_dir / "processed/attachments"
    metadata = {str(x["document_id"]): x for x in load_attachment_metadata(data_dir)}
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output_dir / "reports").glob("*.json"))]
    reports = [repair_ocr_history(report, output_dir) if report["file_type"] == "pdf" else report for report in reports]
    results = []
    for report in reports:
        item = metadata[str(report["document_id"])]
        if report["file_type"] == "doc" and report["status"] == "failed": results.append(recover_doc(item, output_dir))
        elif report["file_type"] == "pdf" and missing_text_pages(report): results.append(recover_pdf(item, output_dir, dpi, max_ocr_pages))
    return {"recovered_attachments": len(results), "status_counts": dict(Counter(str(x["status"]) for x in results)), "doc_conversions": sum(x["file_type"] == "doc" for x in results), "ocr_documents": sum(x["file_type"] == "pdf" for x in results), "ocr_pages": sum(len(x.get("recovery", {}).get("ocr_pages", [])) for x in results), "ocr_failures": sum(len(x.get("recovery", {}).get("failures", [])) for x in results)}

def main():
    parser=argparse.ArgumentParser(description="구형 DOC와 이미지 PDF 페이지를 로컬 변환/OCR로 보완합니다."); parser.add_argument("--data-dir",type=Path,default=DEFAULT_DATA_DIR); parser.add_argument("--output-dir",type=Path); parser.add_argument("--dpi",type=int,default=300); parser.add_argument("--max-ocr-pages",type=int,default=8)
    args=parser.parse_args(); print(json.dumps(recover_all(args.data_dir,args.output_dir,args.dpi,args.max_ocr_pages),ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
