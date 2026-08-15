"""원본 이미지와 직접 대조한 OCR 페이지 교정값을 적용한다."""
from __future__ import annotations
import argparse, json
from datetime import date
from pathlib import Path
DEFAULT_OVERRIDES = Path("eval/ocr_page_overrides.json")
DEFAULT_OUTPUT = Path("data/processed/attachments")

def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def apply_overrides(overrides_path: Path = DEFAULT_OVERRIDES, output_dir: Path = DEFAULT_OUTPUT, verified_at: str | None = None) -> dict[str, object]:
    overrides=json.loads(overrides_path.read_text(encoding="utf-8")); verified_at=verified_at or date.today().isoformat(); changed=0
    for override in overrides:
        document_id=str(override["document_id"]); page=int(override["page"])
        records_path=output_dir/"records"/f"{document_id}.jsonl"; report_path=output_dir/"reports"/f"{document_id}.json"
        records=load_jsonl(records_path); matches=[r for r in records if int(r.get("locator",{}).get("page",-1))==page]
        if len(matches)!=1: raise ValueError(f"{document_id} page {page}: expected one record, found {len(matches)}")
        record=matches[0]; original_confidence=record.get("locator",{}).get("ocr_mean_confidence")
        text=str(override["text"]).strip(); record.update({"text":text,"text_length":len(text),"extraction_method":"human_verified_visual_transcription"})
        record["locator"].update({"ocr":True,"ocr_mean_confidence":original_confidence,"human_verified":True,"verified_at":verified_at,"verification_reason":override["reason"]})
        records_path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")
        report=json.loads(report_path.read_text(encoding="utf-8")); verified=list(report.get("recovery",{}).get("human_verified_pages",[])); verified=[x for x in verified if int(x["page"])!=page]; verified.append({"page":page,"verified_at":verified_at,"reason":override["reason"]}); report.setdefault("recovery",{})["human_verified_pages"]=sorted(verified,key=lambda x:int(x["page"])); report["record_count"]=len(records); report["text_length"]=sum(int(r["text_length"]) for r in records); report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); changed+=1
    return {"override_count":len(overrides),"applied":changed,"verified_at":verified_at}

def main():
    p=argparse.ArgumentParser(description="원본 이미지와 대조한 OCR 페이지 교정값을 적용합니다."); p.add_argument("--overrides",type=Path,default=DEFAULT_OVERRIDES); p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT); p.add_argument("--verified-at")
    a=p.parse_args(); print(json.dumps(apply_overrides(a.overrides,a.output_dir,a.verified_at),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
