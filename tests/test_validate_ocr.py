import json
from pathlib import Path
from sejong_rag.validate_ocr import apply_overrides

def test_apply_override_marks_record_and_report(tmp_path: Path):
    out=tmp_path/"attachments"; (out/"records").mkdir(parents=True); (out/"reports").mkdir()
    record={"record_id":"d-record-1","document_id":"d","text":"bad","text_length":3,"locator":{"page":2,"ocr":True,"ocr_mean_confidence":40.0},"extraction_method":"ocr"}
    (out/"records/d.jsonl").write_text(json.dumps(record)+"\n")
    (out/"reports/d.json").write_text(json.dumps({"document_id":"d","record_count":1,"text_length":3,"recovery":{}}))
    overrides=tmp_path/"overrides.json"; overrides.write_text(json.dumps([{"document_id":"d","page":2,"text":"정확한 문구","reason":"visual"}]))
    result=apply_overrides(overrides,out,"2026-08-09")
    updated=json.loads((out/"records/d.jsonl").read_text())
    report=json.loads((out/"reports/d.json").read_text())
    assert result["applied"]==1
    assert updated["text"]=="정확한 문구"
    assert updated["locator"]["human_verified"] is True
    assert updated["locator"]["ocr_mean_confidence"]==40.0
    assert report["recovery"]["human_verified_pages"][0]["page"]==2
