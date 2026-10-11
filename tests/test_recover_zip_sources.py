import json
import zipfile
from sejong_rag.recover_zip_sources import recover_zip


def test_zip_pdf_ocr_preserves_member_source_and_review(tmp_path,monkeypatch):
    source=tmp_path/'source.zip'
    with zipfile.ZipFile(source,'w') as archive: archive.writestr('guide.pdf',b'pdf')
    output=tmp_path/'out';(output/'reports').mkdir(parents=True);(output/'records').mkdir()
    records=output/'records/doc.jsonl';records.write_text('')
    report={'document_id':'doc','records_path':str(records),'warnings':['guide.pdf: page 2: LookupError: unknown encoding: /KSCms-UHC-H'],'status':'review_required','record_count':0}
    (output/'reports/doc.json').write_text(json.dumps(report))
    monkeypatch.setattr('sejong_rag.recover_zip_sources.ocr_pdf_page',lambda *args:('신청 마감 8월 1일',91))
    result=recover_zip({'document_id':'doc','resolved_raw_path':str(source),'source_url':'official','file_type':'zip'},output)
    row=json.loads(records.read_text())
    assert row['locator']['zip_member']=='guide.pdf' and row['locator']['page']==2
    assert row['source_url']=='official' and row['locator']['ocr']
    assert result['status']=='review_required'
    assert not any('LookupError' in w for w in result['warnings'])
