from sejong_rag.recover_attachments import missing_text_pages, parse_tesseract_tsv

def test_parse_tesseract_tsv_groups_lines_and_confidence():
    value="level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n5\t1\t1\t1\t1\t1\t0\t0\t1\t1\t90\t장학금\n5\t1\t1\t1\t1\t2\t0\t0\t1\t1\t80\t신청\n5\t1\t1\t1\t2\t1\t0\t0\t1\t1\t70\t마감\n"
    text, confidence=parse_tesseract_tsv(value)
    assert text=="장학금 신청\n마감"
    assert confidence==80.0

def test_missing_text_pages_are_unique_and_sorted():
    report={"warnings":["page 4: no embedded text (OCR may be required)","other", "page 1: no embedded text (OCR may be required)","page 4: no embedded text"]}
    assert missing_text_pages(report)==[1,4]


def test_partial_ocr_keeps_unprocessed_pages_and_review_status(tmp_path,monkeypatch):
    import json
    from sejong_rag.recover_attachments import recover_pdf
    output=tmp_path/'out'; (output/'reports').mkdir(parents=True); (output/'records').mkdir()
    report={'warnings':['page 1: no embedded text','page 2: no embedded text'],'records_path':str(output/'records/doc.jsonl')}
    (output/'reports/doc.json').write_text(json.dumps(report))
    monkeypatch.setattr('sejong_rag.recover_attachments.ocr_pdf_page',lambda *args:('인식된 글',None))
    result=recover_pdf({'document_id':'doc','resolved_raw_path':str(tmp_path/'a.pdf')},output,max_pages=1)
    assert missing_text_pages(result)==[2]
    assert result['status']=='review_required'
    assert any('source verification' in warning for warning in result['warnings'])
