from sejong_rag.recover_attachments import missing_text_pages, parse_tesseract_tsv

def test_parse_tesseract_tsv_groups_lines_and_confidence():
    value="level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n5\t1\t1\t1\t1\t1\t0\t0\t1\t1\t90\t장학금\n5\t1\t1\t1\t1\t2\t0\t0\t1\t1\t80\t신청\n5\t1\t1\t1\t2\t1\t0\t0\t1\t1\t70\t마감\n"
    text, confidence=parse_tesseract_tsv(value)
    assert text=="장학금 신청\n마감"
    assert confidence==80.0

def test_missing_text_pages_are_unique_and_sorted():
    report={"warnings":["page 4: no embedded text (OCR may be required)","other", "page 1: no embedded text (OCR may be required)","page 4: no embedded text"]}
    assert missing_text_pages(report)==[1,4]
