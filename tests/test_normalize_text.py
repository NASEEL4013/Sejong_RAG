import json
import pytest
from sejong_rag.normalize_text import build_normalized, normalize_text, quality_flags


def test_normalization_and_idempotence():
    text = '\ufeff신청\u00a0기간  \r\n\r\n\r\n2026.10.10\u202f18:00\r\n'
    result, changes = normalize_text(text)
    assert result == '신청 기간\n\n2026.10.10 18:00'
    assert changes
    assert normalize_text(result) == (result, [])


@pytest.mark.parametrize('text', [
    '  구분\t금액\t\n  A\t10,000원\t',
    '신청 불가\n※ 예외: Ⅰ유형 / Ⅱ유형\n02-3408-3036',
    '<표>\n=SUM(A1:A2)\nhttps://example.org/?a=1&b=2',
    '학교\n학교\n10\n분',
])
def test_meaning_and_structure_preserved(text):
    assert normalize_text(text) == (text, [])


def test_nfc_not_compatibility_folding():
    assert normalize_text('가 ① Ⅰ')[0] == '가 ① Ⅰ'


def test_flags_do_not_delete():
    text = '5 1 1 1 5 4 2483 858 47 36 14 mY\n깨짐\ufffd'
    assert set(quality_flags(text)) == {'possible_ocr_tsv_noise', 'replacement_character'}
    assert normalize_text(text)[0] == text


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


def test_pipeline_preserves_sources_and_review(tmp_path):
    data = tmp_path / 'data'
    for year in (2025, 2026):
        put(data / f'catalog/notices-{year}.json', {'items': [{'document_id': 'n-1'}]})
    put(data / 'metadata/details/n-1.json', {'document_id': 'n-1', 'title': '제목', 'body': '', 'source_url': 'https://example.org'})
    put(data / 'processed/notice-images/n-1.json', {'images': [
        {'image': 1, 'image_url': 'https://example.org/i', 'status': 'failed', 'text': ''},
        {'image': 2, 'image_url': 'https://example.org/j', 'status': 'review_required', 'text': '날짜\u00a0미확인'},
    ]})
    put(data / 'processed/attachments/reports/n-1-attachment-1.json', {'document_id': 'n-1-attachment-1', 'record_count': 1, 'status': 'review_required', 'warnings': ['review']})
    put(data / 'processed/attachments/records/n-1-attachment-1.jsonl', {'record_id': 'r1', 'document_id': 'n-1-attachment-1', 'text': 'a\u00a0b', 'locator': {'page': 1, 'human_verified': True}})
    before = {p: p.read_bytes() for p in data.rglob('*') if p.is_file()}
    out = tmp_path / 'output'; report = build_normalized(data, out)
    rows = [json.loads(line) for line in (out / 'records.jsonl').read_text().splitlines()]
    assert report['counts']['records'] == 3
    assert report['counts']['images_without_text'] == 1
    assert rows[-1]['locator']['human_verified'] is True
    assert rows[-1]['document_extraction_status'] == 'review_required'
    assert rows[0]['normalization']['quality_flags'] == ['empty_text']
    assert all(p.read_bytes() == content for p, content in before.items())
    with pytest.raises(FileExistsError): build_normalized(data, out)
