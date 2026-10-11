"""Conservative normalization before chunking; never changes upstream files."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

VERSION = 'conservative_v1'


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalize_text(text: str) -> tuple[str, list[str]]:
    """Preserve words, indentation, tabs and individual line boundaries."""
    changes = []
    steps = [
        ('unicode_nfc', lambda s: unicodedata.normalize('NFC', s)),
        ('line_endings', lambda s: s.replace('\r\n', '\n').replace('\r', '\n')),
        ('initial_bom', lambda s: s.lstrip('\ufeff')),
        ('nonbreaking_spaces', lambda s: s.replace('\u00a0', ' ').replace('\u202f', ' ')),
        ('trailing_spaces', lambda s: '\n'.join(line.rstrip(' ') for line in s.split('\n'))),
        ('excess_blank_lines', lambda s: re.sub(r'\n{3,}', '\n\n', s)),
        ('boundary_newlines', lambda s: s.strip('\n')),
    ]
    for name, operation in steps:
        updated = operation(text)
        if updated != text:
            changes.append(name)
        text = updated
    return text, changes


def quality_flags(text):
    flags = []
    if not text.strip(): flags.append('empty_text')
    if '\ufffd' in text: flags.append('replacement_character')
    if any(unicodedata.category(c) == 'Cc' and c not in '\n\t' for c in text):
        flags.append('control_character')
    if re.search(r'(?m)^\s*[1-5](?:[ \t]+\d+){9}[ \t]+-?\d+(?:\.\d+)?(?:[ \t]|$)', text):
        flags.append('possible_ocr_tsv_noise')
    return flags


def build_normalized(data_dir: Path, output_dir: Path):
    data_dir, output_dir = data_dir.resolve(), output_dir.resolve()
    if output_dir.exists(): raise FileExistsError(f'Use a fresh output directory: {output_dir}')
    if data_dir == output_dir or data_dir.is_relative_to(output_dir):
        raise ValueError('Output cannot contain the input dataset')
    inputs = {}

    def read(path):
        raw = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    allowed = set()
    for year in (2025, 2026):
        allowed.update(x['document_id'] for x in read(data_dir / f'catalog/notices-{year}.json')['items'])
    details = {}
    for docid in sorted(allowed):
        path = data_dir / f'metadata/details/{docid}.json'
        details[docid] = (read(path), path)
    output_dir.mkdir(parents=True)
    counts, rules, flags = Counter(), Counter(), Counter()
    documents, examples, flagged_examples, seen = [], [], [], set()

    def emit(stream, record, notice, source, kind, status):
        original = record['text']
        cleaned, changes = normalize_text(original)
        issues = quality_flags(cleaned)
        rid = record['record_id']
        if rid in seen: raise ValueError(f'Duplicate record ID: {rid}')
        seen.add(rid)
        result = dict(record)
        result.update(text=cleaned, text_length=len(cleaned), source_kind=kind,
                      document_title=notice.get('title'), published_at=notice.get('published_at'),
                      board=notice.get('board'), document_extraction_status=status)
        result['source_url'] = record.get('source_url') or notice.get('source_url')
        if not result['source_url']: raise ValueError(f'Missing source: {rid}')
        result['normalization'] = dict(version=VERSION, source_path=str(source),
            source_text_sha256=digest(original), text_sha256=digest(cleaned),
            changes=changes, quality_flags=issues, review_inherited=status != 'success')
        stream.write(json.dumps(result, ensure_ascii=False) + '\n')
        counts.update({kind: 1, 'records': 1, 'original_characters': len(original),
            'normalized_characters': len(cleaned), 'changed_records': bool(changes),
            'flagged_records': bool(issues)})
        rules.update(changes); flags.update(issues)
        if changes and len(examples) < 5:
            examples.append(dict(record_id=rid, changes=changes, before=original[:500], after=cleaned[:500]))
        if issues and len(flagged_examples) < 10:
            flagged_examples.append(dict(record_id=rid, flags=issues))

    with (output_dir / 'records.jsonl').open('w', encoding='utf-8') as stream:
        for docid, (notice, path) in sorted(details.items()):
            body = notice.get('body') or ''
            record = dict(record_id=f'{docid}-body', document_id=docid, parent_document_id=docid,
                text=body, locator={'section': 'notice_body'}, extraction_method='official_html_body')
            emit(stream, record, notice, path, 'notice_body', 'success')
            documents.append(dict(document_id=docid, source_kind='notice_body', source_path=str(path), body_empty=not body.strip()))
            image_path = data_dir / f'processed/notice-images/{docid}.json'
            if image_path.exists():
                for img in read(image_path).get('images', []):
                    image_id = f"{docid}-body-image-{img['image']}"
                    documents.append(dict(document_id=image_id, source_kind='notice_image', source_path=str(image_path), image_metadata=img))
                    if not img.get('text'):
                        counts['images_without_text'] += 1
                        continue
                    record = dict(record_id=image_id, document_id=image_id, parent_document_id=docid,
                        text=img['text'], locator=dict(section='notice_body_image', image=img['image'], image_url=img['image_url']),
                        extraction_method=img.get('extraction_method', 'tesseract_kor_eng_notice_image'),
                        image_metadata={k: v for k, v in img.items() if k != 'text'})
                    emit(stream, record, notice, image_path, 'notice_image', img.get('status', 'review_required'))
        for report_path in sorted((data_dir / 'processed/attachments/reports').glob('*.json')):
            docid = report_path.stem
            parent = docid.split('-attachment-', 1)[0]
            if parent not in details: continue
            report = read(report_path)
            path = data_dir / f'processed/attachments/records/{docid}.jsonl'
            raw = path.read_bytes(); inputs[str(path)] = hashlib.sha256(raw).hexdigest()
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if len(records) != report['record_count']: raise ValueError(f'Count mismatch: {docid}')
            for record in records:
                emit(stream, record, details[parent][0], path, 'attachment', report['status'])
            documents.append(dict(document_id=docid, source_kind='attachment', source_path=str(path), extraction_report=report))
            counts['attachment_documents'] += 1
    for path, expected in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Input changed during run: {path}')
    with (output_dir / 'documents.jsonl').open('w', encoding='utf-8') as stream:
        for doc in documents: stream.write(json.dumps(doc, ensure_ascii=False) + '\n')
    report = dict(version=VERSION, created_at=datetime.now(timezone.utc).isoformat(),
        scope='2025/2026 academic and scholarship catalogs', counts=dict(counts), rules=dict(rules),
        quality_flags=dict(flags), examples=examples, flagged_examples=flagged_examples, input_sha256=inputs,
        notes=['Review status preserved', 'No deduplication or record deletion',
               'No semantic rewriting, chunking or embedding', 'Input hashes checked unchanged'])
    (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='Separate normalized text, without chunking or embedding')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/normalized-text'))
    args = parser.parse_args()
    report = build_normalized(args.data_dir, args.output_dir)
    print(json.dumps({k: report[k] for k in ('counts', 'rules', 'quality_flags')}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
