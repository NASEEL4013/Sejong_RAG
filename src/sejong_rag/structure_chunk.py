"""Offline structure-aware chunking of normalized records; no embedding."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

MODEL = 'intfloat/multilingual-e5-small'
HEADING = re.compile(r'^(?:\d{1,2}[.)]\s|[가-하][.)]\s|[■□◆▶]|(?:신청 대상|신청 기간|신청 방법|유의사항|문의처)\s*[:：]?)')
EXCEPTION = re.compile(r'^\s*(?:※|단[,， ]|다만|주의|예외)')


def tier(record):
    flags = record.get('normalization', {}).get('quality_flags', [])
    if any(x != 'empty_text' for x in flags): return 'quarantine'
    if record.get('document_extraction_status') != 'success': return 'review'
    return 'ready'


def group_key(record):
    loc = record.get('locator', {})
    return (record['document_id'], tier(record), loc.get('zip_member'),
            loc.get('sheet'), loc.get('table'), loc.get('page'),
            loc.get('slide'), loc.get('section'), loc.get('image'))


def sections(text):
    """Return exact source character spans. Keep caution lines in preceding section."""
    starts = [0]
    for match in re.finditer(r'(?m)^.+$', text):
        line = match.group().strip()
        if match.start() and HEADING.match(line) and not EXCEPTION.match(line):
            starts.append(match.start())
    return [(a, b) for a, b in zip(starts, starts[1:] + [len(text)]) if text[a:b].strip()]


class Chunker:
    def __init__(self, tokenizer, max_tokens=448, overlap=64):
        if max_tokens < 64 or not 0 <= overlap < max_tokens // 2:
            raise ValueError('Invalid token limits')
        self.tokenizer, self.limit, self.overlap = tokenizer, max_tokens, overlap

    def count(self, text):
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)['input_ids'])

    def clip(self, text, budget):
        if self.count(text) <= budget: return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.count(text[:mid]) <= budget: lo = mid
            else: hi = mid - 1
        return text[:lo]

    def split(self, text, header):
        start = 0
        while start < len(text):
            if self.count(header + text[start:]) <= self.limit:
                yield start, len(text)
                break
            lo, hi = start + 1, len(text)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self.count(header + text[start:mid]) <= self.limit: lo = mid
                else: hi = mid - 1
            end = lo
            if self.count(header + text[start:end]) > self.limit:
                raise ValueError('Header leaves no room for text')
            # Prefer a sentence/line ending in the latter half of the window.
            boundaries = [m.end() + start for m in re.finditer(r'\n|[.!?。][ \t]+', text[start:end])]
            candidates = [p for p in boundaries if p > start + (end-start)//2
                          and not EXCEPTION.match(text[p:])]
            if candidates: end = candidates[-1]
            while self.count(header + text[start:end]) > self.limit: end -= 1
            yield start, end
            offsets = self.tokenizer(text[start:end], add_special_tokens=False,
                                     return_offsets_mapping=True, truncation=False)['offset_mapping']
            offsets = [(a,b) for a,b in offsets if b > a]
            next_start = end
            if self.overlap and len(offsets) > self.overlap:
                next_start = start + offsets[-self.overlap][0]
            start = next_start if next_start > start else end

    def chunks(self, records):
        first = records[0]
        title = self.clip(str(first.get('document_title') or ''), 80)
        filename = self.clip(str(first.get('filename') or ''), 50)
        header = '문서 제목: ' + title + '\n'
        if filename: header += '첨부파일: ' + filename + '\n'
        loc = first.get('locator', {})
        table = 'sheet' in loc or 'table' in loc
        context_records = []
        if table:
            context_records = records[:3]
            context = '\n'.join(r['text'] for r in context_records)
            header += '표 앞부분 참고(열 제목으로 확정하지 않음):\n' + self.clip(context, 70) + '\n'
        header = self.clip(header, self.limit // 2) + '\n'
        current, body = [], ''

        def make(parts, text):
            full = header + text
            assert self.count(full) <= self.limit
            return dict(text=full, text_length=len(full), token_count=self.count(full),
                source_spans=parts, context_record_ids=[r['record_id'] for r in context_records],
                context_note='Table prefix excerpt, not inferred column headers' if table else None)

        for record in records:
            text = record['text']
            spans = [(0,len(text))] if table else sections(text)
            for a,b in spans:
                # A detected heading starts its own group, not the previous topic.
                if current and not table and HEADING.match(text[a:b].strip()):
                    yield make(current, body); current, body = [], ''
                part = text[a:b]
                ref = dict(record_id=record['record_id'], start=a, end=b, locator=record.get('locator', {}))
                candidate = body + ('\n' if body else '') + part
                if self.count(header + candidate) <= self.limit:
                    current.append(ref); body = candidate
                    continue
                if current:
                    yield make(current, body); current, body = [], ''
                if self.count(header + part) <= self.limit:
                    current, body = [ref], part
                else:
                    for x,y in self.split(part, header):
                        yield make([dict(ref, start=a+x, end=a+y)], part[x:y])
        if current: yield make(current, body)


def build(input_dir, output_dir, tokenizer, max_tokens=448, overlap=64):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists(): raise FileExistsError('Use a fresh output directory')
    source = input_dir / 'records.jsonl'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    engine = Chunker(tokenizer, max_tokens, overlap)
    output_dir.mkdir(parents=True)
    counts, reasons = Counter(), Counter()
    seen, coverage, originals = set(), {}, {}
    samples = []
    streams = {k: (output_dir / (k+'.jsonl')).open('w', encoding='utf-8')
               for k in ['ready','review','quarantine','skipped']}

    def flush(group):
        if not group: return
        first = group[0]; quality = tier(first)
        for chunk in engine.chunks(group):
            counts[quality] += 1
            refs = chunk['source_spans']
            identity = json.dumps(refs, sort_keys=True, ensure_ascii=False)
            cid = 'structured-' + hashlib.sha256(identity.encode()).hexdigest()[:24]
            if cid in seen: raise ValueError('Duplicate chunk ID')
            seen.add(cid)
            chunk.update(chunk_id=cid, document_id=first['document_id'],
                parent_document_id=first.get('parent_document_id'), document_title=first.get('document_title'),
                filename=first.get('filename'), source_url=first['source_url'],
                attachment_url=first.get('attachment_url'), published_at=first.get('published_at'),
                board=first.get('board'), quality_tier=quality, source_kind=first['source_kind'],
                document_extraction_status=first['document_extraction_status'],
                record_ids=list(dict.fromkeys(x['record_id'] for x in refs)),
                chunk_method='structure_rules_v1', chunk_max_tokens=max_tokens,
                chunk_overlap_tokens=overlap, tokenizer_model=MODEL)
            chunk['quality_flags'] = sorted({f for r in group if r['record_id'] in chunk['record_ids']
                                            for f in r.get('normalization',{}).get('quality_flags',[])})
            for ref in refs: coverage.setdefault(ref['record_id'], []).append((ref['start'],ref['end']))
            streams[quality].write(json.dumps(chunk, ensure_ascii=False)+'\n')
            if len(samples) < 5: samples.append(chunk)
            counts['maximum_tokens'] = max(counts['maximum_tokens'],chunk['token_count'])

    group = []
    try:
        with source.open() as stream:
            for line in stream:
                record = json.loads(line); rid = record['record_id']
                if rid in originals: raise ValueError('Duplicate record')
                originals[rid] = record['text']; counts['input_records'] += 1
                if not record['text'].strip():
                    streams['skipped'].write(json.dumps(dict(record_id=rid, reason='empty_text'),ensure_ascii=False)+'\n')
                    counts['skipped_empty'] += 1
                    continue
                if group and group_key(group[0]) != group_key(record): flush(group); group = []
                group.append(record)
        flush(group)
    finally:
        for stream in streams.values(): stream.close()
    for rid,text in originals.items():
        if not text.strip(): continue
        cursor = 0
        for a,b in sorted(coverage.get(rid, [])):
            if text[cursor:a].strip(): raise ValueError(f'Coverage gap: {rid}')
            cursor = max(cursor,b)
        if text[cursor:].strip(): raise ValueError(f'Coverage missing: {rid}')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    report = dict(counts=dict(counts), input_sha256=before, tokenizer_model=MODEL,
        max_tokens=max_tokens, overlap=overlap, nonwhitespace_coverage_verified=True,
        notes=['No embedding', 'No deduplication', 'ready means extraction succeeded, not human verified',
               'Rule-based headings, not semantic understanding', 'Context prefix may be truncated'], samples=samples)
    (output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir',type=Path,default=Path('data/processed/normalized-text'))
    p.add_argument('--output-dir',type=Path,default=Path('data/processed/structured-chunks'))
    a=p.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    print(json.dumps(build(a.input_dir,a.output_dir,tokenizer)['counts'],ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
