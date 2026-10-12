"""Select ready chunks and deduplicate byte-identical attachment copies only."""
from __future__ import annotations
from sejong_rag.corpus_policy import allowed_chunk, EXCLUDED_DOCUMENTS

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from sejong_rag.extract_attachments import load_attachment_metadata


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def same_bytes(a, b):
    with Path(a).open('rb') as left, Path(b).open('rb') as right:
        while True:
            x, y = left.read(1024 * 1024), right.read(1024 * 1024)
            if x != y: return False
            if not x: return True


def jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def extraction_signature(records):
    """Compare usable text and structural location, not document-specific IDs."""
    locations = {'page','sheet','row','column','table','section','paragraph','slide','zip_member','image'}
    payload = [(r['text'], {k:v for k,v in r.get('locator',{}).items() if k in locations}) for r in records]
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def build(data_dir, chunks_dir, normalized_dir, output_dir):
    data_dir, chunks_dir, normalized_dir, output_dir = map(lambda p: Path(p).resolve(),
        (data_dir, chunks_dir, normalized_dir, output_dir))
    if output_dir.exists(): raise FileExistsError('Use a fresh output directory')
    inputs = [chunks_dir / (k+'.jsonl') for k in ('ready','review','quarantine','skipped')]
    inputs += [normalized_dir/'records.jsonl', normalized_dir/'documents.jsonl']
    before = {str(p):sha(p) for p in inputs}
    documents = {d['document_id']:d for d in jsonl(normalized_dir/'documents.jsonl') if d['source_kind']=='attachment'}
    metadata = {m['document_id']:m for m in load_attachment_metadata(data_dir) if m['document_id'] in documents}
    if metadata.keys() != documents.keys(): raise ValueError('Attachment metadata missing')
    raw_groups = defaultdict(list)
    for docid,m in sorted(metadata.items()):
        path = Path(m['resolved_raw_path']); fingerprint = sha(path)
        if m.get('content_hash') and m['content_hash'] != fingerprint:
            raise ValueError(f'Raw hash mismatch: {docid}')
        before[str(path)] = fingerprint
        raw_groups[fingerprint].append(docid)
    for ids in raw_groups.values():
        first = metadata[ids[0]]['resolved_raw_path']
        for docid in ids[1:]:
            if not same_bytes(first, metadata[docid]['resolved_raw_path']):
                raise ValueError('Hash match but byte mismatch')
    used = defaultdict(set); chunk_counts = Counter()
    for c in jsonl(chunks_dir/'ready.jsonl'):
        if c['quality_tier'] != 'ready': raise ValueError('Non-ready chunk in ready input')
        if c['source_kind']=='attachment': used[c['document_id']].update(c['record_ids'])
        chunk_counts[c['document_id']] += 1
    usable = defaultdict(list); notice_meta = {}
    for r in jsonl(normalized_dir/'records.jsonl'):
        if r['record_id'] in used.get(r['document_id'],set()): usable[r['document_id']].append(r)
        if r['source_kind']=='notice_body': notice_meta[r['document_id']] = {
            k:r.get(k) for k in ('document_title','published_at','board','source_url')}
    signatures = {docid:extraction_signature(rs) for docid,rs in usable.items()}
    representative, aliases, groups, conflicts = {}, {}, [], []

    def alias(docid):
        m = metadata[docid]; parent = m.get('parent_document_id') or docid.split('-attachment-',1)[0]
        return dict(document_id=docid, parent_document_id=parent, filename=m.get('filename'),
            attachment_url=m.get('attachment_url'), **notice_meta.get(parent,{}),
            document_extraction_status=documents[docid]['extraction_report']['status'],
            eligible_ready_chunks=chunk_counts[docid])

    for fingerprint, ids in sorted(raw_groups.items()):
        partitions = defaultdict(list)
        for docid in ids:
            if docid in signatures: partitions[signatures[docid]].append(docid)
        if len(partitions)>1: conflicts.append(ids)
        for signature, members in partitions.items():
            canonical = min(members)
            aliases[canonical] = [alias(x) for x in sorted(members)]
            for x in members: representative[x] = canonical
        groups.append(dict(raw_sha256=fingerprint, byte_equality_verified=True,
            sources=[alias(x) for x in ids],
            selection_groups=[dict(canonical_document_id=min(v), members=sorted(v)) for v in partitions.values()]))
    output_dir.mkdir(parents=True)
    counts = Counter(); selected_ids = set()
    with (output_dir/'selected.jsonl').open('w',encoding='utf-8') as selected, (output_dir/'omitted-duplicates.jsonl').open('w',encoding='utf-8') as omitted:
        for c in jsonl(chunks_dir/'ready.jsonl'):
            counts['ready_input_chunks'] += 1
            docid = c['document_id']
            if not allowed_chunk(c):
                counts['excluded_by_user_policy_chunks'] += 1
                continue
            if c['source_kind']=='attachment':
                canonical = representative[docid]
                if docid != canonical:
                    omitted.write(json.dumps(dict(chunk_id=c['chunk_id'],document_id=docid,
                        canonical_document_id=canonical, reason='identical_raw_bytes_and_usable_extraction'),ensure_ascii=False)+'\n')
                    counts['omitted_duplicate_chunks'] += 1
                    continue
                c['source_aliases'] = aliases[canonical]
                c['deduplication'] = dict(method='sha256+byte_equality+exact_usable_extraction',
                    representative_policy='lexicographically smallest eligible document ID',
                    title_scope='Displayed title belongs to representative notice; use source_aliases for other notices')
            else:
                c['source_aliases'] = [dict(document_id=docid, parent_document_id=c.get('parent_document_id'),
                    source_url=c['source_url'], document_title=c.get('document_title'), published_at=c.get('published_at'),board=c.get('board'))]
            if c['chunk_id'] in selected_ids: raise ValueError('Duplicate selected chunk ID')
            selected_ids.add(c['chunk_id']);counts['selected_chunks'] += 1
            selected.write(json.dumps(c,ensure_ascii=False)+'\n')
    with (output_dir/'attachment-groups.jsonl').open('w',encoding='utf-8') as stream:
        for group in groups:stream.write(json.dumps(group,ensure_ascii=False)+'\n')
    for quality in ('review','quarantine'):
        counts['excluded_'+quality+'_chunks'] = sum(1 for _ in jsonl(chunks_dir/(quality+'.jsonl')))
    for path, fingerprint in before.items():
        if sha(path)!=fingerprint:raise RuntimeError(f'Input changed: {path}')
    report=dict(counts=dict(counts), attachment_files=len(metadata), unique_raw_files=len(raw_groups),
        duplicate_raw_groups=sum(len(v)>1 for v in raw_groups.values()),
        duplicate_raw_copies=sum(len(v)-1 for v in raw_groups.values()),
        ready_attachment_documents=len(signatures), selected_attachment_documents=len(aliases),
        differing_extraction_groups=conflicts, input_sha256=before,
        notes=['No embedding or index changes','No source files deleted','No fuzzy matching',
               'Whole ZIP files compared, not individual nested members',
               'Notice bodies and repeated text within documents are not deduplicated',
               'Review/quarantine originals retained in structured-chunks',
               'Representative title is not the title/date of every linked notice'])
    assert counts['ready_input_chunks']==counts['selected_chunks']+counts['omitted_duplicate_chunks']
    (output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--chunks-dir',type=Path,default=Path('data/processed/structured-chunks'))
    p.add_argument('--normalized-dir',type=Path,default=Path('data/processed/normalized-text'))
    p.add_argument('--output-dir',type=Path,default=Path('data/processed/selected-chunks'))
    a=p.parse_args();report=build(a.data_dir,a.chunks_dir,a.normalized_dir,a.output_dir)
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
