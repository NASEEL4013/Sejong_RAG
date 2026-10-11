import json
import pytest
from sejong_rag.select_chunks import build, extraction_signature, same_bytes, sha


def write_lines(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))


def test_exact_bytes_and_signature(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.write_bytes(b'2025');b.write_bytes(b'2026')
    assert not same_bytes(a,b)
    b.write_bytes(a.read_bytes());assert same_bytes(a,b)
    r={'record_id':'a','text':'마감 10일','locator':{'page':1}}
    assert extraction_signature([r])==extraction_signature([dict(r,record_id='b')])
    assert extraction_signature([r])!=extraction_signature([dict(r,text='마감 11일')])


@pytest.mark.parametrize('different_extraction',[False,True])
def test_selection_and_provenance(tmp_path,monkeypatch,different_extraction):
    data=tmp_path/'data';norm=data/'normalized';chunks=data/'chunks';meta=[];docs=[];records=[];ready=[]
    for i in (1,2):
        docid=f'n{i}-attachment-1';raw=data/f'{i}.pdf';raw.parent.mkdir(exist_ok=True);raw.write_bytes(b'identical')
        meta.append(dict(document_id=docid,parent_document_id=f'n{i}',resolved_raw_path=str(raw),content_hash=sha(raw),filename=f'{i}.pdf',attachment_url=f'https://example.org/{i}.pdf'))
        docs.append(dict(document_id=docid,source_kind='attachment',extraction_report={'status':'success'}))
        records.append(dict(record_id=f'n{i}-body',document_id=f'n{i}',source_kind='notice_body',text='공지',source_url=f'https://example.org/{i}',document_title=f'공지{i}'))
        records.append(dict(record_id=f'r{i}',document_id=docid,source_kind='attachment',text='다름' if i==2 and different_extraction else '동일',locator={'page':1}))
        ready.append(dict(chunk_id=f'c{i}',document_id=docid,source_kind='attachment',quality_tier='ready',record_ids=[f'r{i}'],text='본문',source_url=f'https://example.org/{i}'))
    monkeypatch.setattr('sejong_rag.select_chunks.load_attachment_metadata',lambda _:meta)
    write_lines(norm/'records.jsonl',records);write_lines(norm/'documents.jsonl',docs)
    write_lines(chunks/'ready.jsonl',ready)
    for k in ('review','quarantine','skipped'):write_lines(chunks/f'{k}.jsonl',[])
    before={p:p.read_bytes() for p in data.rglob('*') if p.is_file()}
    out=data/'selected';report=build(data,chunks,norm,out)
    selected=[json.loads(l) for l in (out/'selected.jsonl').read_text().splitlines()]
    assert len(selected)==(2 if different_extraction else 1)
    assert report['duplicate_raw_copies']==1
    if not different_extraction:assert len(selected[0]['source_aliases'])==2
    else:assert report['differing_extraction_groups']
    assert all(p.read_bytes()==b for p,b in before.items())
    with pytest.raises(FileExistsError):build(data,chunks,norm,out)
