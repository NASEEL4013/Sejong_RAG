import json
import pytest
from sejong_rag.structure_chunk import Chunker, build, sections, tier


class Characters:
    def __call__(self, text, add_special_tokens=True, **kwargs):
        result = {'input_ids': list(range(len(text) + (2 if add_special_tokens else 0)))}
        if kwargs.get('return_offsets_mapping'):
            result['offset_mapping'] = [(i,i+1) for i in range(len(text))]
        return result


def record(rid, text, **kwargs):
    return dict(record_id=rid, document_id='d', text=text, locator={},
                source_url='https://example.org', source_kind='attachment',
                document_extraction_status='success', **kwargs)


def test_caution_stays_with_section():
    text='1. 신청 대상\n재학생\n※ 졸업예정자 제외\n2. 신청 기간\n10월'
    spans=sections(text)
    assert len(spans)==2
    assert '졸업예정자 제외' in text[slice(*spans[0])]


def test_long_split_limit_and_coverage():
    engine=Chunker(Characters(),128,16)
    text='긴 문장입니다. '*100
    chunks=list(engine.chunks([record('r',text)]))
    covered=set()
    for c in chunks:
        assert c['token_count']<=128
        for ref in c['source_spans']:covered.update(range(ref['start'],ref['end']))
    assert all(i in covered for i,c in enumerate(text) if not c.isspace())
    assert chunks[1]['source_spans'][0]['start']<chunks[0]['source_spans'][0]['end']


def test_tier_does_not_promote_verified_page():
    r=record('r','text');r['document_extraction_status']='review_required';r['locator']['human_verified']=True
    assert tier(r)=='review'
    r['normalization']={'quality_flags':['possible_ocr_tsv_noise']}
    assert tier(r)=='quarantine'


def test_build_isolation_and_empty(tmp_path):
    source=tmp_path/'in';source.mkdir()
    rows=[record('r1','첫 문서'),record('r2','두 번째')]
    rows[1]['document_id']='other'
    rows.append(record('empty',''))
    path=source/'records.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    before=path.read_bytes();out=tmp_path/'out'
    report=build(source,out,Characters(),128,16)
    chunks=[json.loads(l) for l in (out/'ready.jsonl').read_text().splitlines()]
    assert len(chunks)==2
    assert report['counts']['skipped_empty']==1
    assert path.read_bytes()==before
    with pytest.raises(FileExistsError):build(source,out,Characters())


def test_table_context_is_explicit_and_preserves_locator():
    rows=[record('r'+str(i),'표 행 '+str(i)) for i in range(4)]
    for i,r in enumerate(rows):r['locator']={'sheet':'시트','row':i+1}
    chunks=list(Chunker(Characters(),256,16).chunks(rows))
    assert chunks[0]['context_record_ids']==['r0','r1','r2']
    assert chunks[0]['source_spans'][0]['locator']['row']==1
    assert '열 제목으로 확정하지 않음' in chunks[0]['text']
