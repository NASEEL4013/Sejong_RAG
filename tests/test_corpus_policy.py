import json
import numpy as np
from sejong_rag.corpus_policy import allowed_chunk, EXCLUDED_DOCUMENTS
from sejong_rag.keyword_search import load_chunks
from sejong_rag.vector_search import load_index

DOC=next(iter(EXCLUDED_DOCUMENTS))

def test_exact_document_and_provenance_excluded_but_other_rosters_retained():
    assert not allowed_chunk({'document_id':DOC})
    assert not allowed_chunk({'source_spans':[{'record_id':DOC+'-record-8'}]})
    assert allowed_chunk({'document_id':'notice3-805951-attachment-217293'})
    assert allowed_chunk({'document_id':'notice3-805951'})

def test_old_index_filter_keeps_vectors_aligned_and_files_intact(tmp_path):
    chunks=[{'chunk_id':'a','document_id':'other'}, {'chunk_id':'b','document_id':DOC}, {'chunk_id':'c','document_id':'third'}]
    path=tmp_path/'chunks.jsonl';path.write_text(''.join(json.dumps(c)+'\n' for c in chunks))
    assert [c['chunk_id'] for c in load_chunks(path)]==['a','c']
    index=tmp_path/'index.npz';vectors=np.eye(3)
    np.savez(index,embeddings=vectors,chunks=np.array([json.dumps(c) for c in chunks]),metadata=np.array(json.dumps({'chunk_count':3})))
    before=index.read_bytes()
    active,cs,meta=load_index(index)
    assert [c['chunk_id'] for c in cs]==['a','c']
    np.testing.assert_array_equal(active,vectors[[0,2]])
    assert meta['policy_excluded_chunks']==1 and meta['active_chunk_count']==2
    assert index.read_bytes()==before
