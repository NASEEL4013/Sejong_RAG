import numpy as np
import pytest
from sejong_rag.embed_selected import validate_vectors


def test_valid_vectors():
    validate_vectors(np.eye(3,dtype=np.float32),3,3)


@pytest.mark.parametrize('vectors,rows,dim',[
    (np.eye(3),2,3),(np.eye(3),3,4),(np.zeros((2,3)),2,3),
    (np.array([[float('nan'),0]]),1,2),(np.ones(3),3,3),
])
def test_bad_vectors_rejected(vectors,rows,dim):
    with pytest.raises(ValueError):validate_vectors(vectors,rows,dim)


def test_build_preserves_aliases_and_reuses_checkpoints(tmp_path,monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from sejong_rag.embed_selected import build
    calls=[]
    class FakeModel:
        max_seq_length=512
        tokenizer=SimpleNamespace(encode=lambda *a,**k:[1,2,3])
        def __init__(self,*a,**kw):assert kw['local_files_only']
        def get_sentence_embedding_dimension(self):return 2
        def encode(self,texts,**kw):
            calls.append(list(texts))
            return np.tile(np.array([[1.,0.]],dtype=np.float32),(len(texts),1))
    monkeypatch.setitem(sys.modules,'sentence_transformers',SimpleNamespace(SentenceTransformer=FakeModel))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(set_num_threads=lambda _:None))
    source=tmp_path/'selected.jsonl'
    rows=[dict(chunk_id=str(i),quality_tier='ready',text='본문',source_url='https://example.org',source_aliases=[{'source_url':'https://example.org/alias'}]) for i in range(2)]
    source.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    checkpoint=tmp_path/'checkpoints';output=tmp_path/'index.npz'
    build(source,output,checkpoint,shard_size=1)
    with np.load(output,allow_pickle=False) as stored:
        assert json.loads(stored['chunks'][0])['source_aliases']==rows[0]['source_aliases']
        assert stored['embeddings'].shape==(2,2)
    calls.clear()
    build(source,tmp_path/'resumed.npz',checkpoint,shard_size=1)
    assert len(calls)==1 and all(x.startswith('query: ') for x in calls[0])
    with pytest.raises(FileExistsError):build(source,output,checkpoint)
    source.write_text(source.read_text()+'\n')
    with pytest.raises(ValueError,match='Checkpoint'):build(source,tmp_path/'changed.npz',checkpoint,shard_size=1)
