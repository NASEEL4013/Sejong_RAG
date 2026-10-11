"""Offline E5 embedding with restartable, validated checkpoints."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from sejong_rag.embed_chunks import DEFAULT_MODEL, build_passages, file_sha256, save_index, token_length_stats
from sejong_rag.keyword_search import load_chunks


def validate_vectors(vectors, rows, dimension=None):
    if vectors.ndim != 2 or vectors.shape[0] != rows:
        raise ValueError('Vector row count mismatch')
    if dimension is not None and vectors.shape[1] != dimension:
        raise ValueError('Vector dimension mismatch')
    if not np.isfinite(vectors).all(): raise ValueError('Nonfinite vectors')
    if not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4):
        raise ValueError('Vectors are not unit normalized')


def build(chunks_path, output, checkpoint_dir, batch_size=32, shard_size=512, threads=8):
    import torch
    from sentence_transformers import SentenceTransformer
    chunks_path, output, checkpoint_dir = map(Path,(chunks_path,output,checkpoint_dir))
    if output.exists(): raise FileExistsError(f'Index already exists: {output}')
    if min(batch_size,shard_size,threads)<=0: raise ValueError('Settings must be positive')
    source_hash=file_sha256(chunks_path)
    chunks=load_chunks(chunks_path)
    if not chunks or any(c.get('quality_tier')!='ready' for c in chunks):
        raise ValueError('Expected nonempty selected ready chunks only')
    if len({c['chunk_id'] for c in chunks})!=len(chunks): raise ValueError('Duplicate chunk IDs')
    state=dict(model=DEFAULT_MODEL,source_sha256=source_hash,chunk_count=len(chunks),
               shard_size=shard_size,passage_prefix='passage: ',normalize_embeddings=True)
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    state_path=checkpoint_dir/'state.json'
    if state_path.exists():
        if json.loads(state_path.read_text())!=state: raise ValueError('Checkpoint settings/input mismatch')
    else:
        if any(checkpoint_dir.iterdir()): raise ValueError('Unrecognized checkpoint files')
        state_path.write_text(json.dumps(state,indent=2)+'\n')
    torch.set_num_threads(threads)
    print(f'Loading cached {DEFAULT_MODEL}; CPU threads={threads}',flush=True)
    model=SentenceTransformer(DEFAULT_MODEL,device='cpu',local_files_only=True)
    dimension=model.get_sentence_embedding_dimension()
    passages=build_passages(chunks)
    stats=token_length_stats(model,passages)
    print(json.dumps(dict(token_lengths=stats,dimension=dimension)),flush=True)
    if stats['truncated_count']:raise ValueError('Refusing silent truncation')
    started=time.monotonic();arrays=[]
    for start in range(0,len(chunks),shard_size):
        end=min(start+shard_size,len(chunks));path=checkpoint_dir/f'{start:06d}-{end:06d}.npy'
        if path.exists():
            vectors=np.load(path,allow_pickle=False)
        else:
            vectors=model.encode(passages[start:end],batch_size=batch_size,show_progress_bar=False,
                normalize_embeddings=True,convert_to_numpy=True)
            validate_vectors(vectors,end-start,dimension)
            temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as stream:np.save(stream,np.asarray(vectors,dtype=np.float32),allow_pickle=False)
            temporary.replace(path)
        validate_vectors(vectors,end-start,dimension);arrays.append(vectors)
        print(f'Embedded {end}/{len(chunks)} ({end/len(chunks):.1%}); elapsed {time.monotonic()-started:.1f}s',flush=True)
    embeddings=np.concatenate(arrays)
    validate_vectors(embeddings,len(chunks),dimension)
    if file_sha256(chunks_path)!=source_hash:raise ValueError('Source changed during embedding')
    metadata=dict(model=DEFAULT_MODEL,dimension=dimension,chunk_count=len(chunks),normalized=True,
        query_prefix='query: ',passage_prefix='passage: ',source_chunks=str(chunks_path.resolve()),
        source_sha256=source_hash,token_lengths=stats,created_at=datetime.now(timezone.utc).isoformat(),
        selection='ready chunks with exact attachment deduplication',local_files_only=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.stem+'.building.npz')
    if temporary.exists():raise FileExistsError(f'Temporary index exists: {temporary}')
    save_index(temporary,embeddings,chunks,metadata)
    with np.load(temporary,allow_pickle=False) as stored:
        validate_vectors(stored['embeddings'],len(chunks),dimension)
        assert len(stored['chunks'])==len(chunks)
    if output.exists():raise FileExistsError(output)
    temporary.replace(output)
    print(f'Index saved: {output}; {embeddings.shape}',flush=True)
    queries=['휴학은 어떻게 신청하나요?', '2026년 2학기 전자출결 앱은 무엇을 설치해야 하나요?', '국가장학금 신청은 전자서명까지 해야 완료되나요?']
    qvectors=model.encode(['query: '+q for q in queries],normalize_embeddings=True,convert_to_numpy=True)
    results=[]
    for query,qv in zip(queries,qvectors):
        scores=embeddings@qv;order=np.argsort(-scores,kind='stable')[:5]
        results.append(dict(query=query,results=[dict(chunk_id=chunks[i]['chunk_id'],score=float(scores[i]),
            title=chunks[i].get('document_title'),source_url=chunks[i]['source_url'],
            text=chunks[i]['text'],source_aliases=chunks[i].get('source_aliases',[])) for i in order]))
    report=dict(index=str(output),metadata=metadata,smoke_search=results,
                note='Search smoke test, not a formal relevance evaluation')
    (checkpoint_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--chunks',type=Path,default=Path('data/processed/selected-chunks/selected.jsonl'))
    p.add_argument('--output',type=Path,default=Path('data/index/sejong-2025-2026-selected-e5-small.npz'))
    p.add_argument('--checkpoint-dir',type=Path,default=Path('data/processed/embedding-selected-e5'))
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--shard-size',type=int,default=512)
    p.add_argument('--threads',type=int,default=8)
    a=p.parse_args();build(a.chunks,a.output,a.checkpoint_dir,a.batch_size,a.shard_size,a.threads)


if __name__=='__main__':main()
