"""Run the selected 2025/2026 retriever through a source-bound Gemini answer."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from sejong_rag.gemini_rag import DEFAULT_GENERATION_MODEL, get_api_key
from sejong_rag.selected_search import DEFAULT_STRATEGY, VERSION, load_search
from sejong_rag.selected_generation import build_request, generate_answer

DEFAULT_INDEX = Path('data/index/sejong-2025-2026-selected-e5-small.npz')
DEFAULT_NORMALIZED = Path('data/processed/normalized-text/records.jsonl')


def load_runtime(index=DEFAULT_INDEX, normalized=DEFAULT_NORMALIZED):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(8)
    engine, metadata = load_search(index, normalized)
    model = SentenceTransformer(metadata['model'], device='cpu', local_files_only=True)
    return engine, metadata, model


def prepare_request(query, engine, metadata, embedding_model):
    if not isinstance(query, str) or not query.strip():
        raise ValueError('질문은 비어 있을 수 없어.')
    query = query.strip()
    from sejong_rag.syllabus_originals import guidance_request
    original=guidance_request(query,getattr(engine,'syllabus_catalog',[]))
    if original is not None:return original
    encoded_query = metadata['query_prefix'] + query
    if engine.count(encoded_query) > embedding_model.max_seq_length:
        raise ValueError('질문이 임베딩 입력 한도를 넘었어. 조금 짧게 나눠서 질문해줘.')
    vector = embedding_model.encode(encoded_query, normalize_embeddings=True, convert_to_numpy=True)
    bundles = engine.search(query, vector, strategy=DEFAULT_STRATEGY)
    # Enrich provenance only; do not re-rank, re-chunk, or add extra evidence.
    for bundle in bundles:
        anchor = engine.chunks[engine.by_id[bundle['anchor_chunk_id']]]
        for field in ('filename', 'quality_tier', 'source_kind', 'context_note',
                      'document_extraction_status', 'quality_flags'):
            if field in anchor:
                bundle[field] = deepcopy(anchor[field])
    request = build_request(query, bundles)
    request['prompt_tokens_e5_estimate'] = sum(engine.count(m['content']) for m in request['messages'])
    request['retrieval'] = dict(version=VERSION, strategy=DEFAULT_STRATEGY,
        index_model=metadata['model'], source_sha256=metadata['source_sha256'],
        corpus_chunks=len(engine.chunks), top_k=5, max_bundle_tokens=512,
        total_evidence_tokens=2200,
        procedure_decisions=deepcopy(getattr(engine,'last_procedure_decisions',[])))
    return request


def create_client():
    from google import genai
    from google.genai import types

    return genai.Client(api_key=get_api_key(), http_options=types.HttpOptions(
        timeout=45000, retry_options=types.HttpRetryOptions(attempts=1)))


def save_json(path, value):
    # Never overwrite an earlier prompt/answer or accidentally open a symlink.
    with Path(path).open('x', encoding='utf-8') as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query')
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--normalized', type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument('--model', default=DEFAULT_GENERATION_MODEL)
    parser.add_argument('--max-output-tokens', type=int, default=1600)
    parser.add_argument('--dry-run', action='store_true', help='실제 프롬프트만 출력; 외부 API 호출 없음')
    parser.add_argument('--json', action='store_true', help='답변과 실제 출처 매핑을 JSON으로 출력')
    parser.add_argument('--output-dir', type=Path, help='프롬프트·답변을 기록할 새 폴더')
    args = parser.parse_args()
    if not args.query.strip() or args.max_output_tokens <= 0:
        parser.error('빈 질문이나 잘못된 출력 길이야.')
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    runtime = load_runtime(args.index, args.normalized)
    request = prepare_request(args.query, *runtime)
    if args.output_dir:
        save_json(args.output_dir / 'request.json', request)
    if args.dry_run:
        print(json.dumps(request, ensure_ascii=False, indent=2))
        return
    try:
        if 'original_document_response' in request:
            result = generate_answer(None, request, model=args.model, max_output_tokens=args.max_output_tokens)
        else:
            with create_client() as client:
                result = generate_answer(client, request, model=args.model,
                                         max_output_tokens=args.max_output_tokens)
    except Exception as error:
        # Provider exception messages can include request URLs. Never print keys,
        # headers, or the provider's raw error body.
        parser.exit(1, f'답변 생성 요청을 완료하지 못했어 ({type(error).__name__}). 키·모델·네트워크 설정을 확인해줘.\n')
    if args.output_dir:
        save_json(args.output_dir / 'answer.json', result)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(result['answer'])
    for source in result['cited_sources']:
        print(f"\n[근거 {source['id']}] {source.get('document_title') or source.get('title', '')}")
        print('공지:', source['source_url'])
        if source.get('attachment_url'):
            print('첨부:', source['attachment_url'])
        print('위치:', json.dumps(source.get('locations', []), ensure_ascii=False))
    print('\n[실제 API 사용량]', json.dumps(result['usage'], ensure_ascii=False))


if __name__ == '__main__':
    main()
