"""통합 검색 결과를 출처가 명확한 Gemini용 근거 문맥으로 조립한다."""
from __future__ import annotations
from collections import defaultdict
from typing import Iterable
from sejong_rag.unified_search import format_locations

DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_MAX_CONTEXT_CHUNKS = 8
DEFAULT_MAX_CONTEXT_CHARS = 12_000
DEFAULT_MAX_RETRIEVED_PER_NOTICE = 3

def _parent_id(chunk: dict[str, object]) -> str:
    return str(chunk.get("parent_document_id") or chunk.get("document_id") or "")

def _chunk_index(chunk: dict[str, object]) -> int | None:
    try:
        return int(chunk["chunk_index"])
    except (KeyError, TypeError, ValueError):
        return None

def select_unified_context(results: list[dict[str, object]], all_chunks: list[dict[str, object]], *, neighbor_window: int = 1, max_chunks: int = DEFAULT_MAX_CONTEXT_CHUNKS, max_chars: int = DEFAULT_MAX_CONTEXT_CHARS, max_retrieved_per_notice: int = DEFAULT_MAX_RETRIEVED_PER_NOTICE) -> list[dict[str, object]]:
    """검색 청크와 같은 파일의 인접 청크를 예산 안에서 선택한다."""
    if neighbor_window < 0:
        raise ValueError("neighbor_window는 0 이상이어야 합니다.")
    if max_chunks <= 0 or max_chars <= 0 or max_retrieved_per_notice <= 0:
        raise ValueError("문맥 제한값은 모두 1 이상이어야 합니다.")
    by_file_index = {}
    by_file_page = defaultdict(list)
    for chunk in all_chunks:
        document_id = str(chunk.get("document_id", ""))
        pages = {locator.get("page") for locator in chunk.get("locators", []) if locator.get("page") is not None}
        for page in pages:
            by_file_page[(document_id, page)].append(chunk)
        index = _chunk_index(chunk)
        if index is not None:
            by_file_index[(str(chunk.get("document_id", "")), index)] = chunk
    selected, selected_ids = [], set()
    retrieved_per_notice = defaultdict(int)
    used_chars = 0
    def add(chunk, relation, rank):
        nonlocal used_chars
        chunk_id = str(chunk.get("chunk_id", ""))
        if not chunk_id or chunk_id in selected_ids:
            return True
        text_length = len(str(chunk.get("text", "")))
        if len(selected) >= max_chunks or (selected and used_chars + text_length > max_chars):
            return False
        enriched = dict(chunk)
        enriched.update(context_relation=relation, retrieval_rank=rank)
        selected.append(enriched)
        selected_ids.add(chunk_id)
        used_chars += text_length
        return True
    retained = []
    for rank, result in enumerate(results, 1):
        chunk = result.get("chunk")
        if not isinstance(chunk, dict):
            continue
        parent = _parent_id(chunk)
        if retrieved_per_notice[parent] >= max_retrieved_per_notice:
            continue
        if not add(chunk, "retrieved", rank):
            break
        retrieved_per_notice[parent] += 1
        retained.append((rank, chunk))
    # 직접 검색된 근거를 먼저 확보한 뒤 남은 예산에만 이웃을 넣는다.
    # 같은 PDF 페이지가 여러 청크로 나뉜 경우 다른 페이지보다 먼저 보충한다.
    for rank, chunk in retained:
        document_id = str(chunk.get("document_id", ""))
        pages = {locator.get("page") for locator in chunk.get("locators", []) if locator.get("page") is not None}
        for page in pages:
            for sibling in by_file_page.get((document_id, page), []):
                add(sibling, "same_page", rank)
    for rank, chunk in retained:
        index = _chunk_index(chunk)
        if index is None:
            continue
        document_id = str(chunk.get("document_id", ""))
        for distance in range(1, neighbor_window + 1):
            for neighbor_index in (index - distance, index + distance):
                neighbor = by_file_index.get((document_id, neighbor_index))
                if neighbor is not None:
                    add(neighbor, "neighbor", rank)
    return selected

def source_records(chunks: Iterable[dict[str, object]]) -> list[dict[str, str]]:
    """사용한 근거의 공식 URL과 위치를 중복 없이 출력 가능한 형태로 만든다."""
    records, seen = [], set()
    for chunk in chunks:
        record = {
            "title": str(chunk.get("document_title") or "제목 없음"),
            "filename": str(chunk.get("filename") or "공지 본문"),
            "location": format_locations(chunk) or "문서",
            "source_url": str(chunk.get("source_url") or ""),
            "attachment_url": str(chunk.get("attachment_url") or ""),
        }
        key = (record["source_url"], record["filename"], record["location"])
        if key not in seen:
            seen.add(key)
            records.append(record)
    return records
