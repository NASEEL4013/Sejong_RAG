import pytest

from sejong_rag.context_builder import format_context, select_context_chunks


def sample_chunks():
    return [
        {
            "chunk_id": f"p1-{index}",
            "document_id": "doc",
            "page": 1,
            "page_chunk_index": index,
            "text": f"내용 {index}",
            "source_url": "https://example.test/notice",
        }
        for index in range(1, 4)
    ]


def test_select_context_adds_same_page_neighbors_and_deduplicates():
    chunks = sample_chunks()
    results = [{"chunk": chunks[1]}, {"chunk": chunks[2]}]
    selected = select_context_chunks(results, chunks, neighbor_window=1)
    assert [chunk["chunk_id"] for chunk in selected] == ["p1-2", "p1-1", "p1-3"]
    assert selected[0]["context_relation"] == "retrieved"
    assert selected[1]["context_relation"] == "neighbor"


def test_select_context_respects_limits():
    chunks = sample_chunks()
    selected = select_context_chunks(
        [{"chunk": chunks[1]}], chunks, neighbor_window=1, max_chunks=2
    )
    assert len(selected) == 2


def test_format_context_keeps_provenance():
    chunk = sample_chunks()[0] | {
        "context_relation": "retrieved",
        "retrieval_rank": 1,
    }
    context = format_context([chunk])
    assert "[근거 1]" in context
    assert "페이지: 1" in context
    assert "청크 ID: p1-1" in context
    assert "https://example.test/notice" in context


def test_select_context_rejects_invalid_limit():
    with pytest.raises(ValueError):
        select_context_chunks([], [], max_chunks=0)


def test_compressed_context_defaults_are_shared():
    from sejong_rag.context_builder import (
        DEFAULT_MAX_CONTEXT_CHARS,
        DEFAULT_MAX_CONTEXT_CHUNKS,
        DEFAULT_RETRIEVAL_TOP_K,
    )
    assert DEFAULT_RETRIEVAL_TOP_K == 3
    assert DEFAULT_MAX_CONTEXT_CHUNKS == 6
    assert DEFAULT_MAX_CONTEXT_CHARS == 6_000
