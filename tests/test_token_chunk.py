import pytest

from sejong_rag.token_chunk import window_token_spans


def test_window_token_spans_uses_overlap():
    offsets = [(index, index + 1) for index in range(10)]
    spans = window_token_spans(offsets, max_tokens=4, overlap_tokens=1)
    assert spans == [(0, 4, 4), (3, 7, 4), (6, 10, 4)]


def test_window_token_spans_ignores_empty_special_offsets():
    assert window_token_spans([(0, 0), (0, 2), (2, 4)], 3, 0) == [
        (0, 4, 2)
    ]


def test_window_token_spans_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        window_token_spans([(0, 1)], max_tokens=4, overlap_tokens=4)
