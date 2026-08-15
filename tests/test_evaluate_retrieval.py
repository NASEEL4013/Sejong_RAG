import pytest

from sejong_rag.evaluate_retrieval import metrics_from_ranks, relevant_chunk_ids


def test_relevant_chunk_requires_page_and_all_phrases():
    chunks = [
        {"chunk_id": "right", "page": 34, "text": "성적은 A0를 초과할 수 없다"},
        {"chunk_id": "wrong-page", "page": 35, "text": "성적은 A0를 초과할 수 없다"},
        {"chunk_id": "missing", "page": 34, "text": "재수강 안내"},
    ]
    case = {"expected_page": 34, "required_phrases": ["A0", "초과할 수 없다"]}
    assert relevant_chunk_ids(chunks, case) == {"right"}


def test_metrics_calculate_hit_rate_and_mrr():
    metrics = metrics_from_ranks([1, 2, 6, None], hit_k=5)
    assert metrics["hit_rate@5"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx((1 + 0.5 + 1 / 6) / 4)


def test_metrics_reject_empty_evaluation():
    with pytest.raises(ValueError):
        metrics_from_ranks([])
