import numpy as np
import pytest

from sejong_rag.embed_chunks import build_passages
from sejong_rag.vector_search import rank_by_similarity


def test_build_passages_adds_e5_prefix():
    assert build_passages([{"text": "졸업학점 안내"}]) == ["passage: 졸업학점 안내"]


def test_rank_by_similarity_returns_highest_dot_product_first():
    documents = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32)
    results = rank_by_similarity(documents, np.array([1.0, 0.0]), top_k=2)
    assert [index for index, _ in results] == [0, 2]
    assert results[0][1] == pytest.approx(1.0)


def test_rank_by_similarity_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="차원"):
        rank_by_similarity(np.zeros((2, 3)), np.zeros(2), top_k=1)
