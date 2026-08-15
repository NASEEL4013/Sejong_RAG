import pytest

from sejong_rag.hybrid_search import expand_query_for_keyword, reciprocal_rank_fusion


def test_expand_query_maps_student_wording_to_official_term():
    query = "과목을 다시 들으면 받을 수 있는 최고 성적은?"
    expanded = expand_query_for_keyword(query)
    assert "재수강" in expanded
    assert "성적 제한" in expanded
    assert "A0" not in expanded


def test_rrf_rewards_result_found_by_both_searches():
    results = reciprocal_rank_fusion(
        vector_ids=["vector-only", "both"],
        keyword_ids=["both", "keyword-only"],
        rrf_k=60,
    )
    assert results[0]["chunk_id"] == "both"
    assert results[0]["vector_rank"] == 2
    assert results[0]["keyword_rank"] == 1


def test_rrf_rejects_invalid_constant():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], [], rrf_k=0)
