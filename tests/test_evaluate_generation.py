import pytest

from sejong_rag.evaluate_generation import evaluate_answer, metric_summary


def test_evaluate_answer_accepts_alternative_and_citation():
    answer = "최대 24학점입니다. [p.5]\n출처: https://www.sejong.ac.kr/x"
    result = evaluate_answer(answer, 5, [["24학점", "스물네 학점"]])
    assert result["answer_values_ok"] is True
    assert result["page_citation_ok"] is True
    assert result["official_source_ok"] is True


def test_evaluate_answer_requires_every_concept():
    result = evaluate_answer("최대 21학점", 6, [["16학점"], ["21학점"]])
    assert result["concept_results"] == [False, True]
    assert result["answer_values_ok"] is False


def test_metric_summary_counts_all_checks():
    rows = [
        {"answer_values_ok": True, "page_citation_ok": True, "official_source_ok": True},
        {"answer_values_ok": True, "page_citation_ok": False, "official_source_ok": True},
    ]
    summary = metric_summary(rows)
    assert summary["answer_value_accuracy"] == 1.0
    assert summary["page_citation_rate"] == 0.5
    assert summary["all_automatic_checks_rate"] == 0.5


def test_metric_summary_rejects_empty_rows():
    with pytest.raises(ValueError):
        metric_summary([])


def test_evaluate_answer_accepts_multiple_pages_in_one_citation():
    answer = "개강일은 9월 1일입니다 [p. 3, p. 5]. 출처: https://www.sejong.ac.kr/x"
    result = evaluate_answer(answer, 5, [["9월 1일"]])
    assert result["page_citation_ok"] is True
