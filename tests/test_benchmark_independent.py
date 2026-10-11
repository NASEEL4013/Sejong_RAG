"""Independent measurement tests; synthetic questions and sources only.

Never open the frozen benchmark questions, load an embedding model, or call an
external generation service in this module. Exact phrase checks are deliberately
not treated as semantic correctness.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from sejong_rag import benchmark


def options():
    return [
        {"notice_id": "notice1-1", "evidence_any": [["필수 대상", "정확한 기한"]]},
        {"notice_id": "notice1-2", "evidence_any": [["대체 근거"]]},
    ]


def source(body, *, number=1, notice="notice1-1", title="가상의 공지"):
    return {
        "id": number,
        "body": body,
        "text": f"문서 제목: {title}\n\n{body}",
        "document_title": title,
        "source_url": f"https://example.test/{notice}",
        "source_aliases": [{"parent_document_id": notice}],
        "source_spans": [{"record_id": f"record-{number}", "start": 0,
                          "end": len(body), "locator": {"page": number}}],
        "chunk_ids": [f"chunk-{number}"],
        "token_count": 20,
    }


def test_retrieval_does_not_count_matching_title_as_answer_evidence():
    bundles = [source("필수 대상", title="정확한 기한")]
    row = benchmark.retrieval_entry(bundles, options())
    assert row["first_relevant_rank"] is None
    assert row["top5"][0]["is_evidence"] is False


def test_retrieval_keeps_scope_and_phrase_label_paired():
    assert benchmark.retrieval_entry([source("대체 근거", notice="notice1-1")], options())["first_relevant_rank"] is None
    assert benchmark.retrieval_entry([source("필수 대상 정확한 기한", notice="notice1-2")], options())["first_relevant_rank"] is None
    assert benchmark.retrieval_entry([source("대체 근거", notice="notice1-2")], options())["first_relevant_rank"] == 1


def test_retrieval_cannot_join_incomplete_different_bundles_into_one_gold_match():
    bundles = [source("필수 대상", number=1), source("정확한 기한", number=2)]
    assert benchmark.retrieval_entry(bundles, options())["first_relevant_rank"] is None


def test_retrieval_matching_is_whitespace_insensitive_but_not_scope_insensitive():
    body = "필수\n대상\t정확한\n기한"
    assert benchmark.retrieval_entry([source(body)], options())["first_relevant_rank"] == 1
    assert benchmark.retrieval_entry([source(body, notice="notice1-999")], options())["first_relevant_rank"] is None


def test_retrieval_accepts_full_answer_in_an_alias_without_modifying_input():
    item = source("필수 대상 정확한 기한", notice="notice1-999")
    item["source_aliases"].append({"parent_document_id": "notice1-1"})
    before = deepcopy(item)
    result = benchmark.retrieval_entry([item], options())
    assert result["first_relevant_rank"] == 1
    assert item == before


def test_empty_or_unlabelled_context_has_no_relevant_rank():
    assert benchmark.retrieval_entry([], options())["first_relevant_rank"] is None
    assert benchmark.retrieval_entry([source("필수 대상 정확한 기한")], [])["first_relevant_rank"] is None


def test_rank_six_is_not_a_hit_at_five():
    bundles = [source("관련 없는 내용", number=i) for i in range(1, 6)]
    bundles.append(source("필수 대상 정확한 기한", number=6))
    result = benchmark.retrieval_entry(bundles, options())
    assert result["first_relevant_rank"] is None
    assert len(result["top5"]) <= 5


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
def test_retrieval_rank_is_one_based(rank):
    bundles = [source("관련 없는 내용", number=i) for i in range(1, rank)]
    bundles.append(source("필수 대상 정확한 기한", number=rank))
    assert benchmark.retrieval_entry(bundles, options())["first_relevant_rank"] == rank


def answer_case(case_id="SYN1", *, kind="answerable"):
    return {
        "id": case_id, "kind": kind, "category": "synthetic",
        "query": "가상 공지의 안내는 뭐야?", "answer_note": "라벨 전용 정답 설명",
        "expected_status": ["answered"] if kind == "answerable" else ["insufficient_evidence"],
        "answer_concepts": [["정확한 기한"]] if kind == "answerable" else [],
        "evidence_options": options() if kind == "answerable" else [],
        "gold_sources": [], "year": None, "rubric": "합성 평가용",
    }


def request():
    return {"query": "가상 공지의 안내는 뭐야?",
            "sources": [source("필수 대상 정확한 기한")],
            "messages": [{"role": "system", "content": "합성 시스템"},
                         {"role": "user", "content": "질문과 검색 근거만"}]}


def result(*, status="answered", citations=None):
    claims = [{"text": "정확한 기한이 있어.", "citations": [1] if citations is None else citations}]
    if status != "answered":
        claims = []
    return {"status": status, "claims": claims,
            "answer": "정확한 기한이 있어." if status == "answered" else "제공된 근거가 없어.",
            "message": "" if status == "answered" else "제공된 근거가 없어.",
            "validation_passed": True, "cited_sources": [], "usage": {}}


def completed(case, *, status="answered", citations=None, wall_ms=20):
    generated = result(status=status, citations=citations)
    return {"state": "completed", "result": generated, "wall_ms": wall_ms,
            **benchmark.grade(case, request(), generated)}


def test_grade_requires_gold_scope_not_just_a_valid_citation_number():
    built = request()
    built["sources"][0]["source_aliases"] = [{"parent_document_id": "notice1-999"}]
    grade = benchmark.grade(answer_case(), built, result())
    assert grade["checks"]["citation_ids_valid"] is True
    assert grade["checks"]["cited_gold_proxy"] is False
    assert grade["automatic_proxy_pass"] is False


@pytest.mark.parametrize("citations", [[], [0], [2], [True], ["1"]])
def test_grade_does_not_assume_valid_citations_from_validation_flag(citations):
    graded = benchmark.grade(answer_case(), request(), result(citations=citations))
    assert graded["claims_with_valid_citations"] == 0
    assert graded["claim_citation_coverage"] == 0
    assert graded["automatic_proxy_pass"] is False


def test_grade_exposes_phrase_proxy_without_claiming_semantic_correctness():
    value = result()
    value["claims"][0]["text"] = "정확한 기한이라는 안내는 없어."
    value["answer"] = value["claims"][0]["text"]
    graded = benchmark.grade(answer_case(), request(), value)
    assert graded["checks"]["content_concepts_match"] is True
    assert "semantic_correct" not in graded or graded["semantic_correct"] is None
    assert "faithfulness" not in graded or graded["faithfulness"] is None


def test_grade_detects_one_uncited_claim_even_if_another_cites_gold():
    value = result()
    value["claims"].append({"text": "추가로 수수료가 없어.", "citations": []})
    graded = benchmark.grade(answer_case(), request(), value)
    assert graded["claim_count"] == 2
    assert graded["claims_with_valid_citations"] == 1
    assert graded["claim_citation_coverage"] == .5
    assert graded["automatic_proxy_pass"] is False


def measurement_fixture():
    cases = [answer_case("SYN1"), answer_case("SYN2"),
             answer_case("SYN3", kind="unanswerable"),
             answer_case("SYN4", kind="clarification")]
    rows = []
    for case, raw_rank, current_rank in zip(cases, [None, 3, 1, 1], [1, None, 1, 1]):
        rows.append({"id": case["id"], "kind": case["kind"], "query": case["query"],
                     "baseline": {"first_relevant_rank": raw_rank, "top5": [],
                                  "search_ms": 3, "extra_query_embedding_ms": 2},
                     "current": {"first_relevant_rank": current_rank, "top5": [],
                                 "retrieval_with_embedding_ms": 7},
                     "generation": {"state": "not_run"},
                     "semantic_correct": None, "faithfulness": None})
    rows[0]["generation"] = completed(cases[0], wall_ms=20)
    rows[1]["generation"] = {"state": "error", "wall_ms": 30}
    rows[2]["generation"] = completed(cases[2], status="insufficient_evidence", wall_ms=10)
    return rows, cases


def summary(rows, cases, *, live=True, api_records=None):
    return benchmark.summarize(rows, cases, live=live, api_records=api_records or [],
                               load_ms=100, warmup_ms=50)


def test_retrieval_metrics_exclude_all_behavior_questions_from_denominator():
    rows, cases = measurement_fixture()
    report = summary(rows, cases)
    assert report["answerable_count"] == 2
    assert report["behavior_count"] == 2
    for name in ("baseline", "current"):
        assert report["retrieval"][name]["denominator"] == 2
        assert report["retrieval"][name]["completed"] == 2
        assert report["retrieval"][name]["hit_at_5"] == .5
    assert report["retrieval"]["current"]["hit_at_1"] == .5
    assert report["retrieval"]["current"]["mrr_at_5"] == .5
    assert report["retrieval"]["baseline"]["hit_at_1"] == 0
    assert report["retrieval"]["baseline"]["mrr_at_5"] == pytest.approx(1 / 6)


def test_partial_generation_uses_planned_denominator_and_exposes_errors():
    rows, cases = measurement_fixture()
    report = summary(rows, cases)
    assert report["planned"] == 4
    assert report["generation_completed"] == 2
    assert report["generation_errors"] == 1
    assert report["generation_proxy"]["denominator"] == 4
    assert report["generation_proxy"]["completed"] == 2
    assert report["generation_proxy"]["pass_count"] == 2
    assert report["generation_proxy"]["pass_rate"] == .5
    assert report["generation_proxy"]["status_match_rate"] == .5
    assert report["generation_proxy"]["answerable_content_match_rate"] == .5
    assert report["generation_proxy"]["behavior_status_match_rate"] == .5
    assert report["semantic_accuracy"] is None
    assert report["faithfulness"] is None


def test_offline_preparation_never_looks_like_zero_or_perfect_generation_accuracy():
    rows, cases = measurement_fixture()
    for row in rows:
        row["generation"] = {"state": "not_run"}
    report = summary(rows, cases, live=False)
    assert report["generation_completed"] == 0
    assert report["generation_proxy"]["pass_count"] is None
    for key, value in report["generation_proxy"].items():
        if key.endswith("rate"):
            assert value is None
    assert report["semantic_accuracy"] is None


def test_incomplete_retrieval_is_not_reported_as_a_full_run():
    rows, cases = measurement_fixture()
    del rows[1]["current"]
    report = summary(rows, cases)
    assert report["retrieval"]["current"]["denominator"] == 2
    assert report["retrieval"]["current"]["completed"] == 1
    assert report["retrieval"]["current"]["hit_at_5"] is None
    assert report["retrieval"]["current"]["mrr_at_5"] is None


def test_behavior_only_run_has_no_fabricated_search_accuracy():
    rows, cases = measurement_fixture()
    report = summary(rows[2:], cases[2:])
    for metric in report["retrieval"].values():
        assert metric["denominator"] == 0
        assert metric["hit_at_5"] is None
        assert metric["mrr_at_5"] is None


def test_no_completed_generation_still_keeps_planned_cases_in_live_denominator():
    cases = [answer_case()]
    report = summary([], cases)
    assert report["planned"] == 1
    assert report["generation_completed"] == 0
    assert report["generation_proxy"]["denominator"] == 1
    assert report["generation_proxy"]["pass_count"] == 0
    assert report["generation_proxy"]["pass_rate"] == 0


def api_record(number, *, status="returned", prompt=None, candidates=None, total=None):
    return {"case_id": "SYN1", "call_index": number, "status": status,
            "usage": {"prompt_token_count": prompt, "candidates_token_count": candidates,
                      "total_token_count": total}, "elapsed_ms": 10}


def test_usage_counts_all_attempts_and_distinguishes_unknown_from_zero():
    rows, cases = measurement_fixture()
    records = [api_record(1, prompt=100, candidates=5, total=105),
               api_record(2, prompt=120, candidates=10, total=130),
               api_record(3, status="error"), api_record(4, candidates=2)]
    usage = summary(rows, cases, api_records=records)["usage"]
    assert usage["api_call_count"] == 4
    assert usage["response_count"] == 3
    assert usage["prompt_token_count"] == 220
    assert usage["candidates_token_count"] == 17
    assert usage["total_token_count"] == 235
    assert usage["missing_count"]["prompt_token_count"] == 2
    assert usage["reported_count"]["prompt_token_count"] == 2


def test_usage_entirely_missing_is_null_not_zero():
    rows, cases = measurement_fixture()
    usage = summary(rows, cases, api_records=[api_record(1)])["usage"]
    for key in ("prompt_token_count", "candidates_token_count", "total_token_count"):
        assert usage[key] is None
        assert usage["missing_count"][key] == 1
        assert usage["reported_count"][key] == 0


def test_latency_keeps_load_and_query_generation_costs_separate():
    rows, cases = measurement_fixture()
    latency = summary(rows, cases)["latency_ms"]
    assert latency["initial_load"] == 100
    assert latency["warmup"] == 50
    assert latency["current_retrieval_with_embedding"]["count"] == 4
    assert latency["current_retrieval_with_embedding"]["p50"] == 7
    assert latency["baseline_search_only"]["p50"] == 3
    assert latency["generation_wall"]["count"] == 3


def test_ledger_bounds_calls_and_retains_usage_from_failed_question():
    class Models:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("sensitive-provider-message")
            return SimpleNamespace(text="{}", usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=3, total_token_count=13))

    original = SimpleNamespace(models=Models())
    ledger = benchmark.UsageLedger(original, max_calls=2)
    ledger.set_case("SYN1")
    ledger.models.generate_content(model="fake", contents="synthetic")
    ledger.set_case("SYN2")
    with pytest.raises(Exception):
        ledger.models.generate_content(model="fake", contents="synthetic")
    with pytest.raises(Exception):
        ledger.models.generate_content(model="fake", contents="synthetic")
    assert original.models.calls == 2
    assert len(ledger.records) == 2
    assert ledger.records[0]["case_id"] == "SYN1"
    assert ledger.records[0]["usage"]["total_token_count"] == 13
    assert ledger.records[1]["case_id"] == "SYN2"
    assert ledger.records[1]["status"] == "error"
    assert "sensitive-provider-message" not in repr(ledger.records)


def synthetic_evaluation(tmp_path, monkeypatch, *, responses=None, tamper_cases=False, bad_quote=False):
    """Run real freezing/scoring/ledger with fake search and fake SDK responses."""
    cases = [answer_case(f"SYN{i}") for i in range(1, 4)]
    original = tmp_path / "notice1-1.html"
    original.write_text("<p>필수 대상 정확한 기한</p>", encoding="utf-8")
    for index, case in enumerate(cases, 1):
        case["query"] = f"가상 질문 {index}"
        case["answer_note"] = "EVALUATOR_ONLY_SECRET_LABEL"
        case["gold_sources"] = [{"path": str(original), "notice_id": "notice1-1",
                                 "quotes": ["없는 인용" if bad_quote else "필수 대상 정확한 기한"]}]
    cases_path = tmp_path / "synthetic_cases.jsonl"
    cases_path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    index_path, normalized, selected = [tmp_path / filename for filename in
                                        ("index.npz", "normalized.jsonl", "selected.jsonl")]
    for path in (index_path, normalized, selected):
        path.write_text("synthetic fixture, never loaded as an actual index", encoding="utf-8")
    output = tmp_path / "output"
    state = {"runtime_calls": 0, "prepare_calls": [], "closed": False, "sdk_calls": []}

    class Engine:
        def search(self, query, vector, **kwargs):
            assert kwargs == {"strategy": "vector", "top_k": 5, "candidate_k": 100,
                              "max_bundle_tokens": 512, "total_tokens": 2200}
            return [source("필수 대상 정확한 기한")]

    class Model:
        def encode(self, query, **kwargs):
            assert query.startswith("query: ")
            assert "EVALUATOR_ONLY_SECRET_LABEL" not in query
            return [1., 0.]

    def load_runtime(*args):
        state["runtime_calls"] += 1
        assert (output / "code-snapshot.json").is_file()
        return Engine(), {"source_chunks": str(selected), "query_prefix": "query: ",
                          "model": "synthetic-no-download"}, Model()

    def prepare(query, *runtime):
        state["prepare_calls"].append(query)
        assert query in {case["query"] for case in cases}
        if tamper_cases:
            cases_path.write_text(cases_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        built = request()
        built["query"] = query
        built["messages"][1]["content"] = json.dumps({"question": query, "evidence": built["sources"]}, ensure_ascii=False)
        return built

    class Models:
        def generate_content(self, **kwargs):
            state["sdk_calls"].append(kwargs)
            frozen = json.loads((output / "frozen.json").read_text(encoding="utf-8"))
            assert len(frozen["cases"]) == 3
            assert len(list(output.glob("SYN*-request.json"))) == 3
            assert "EVALUATOR_ONLY_SECRET_LABEL" not in kwargs["contents"]
            assert "answer_concepts" not in kwargs["contents"]
            assert "gold_sources" not in kwargs["contents"]
            content = responses.pop(0) if responses is not None else json.dumps(
                {"status": "answered", "claims": [{"text": "정확한 기한이 있어.", "citations": [1]}], "message": ""},
                ensure_ascii=False)
            if isinstance(content, Exception):
                raise content
            return SimpleNamespace(text=content, usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=3, total_token_count=13))

    class Client:
        models = Models()

        def close(self):
            state["closed"] = True

    def create_client():
        assert (output / "frozen.json").is_file()
        return Client()

    monkeypatch.setattr(benchmark, "load_runtime", load_runtime)
    monkeypatch.setattr(benchmark, "prepare_request", prepare)
    monkeypatch.setattr(benchmark, "create_client", create_client)
    return {"cases_path": cases_path, "output_dir": output,
            "index": index_path, "normalized": normalized, "model": "synthetic-fake"}, state


def test_synthetic_end_to_end_freezes_requests_before_any_sdk_call_and_hides_labels(tmp_path, monkeypatch):
    kwargs, state = synthetic_evaluation(tmp_path, monkeypatch)
    report = benchmark.evaluate(**kwargs, live=True)
    assert report["completed"] is True
    assert report["integrity"]["passed"] is True
    assert report["error"] is None
    assert len(state["sdk_calls"]) == 3
    assert state["closed"] is True
    assert report["summary"]["generation_proxy"]["pass_rate"] == 1
    assert report["summary"]["semantic_accuracy"] is None
    assert report["summary"]["faithfulness"] is None
    assert report["summary"]["usage"]["api_call_count"] == 3
    frozen = json.loads((kwargs["output_dir"] / "frozen.json").read_text(encoding="utf-8"))
    assert str(kwargs["cases_path"].resolve()) in frozen["files_sha256"]
    for index in range(1, 4):
        path = kwargs["output_dir"] / f"SYN{index}-request.json"
        assert frozen["artifacts_sha256"][str(path.resolve())] == benchmark.file_sha256(path)


def test_partial_sdk_failure_preserves_all_cases_and_prior_attempt_usage(tmp_path, monkeypatch):
    good = json.dumps({"status": "answered", "claims": [{"text": "정확한 기한이 있어.", "citations": [1]}], "message": ""})
    kwargs, state = synthetic_evaluation(tmp_path, monkeypatch, responses=[
        good, "invalid json", RuntimeError("secret-provider-response-must-not-escape")])
    report = benchmark.evaluate(**kwargs, live=True)
    assert report["completed"] is False
    assert [row["generation"]["state"] for row in report["results"]] == ["completed", "error", "not_run"]
    assert len(report["results"]) == report["summary"]["planned"] == 3
    assert report["summary"]["generation_proxy"]["pass_rate"] == pytest.approx(1 / 3)
    assert report["summary"]["usage"]["api_call_count"] == 3
    assert report["summary"]["usage"]["total_token_count"] == 26
    assert report["summary"]["usage"]["missing_count"]["total_token_count"] == 1
    assert state["closed"] is True
    assert (kwargs["output_dir"] / "SYN1-generation.json").is_file()
    assert (kwargs["output_dir"] / "SYN2-generation.json").is_file()
    assert "secret-provider-response-must-not-escape" not in json.dumps(report)


def test_offline_evaluation_makes_no_sdk_calls(tmp_path, monkeypatch):
    kwargs, state = synthetic_evaluation(tmp_path, monkeypatch)
    report = benchmark.evaluate(**kwargs, live=False)
    assert report["completed"] is True
    assert state["sdk_calls"] == []
    assert state["closed"] is False
    assert report["summary"]["generation_proxy"]["pass_rate"] is None
    assert report["summary"]["usage"]["api_call_count"] == 0


def test_changed_case_file_prevents_all_external_calls_and_marks_integrity_failure(tmp_path, monkeypatch):
    kwargs, state = synthetic_evaluation(tmp_path, monkeypatch, tamper_cases=True)
    report = benchmark.evaluate(**kwargs, live=True)
    assert report["completed"] is False
    assert report["integrity"]["passed"] is False
    assert state["sdk_calls"] == []
    assert report["error"] is not None


def test_unsupported_original_quote_fails_before_runtime_loading(tmp_path, monkeypatch):
    kwargs, state = synthetic_evaluation(tmp_path, monkeypatch, bad_quote=True)
    report = benchmark.evaluate(**kwargs, live=True)
    assert report["completed"] is False
    assert report["error"] is not None
    assert state["runtime_calls"] == 0
    assert state["sdk_calls"] == []


def test_existing_evaluation_directory_is_never_overwritten(tmp_path, monkeypatch):
    kwargs, state = synthetic_evaluation(tmp_path, monkeypatch)
    kwargs["output_dir"].mkdir()
    sentinel = kwargs["output_dir"] / "keep.txt"
    sentinel.write_text("keep previous results", encoding="utf-8")
    with pytest.raises(FileExistsError):
        benchmark.evaluate(**kwargs, live=True)
    assert sentinel.read_text(encoding="utf-8") == "keep previous results"
    assert state["runtime_calls"] == 0
    assert state["sdk_calls"] == []
