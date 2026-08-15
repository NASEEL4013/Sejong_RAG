from types import SimpleNamespace

import pytest

from sejong_rag.gemini_rag import generate_with_gemini, unique_sources
from sejong_rag.prompt_builder import build_messages


class FakeModels:
    def __init__(self):
        self.call = None

    def generate_content(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(
            text="재수강 최고 성적은 A0입니다. [p.34]",
            usage_metadata=SimpleNamespace(total_token_count=123),
        )


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def test_generate_with_gemini_passes_prompt_and_config():
    client = FakeClient()
    messages = build_messages("재수강 최고 성적은?", [])
    answer, usage = generate_with_gemini(client, messages)
    call = client.models.call
    assert answer.startswith("재수강 최고 성적")
    assert usage.total_token_count == 123
    assert call["model"] == "gemini-3.5-flash-lite"
    assert "공식 근거" in call["contents"]
    assert call["config"].temperature is None
    assert call["config"].max_output_tokens == 1024
    assert "근거의 내용만" in call["config"].system_instruction


def test_generate_with_gemini_rejects_wrong_message_order():
    with pytest.raises(ValueError):
        generate_with_gemini(FakeClient(), [{"role": "user", "content": "질문"}])


def test_generate_with_gemini_rejects_empty_response():
    client = FakeClient()
    client.models.generate_content = lambda **kwargs: SimpleNamespace(text="")
    with pytest.raises(RuntimeError, match="텍스트 답변"):
        generate_with_gemini(client, build_messages("질문", []))


def test_unique_sources_deduplicates_page_and_url():
    chunks = [
        {"page": 34, "source_url": "https://example.test"},
        {"page": 34, "source_url": "https://example.test"},
        {"page": 35, "source_url": "https://example.test"},
    ]
    assert unique_sources(chunks) == [
        (34, "https://example.test"),
        (35, "https://example.test"),
    ]

def test_validate_answer_rejects_scope_conflict_old_value_and_bad_citation():
    from sejong_rag.gemini_rag import validate_answer
    prompt = '<evidence id="1">x</evidence><answer_scope>x</answer_scope><resolved_conflict>x</resolved_conflict>'
    errors = validate_answer("영어영문학 TEPS 556점 [근거 9]", prompt)
    assert len(errors) == 4

def test_normalize_citations():
    from sejong_rag.gemini_rag import normalize_citations
    assert normalize_citations("내용 [1] [근거 2]") == "내용 [근거 1] [근거 2]"

def test_apply_resolved_constraints_removes_specific_major_and_old_teps():
    from sejong_rag.gemini_rag import apply_resolved_constraints
    prompt = "<answer_scope>x</answer_scope><resolved_conflict>x</resolved_conflict>"
    answer = apply_resolved_constraints("공통 TEPS 556점이다. 영어영문학은 637점이다. 총 130학점이다.", prompt)
    assert "TEPS 301점" in answer
    assert "영어영문학" not in answer
    assert "총 130학점" in answer

def test_apply_resolved_constraints_removes_only_invalid_citations():
    from sejong_rag.gemini_rag import apply_resolved_constraints
    prompt = '<evidence id="1">x</evidence><evidence id="2">y</evidence>'
    answer = apply_resolved_constraints("맞음 [근거 1], 잘못됨 [9]", prompt)
    assert "[근거 1]" in answer
    assert "[9]" not in answer

def test_apply_resolved_constraints_cleans_composite_citations_and_teps_source():
    from sejong_rag.gemini_rag import apply_resolved_constraints
    prompt = '<evidence id="1">TEPS 556</evidence><evidence id="2">TEPS 301</evidence><resolved_conflict>x</resolved_conflict>'
    answer = apply_resolved_constraints("TEPS 556점 [근거 1, resolved_conflict]\n요청 [근거 1, 9]\n- (", prompt)
    assert "TEPS 301점 [근거 2]" in answer
    assert "resolved_conflict" not in answer
    assert "9" not in answer
    assert "- (" not in answer

def test_validate_answer_requires_all_resolved_single_major_values():
    from sejong_rag.gemini_rag import validate_answer
    prompt = '<evidence id="1">x</evidence><resolved_single_major department="소프트웨어학과">교양필수=14학점, 전공합계=72학점</resolved_single_major>'
    errors = validate_answer("전공합계는 72학점이다 [근거 1]", prompt)
    assert "학과별 표의 교양필수 14학점을 누락함" in errors

def test_apply_constraints_accepts_resolved_single_major_block():
    from sejong_rag.gemini_rag import apply_resolved_constraints
    prompt = '<evidence id="1">x</evidence><resolved_single_major>교양필수=14학점</resolved_single_major>'
    assert apply_resolved_constraints("교양필수 14학점 [근거 1]", prompt)

def test_validate_answer_requires_classic_reading_when_evidence_has_it():
    from sejong_rag.gemini_rag import validate_answer
    prompt = '<question>졸업요건</question><evidence id="1">고전독서인증 검증시험 통과</evidence>'
    assert "공식 근거의 고전독서인증 요건을 누락함" in validate_answer("영어인증만 필요 [근거 1]", prompt)
    assert not validate_answer("고전독서인증이 필요 [근거 1]", prompt)

def test_validate_answer_requires_courses_from_verified_structured_evidence():
    from sejong_rag.gemini_rag import validate_answer
    prompt = '<evidence id="1">\n검증된 과목별 졸업요건\n- 기초미적분학: 3학점, 1학년 1학기\n- 공업수학1: 3학점, 1학년 2학기\n</evidence>'
    errors = validate_answer("기초미적분학을 이수한다 [근거 1]", prompt)
    assert "검증된 필수 과목 공업수학1을 누락함" in errors

def test_apply_constraints_injects_all_verified_credit_facts():
    from sejong_rag.gemini_rag import apply_resolved_constraints, validate_answer
    prompt = '<evidence id="1">검증된 단일전공 졸업 이수학점표:</evidence><resolved_single_major>공통교양필수=13학점, 전공합계=72학점, 졸업학점=130학점</resolved_single_major>'
    answer = apply_resolved_constraints("전공 설명입니다.", prompt)
    assert "공통교양필수=13학점" in answer
    assert "졸업학점=130학점" in answer
    assert "[근거 1]" in answer
    assert not validate_answer(answer, prompt)

def test_classic_reading_validation_uses_question_not_unrelated_evidence():
    from sejong_rag.gemini_rag import validate_answer
    prompt = '<question>계절학기 최대 학점은?</question><evidence id="1">졸업요건 고전독서인증</evidence>'
    assert not validate_answer("최대 3학점 [근거 1]", prompt)

def test_validate_answer_rejects_personal_identifier_request():
    from sejong_rag.gemini_rag import validate_answer
    prompt='<question>내가 졸업 가능한지 확인해줘</question><evidence id="1">공식 공지</evidence>'
    assert "개인 학적 판정을 위해 개인정보 제공을 요청함" in validate_answer("학번 또는 성명을 알려주세요 [근거 1]",prompt)

def test_validate_answer_requires_conditions_for_generic_graduation_question():
    from sejong_rag.gemini_rag import validate_answer
    prompt='<question>졸업하려면 몇 학점?</question><answer_scope>공통만</answer_scope><evidence id="1">공식 공지</evidence>'
    assert any("전공 형태" in x for x in validate_answer("알 수 없습니다 [근거 1]",prompt))
