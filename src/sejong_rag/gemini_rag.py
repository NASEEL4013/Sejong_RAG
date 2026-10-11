"""통합 검색·문맥·프롬프트 파이프라인을 Gemini 답변 생성에 연결한다."""
from __future__ import annotations
import argparse, os, re
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from sejong_rag.prompt_builder import build_messages
from sejong_rag.graduation_requirements import credit_structured_chunk, structured_chunk
from sejong_rag.unified_context import DEFAULT_MAX_CONTEXT_CHARS, DEFAULT_MAX_CONTEXT_CHUNKS, DEFAULT_RETRIEVAL_TOP_K, select_unified_context, source_records
from sejong_rag.unified_search import DEFAULT_REVIEW, DEFAULT_TRUSTED, ambiguity_warning, search_indexes
from sejong_rag.vector_search import load_index

DEFAULT_GENERATION_MODEL = "gemini-3.5-flash-lite"

def get_api_key() -> str:
    """환경변수에서 키를 읽되 값 자체는 로그에 남기지 않는다."""
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Gemini API 키가 없습니다. .env.example을 .env로 복사한 뒤 GEMINI_API_KEY에 Google AI Studio 키를 입력하세요.")
    return key

def _evidence_count(user_prompt: str) -> int:
    return len(re.findall(r'<evidence id="\d+">', user_prompt))

def validate_answer(answer: str, user_prompt: str) -> list[str]:
    """출력 범위, 해결된 충돌, 인용 번호를 기계적으로 검사한다."""
    violations = []
    if "<answer_scope>" in user_prompt and ("학과" not in answer or not re.search(r"단일|복수|전공 형태", answer)):
        violations.append("학과 미지정 졸업 질문에 필요한 학과와 전공 형태 안내를 누락함")
    question_match = re.search(r"<question>(.*?)</question>", user_prompt, re.DOTALL)
    personal_question = question_match and re.search(r"내가|내 졸업|본인", question_match.group(1))
    if personal_question and re.search(r"학번|성명", answer) and re.search(r"알려|제공|입력", answer):
        violations.append("개인 학적 판정을 위해 개인정보 제공을 요청함")
    if "<answer_scope>" in user_prompt and "영어영문학" in answer:
        violations.append("학과 미지정 질문에 특정 학과 예외를 포함함")
    if "<resolved_conflict>" in user_prompt and re.search(r"TEPS\s*556", answer, re.IGNORECASE):
        violations.append("폐기된 TEPS 556점 기준을 사용함")
    question_match = re.search(r"<question>(.*?)</question>", user_prompt, re.DOTALL)
    question_text = question_match.group(1) if question_match else user_prompt
    if re.search(r"졸업\s*(?:요건|기준)", question_text) and "고전독서인증" in user_prompt and "고전독서" not in answer:
        violations.append("공식 근거의 고전독서인증 요건을 누락함")
    verified_block = re.search(r"<evidence id=\"\d+\">(?:(?!</evidence>).)*검증된 과목별 졸업요건(?:(?!</evidence>).)*</evidence>", user_prompt, re.DOTALL)
    if verified_block:
        required_courses = re.findall(r"^- ([^:\n]+): \d+학점", verified_block.group(0), re.MULTILINE)
        for course in required_courses:
            if course not in answer:
                violations.append(f"검증된 필수 과목 {course}을 누락함")
    table = re.search(r"<resolved_single_major[^>]*>(.*?)</resolved_single_major>", user_prompt, re.DOTALL)
    if table:
        aliases = {"전공합계": r"(?:전공합계|전공학점계|전공[^\n]{0,10}합계)", "졸업학점": r"(?:졸업학점|졸업이수학점|총[^\n]{0,10}졸업학점)"}
        for label, value in re.findall(r"([^,=\n]+)=(\d+)학점", table.group(1)):
            label = label.strip()
            label_pattern = aliases.get(label, re.escape(label))
            if not re.search(label_pattern + r".{0,40}" + re.escape(value) + r"\s*학점", answer, re.DOTALL):
                violations.append(f"학과별 표의 {label} {value}학점을 누락함")
    maximum = _evidence_count(user_prompt)
    cited = [int(value) for value in re.findall(r"\[(?:근거\s*)?(\d+)\]", answer)]
    if any(number < 1 or number > maximum for number in cited):
        violations.append("존재하지 않는 근거 번호를 인용함")
    return violations

def _inject_verified_credit_summary(answer: str, user_prompt: str) -> str:
    table = re.search(r"<resolved_single_major[^>]*>(.*?)</resolved_single_major>", user_prompt, re.DOTALL)
    if not table:
        return answer
    facts = ", ".join(part.strip() for part in table.group(1).split(",") if part.strip())
    source_number = None
    for number, body in re.findall(r"<evidence id=\"(\d+)\">(.*?)</evidence>", user_prompt, re.DOTALL):
        if "검증된 단일전공 졸업 이수학점표:" in body:
            source_number = number
            break
    citation = f" [근거 {source_number}]" if source_number else ""
    return f"검증된 단일전공 학점 기준: {facts}{citation}\n\n{answer}"


def apply_resolved_constraints(answer: str, user_prompt: str) -> str:
    """프로그램이 확정한 좁은 충돌·범위 규칙을 답변에 결정적으로 적용한다."""
    if "<answer_scope>" in user_prompt:
        answer = re.sub(r"(?:단,?\s*)?영어영문학[^.\n]*(?:\.)?", "", answer)
    maximum = _evidence_count(user_prompt)
    def clean_citation(match):
        numbers = [int(value) for value in re.findall(r"\d+", match.group(1))]
        valid = list(dict.fromkeys(number for number in numbers if 1 <= number <= maximum))
        return "[" + ", ".join(f"근거 {number}" for number in valid) + "]" if valid else ""
    answer = re.sub(r"\[(?:근거\s*)?([^]\n]+)\]", clean_citation, answer)
    if "<resolved_conflict>" in user_prompt:
        answer = re.sub(r"TEPS\s*556(?=점?)", "TEPS 301", answer, flags=re.IGNORECASE)
        match = re.search(r"<evidence id=\"(\d+)\">(?:(?!</evidence>).)*TEPS\s*301", user_prompt, re.IGNORECASE | re.DOTALL)
        if match:
            answer = "\n".join(re.sub(r"\[(?:근거\s*)?[^]\n]+\]", f"[근거 {match.group(1)}]", line) if re.search(r"TEPS\s*301", line, re.IGNORECASE) else line for line in answer.splitlines())
    answer = "\n".join(line for line in answer.splitlines() if not re.fullmatch(r"\s*-?\s*\(?\s*", line))
    return _inject_verified_credit_summary(answer, user_prompt)
def normalize_citations(answer: str) -> str:
    """[1] 형식은 표준 [근거 1] 형식으로 바꾼다."""
    return re.sub(r"\[(\d+)\]", r"[근거 \1]", answer)

def generate_with_gemini(client: Any, messages: list[dict[str, str]], model: str = DEFAULT_GENERATION_MODEL, max_output_tokens: int = 1024) -> tuple[str, Any]:
    """Gemini 답변을 생성하고 위반 시 한 번 교정한 뒤 다시 검증한다."""
    if [message.get("role") for message in messages] != ["system", "user"]:
        raise ValueError("Gemini에는 system, user 순서의 메시지 두 개가 필요합니다.")
    from google.genai import types
    user_content = messages[1]["content"]
    usage = None
    for attempt in range(2):
        response = client.models.generate_content(model=model, contents=user_content, config=types.GenerateContentConfig(system_instruction=messages[0]["content"], max_output_tokens=max_output_tokens))
        answer = (response.text or "").strip()
        answer = apply_resolved_constraints(answer, messages[1]["content"])
        usage = getattr(response, "usage_metadata", None)
        if not answer:
            raise RuntimeError("Gemini가 텍스트 답변을 반환하지 않았습니다.")
        violations = validate_answer(answer, messages[1]["content"])
        if not violations:
            return normalize_citations(answer), usage
        if attempt == 0:
            user_content += "\n\n<correction>\n이전 답변의 다음 오류를 모두 수정해 다시 답하세요: " + "; ".join(violations) + "\n</correction>"
    raise RuntimeError("Gemini 답변이 정확성 검증을 통과하지 못했습니다: " + "; ".join(violations))

def unique_sources(chunks: list[dict[str, object]]) -> list[tuple[int, str]]:
    """이전 단일 PDF 파이프라인용 페이지·URL 중복 제거 함수."""
    seen, sources = set(), []
    for chunk in chunks:
        item = (int(chunk["page"]), str(chunk["source_url"]))
        if item not in seen:
            seen.add(item); sources.append(item)
    return sources

def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini 기반 세종대학교 통합 RAG 질문")
    parser.add_argument("query", help="자연어 질문")
    parser.add_argument("--trusted-index", type=Path, default=DEFAULT_TRUSTED)
    parser.add_argument("--review-index", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--include-review", action="store_true", help="사람 검토가 덜 된 OCR 근거도 검색")
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--retrieval-top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=300)
    parser.add_argument("--neighbor-window", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CONTEXT_CHUNKS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    args = parser.parse_args()
    try:
        api_key = get_api_key()
    except RuntimeError as error:
        parser.error(str(error))
    _, chunks, _ = load_index(args.trusted_index)
    if args.include_review:
        _, review_chunks, _ = load_index(args.review_index)
        chunks += review_chunks
    results = search_indexes(args.query, args.trusted_index, args.review_index if args.include_review else None, top_k=args.retrieval_top_k, candidate_k=args.candidate_k)
    warning = ambiguity_warning(args.query, results)
    if warning:
        print(f"\n[질문 확인 필요]\n\n{warning}")
        return
    context_chunks = select_unified_context(results, chunks, neighbor_window=args.neighbor_window, max_chunks=args.max_chunks, max_chars=args.max_chars)
    credit_chunk = credit_structured_chunk(args.query)
    if credit_chunk is not None:
        context_chunks.append(credit_chunk)
    verified_chunk = structured_chunk(args.query)
    if verified_chunk is not None:
        context_chunks.append(verified_chunk)
    messages = build_messages(args.query, context_chunks)
    from google import genai
    answer, usage = generate_with_gemini(genai.Client(api_key=api_key), messages, model=args.model, max_output_tokens=args.max_output_tokens)
    print("\n[답변]\n"); print(answer)
    print("\n[프로그램이 전달한 공식 근거]\n")
    for source in source_records(context_chunks):
        print(f"- {source['title']} / {source['filename']} / {source['location']}")
        print(f"  공지: {source['source_url']}")
        if source["attachment_url"]: print(f"  첨부: {source['attachment_url']}")
        if source.get("image_url"): print(f"  본문 이미지: {source['image_url']}")
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        answer_tokens = getattr(usage, "candidates_token_count", None)
        total_tokens = getattr(usage, "total_token_count", None)
        print("\n[사용량]"); print(f"입력={prompt_tokens or '-'} 출력={answer_tokens or '-'} 전체={total_tokens or '-'} tokens")
if __name__ == "__main__": main()
