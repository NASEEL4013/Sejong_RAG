# Gemini API 연결
> 현재 기준 안내(2026-08-10): 파이프라인 완성 후 전체 자동 테스트는 148개 통과했다. 본문에 더 작은 테스트 수가 있으면 해당 기능을 만들던 당시의 기록이다. 최신 전체 성능은 [독립 평가 보고서](HOLDOUT_EVALUATION.md)를 참고한다.


## 선택

- 생성 모델: `gemini-3.5-flash-lite`
- SDK: Google 공식 `google-genai`
- 최대 출력: 1,024 tokens
- 키 환경변수: `GEMINI_API_KEY` 또는 `GOOGLE_API_KEY`

Gemini 3.5부터 sampling parameter인 temperature가 폐기되어 전달하지 않는다.

## API 키 설정

1. Google AI Studio에서 Gemini API key를 발급한다.
2. 프로젝트 루트에서 예제 파일을 복사한다.

```bash
cp .env.example .env
```

3. `.env`를 열어 본인의 키를 입력한다.

```dotenv
GEMINI_API_KEY=발급받은_키
```

`.env`는 `.gitignore`에 포함된다. 키를 코드, README, 터미널 출력 또는 Git commit에 넣지 않는다.

## 실행

```bash
.venv/bin/sejong-ask \
  "과목을 다시 들으면 받을 수 있는 최고 성적은?"
```

실행 과정:

1. `.env`에서 API key 존재 여부 확인
2. E5로 질문 임베딩
3. 키워드+벡터 RRF 검색
4. 같은 페이지 인접 chunk 확장
5. 근거 전용 prompt 생성
6. Gemini 2.5 Flash-Lite 호출
7. 모델 답변, 실제 전달한 공식 페이지·URL, token 사용량 출력

## 데이터 주의사항

Gemini 무료 서비스에 보낸 입력과 출력은 Google 제품 개선에 사용될 수 있으며 사람이 검토할 수 있다. 현재 공개 수강편람은 전송할 수 있지만 이름, 학번, 연락처, 개인 성적, 상담 내용 등 개인정보나 민감정보를 질문에 포함하지 않는다.

## 현재 검증 상태

- 공식 SDK 설치: 완료
- 가짜 client를 사용한 요청 파라미터 테스트: 완료
- 키 누락 시 외부 호출 전 중단: 완료
- 실제 Gemini 응답: 성공
- 15문항 생성 평가: 자동 검사 15/15, 수동 근거 대조 15/15
- 전체 token: 55,708
- 전체 자동 테스트: 36 passed

## GEMINI-001 실제 호출 결과

기존 `gemini-2.5-flash-lite`는 신규 사용자에게 제공되지 않아 404를 반환했다. 공식 안정 대체 모델인 `gemini-3.5-flash-lite`로 변경하고, Gemini 3.5에서 폐기된 `temperature` 설정을 제거했다.

질문:

```text
과목을 다시 들으면 받을 수 있는 최고 성적은?
```

결과:

- API 호출: 성공
- 답변 핵심: A0를 초과할 수 없음
- 예외: 학점교류 제외
- 인용: `[p.34]`
- 공식 URL: 포함
- 입력: 7,252 tokens
- 출력: 81 tokens
- 전체: 7,333 tokens

이 호출은 초기 연결 검증 기록이다. 이후 통합 근거 압축과 32문항 답변 평가를 완료했으며 현재 결과는 [독립 평가 보고서](HOLDOUT_EVALUATION.md)를 참고한다.
