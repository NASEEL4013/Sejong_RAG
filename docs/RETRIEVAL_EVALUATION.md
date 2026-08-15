# 검색 평가 기준선
> 역사적 단계 기록: 이 문서는 개발 당시의 구현과 수치를 보존한다. 현재 상태와 실행 방법은 [프로젝트 README](../README.md) 및 [문서 지도](README.md)를 기준으로 한다.


## 평가 목적

검색 변경이 일부 예시가 아니라 전체 질문 집합에서 실제로 개선되는지 측정한다. 이 평가는 LLM 답변 품질이 아니라 정답 근거 chunk를 찾는 retrieval 품질만 측정한다.

## 데이터셋

- 파일: `eval/questions.jsonl`
- 질문 수: 15개
- 출처: 2026-2학기 수강편람
- 영역: 수강 일정, 신청 학점, 장학금 기본이수학점, 학점 이월, 철회, 대학원 과목, 강의 시간, e-러닝, K-MOOC, 재수강, 학점교류
- 정답 판정: 지정 페이지와 모든 필수 문구를 함께 만족하는 chunk

평가 질문은 현재 한 문서에서 수동 작성했다. 개발 중 확인한 내용과 일부 겹치므로 완전히 독립적인 test set은 아니다.

## 지표

- Hit Rate@5: 정답 chunk가 상위 5개 안에 든 질문의 비율
- MRR: 각 질문의 첫 정답 순위 역수 평균

## EVAL-001 결과

- 검색기: TOKEN-001 chunk + E5 + 키워드 + RRF
- 질문 수: 15
- Hit@5: 14/15
- Hit Rate@5: 0.9333
- MRR: 0.7556
- 1위 정답: 9/15

| ID | 정답 순위 | 결과 |
|---|---:|---|
| course_start_date | 1 | HIT |
| wishlist_credit_limit | 1 | HIT |
| normal_credit_range | 3 | HIT |
| high_grade_credit_limit | 2 | HIT |
| scholarship_minimum_credits | 3 | HIT |
| credit_rollover_limit | 1 | HIT |
| withdraw_replace_course | 1 | HIT |
| assigned_course_cancel | 1 | HIT |
| graduate_course_credits | 1 | HIT |
| seventy_five_minute_class | 2 | HIT |
| elearning_attendance_ratio | 1 | HIT |
| external_kmooc_credit_limit | 1 | HIT |
| retake_eligibility | 1 | HIT |
| retake_max_grade | 2 | HIT |
| exchange_student_gpa | 6 | MISS |

## 실패 분석

학점교류 질문에서는 같은 157쪽의 신청 절차와 학점 제한 chunk가 2·3위였지만, 실제 자격 문구 `총 평점평균 B0(3.0) 이상`이 든 chunk는 6위였다. 문서의 한 페이지가 여러 token chunk로 나뉘었고 관련 내용이 이웃 chunk에 분산된 사례다.

단일 실패에 맞춘 질의 확장 규칙을 즉시 추가하지 않는다. 같은 평가셋에 반복적으로 맞추면 과적합되기 때문이다. 다음 RAG 문맥 구성 단계에서 상위 결과의 같은 페이지 인접 chunk를 함께 수집하는 방법을 실험한다.

## 실행

```bash
.venv/bin/sejong-evaluate
```

## 해석 시 주의사항

- 93.33%는 세종대학교의 모든 질문에 대한 정확도가 아니다.
- 검색 성공은 LLM이 올바른 답을 생성한다는 보장이 아니다.
- 현재 문서 하나와 15개 질문뿐이므로 데이터 범위를 늘려야 한다.
- 질문 작성자와 개발자가 같아 독립 평가보다 점수가 높게 나올 수 있다.
