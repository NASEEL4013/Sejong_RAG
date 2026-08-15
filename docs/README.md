# 문서 지도

문서가 많아진 이유는 RAG를 한 단계씩 만들며 실험 결과를 남겼기 때문입니다. 현재 운영 기준과 과거 학습 기록을 구분해 읽으면 됩니다.

## 먼저 읽을 문서

1. [프로젝트 README](../README.md): 현재 상태와 실행 방법
2. [전체 제작 로드맵](../RAG_ROADMAP.md): 완료된 단계와 다음 작업
3. [데이터 출처](DATA_SOURCES.md): 공식 출처와 수집 정책
4. [정확도·버전 정책](ACCURACY_AND_VERSION_POLICY.md): 최신 문서와 입학연도 규칙
5. [독립 평가 보고서](HOLDOUT_EVALUATION.md): 현재 검색·답변 성능

## 현재 시스템 설명

- [범위와 분류](SCOPE_AND_CLASSIFICATION.md)
- [승인된 첨부파일 다운로드](AUTHORIZED_ATTACHMENT_DOWNLOAD.md)
- [첨부파일 다운로드 현황](ATTACHMENT_DOWNLOAD_CHECKLIST.md)
- [첨부파일 텍스트 추출](ATTACHMENT_TEXT_EXTRACTION.md)
- [OCR 및 문서 복구](OCR_AND_DOC_RECOVERY.md)
- [OCR 검증](OCR_VALIDATION.md)
- [통합 인덱스](UNIFIED_INDEX.md)
- [통합 검색 평가](UNIFIED_RETRIEVAL_EVALUATION.md)
- [문맥 구성기](CONTEXT_BUILDER.md)
- [답변 프롬프트](GENERATION_PROMPT.md)
- [통합 답변 생성](UNIFIED_GENERATION.md)
- [Gemini 설정](GEMINI_SETUP.md)
- [학과별 졸업요건](DEPARTMENT_GRADUATION_REQUIREMENTS.md)
- [졸업요건 질문 보정](GRADUATION_QUERY_FIX.md)
- [2021년 소프트웨어학과 필수과목](SOFTWARE_2021_REQUIRED_COURSES.md)

## 과거 단계의 실험 기록

아래 문서는 현재 사용법이 아니라 성능이 어떻게 개선됐는지 남긴 기록입니다. 문서 안의 데이터 수나 테스트 수는 실험 당시 값일 수 있습니다.

- [공지 목록 수집기](NOTICE_LIST_CRAWLER.md)
- [공지 상세 수집](DETAIL_COLLECTION.md)
- [수동 첨부파일 반입](MANUAL_ATTACHMENT_IMPORT.md)
- [2026년 자료 수집](YEAR_2026_COLLECTION.md)
- [키워드 검색 기준선](KEYWORD_BASELINE.md)
- [임베딩 검색 기준선](EMBEDDING_BASELINE.md)
- [혼합 검색 기준선](HYBRID_BASELINE.md)
- [초기 검색 평가](RETRIEVAL_EVALUATION.md)
- [초기 답변 평가](GENERATION_EVALUATION.md)

`기준선(baseline)`은 개선 전후를 비교하기 위한 출발점 성능입니다. 과거 기록은 삭제하면 개선 과정을 설명하거나 같은 실험을 재현하기 어려우므로 보존합니다.
