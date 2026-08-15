# 2026 학사·장학 통합 청킹 및 벡터 인덱스
> 현재 기준 안내(2026-08-10): 파이프라인 완성 후 전체 자동 테스트는 148개 통과했다. 본문에 더 작은 테스트 수가 있으면 해당 기능을 만들던 당시의 기록이다. 최신 전체 성능은 [독립 평가 보고서](HOLDOUT_EVALUATION.md)를 참고한다.


## 목적

기존 수강편람 PDF 한 개의 인덱스를 2026년 학사·장학 공지 본문과 첨부 전체로 확장한다. 정확도 상태가 다른 자료를 섞어 같은 신뢰도로 취급하지 않도록 신뢰·검토 인덱스를 분리한다.

## 통합 청킹

실행:

    .venv/bin/sejong-unified-chunk

입력:

- 공지 상세 145건
- 첨부 146개
- 첨부 추출 record 50,808개

출력:

- `data/processed/unified-chunks/trusted.jsonl`
- `data/processed/unified-chunks/review.jsonl`

결과:

- 공지 문서: 145건
- 중복 제거 후 첨부 문서: 130개
- 내용 해시가 같아 중복 임베딩하지 않은 첨부: 16개
- 신뢰 청크: 10,663개
- 검토 청크: 3,391개
- 전체 청크: 14,054개
- 실제 최대 토큰: 444
- 448토큰 초과: 0개
- 공식 출처 누락: 0개
- 고유 청크 ID: 14,054개
- 복수 원본 출처를 가진 청크: 3,099개

동일 내용 첨부는 벡터를 한 번만 만들지만 `source_aliases`에 모든 부모 공지와 공식 URL을 보존한다.

## 벡터 인덱스

신뢰 인덱스:

    .venv/bin/sejong-embed \
      --chunks data/processed/unified-chunks/trusted.jsonl \
      --output data/index/sejong-2026-trusted-e5-small.npz \
      --batch-size 64

검토 인덱스:

    .venv/bin/sejong-embed \
      --chunks data/processed/unified-chunks/review.jsonl \
      --output data/index/sejong-2026-review-e5-small.npz \
      --batch-size 64

- 모델: `intfloat/multilingual-e5-small`
- 벡터 차원: 384
- 신뢰 행렬: 10,663 × 384, 약 19MB
- 검토 행렬: 3,391 × 384, 약 8.4MB
- 모델 토큰 잘림: 양쪽 모두 0개

## 검색

기본 검색은 신뢰 인덱스만 사용한다.

    .venv/bin/sejong-unified-search \
      "2026학년도 2학기 수강신청 일정은 언제야?"

검토 자료까지 명시적으로 포함:

    .venv/bin/sejong-unified-search \
      "미혼이면 누구 명의 가족관계증명서를 제출해?" \
      --include-review

검색 결과는 품질 계층, 문서 제목, 파일명, 페이지·시트·행·슬라이드 등의 위치와 공식 URL을 표시한다.

## 표본 결과와 한계

- 수강신청 일정 질문에서 정확한 수강편람 5페이지가 벡터 검색 3위에 포함됐다.
- 미혼 신청자의 가족관계증명서 질문에서는 OCR 교정 자료가 검토 결과 1위였고, 정상 추출 모바일 매뉴얼이 신뢰 결과 상위권에서 같은 사실을 뒷받침했다.
- 일반적인 “장학금 서류” 질문은 서로 다른 장학사업 문서가 섞였다. 장학사업명이 없으면 재질문하거나 사업별 필터가 필요하다.
- 벡터 검색만으로 정확한 날짜·문서 목적·최신 버전을 항상 1위로 보장할 수 없다.

다음 단계에서는 문서 제목·게시일·카테고리·품질 계층을 사용하는 하이브리드 검색과 학교 전체 평가 질문을 추가한다.
