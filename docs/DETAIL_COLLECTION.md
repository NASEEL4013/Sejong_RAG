# 공지 상세 HTML 수집
> 역사적 단계 기록: 이 문서는 개발 당시의 구현과 수치를 보존한다. 현재 상태와 실행 방법은 [프로젝트 README](../README.md) 및 [문서 지도](README.md)를 기준으로 한다.


## 목적

목록에는 제목과 날짜 정도만 있어 신청 대상, 기간, 성적 조건을 알 수 없다. 상세 HTML 수집 단계는 분류된 후보의 본문과 첨부파일 목록을 확보한다. 첨부파일 내용은 아직 수집하지 않는다.

## 실행 흐름

```text
classified-notices.json
  → 우선순위·카테고리로 후보 선택
  → 상세 URL 안전성 검사
  → 10초 간격으로 HTML 요청
  → 본문과 첨부 메타데이터 파싱
  → 내용 해시로 new/updated/unchanged 판정
```

본문 해시는 제목, 담당 부서, 게시일, 정제된 본문, 첨부 목록으로 계산한다. 조회수와 화면 장식은 제외해 내용이 그대로인데 변경됐다고 오판하는 일을 줄인다.

## 명령

네트워크 요청 전 후보만 확인한다.

    .venv/bin/sejong-collect-details --max-items 5 --dry-run

실제로 상세 HTML을 수집한다.

    .venv/bin/sejong-collect-details --max-items 5 --delay 10

특정 분야만 선택할 수 있다.

    .venv/bin/sejong-collect-details \
      --category scholarship \
      --max-items 5 \
      --delay 10

실제 요청에서 10초보다 작은 간격은 거부하며, 최대 문서 수는 100개로 제한한다. 외부 호스트, 알 수 없는 게시판, `mode=download` URL도 거부한다.

## 첫 실제 결과

- `notice3-891311`: 등록금 공지, 본문 19,755자, 첨부 메타데이터 2개
- `notice3-891273`: 조기졸업 공지, 본문 2,349자, 첨부 메타데이터 0개
- 실제 첨부 다운로드: 0개

저장 위치:

- 원본 HTML: `data/raw/notices/{document_id}.html`
- 상세 메타데이터: `data/metadata/details/{document_id}.json`

## 첨부파일을 넣는 시점

다음 단계에서 수동 첨부 import 명령을 구현한 후 넣는다. 사용자가 공식 페이지에서 내려받은 파일을 `data/manual-input/`에 두면, 도구가 상세 메타데이터의 공식 파일명·첨부번호와 비교해 부모 공지에 연결한다.

수동 파일도 자동으로 신뢰하지 않는다. 일치 여부, SHA-256 해시, 파일 형식, 처리 성공 여부를 기록한 뒤 텍스트 추출과 청킹으로 넘긴다.
