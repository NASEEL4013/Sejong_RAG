from sejong_rag.ingest import parse_notice_html


NOTICE_URL = (
    "https://www.sejong.ac.kr/kor/intro/notice3.do"
    "?mode=view&articleNo=891485"
)


def test_parse_notice_and_attachments():
    html = """
    <div class="b-view-common-box">
      <input type="hidden" name="articleNo" value="891485">
      <div class="b-title-box">
        <p class="b-title">2026-2학기 수강신청 안내</p>
        <ul class="b-info">
          <li>학사지원과</li>
          <li class="b-date"><span>2026.08.07</span></li>
          <li class="b-hit"><span>575</span></li>
        </ul>
      </div>
      <div class="b-content-box"><p>수강편람을 확인하세요.</p></div>
      <div class="b-file-box"><ul>
        <li><a href="?mode=download&amp;articleNo=891485&amp;attachNo=236669">
          <span>2026-2학기 수강편람_20260730.pdf</span>
        </a></li>
      </ul></div>
    </div>
    """

    notice = parse_notice_html(html, NOTICE_URL)

    assert notice.article_no == "891485"
    assert notice.title == "2026-2학기 수강신청 안내"
    assert notice.department == "학사지원과"
    assert notice.published_at == "2026.08.07"
    assert notice.body == "수강편람을 확인하세요."
    assert len(notice.attachments) == 1
    attachment = notice.attachments[0]
    assert attachment.attach_no == "236669"
    assert attachment.file_type == "pdf"
    assert attachment.version_date == "2026-07-30"
    assert "mode=download" in attachment.download_url



def test_chunk_text_preserves_content_and_adds_overlap():
    from sejong_rag.chunking import chunk_text

    text = "첫 번째 줄입니다.\n두 번째 줄입니다.\n세 번째 줄입니다."
    chunks = chunk_text(text, max_chars=22, overlap_chars=10)

    assert len(chunks) >= 2
    assert "첫 번째 줄입니다." in chunks[0]
    assert any("세 번째 줄입니다." in chunk for chunk in chunks)
    assert any(
        line in chunks[index + 1]
        for index, chunk in enumerate(chunks[:-1])
        for line in chunk.splitlines()[-1:]
    )


def test_chunk_text_rejects_invalid_settings():
    import pytest
    from sejong_rag.chunking import chunk_text

    with pytest.raises(ValueError):
        chunk_text("내용", max_chars=100, overlap_chars=100)


def test_chunk_page_text_repeats_page_context():
    from sejong_rag.chunking import chunk_page_text

    text = (
        "졸업 이수학점 표\n"
        "학과명 전공필수 전공선택 졸업학점\n"
        "소프트웨어학과 27 33 130\n"
        "컴퓨터공학과 21 39 130\n"
        "정보보호학과 24 36 130"
    )
    chunks = chunk_page_text(
        text,
        max_chars=70,
        overlap_chars=10,
        context_lines=2,
    )

    assert len(chunks) >= 2
    assert all(chunk.startswith("졸업 이수학점 표") for chunk in chunks)
    assert all("학과명 전공필수 전공선택 졸업학점" in chunk for chunk in chunks)
    assert all(len(chunk) <= 70 for chunk in chunks)


def test_keyword_search_ranks_chunk_with_all_terms_first():
    from sejong_rag.keyword_search import search_chunks

    chunks = [
        {
            "chunk_id": "one",
            "page": 1,
            "text": "소프트웨어학과의 졸업학점은 표를 확인한다.",
        },
        {
            "chunk_id": "two",
            "page": 2,
            "text": "소프트웨어학과 수강신청 안내",
        },
        {
            "chunk_id": "three",
            "page": 3,
            "text": "다른 학과의 졸업학점 안내",
        },
    ]

    results = search_chunks(chunks, "소프트웨어학과 졸업학점", top_k=3)

    assert results[0]["chunk"]["chunk_id"] == "one"
    assert results[0]["coverage"] == 1.0
    assert results[0]["matched_terms"] == ["소프트웨어학과", "졸업학점"]


def test_keyword_search_returns_empty_for_unknown_term():
    from sejong_rag.keyword_search import search_chunks

    chunks = [{"chunk_id": "one", "page": 1, "text": "수강신청 안내"}]

    assert search_chunks(chunks, "기숙사식단") == []


def test_parse_notice_infers_scholarship_board_and_document_id():
    html = """
    <div class="b-view-common-box">
      <div class="b-title-box">
        <p class="b-title">국가장학금 신청 안내</p>
        <ul class="b-info">
          <li>학생지원과</li><li class="b-date">2026.08.01</li>
        </ul>
      </div>
      <div class="b-content-box"><p>신청 내용을 확인하세요.</p></div>
    </div>
    """
    url = "https://www.sejong.ac.kr/kor/intro/notice7.do?mode=view&articleNo=123"

    notice = parse_notice_html(html, url)

    assert notice.document_id == "notice7-123"
    assert notice.board == "scholarship_notice"
