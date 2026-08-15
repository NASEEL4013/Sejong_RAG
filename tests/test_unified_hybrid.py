import numpy as np
from sejong_rag.unified_hybrid import hybrid_rank,infer_categories

def test_infer_categories():
    assert infer_categories('등록금 납부')=={'tuition'}
    assert 'scholarship' in infer_categories('국가장학금 신청')

def test_title_and_period_can_promote_exact_document():
    chunks=[{'chunk_id':'a','document_id':'a','parent_document_id':'a','document_title':'일반 안내','text':'수강 일정'},{'chunk_id':'b','document_id':'b','parent_document_id':'b','document_title':'2026년 2학기 수강신청 안내','text':'수강 일정'}]
    embeddings=np.array([[1.,0.],[.9,.1]],dtype=np.float32); q=np.array([1.,0.],dtype=np.float32)
    meta={'a':{'categories':['course_registration']},'b':{'categories':['course_registration'],'academic_year':2026,'semester':2}}
    assert hybrid_rank(chunks,embeddings,q,'2026년 2학기 수강신청',meta,2)[0]['chunk']['chunk_id']=='b'

def test_expand_query_uses_official_intent_terms():
    from sejong_rag.unified_hybrid import expand_query
    assert "신청기간" in expand_query("신청은 언제야?")
    assert "총 평점평균" in expand_query("학점교류 평점 조건은?")
    assert "제출서류" in expand_query("무슨 서류를 가져가?")

def test_expand_graduation_query_normalizes_admission_year_expression():
    from sejong_rag.unified_hybrid import expand_query
    expanded = expand_query("2021년 입학생의 졸업요건을 알려줘")
    assert "2021학년도 입학자 교과과정 졸업 기준" in expanded
    assert "영어졸업인증 이수 조건 면제 대상" in expanded

def test_graduation_requirement_query_penalizes_result_roster():
    import numpy as np
    from sejong_rag.unified_hybrid import hybrid_rank
    chunks = [
        {"chunk_id":"rule", "text":"2021학년도 입학자 교과과정 1. 졸업 기준", "document_title":"2026 수강편람", "filename":"수강편람.pdf", "extraction_methods":["pypdf"]},
        {"chunk_id":"roster", "text":"졸업가능 취득학점미달", "document_title":"졸업대상자 졸업심사 결과", "filename":"졸업예정자 명단.xlsx", "extraction_methods":["openpyxl_row"]},
    ]
    embeddings = np.array([[1.0,0.0],[0.0,1.0]], dtype=float)
    result = hybrid_rank(chunks, embeddings, np.array([0.0,1.0]), "2021년 입학생의 졸업요건", {}, top_k=2, candidate_k=2)
    assert result[0]["chunk"]["chunk_id"] == "rule"

def test_exact_admission_year_graduation_table_gets_structured_boost():
    import numpy as np
    from sejong_rag.unified_hybrid import hybrid_rank
    chunks = [
        {"chunk_id":"generic", "text":"졸업요건자가진단", "document_title":"졸업요건 안내", "extraction_methods":[]},
        {"chunk_id":"exact", "text":"2020학년도 입학자 교과과정 1. 졸업 기준", "document_title":"수강편람", "extraction_methods":[]},
    ]
    result = hybrid_rank(chunks, np.array([[1.0,0.0],[0.0,1.0]]), np.array([1.0,0.0]), "2020년 입학생 졸업요건", {}, top_k=2, candidate_k=2)
    assert result[0]["chunk"]["chunk_id"] == "exact"

def test_infer_admission_year_sections_continues_until_next_heading():
    from sejong_rag.unified_hybrid import infer_admission_year_sections
    chunks = [
        {"document_id":"guide", "chunk_index":1, "text":"❚ 2021학년도 입학자 교과과정"},
        {"document_id":"guide", "chunk_index":2, "text":"단일전공 이수 시 학과별 졸업 이수학점 소프트웨어학과"},
        {"document_id":"guide", "chunk_index":3, "text":"❚ 2020학년도 입학자 교과과정"},
        {"document_id":"guide", "chunk_index":4, "text":"단일전공 이수 시 학과별 졸업 이수학점"},
    ]
    assert infer_admission_year_sections(chunks) == {0:2021, 1:2021, 2:2020, 3:2020}

def test_single_major_graduation_query_expands_table_heading():
    from sejong_rag.unified_hybrid import expand_query
    assert "단일전공 이수 시 학과별 졸업 이수학점" in expand_query("2021년 소프트웨어학과 단일전공 졸업요건")

def test_graduation_credit_wording_is_treated_as_requirement_table():
    import numpy as np
    from sejong_rag.unified_hybrid import hybrid_rank
    chunks = [
        {"chunk_id":"roster", "document_id":"roster", "document_title":"졸업대상자 졸업심사 결과", "filename":"졸업예정자 명단.xlsx", "text":"소프트웨어학과 졸업가능"},
        {"chunk_id":"heading", "document_id":"guide", "chunk_index":1, "document_title":"수강편람", "filename":"수강편람.pdf", "text":"2019학년도 입학자 교과과정"},
        {"chunk_id":"row", "document_id":"guide", "chunk_index":2, "document_title":"수강편람", "filename":"수강편람.pdf", "text":"소프트웨어학과 14 9 72 36 36 140"},
    ]
    result = hybrid_rank(chunks, np.array([[1.,0.],[0.,1.],[0.,1.]]), np.array([1.,0.]), "2019학년도 입학 소프트웨어학과 단일전공 전공필수와 졸업학점", {}, top_k=3, candidate_k=3)
    assert result[0]["chunk"]["chunk_id"] == "row"

def test_expand_query_maps_degree_certificate_and_seasonal_period_intents():
    from sejong_rag.unified_hybrid import expand_query
    assert "학위수여식" in expand_query("학위증은 언제 받아?")
    assert "등록 및 수강신청" in expand_query("하계 계절학기 수업 기간은?")

def test_expanded_intent_matching_title_promotes_official_document():
    import numpy as np
    from sejong_rag.unified_hybrid import hybrid_rank
    chunks = [
        {"chunk_id":"wrong", "document_id":"w", "document_title":"조기취업 공결 안내", "text":"졸업예정증명서 발급"},
        {"chunk_id":"right", "document_id":"r", "document_title":"학위수여식 안내", "text":"학위증서 배부"},
    ]
    result = hybrid_rank(chunks, np.array([[1.,0.],[0.,1.]]), np.array([1.,0.]), "학위증은 언제 받아?", {}, top_k=2, candidate_k=2)
    assert result[0]["chunk"]["chunk_id"] == "right"
