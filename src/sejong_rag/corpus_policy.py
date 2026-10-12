"""Explicit user exclusions. Raw sources and historical artifacts are retained."""
EXCLUDED_DOCUMENTS = {
    'notice3-805951-attachment-217294':
        '사용자 요청: 학생 개인정보가 포함된 학기별 고급프로그래밍활용 실습조 명단 제외',
}


SYLLABUS_DOCUMENTS = {
    'notice3-805885-attachment-217238', 'notice3-805900-attachment-217253',
    'notice3-805978-attachment-217314', 'notice3-854318-attachment-229008',
    'notice3-859604-attachment-230886', 'notice3-861957-attachment-231568',
    'notice3-861957-attachment-231569', 'notice3-862742-attachment-231905',
    'notice3-862742-attachment-231955', 'notice3-891421-attachment-236615',
}
EXCLUDED_DOCUMENTS.update({doc:'사용자 요청: 강의계획서 본문 검색 제외, 과목별 원문 안내로 전환'
                           for doc in SYLLABUS_DOCUMENTS})


def allowed_chunk(chunk):
    import re
    if re.search(r'(?:수업|강의)\s*계획서',chunk.get('filename') or ''):
        return False
    if chunk.get('document_id') in EXCLUDED_DOCUMENTS:
        return False
    # Some intermediate chunks expose provenance only through record IDs.
    return not any(s['record_id'].startswith(doc+'-record-')
                   for s in chunk.get('source_spans',[])+chunk.get('context_source_spans',[])
                   for doc in EXCLUDED_DOCUMENTS)
