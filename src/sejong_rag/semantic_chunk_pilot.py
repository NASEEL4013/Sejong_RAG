"""Offline pilot: source-exact section chunks for five development notice bodies.

No answer labels or queries enter the splitter. This is structure-guided splitting,
not an LLM semantic classifier. Production index and defaults remain unchanged.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import re
from sejong_rag.structure_chunk import Chunker

# Source-visible section/table-row labels, kept with all following values and notes.
PLAIN_HEADINGS = {
    '대상', '서류접수기간', '전형방법', '결과고지', '장학금지급액', '지원서교부',
    '구비서류', '우편접수주소', '문의사항', '유의사항', '반환금액',
    '수업기간', '개설과목', '수강신청', '폐강공고', '수강변경', '수강료',
    '수강료 납부', '수강취소',
}


def section_spans(text):
    """Exact character spans; do not break at sentence-wrapping newlines or ※/단."""
    starts={0}
    for match in re.finditer(r'(?m)^.*\S.*$',text):
        line=match.group().strip();start=match.start()
        if re.match(r'^(?:\d{1,2}[.)](?:\s|$)|[■□◆](?:\s|$))',line):
            starts.add(start)
        elif line in PLAIN_HEADINGS:
            # Lone heading text following a marker belongs to that marker.
            previous=text[:start].rstrip().splitlines()[-1:]
            if not previous or not re.fullmatch(r'(?:\d{1,2}[.)]|[■□◆※\[<])',previous[0].strip()):
                starts.add(start)
    # Bracketed multi-line headings, not inline <대학영어> mentions.
    for match in re.finditer(r'(?m)^[<\[]\s*\n([^<>\[\]\n]{2,35})\n[>\]]\s*\n',text):
        label=match.group(1).strip()
        if re.search(r'소개|주의사항|문의|유의사항',label):starts.add(match.start())
    # Course table starts after the prose exception, not inside it.
    for match in re.finditer(r'(?m)^20\d{2}\n학년도\n1\n학년\n1\n학기\n<\n대학영어\n>\n수강대상',text):
        starts.add(match.start())
    starts=sorted(starts)
    return [(a,b) for a,b in zip(starts,starts[1:]+[len(text)]) if text[a:b].strip()]


def split_record(record, template, tokenizer):
    """Keep original document header and exact source text; no synthetic facts."""
    helper=Chunker(tokenizer,max_tokens=448,overlap=64)
    # Notice-body headers use the same title budget and wording as production.
    header='문서 제목: '+helper.clip(str(record.get('document_title') or ''),80)+'\n\n'
    chunks=[]
    for a,b in section_spans(record['text']):
        body=record['text'][a:b]
        for x,y in helper.split(body,header):
            ref=dict(record_id=record['record_id'],start=a+x,end=a+y,locator=record['locator'])
            chunk=deepcopy(template)
            value=header+record['text'][ref['start']:ref['end']]
            chunk.update(text=value,text_length=len(value),token_count=helper.count(value),
                source_spans=[ref],record_ids=[record['record_id']],context_record_ids=[],context_note=None,
                chunk_id='semantic-pilot-'+hashlib.sha256(str(ref).encode()).hexdigest()[:24],
                chunk_method='semantic_section_pilot_v1',chunk_max_tokens=448,chunk_overlap_tokens=64,
                semantic_section_span=[a,b])
            chunks.append(chunk)
    cursor=0
    for chunk in chunks:
        ref=chunk['source_spans'][0]
        assert not record['text'][cursor:ref['start']].strip()
        cursor=max(cursor,ref['end'])
        assert chunk['text'].endswith(record['text'][ref['start']:ref['end']])
    assert not record['text'][cursor:].strip()
    return chunks
