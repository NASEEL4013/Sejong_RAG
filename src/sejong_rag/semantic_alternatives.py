"""O: mutually exclusive procedures with source-local shared conditions.

The first verified profile is zero-tuition registration. Unknown profiles remain
queued; number labels alone never establish that procedures are alternatives.
"""
import re
from sejong_rag.semantic_rechunk import NeedsReview, make_chunk

RULE = 'O_alternative_registration_methods'
CHOICE = re.compile(r'(?:가지|방법)중(?:1|한|하나)')
METHOD = re.compile(r'(?m)^(?:납부)?방법\s*(\d+)\s*\.\s*')
CHAPTER = re.compile(r'(?m)^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ](?:[. \t]*\n|[. \t]+)')
LABELS = ['자율경비 납부', '국민은행 공과금 납부', '국민은행 지점 직접 방문']


def display(text):
    # Page furniture is not content. Source spans still preserve it for audit.
    text = re.sub(r'(?m)^\s*-\s*\d+\s*-\s*$', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def split_alternatives(records, template, helper):
    from sejong_rag.semantic_bulk import ordered_text, refs_for
    text, ranges = ordered_text(records)
    compact = re.sub(r'\s+', '', text)
    if not CHOICE.search(compact):
        return [], None
    methods = list(METHOD.finditer(text))
    if not methods:
        return [], 'alternative_profile_not_verified'
    if [int(m[1]) for m in methods] != [1, 2, 3]:
        raise NeedsReview('alternative_method_numbers_require_review')
    first = methods[0].start()
    headings = list(CHAPTER.finditer(text))
    previous = [m.start() for m in headings if m.start() < first]
    following = [m.start() for m in headings if m.start() > first]
    start = previous[-1] if previous else 0
    end = following[0] if following else len(text)
    if methods[-1].start() >= end:
        raise NeedsReview('alternative_methods_cross_chapters')
    prefix = text[start:first]
    joined = re.sub(r'\s+', '', prefix)
    if not (re.search(r'3가지(?:방법)?중1가지', joined) and '등록' in joined
            and re.search(r'[“"‘]?0원', joined)):
        return [], 'alternative_profile_not_verified'
    notes = re.search(r'※\s*유의사항', text[methods[-1].end():end])
    if not notes:
        raise NeedsReview('alternative_shared_notes_boundary_unknown')
    notes_start = methods[-1].end() + notes.start()
    bounds = list(zip([m.start() for m in methods],
                      [m.start() for m in methods[1:]] + [notes_start]))
    bodies = [text[a:b] for a,b in bounds]
    expected = [('자율경비', '가상계좌'), ('공과금', '조회', '납부', '클릭'),
                ('고지서', '지점', '방문', '요청')]
    if any(not all(word in re.sub(r'\s+', '', body) for word in words)
           for body,words in zip(bodies, expected)):
        raise NeedsReview('alternative_method_text_incomplete')
    if '<표>' in text[start:end] or re.search(r'\[\s*\]', text[start:end]):
        raise NeedsReview('alternative_missing_extracted_content')
    # Cross-page joining must not silently skip an unextracted page.
    participating = [r for a,b,r in ranges if max(start,a)<min(end,b)]
    pages = [r.get('locator',{}).get('page') for r in participating]
    if all(isinstance(p,int) for p in pages) and pages != list(range(pages[0],pages[-1]+1)):
        raise NeedsReview('alternative_nonconsecutive_pages')
    common = display(prefix)
    notes_text = display(text[notes_start:end])
    outline = '선택 관계: 아래 세 방법 중 하나로 완료. 선택지: ' + ' / '.join(LABELS)
    common_spans = refs_for(ranges,start,first)+refs_for(ranges,notes_start,end)
    # Outline labels also carry the source locations of all three alternatives.
    label_spans = [s for a,b in bounds for s in refs_for(ranges,a,b)]
    chunks = []
    for i,((a,b),body) in enumerate(zip(bounds,bodies),1):
        primary = next(r for x,y,r in ranges if x <= a < y)
        value = '\n'.join([common, outline, '현재 선택지: '+LABELS[i-1], display(body), notes_text])
        spans = refs_for(ranges,a,b)
        # Only actual included body/common content counts as coverage. Other
        # alternatives' outline provenance is separate, not full-text coverage.
        chunk = make_chunk(template,primary,spans,value,common_spans+spans,value,helper,RULE)
        chunk.update(alternative_group_id=primary['document_id']+':zero-registration',
                     alternative_index=i,alternative_count=3,alternative_relation='choose_one',
                     alternative_labels=LABELS,alternative_outline_source_spans=label_spans)
        chunks.append(chunk)
    return chunks, None
