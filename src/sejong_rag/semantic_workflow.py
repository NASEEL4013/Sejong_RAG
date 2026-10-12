"""K: page-spanning workflows grouped by source headings, with ordered siblings."""
import hashlib
import re


class WorkflowReview(ValueError):
    pass


def compact(text):
    return re.sub(r'\s+', '', text)


def candidate(records):
    # Identify the approved layout by its section headings, never by document ID.
    text = compact('\n'.join(r['text'] for r in records))
    return all(s in text for s in ('소속기관등록', '이수증기관등록', '발송이력확인'))


def workflow_plans(records):
    if len({r['document_id'] for r in records}) != 1:
        raise WorkflowReview('K_mixed_documents')
    if any(not isinstance(r.get('locator', {}).get('page'), int) for r in records):
        raise WorkflowReview('K_page_structure_requires_review')
    records = sorted(records, key=lambda r: r['locator']['page'])
    if [r['locator']['page'] for r in records] != list(range(1, len(records)+1)):
        raise WorkflowReview('K_missing_or_duplicate_pages')
    # Only the approved cover/index + 2/2/1 procedure pages + contact layout.
    if len(records) != 8 or not candidate(records):
        raise WorkflowReview('K_unrecognized_layout')
    cover, index, *_, contact = records
    if not re.search(r'Index|Contents', index['text'], re.I):
        raise WorkflowReview('K_index_missing')
    if '감사합니다' not in compact(contact['text']) or not re.search(r'\d{2,3}-\d{3,4}-\d{4}', contact['text']):
        raise WorkflowReview('K_contact_page_requires_review')
    titles = []
    for number in range(1, 4):
        match = re.search(rf'(?m)^\s*{number}[.\s]+([^\n]+)', index['text'])
        if not match:
            raise WorkflowReview('K_index_steps_missing')
        titles.append(match[1].strip())
    expected = [1, 1, 2, 2, 3]
    details = []
    for record, step, marker in zip(records[2:7], expected, ['①', '②', '①', '②', '①']):
        first, sep, rest = record['text'].partition('\n')
        if not re.match(rf'^{step}\.\s*', first) or not sep:
            raise WorkflowReview('K_section_order_requires_review')
        if compact(titles[step-1]) not in compact(first):
            raise WorkflowReview('K_heading_index_mismatch')
        if not rest.lstrip().startswith(marker) or re.search(r'\[\s*\]', rest):
            raise WorkflowReview('K_missing_extracted_buttons')
        details.append(rest)
    order = ' → '.join(f'{n}. {title}' for n, title in enumerate(titles, 1))
    doc = records[0]['document_id']
    revision = hashlib.sha256('\n'.join(r['text'] for r in records).encode()).hexdigest()[:16]
    workflow_id = f'{doc}:workflow:{revision}'
    common = [cover, index, contact]
    plans = []
    for n, (a, b) in enumerate([(0, 2), (2, 4), (4, 5)], 1):
        body = f'전체 순서: {order}\n현재 단계: {n}. {titles[n-1]}\n'
        body += '\n'.join(details[a:b])
        body += '\n문의: ' + contact['text'].split('감사합니다.', 1)[-1].strip()
        plans.append(dict(body=body, records=records[2+a:2+b], common=common,
                          procedure_id=workflow_id, procedure_step=n,
                          procedure_step_count=3, procedure_step_title=titles[n-1],
                          procedure_outline=order))
    return plans


def reference_list_candidate(record):
    text = compact(record['text'])
    return '맨아래참고' in text and '학과별표시과목' in text and '신청서출력' in text


def reference_list_plans(record):
    """L: attach a trailing lookup list to the input step that references it."""
    text = record['text']
    sections = list(re.finditer(r'(?m)^(\d+)\.[ \t]*', text))
    if [int(m[1]) for m in sections] != [1, 2, 3]:
        raise WorkflowReview('L_section_structure_requires_review')
    lists = list(re.finditer(r'(?m)^※[ \t]*학과별[ \t]*표시과목[^\n]*', text))
    if len(lists) != 1 or lists[0].start() <= sections[2].start():
        raise WorkflowReview('L_reference_list_requires_review')
    start, second, third = [m.start() for m in sections]
    lookup = lists[0].start()
    if '맨아래참고' not in compact(text[second:third]):
        raise WorkflowReview('L_reference_owner_requires_review')
    if not all(word in text[third:lookup] for word in ('출력', '날인', '제출')):
        raise WorkflowReview('L_submission_structure_requires_review')
    outline = '1. 온라인 신청 → 2. 신청서 출력·학과장 확인 및 날인 → 방문 제출'
    revision = hashlib.sha256(text.encode()).hexdigest()[:16]
    identity = f"{record['document_id']}:workflow:{revision}"
    plans = []
    for step, title, spans in [(1, '온라인 신청 및 표시과목 확인', [(start, third), (lookup, len(text))]),
                               (2, '신청서 출력·날인·방문 제출', [(third, lookup)])]:
        common = [(0, start)] if start else []
        body = text[:start] + f'전체 순서: {outline}\n현재 단계: {step}. {title}\n'
        body += '\n'.join(text[a:b] for a,b in spans)
        plans.append(dict(body=body, spans=spans, common=common,
                          procedure_id=identity, procedure_step=step, procedure_step_count=2,
                          procedure_step_title=title, procedure_outline=outline))
    return plans
