"""F: retain deadline cautions locally; copy only the referenced period core.

Returns source-exact plans. Multiple cohort schedules require user review rather
than selecting one schedule. Text outside these plans is not implicitly covered.
"""
from __future__ import annotations
import re


class ReferenceReview(ValueError):
    pass


REFERENCE = re.compile(r'상기\s*기간')
MAJOR = re.compile(r'(?m)^[ \t]*\d{1,2}[.)](?!\d)[ \t]*')
TOPIC = re.compile(r'(?m)^[ \t]*(?:\d{1,2}|[가-하])[.)](?!\d)[ \t]*')
PERIOD = re.compile(r'(?m)^[ \t]*(?:\d{1,2}[.)]\s*)?(?:응시|시험)\s*(?:기간|일정)\s*[:：]')


def reference_plans(text):
    references=list(REFERENCE.finditer(text))
    if not references: raise ReferenceReview('F_no_reference')
    periods=list(PERIOD.finditer(text))
    if len(periods)!=1: raise ReferenceReview('F_period_target_missing_or_multiple')
    period=periods[0]
    next_major=next((m.start() for m in MAJOR.finditer(text) if m.start()>period.end()),len(text))
    whole=(period.start(),next_major)
    period_text=text[whole[0]:whole[1]]
    if re.search(r'(?m)^[ \t]*[가-하][.)]',period_text):
        raise ReferenceReview('F_multiple_cohort_periods')
    # A note belongs in 1+2, but is not copied into 1+3+4.
    note=re.search(r'(?m)^[ \t]*※',period_text)
    core=(whole[0],whole[0]+note.start() if note else whole[1])
    if not re.search(r'\d',text[period.end():core[1]]):
        raise ReferenceReview('F_period_has_no_explicit_date')
    plans=[dict(kind='F_period_with_cautions',spans=[whole],period_core=core)]
    seen=set()
    for match in references:
        if match.start()<whole[1]: raise ReferenceReview('F_reference_precedes_period')
        headings=[m for m in TOPIC.finditer(text) if m.start()<=match.start()]
        if not headings: raise ReferenceReview('F_reference_topic_missing')
        start=headings[-1].start()
        stop=next((m.start() for m in TOPIC.finditer(text) if m.start()>match.end()),len(text))
        ending=re.search(r'(?:합니다|바랍니다|됩니다|해야\s*함|하여야\s*함|가능함)\s*[.。]?',text[match.end():stop])
        if not ending: raise ReferenceReview('F_reference_sentence_boundary_unclear')
        end=match.end()+ending.end()
        if start<whole[1] or not re.search(r'레벨\s*테스트|시험|응시',text[start:end]):
            raise ReferenceReview('F_reference_topic_unclear')
        if (start,end) not in seen:
            seen.add((start,end))
            plans.append(dict(kind='F_period_core_with_referencing_topic',spans=[core,(start,end)],period_core=core))
    return plans


def uncovered_spans(text, spans):
    cursor=0; gaps=[]
    for a,b in sorted(spans):
        if not 0<=a<=b<=len(text): raise ValueError('Invalid source span')
        if cursor<a and text[cursor:a].strip(): gaps.append((cursor,a))
        cursor=max(cursor,b)
    if text[cursor:].strip(): gaps.append((cursor,len(text)))
    return gaps


def cohort_reference_plans(text):
    """G: one cohort + common obligation; late-admission cohort also gets exclusion.

Only the approved two-cohort structure is recognized. All conditions, dates and
fallback instructions are source slices; no values are borrowed from other files.
Repeated referring statements are preserved together within each cohort chunk.
"""
    periods=list(PERIOD.finditer(text))
    if len(periods)!=1: raise ReferenceReview('G_period_target_missing_or_multiple')
    period=periods[0]
    end=next((m.start() for m in MAJOR.finditer(text) if m.start()>period.end()),len(text))
    labels=list(re.finditer(r'(?m)^[ \t]*[가-하][.)][ \t]*',text[period.end():end]))
    if len(labels)!=2: raise ReferenceReview('G_cohort_structure_not_confirmed')
    branches=[]
    for i,m in enumerate(labels):
        a=period.end()+m.start()
        b=period.end()+labels[i+1].start() if i+1<len(labels) else end
        body=re.sub(r'\s+','',text[a:b])
        if not re.search(r'(?:레벨테스트|시험|응시)기간[:：]',body):
            raise ReferenceReview('G_cohort_period_label_missing')
        if not re.search(r'\d',body): raise ReferenceReview('G_cohort_date_missing')
        if '추가합격자' in body: cohort='additional_admission'
        elif '신입생' in body: cohort='regular_admission'
        else: raise ReferenceReview('G_cohort_identity_unclear')
        branches.append((cohort,(a,b)))
    if {c for c,s in branches}!={'regular_admission','additional_admission'}:
        raise ReferenceReview('G_cohorts_not_distinct')
    refs=[]
    for match in REFERENCE.finditer(text):
        if match.start()<end: raise ReferenceReview('G_reference_precedes_period')
        headings=[m for m in TOPIC.finditer(text) if m.start()<=match.start()]
        if not headings: raise ReferenceReview('G_reference_topic_missing')
        a=headings[-1].start()
        stop=next((m.start() for m in TOPIC.finditer(text) if m.start()>match.end()),len(text))
        ending=re.search(r'(?:합니다|바랍니다|됩니다|해야\s*함|하여야\s*함|가능함)\s*[.。]?',text[match.end():stop])
        if not ending: raise ReferenceReview('G_reference_boundary_unclear')
        b=match.end()+ending.end()
        if not re.search(r'레벨\s*테스트|시험|응시',text[a:b]):
            raise ReferenceReview('G_reference_topic_unclear')
        if (a,b) not in refs: refs.append((a,b))
    if not refs: raise ReferenceReview('G_no_reference')
    topics=list(TOPIC.finditer(text)); exclusions=[]
    for i,m in enumerate(topics):
        if m.start()<end: continue
        a=m.start();b=topics[i+1].start() if i+1<len(topics) else len(text)
        body=re.sub(r'\s+','',text[a:b])
        if '추가합격자' in body and re.search(r'응시.{0,8}(?:불가능|불가)',body):
            # Keep the complete instruction, including the alternative registration.
            if b==len(text): raise ReferenceReview('G_exclusion_boundary_unclear')
            exclusions.append((a,b))
    if len(exclusions)!=1: raise ReferenceReview('G_exclusion_missing_or_multiple')
    return [dict(kind='G_cohort_period_reference_and_exclusion' if cohort=='additional_admission'
                 else 'G_cohort_period_with_reference',cohort=cohort,period_core=span,
                 spans=[span]+refs+(exclusions if cohort=='additional_admission' else []))
            for cohort,span in branches]
