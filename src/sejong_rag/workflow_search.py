"""Deterministic scope selection for source-scoped application workflows."""
import re


def whole_workflow(query):
    text = re.sub(r'\s+', '', query)
    if re.search(r'전체|처음부터|모든단계|전과정|처음.*끝|신청순서', text):
        return True
    if re.search(r'소속기관|개인정보|이수증|발송|연계|코드|학과|학번|이름|정정|수정|설문|'
                 r'기간|마감|언제|어디|문의|전화|조건|자격|서류|출력|날인|학과장|표시과목|제출처|방문|몇|최대|[1-9]단계', text):
        return False
    return bool(re.search(r'신청|절차|순서|어떻게', text))


def ordered_members(chunks, members):
    """Reject incomplete/corrupt groups instead of claiming a full procedure."""
    ordered = sorted(members, key=lambda i: chunks[i]['procedure_step'])
    counts = {chunks[i]['procedure_step_count'] for i in ordered}
    if len(counts) != 1:
        return None
    count = next(iter(counts))
    if [chunks[i]['procedure_step'] for i in ordered] != list(range(1, count+1)):
        return None
    return ordered
