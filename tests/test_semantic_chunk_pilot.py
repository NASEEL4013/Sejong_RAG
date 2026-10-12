from sejong_rag.semantic_chunk_pilot import section_spans


def pieces(text):return [text[a:b] for a,b in section_spans(text)]


def test_wrapped_number_heading_is_boundary_but_wrapped_sentence_is_not():
    text='안내\n1.\n신청 방법\n포털에서\n신청합니다.\n2.\n문의처\n02-1234-5678'
    assert pieces(text)==['안내\n','1.\n신청 방법\n포털에서\n신청합니다.\n','2.\n문의처\n02-1234-5678']


def test_exceptions_and_amount_dates_stay_in_same_section():
    text='반환금액\n2025. 12. 28. ~ 2026. 1. 2.\n수강료의\n2/3\n※\n본인 계좌\n단\n, 오류시 지연\n6.\n운영 원칙'
    assert len(pieces(text))==2
    assert '단\n, 오류시 지연' in pieces(text)[0]
    assert ''.join(pieces(text))==text


def test_marker_and_label_not_split_and_parenthetical_course_name_is_not_heading():
    text='■\n문의사항\n<\n대학영어\n>\n수강신청 문의\n■\n신청기간\n내일까지'
    assert len(pieces(text))==2
    assert pieces(text)[0].startswith('■\n문의사항')
