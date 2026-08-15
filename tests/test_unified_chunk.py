from sejong_rag.unified_chunk import build_units, pack_units, split_long
class FakeTokenizer:
    def __call__(self,text,**kwargs):
        return {"offset_mapping":[(i,i+1) for i,c in enumerate(text) if not c.isspace()]}

def test_split_long_respects_limit_and_overlap():
    parts=split_long("abcdefghij",FakeTokenizer(),4,1)
    assert [n for _,n in parts]==[4,4,4]
    assert parts[0][0]=="abcd" and parts[1][0]=="defg"

def test_pack_combines_short_units_without_exceeding_limit():
    records=[{"record_id":"1","text":"abc","locator":{"row":1},"extraction_method":"x"},{"record_id":"2","text":"de","locator":{"row":2},"extraction_method":"x"}]
    header,ht,units=build_units(records,"T",None,FakeTokenizer(),20,2)
    packed=pack_units(units,header,ht,20)
    assert len(packed)==1
    assert packed[0][1]<=20
