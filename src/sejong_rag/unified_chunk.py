"""공지 본문과 모든 첨부 record를 품질 계층별 통합 chunk로 변환한다."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
from sejong_rag.extract_attachments import load_attachment_metadata
from sejong_rag.token_chunk import window_token_spans
DEFAULT_MODEL="intfloat/multilingual-e5-small"
DEFAULT_DATA=Path("data")
DEFAULT_OUTPUT=Path("data/processed/unified-chunks")

def load_jsonl(path: Path):
    if not path.is_file(): return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def tokenize(tokenizer, text: str):
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,truncation=False)
    return [tuple(x) for x in encoded["offset_mapping"] if x[1]>x[0]]

def split_long(text: str, tokenizer, max_tokens: int, overlap: int):
    offsets=tokenize(tokenizer,text)
    if len(offsets)<=max_tokens: return [(text,len(offsets))]
    return [(text[a:b].strip(),n) for a,b,n in window_token_spans(offsets,max_tokens,overlap)]

def location_label(locator: dict[str,object]) -> str:
    return ", ".join(f"{k}={v}" for k,v in locator.items() if k in {"page","sheet","row","slide","paragraph","table","zip_member","section"}) or "document"

def build_units(records: list[dict[str,object]], title: str, filename: str|None, tokenizer, max_tokens: int, overlap: int):
    units=[]
    header=f"문서 제목: {title}" + (f"\n첨부파일: {filename}" if filename else "")
    header_tokens=len(tokenize(tokenizer,header))+2; body_limit=max(32,max_tokens-header_tokens-4)
    for record in records:
        text=str(record.get("text","")).strip()
        if not text: continue
        for part,count in split_long(text,tokenizer,body_limit,overlap):
            units.append({"text":part,"token_count":count,"record_id":record.get("record_id"),"locator":record.get("locator",{}),"extraction_method":record.get("extraction_method")})
    return header,header_tokens,units

def pack_units(units, header: str, header_tokens: int, max_tokens: int):
    packed=[]; current=[]; count=header_tokens
    for unit in units:
        extra=int(unit["token_count"])+1
        if current and count+extra>max_tokens:
            packed.append((current,count)); current=[]; count=header_tokens
        current.append(unit); count+=extra
    if current: packed.append((current,count))
    return packed

def attachment_groups(metadata: list[dict[str,object]]):
    groups=defaultdict(list)
    for item in metadata:
        key=str(item.get("content_hash") or item["document_id"])
        groups[key].append(item)
    return groups

def notice_records(details_dir: Path):
    for path in sorted(details_dir.glob("*.json")):
        item=json.loads(path.read_text(encoding="utf-8")); body=str(item.get("body","")).strip()
        text = body or str(item.get("title", "")).strip()
        if text:
            yield item,[{"record_id":f"{item['document_id']}-body","text":text,"locator":{"section":"notice_body" if body else "notice_title_only"},"extraction_method":"official_html_body" if body else "official_html_title_only"}]

def make_chunks(data_dir: Path, output_dir: Path, tokenizer, max_tokens=448, overlap=64):
    output_dir.mkdir(parents=True,exist_ok=True); trusted_path=output_dir/"trusted.jsonl"; review_path=output_dir/"review.jsonl"
    metadata=load_attachment_metadata(data_dir); by_id={str(x["document_id"]):x for x in metadata}
    aliases={}
    for group in attachment_groups(metadata).values():
        canonical=sorted(group,key=lambda x:str(x["document_id"]))[0]
        aliases[str(canonical["document_id"])]=[{"document_id":x["document_id"],"parent_document_id":x.get("parent_document_id"),"source_url":x.get("source_url"),"attachment_url":x.get("attachment_url"),"filename":x.get("filename")} for x in group]
    reports={p.stem:json.loads(p.read_text(encoding="utf-8")) for p in (data_dir/"processed/attachments/reports").glob("*.json")}
    chunks=[]; skipped_duplicates=0; sequence=0
    def add_document(document_id,title,filename,records,quality,source_url,attachment_url,parent_id,source_aliases,file_type):
        nonlocal sequence
        header,header_tokens,units=build_units(records,title,filename,tokenizer,max_tokens,overlap)
        for local_index,(group,token_count) in enumerate(pack_units(units,header,header_tokens,max_tokens),1):
            sequence+=1; body="\n".join(str(x["text"]) for x in group); text=f"{header}\n{body}".strip()
            chunks.append({"chunk_id":f"unified-{sequence:06d}","document_id":document_id,"parent_document_id":parent_id,"document_title":title,"filename":filename,"file_type":file_type,"quality_tier":quality,"text":text,"text_length":len(text),"token_count":token_count,"locators":[x["locator"] for x in group],"record_ids":[x["record_id"] for x in group],"extraction_methods":sorted({str(x["extraction_method"]) for x in group}),"source_url":source_url,"attachment_url":attachment_url,"source_aliases":source_aliases,"chunk_index":local_index,"chunk_method":"e5_token_pack_v1","chunk_max_tokens":max_tokens,"chunk_overlap_tokens":overlap})
    for notice,records in notice_records(data_dir/"metadata/details"):
        add_document(str(notice["document_id"]),str(notice["title"]),None,records,"trusted",notice.get("source_url"),None,notice["document_id"],[{"document_id":notice["document_id"],"source_url":notice.get("source_url")}],"html")
    canonical_ids=set(aliases)
    for document_id,report in sorted(reports.items()):
        if document_id not in canonical_ids:
            if document_id in by_id: skipped_duplicates+=1
            continue
        item=by_id[document_id]; records=load_jsonl(Path(str(report["records_path"])))
        quality="trusted" if report["status"]=="success" else "review"
        add_document(document_id,str(item.get("parent_title") or item.get("filename") or document_id),str(item.get("filename")),records,quality,item.get("source_url"),item.get("attachment_url"),item.get("parent_document_id"),aliases[document_id],report.get("file_type"))
    with trusted_path.open("w",encoding="utf-8") as trusted, review_path.open("w",encoding="utf-8") as review:
        for chunk in chunks:
            target=trusted if chunk["quality_tier"]=="trusted" else review
            target.write(json.dumps(chunk,ensure_ascii=False)+"\n")
    counts=Counter(str(x["quality_tier"]) for x in chunks)
    return {"trusted_path":str(trusted_path),"review_path":str(review_path),"notice_documents":sum(1 for _ in notice_records(data_dir/"metadata/details")),"attachment_documents":len(canonical_ids),"skipped_duplicate_attachments":skipped_duplicates,"trusted_chunks":counts["trusted"],"review_chunks":counts["review"],"total_chunks":len(chunks),"maximum_tokens":max((int(x["token_count"]) for x in chunks),default=0),"over_limit":sum(int(x["token_count"])>max_tokens for x in chunks),"missing_source":sum(not x.get("source_url") for x in chunks)}

def main():
    p=argparse.ArgumentParser(description="공지 본문과 첨부를 품질 계층별 통합 청킹합니다."); p.add_argument("--data-dir",type=Path,default=DEFAULT_DATA); p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT); p.add_argument("--model",default=DEFAULT_MODEL); p.add_argument("--max-tokens",type=int,default=448); p.add_argument("--overlap",type=int,default=64)
    a=p.parse_args(); from transformers import AutoTokenizer; tokenizer=AutoTokenizer.from_pretrained(a.model); print(json.dumps(make_chunks(a.data_dir,a.output_dir,tokenizer,a.max_tokens,a.overlap),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
