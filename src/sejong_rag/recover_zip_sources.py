"""압축 안의 스캔 PDF·구형 엑셀만 로컬 복구하고 원본 위치를 보존한다."""
from __future__ import annotations
import json
import re
import tempfile
import zipfile
from pathlib import Path
from sejong_rag.extract_attachments import _safe_zip_members, _zip_display_name, extract_xls
from sejong_rag.recover_attachments import ocr_pdf_page, _read_records, _write_records, _base_record


def recover_zip(metadata: dict, output_dir: Path) -> dict:
    identifier=metadata['document_id']
    report_path=output_dir/'reports'/f'{identifier}.json'
    report=json.loads(report_path.read_text())
    records_path=Path(report['records_path'])
    records=_read_records(records_path)
    warnings=list(report.get('warnings',[]))
    results=[]
    with zipfile.ZipFile(metadata['resolved_raw_path']) as archive:
        for member in _safe_zip_members(archive):
            name=_zip_display_name(member.filename)
            member_warnings=[w for w in warnings if w.startswith(name+': ')]
            if not member_warnings: continue
            suffix=Path(name).suffix.lower()
            pages=sorted({int(m.group(1)) for w in member_warnings
                if (m:=re.search(r': page (\d+): (?:no embedded text|LookupError: unknown encoding)',w))})
            payload=archive.read(member)
            old_excel=suffix=='.xls' or (suffix=='.xlsx' and payload.startswith(bytes.fromhex('D0CF11E0A1B11AE1')))
            if not (suffix=='.pdf' and pages) and not old_excel: continue
            with tempfile.TemporaryDirectory(prefix='sejong-zip-recovery-') as temporary:
                work=Path(temporary); source=work/('source.xls' if old_excel else 'source.pdf');source.write_bytes(payload)
                if old_excel:
                    try:
                        extracted,notes=extract_xls(source)
                        if not extracted: raise ValueError('변환 후 읽을 수 있는 셀이 없습니다.')
                        records=[r for r in records if r.get('locator',{}).get('zip_member')!=name]
                        records.extend(_base_record(metadata,r['text'],{'zip_member':name,**r['locator']},'xlrd_row') for r in extracted)
                        warnings=[w for w in warnings if w not in member_warnings]
                        warnings.extend(name+': '+note for note in notes)
                        results.append({'zip_member':name,'method':'xlrd_row','records':len(extracted)})
                    except Exception as error:
                        warnings.append(f'{name}: XLS extraction failed: {type(error).__name__}: {error}')
                else:
                    for page in pages:
                        try:
                            text,confidence=ocr_pdf_page(source,page,work)
                            if not text: raise ValueError('OCR returned no text')
                            records=[r for r in records if not (r.get('locator',{}).get('zip_member')==name and r.get('locator',{}).get('page')==page)]
                            records.append(_base_record(metadata,text,{'zip_member':name,'page':page,'ocr':True,'ocr_mean_confidence':confidence},'pdftoppm+tesseract_kor_eng'))
                            warnings=[w for w in warnings if not w.startswith(f'{name}: page {page}: ')]
                            warnings.append(f'{name}: page {page}: OCR text requires source verification')
                            results.append({'zip_member':name,'page':page,'mean_confidence':confidence})
                        except Exception as error:
                            warnings.append(f'{name}: page {page}: OCR failed: {type(error).__name__}: {error}')
    _write_records(records_path,identifier,records)
    report.update(status='review_required' if warnings or not records else 'success',
        record_count=len(records),text_length=sum(r['text_length'] for r in records),warnings=list(dict.fromkeys(warnings)))
    report.setdefault('recovery',{}).setdefault('zip_members',[]).extend(results)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    import argparse
    from sejong_rag.extract_attachments import load_attachment_metadata
    parser=argparse.ArgumentParser(description='압축 내부 PDF OCR 및 구형 XLS 복구')
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    args=parser.parse_args();output=args.data_dir/'processed/attachments'
    for item in load_attachment_metadata(args.data_dir):
        path=output/'reports'/f"{item['document_id']}.json"
        if item['file_type']!='zip' or not path.is_file(): continue
        if json.loads(path.read_text())['status']!='review_required':continue
        result=recover_zip(item,output)
        print(json.dumps({key:result[key] for key in ('document_id','status','record_count')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
