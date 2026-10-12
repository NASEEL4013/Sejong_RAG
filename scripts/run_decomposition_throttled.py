"""Run the unchanged A/B experiment at <=10 API requests/minute.

This wrapper changes request pacing only. Model/prompts/retrieval stay frozen.
"""
from pathlib import Path
import sys

from sejong_rag import compare_decomposed
from sejong_rag.ask_selected import save_json
from sejong_rag.embed_chunks import file_sha256
from resume_decomposition import Ledger

if __name__=='__main__':
    output=Path(sys.argv[sys.argv.index('--output-dir')+1])
    if output.exists():
        raise ValueError('Output directory already exists')
    compare_decomposed.UsageLedger=Ledger
    try:
        compare_decomposed.main()
    finally:
        if output.exists():
            save_json(output/'pacing.json',dict(min_request_interval_seconds=6,
                wrapper_sha256=file_sha256(Path(__file__)),
                ledger_sha256=file_sha256(Path(__file__).with_name('resume_decomposition.py'))))
