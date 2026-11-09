"""Read-only stdlib verification for a versioned D improvement snapshot."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    root=Path(__file__).resolve().parents[1]
    read=lambda p:json.loads((root/p).read_text(encoding='utf-8-sig'))
    m=read('release_manifest.json')
    for entry in m['files']:
        path=(root/entry['path']).resolve()
        if not path.is_relative_to(root) or sha(path)!=entry['sha256']:raise ValueError('Changed '+entry['path'])
    q=read('quality_report.json');c=read('input_contract.json')
    if sha(root/c['model_json'])!=c['model_sha256']:raise ValueError('Mixed model identity differs')
    if c['input']['storage_bytes']!=16000 or c['full_window_direct_int8_input_supported']:raise ValueError('Wrong input ABI')
    print(json.dumps(dict(status='passed',scope='offline improvement only',files_verified=len(m['files']),
        offline_quality_passed=q['rows']['mixed']['quality_budget_passed'],pure_int8=q['pure_int8'],
        board_tested=False,npu_speedup=None,external_input_dtype='float32',external_input_bytes=16000),indent=2))

if __name__=='__main__':main()
