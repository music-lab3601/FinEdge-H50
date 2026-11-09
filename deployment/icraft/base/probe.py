"""Stdlib-only, read-only intake probe; safe from a hashed G snapshot."""
from pathlib import Path
import hashlib
import json
import sys
sys.dont_write_bytecode=True


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    read=lambda p:json.loads((root/p).read_text(encoding='utf-8-sig'))
    manifest=read('candidate.json')
    for entry in manifest['files']:
        path=(root/entry['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path)!=entry['sha256']:
            raise ValueError('Candidate file missing/changed: '+entry['path'])
    c=read('input_contract.json');q=read('quality_report.json')
    n=read('artifacts/compile/int8/tcn_d_ZG.json')
    if c['input']['compiled_dtype']!=n['ops'][0]['outputs'][0]['dtype']:raise ValueError('Input contract differs from compiled graph')
    if c['input']['storage_bytes']!=16000 or c['direct_int8_input_supported_by_current_full_graph']:
        raise ValueError('Unsupported direct integer input claim')
    if q['board_accessed'] or q['hardware_calls']!=0:raise ValueError('Unexpected D device-access claim')
    local=read('artifacts/local_test_report.json')
    if local['status']!='passed':raise ValueError('Local implementation tests not passed')
    result=dict(status='passed',scope='local_candidate_integrity_and_contract_only',
        files_verified=len(manifest['files']),implementation_tests=local['tests_run'],
        int8_quality_passed=q['int8_quality_passed'],
        production_enable_allowed=False,board_tested=False,
        hardware_performance_measured=False,external_input_dtype=c['input']['dtype'],
        external_input_bytes=c['input']['storage_bytes'],
        quality_status=q['status'])
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
