"""Publish the already generated raw reference files after independent, non-ML verification."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'artifacts/tcn_hardware_raw_handoff_2026-10-20'
PACKAGE = STAGE / 'package'
NAME = 'TCN_ED4_seed42_RAW1024_Hardware_Handoff_20261020'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def readj(path):
    return json.loads(path.read_text('utf-8'))


def writej(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    audit_path = STAGE / 'independent_audit.json'
    audit = readj(audit_path)
    assert audit.get('status') == 'passed', 'Independent audit did not pass'
    acceptance = readj(PACKAGE / 'acceptance.json')
    assert acceptance['status'] == 'passed' and acceptance['model_forward_calls'] == 2
    assert acceptance['model_forward_samples'] == 12 and acceptance['training_runs'] == 0
    m = readj(PACKAGE / 'manifest.json')
    for relative, ref in m['files'].items():
        assert binding(PACKAGE / relative) == ref, relative
    for path, ref in acceptance['frozen_software_files_unchanged'].items():
        assert binding(Path(path)) == ref, path
    shutil.copyfile(audit_path, PACKAGE / 'independent_audit.json')
    audit_code = STAGE / 'independent_audit_check.py'
    if audit_code.is_file():
        shutil.copyfile(audit_code, PACKAGE / 'provenance/independent_audit_check.py')
    shutil.copyfile(__file__, PACKAGE / 'provenance' / Path(__file__).name)
    writej(PACKAGE / 'execution_receipts.json', {
        'recorded_after_actual_tool_completions': True,
        'builder': {'script': 'build_tcn_raw_handoff_20261020.py', 'tool_session_id': 69937,
                    'actual_exit_code': 0, 'completed_records': 1024, 'completed_windows': 6,
                    'TorchScript_loads': 1, 'forward_calls': 2, 'forward_samples': 12},
        'portable_verifier': {'script': 'verify_reference.py', 'tool_session_id': 23738,
                              'actual_exit_code': 0, 'mode': 'no_model', 'verified_file_hashes': 53,
                              'preprocessing_byte_identical': True, 'model_forward_calls': 0},
        'independent_audit': {'sha256': sha(audit_path), 'status': audit['status']},
        'old_software_training_budget_reopened': False,
    })
    (PACKAGE / 'source_review.txt').write_text(
        'Independent static review of the two new extraction/verifier scripts passed before execution.\n'
        'Reviewed fixed 1024 original rows, uint64/float32 endianness, source-position and EST clock semantics,\n'
        'eleven preprocessing windows covering all1024 rows, historical-cache mapping, six model windows,\n'
        'one TorchScript load and two forwards/12 samples; no training, old writes or DEV/17-19 data streams.\n'
        'Static review did not itself execute models or read the large source files.\n'
        'Mask full-file SHA I/O is separate from the4096-byte numeric mask slice.\n', encoding='utf-8', newline='\n')
    references = {p.relative_to(PACKAGE).as_posix(): binding(p)
                  for p in sorted(PACKAGE.rglob('*')) if p.is_file()}
    checksums = PACKAGE / 'SHA256SUMS.txt'
    with checksums.open('x', encoding='utf-8', newline='\n') as f:
        for name, ref in references.items():
            f.write(ref['sha256'] + '  ' + name + '\n')
    references['SHA256SUMS.txt'] = binding(checksums)
    archive = STAGE / (NAME + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(references):
            z.write(PACKAGE / name, NAME + '/' + name)
    checked = []
    with zipfile.ZipFile(archive, 'r') as z:
        assert len(z.infolist()) == len(references)
        for info in z.infolist():
            name = info.filename.removeprefix(NAME + '/')
            assert name in references and info.file_size == references[name]['bytes']
            h = hashlib.sha256()
            count = 0
            with z.open(info) as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
                    count += len(block)
            assert h.hexdigest() == references[name]['sha256'] and count == references[name]['bytes']
            checked.append(name)
    archive_ref = binding(archive)
    (STAGE / (NAME + '.zip.sha256')).write_text(archive_ref['sha256'] + '  ' + archive.name + '\n',
                                               encoding='utf-8', newline='\n')
    completion = {'status': 'completed', 'completed_shanghai': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'archive': {'path': str(archive), **archive_ref}, 'archive_file_count': len(checked),
        'every_archive_stream_SHA_bytes_CRC_verified': True,
        'package_manifest': binding(PACKAGE / 'manifest.json'), 'checksum_file': binding(checksums),
        'independent_audit': binding(audit_path), 'raw_records': 1024, 'reference_windows': 6,
        'date': m['date'], 'symbol': m['symbol'], 'source_rows': m['source_rows'],
        'first_time': m['first_row']['exchange_datetime_est'], 'last_time': m['last_row']['exchange_datetime_est'],
        'preprocessing_matches_ED4_cache_byte_for_byte': True,
        'TorchScript_loads': 1, 'forward_calls': 2, 'forward_samples': 12,
        'new_training_or_accuracy_evaluation': False, 'NPU_or_FPGA_verified': False,
        'frozen_software_plan_state_budget_unchanged': True, 'files': references}
    writej(STAGE / 'completion_acceptance.json', completion)
    print(json.dumps({k: completion[k] for k in ('status', 'archive', 'archive_file_count', 'raw_records',
                                                'reference_windows', 'first_time', 'last_time')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
