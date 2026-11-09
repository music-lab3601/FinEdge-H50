"""Single-use actual capture for the separate ED6 resource posterior mock suite.

Preparation only. This controller performs no Windows resource API, model,
checkpoint, market-data, training-budget or publication operation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RES = Path(__file__).resolve().parent
ROOT = RES.parents[3]
PHASE = 'resource_postrun_tests'


def local(value):
    path = Path(value); path = path if path.is_absolute() else ROOT / path
    if '..' in path.parts or not path.is_relative_to(ROOT):
        raise ValueError('Resource posterior test metadata outside workspace')
    for ancestor in (path, *path.parents):
        if ancestor.exists() and (ancestor.is_symlink() or getattr(ancestor.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Resource posterior test reparse path forbidden')
        if ancestor == ROOT:
            break
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Resolved resource posterior test path escaped workspace')
    return path


def binding(value):
    path = local(value)
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError('Oversized posterior test source/log')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


def put(value, data):
    path = local(value)
    if not path.is_relative_to(RES):
        raise ValueError('Posterior test capture may write only its resource stage')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--entry-sha256', required=True)
    args = parser.parse_args(argv)
    entry = ROOT / 'scripts/test_tcn_existing_data_ed6_resource_supplement.py'
    sources = [entry, ROOT / 'scripts/audit_tcn_existing_data_ed6_resource_supplement.py',
               ROOT / 'scripts/run_tcn_existing_data_ed6_resource_publication_gate.py', Path(__file__)]
    before = binding(entry); sources_before = [binding(path) for path in sources]
    if before['sha256'] != args.expected_sha256 or before['sha256'] != args.entry_sha256:
        raise ValueError('Explicit resource posterior test input/source differs')
    receipt_path = RES / (PHASE + '_execution_receipt.json')
    reservation_path = RES / (PHASE + '_capture_reservation.json')
    if receipt_path.exists() or reservation_path.exists():
        raise ValueError('Resource posterior test already permanently reserved; no retry')
    command = [sys.executable, '-u', '-B', str(entry)]
    started = datetime.now(timezone.utc).isoformat(); timer = time.monotonic()
    put(reservation_path, {'phase': PHASE, 'started_utc': started, 'input': before, 'entry': before,
        'sources_before': sources_before, 'permanent_single_use': True, 'automatic_retry': False})
    stdout = RES / (PHASE + '.stdout.log'); stderr = RES / (PHASE + '.stderr.log')
    with stdout.open('xb') as out, stderr.open('xb') as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=err,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1'))
        put(RES / (PHASE + '_launch.json'), {'phase': PHASE, 'started_utc': started, 'command': command,
            'capture_pid': os.getpid(), 'launcher_pid': process.pid, 'worker_pid_may_differ_from_launcher': True,
            'input': before, 'entry': before})
        for line in iter(process.stdout.readline, b''):
            out.write(line); out.flush(); print(line.decode('utf-8', errors='replace').rstrip(), flush=True)
        process.stdout.close(); code = process.wait()
    after = binding(entry); sources_after = [binding(path) for path in sources]
    put(receipt_path, {'version': 'TCN-ED6-resource-postrun-tests-actual-command-receipt-1.0', 'phase': PHASE,
        'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.monotonic() - timer,
        'exit_code': code, 'capture_pid': os.getpid(), 'launcher_pid': process.pid, 'actual_worker_pid': None,
        'worker_pid_not_inferred_from_launcher': True, 'command': command, 'input': before,
        'entry_before': before, 'entry_after': after, 'entry_unchanged': before == after,
        'sources_before': sources_before, 'sources_after': sources_after, 'sources_unchanged': sources_before == sources_after,
        'stdout': binding(stdout), 'stderr': binding(stderr), 'outputs': {}, 'capture_source': binding(Path(__file__)),
        'automatic_retry': False, 'actual_model_or_optimizer_operations': 0, 'checkpoint_loads': 0,
        'real_records_read': False, 'real_Windows_API_calls': 0, 'training_or_budget_mutations': 0})
    print(json.dumps({'phase': PHASE, 'actual_exit_code': code, 'receipt': binding(receipt_path)}, ensure_ascii=True), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
