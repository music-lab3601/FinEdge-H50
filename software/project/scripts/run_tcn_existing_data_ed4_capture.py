"""Root-owned actual CLI capture for the finite ED4 feature experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'artifacts/tcn_existing_data_optimization_2026-10-17/step04_features'


def binding(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return {'path': str(path), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def put(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['feature_tests', 'training_tests', 'preparation', 'training', 'audit'])
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args()
    phase = args.phase
    receipt_path = STAGE / (phase + '_execution_receipt.json')
    if receipt_path.exists():
        raise ValueError('Existing terminal receipt; do not repeat a phase')
    outputs = {}
    if phase.endswith('_tests'):
        name = 'features' if phase == 'feature_tests' else 'train'
        entry = ROOT / f'scripts/test_tcn_existing_data_ed4_{name}.py'
        target = entry
        command = [sys.executable, '-u', '-B', str(entry)]
    else:
        target = STAGE / ('preparation_registration.json' if phase == 'preparation' else 'training_registration.json')
        reg = json.loads(target.read_text('utf-8'))
        if binding(target)['sha256'] != args.expected_sha256:
            raise ValueError('Explicit frozen registration differs')
        if phase == 'preparation':
            entry = ROOT / 'scripts/tcn_existing_data_ed4_features.py'
            source = reg['prepared_manifest_binding']
            command = [sys.executable, '-u', '-B', str(entry), 'prepare',
                       '--manifest', source['path'], '--manifest-sha256', source['sha256'],
                       '--output', str(STAGE / 'prepared'), '--registration', str(target),
                       '--registration-sha256', args.expected_sha256]
            outputs['feature_cache_manifest'] = STAGE / 'prepared/cache_manifest.json'
        elif phase == 'training':
            entry = ROOT / 'scripts/tcn_existing_data_ed4_train.py'
            command = [sys.executable, '-u', '-B', str(entry), '--registration', str(target),
                       '--registration-sha256', args.expected_sha256]
            outputs = {'training_acceptance': STAGE / 'run/training_acceptance.json',
                       'history': STAGE / 'run/history.json'}
        else:
            entry = STAGE / 'independent_audit.py'
            command = [sys.executable, '-u', '-B', str(entry), '--registration-sha256', args.expected_sha256]
            outputs['audit'] = STAGE / 'independent_audit.json'
        if phase != 'audit':
            name = entry.relative_to(ROOT).as_posix()
            if binding(entry)['sha256'] != reg['source_bindings'][name]:
                raise ValueError('Registered numerical entry differs')
    if binding(target)['sha256'] != args.expected_sha256:
        raise ValueError('Explicit command input differs')
    before = binding(entry)
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    # A failed or interrupted launch still consumes this unique CLI reservation.
    put(STAGE / (phase + '_capture_reservation.json'), {
        'phase': phase, 'started_utc': started, 'entry': before, 'input': binding(target),
        'permanent_single_use': True, 'automatic_retry': False})
    out_path, err_path = STAGE / (phase + '.stdout.log'), STAGE / (phase + '.stderr.log')
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONIOENCODING='utf-8', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with out_path.open('xb') as out, err_path.open('xb') as err:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=err)
        put(STAGE / (phase + '_launch.json'), {
            'phase': phase, 'started_utc': started, 'capture_pid': os.getpid(),
            'launcher_pid': child.pid, 'worker_pid_may_differ_from_launcher': True,
            'command': command, 'input': binding(target), 'entry': before})
        for line in iter(child.stdout.readline, b''):
            out.write(line)
            out.flush()
            print(line.decode('utf-8', errors='replace').rstrip(), flush=True)
        child.stdout.close()
        code = child.wait()
    refs = {key: binding(path) for key, path in outputs.items()} if code == 0 else {}
    worker = None
    if phase == 'training' and code == 0:
        worker = json.loads(outputs['training_acceptance'].read_text('utf-8')).get('actual_worker_pid')
    after = binding(entry)
    put(receipt_path, {
        'version': 'TCN-ED4-actual-command-receipt-1.0', 'phase': phase,
        'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': time.perf_counter() - start, 'exit_code': code,
        'capture_pid': os.getpid(), 'launcher_pid': child.pid, 'actual_worker_pid': worker,
        'command': command, 'input': binding(target), 'entry_before': before, 'entry_after': after,
        'entry_unchanged': before == after, 'stdout': binding(out_path), 'stderr': binding(err_path),
        'outputs': refs, 'capture_source': binding(Path(__file__)), 'automatic_retry': False})
    print(json.dumps({'phase': phase, 'actual_exit_code': code,
                      'receipt': binding(receipt_path)}, ensure_ascii=True), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
