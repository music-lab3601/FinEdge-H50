"""Frozen root-owned ED6 cache/tests and sequential paired-training capture.

Every phase has a permanent one-use reservation and actual terminal receipt.
Postrun CPU/audit/publication phases use a different capture source.
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

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'artifacts/tcn_existing_data_optimization_2026-10-17/step06_replication'
PHASES = ('cache_tests', 'training_tests', 'preparation', 'training_control43', 'training_candidate43')


def local(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    if '..' in p.parts or not p.is_relative_to(ROOT):
        raise ValueError('ED6 capture path outside workspace')
    for q in (p, *p.parents):
        if q.exists() and (q.is_symlink() or getattr(q.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('ED6 capture reparse path forbidden')
        if q == ROOT:
            break
    if not p.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Resolved ED6 capture path escaped workspace')
    return p


def binding(value):
    p = local(value);h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return {'path': str(p), 'sha256': h.hexdigest(), 'bytes': p.stat().st_size}


def read(value):
    p = local(value)
    if p.stat().st_size > 16 * 1024**2:
        raise ValueError('Oversized capture metadata')
    return json.loads(p.read_text('utf-8-sig'))


def put(value, data):
    p = local(value)
    if not p.is_relative_to(STAGE):
        raise ValueError('Capture may write only the new ED6 stage')
    with p.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=PHASES)
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args(argv);phase = args.phase
    slot = phase.removeprefix('training_') if phase.startswith('training_') and phase != 'training_tests' else None
    phase_root = STAGE / 'trials' / slot if slot is not None else STAGE
    actual_phase = 'training' if slot is not None else phase
    receipt_path = phase_root / (actual_phase + '_execution_receipt.json')
    if receipt_path.exists() or (phase_root / (actual_phase + '_capture_reservation.json')).exists():
        raise ValueError('ED6 phase already permanently reserved; inspect original execution')
    outputs = {}
    if phase in ('cache_tests', 'training_tests'):
        suffix = 'data' if phase == 'cache_tests' else 'train'
        entry = ROOT / f'scripts/test_tcn_existing_data_ed6_{suffix}.py'
        target = entry;command = [sys.executable, '-u', '-B', str(entry)]
    else:
        if phase == 'preparation':
            target = STAGE / 'preparation_registration.json'
            entry = ROOT / 'scripts/tcn_existing_data_ed6_data.py'
        else:
            target = STAGE / 'trials' / slot / 'training_registration.json'
            entry = ROOT / 'scripts/tcn_existing_data_ed6_train.py'
        if binding(target)['sha256'] != args.expected_sha256:
            raise ValueError('Explicit ED6 registration differs')
        reg = read(target)
        if reg.get('step') != 'ED6':
            raise ValueError('Wrong registered phase')
        if binding(entry)['sha256'] != reg['source_bindings'][entry.relative_to(ROOT).as_posix()]:
            raise ValueError('Registered numerical source differs')
        if binding(Path(__file__))['sha256'] != reg['capture_source_binding']['sha256']:
            raise ValueError('Registered primary capture source differs')
        if phase == 'preparation':
            source = reg['prepared_manifest_binding']
            command = [sys.executable, '-u', '-B', str(entry), 'prepare-control',
                       '--manifest', source['path'], '--manifest-sha256', source['sha256'],
                       '--output', str(STAGE / 'prepared_control'), '--registration', str(target),
                       '--registration-sha256', args.expected_sha256]
            outputs['feature_cache_manifest'] = STAGE / 'prepared_control/cache_manifest.json'
        else:
            command = [sys.executable, '-u', '-B', str(entry), '--slot', slot,
                       '--registration', str(target), '--registration-sha256', args.expected_sha256]
            folder = STAGE / 'trials' / slot / 'run'
            outputs = {'training_acceptance': folder / 'training_acceptance.json', 'history': folder / 'history.json'}
    if binding(target)['sha256'] != args.expected_sha256:
        raise ValueError('Explicit phase input differs')
    before = binding(entry);started = datetime.now(timezone.utc).isoformat();start = time.perf_counter()
    put(phase_root / (actual_phase + '_capture_reservation.json'), {
        'phase': actual_phase, 'requested_phase': phase, 'slot_id': slot, 'started_utc': started, 'entry': before,
        'input': binding(target), 'permanent_single_use': True, 'automatic_retry': False})
    stdout = phase_root / (actual_phase + '.stdout.log');stderr = phase_root / (actual_phase + '.stderr.log')
    with stdout.open('xb') as out, stderr.open('xb') as err:
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=err,
            env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                     PYTHONIOENCODING='utf-8', CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        put(phase_root / (actual_phase + '_launch.json'), {
            'phase': actual_phase, 'requested_phase': phase, 'slot_id': slot,
            'started_utc': started, 'capture_pid': os.getpid(),
            'launcher_pid': child.pid, 'worker_pid_may_differ_from_launcher': True,
            'command': command, 'input': binding(target), 'entry': before})
        for line in iter(child.stdout.readline, b''):
            out.write(line);out.flush();print(line.decode('utf-8', errors='replace').rstrip(), flush=True)
        child.stdout.close();code = child.wait()
    refs = {name: binding(value) for name, value in outputs.items()} if code == 0 else {}
    worker = None
    if code == 0 and phase == 'preparation':
        worker = read(outputs['feature_cache_manifest']).get('worker_pid')
    elif code == 0 and slot is not None:
        worker = read(outputs['training_acceptance']).get('actual_worker_pid')
    after = binding(entry)
    put(receipt_path, {'version': 'TCN-ED6-actual-command-receipt-1.0', 'phase': actual_phase,
        'requested_phase': phase, 'slot_id': slot,
        'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': time.perf_counter() - start, 'exit_code': code,
        'capture_pid': os.getpid(), 'launcher_pid': child.pid, 'actual_worker_pid': worker,
        'command': command, 'input': binding(target), 'entry_before': before, 'entry_after': after,
        'entry_unchanged': before == after, 'stdout': binding(stdout), 'stderr': binding(stderr),
        'outputs': refs, 'capture_source': binding(Path(__file__)), 'automatic_retry': False})
    print(json.dumps({'phase': phase, 'slot_id': slot, 'actual_exit_code': code,
                      'receipt': binding(receipt_path)}, ensure_ascii=True), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
