"""Actual capture for two registered new-checkpoint CPU container inspections."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.reference.training.scripts.run_tcn_existing_data_ed4_capture import ROOT, STAGE, binding, put


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args()
    phase = 'checkpoint_validation'
    target = STAGE / 'checkpoint_validation_registration.json'
    entry = STAGE / 'checkpoint_validation.py'
    receipt_path = STAGE / (phase + '_execution_receipt.json')
    if receipt_path.exists() or binding(target)['sha256'] != args.expected_sha256:
        raise ValueError('Already inspected or wrong explicit checkpoint registration')
    reg = json.loads(target.read_text('utf-8'))
    if binding(entry)['sha256'] != reg['source_binding']['sha256']:
        raise ValueError('Registered checkpoint helper source differs')
    actual = json.loads(Path(reg['training_receipt_binding']['path']).read_text('utf-8'))
    if actual['exit_code'] != 0 or not actual['entry_unchanged']:
        raise ValueError('Successful actual training terminal required')
    command = [sys.executable, '-u', '-B', str(entry), '--registration-sha256', args.expected_sha256]
    before = binding(entry)
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    put(STAGE / (phase + '_capture_reservation.json'), {
        'phase': phase, 'started_utc': started, 'input': binding(target),
        'entry': before, 'permanent_single_use': True, 'automatic_retry': False})
    out_path, err_path = STAGE / (phase + '.stdout.log'), STAGE / (phase + '.stderr.log')
    with out_path.open('xb') as out, err_path.open('xb') as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=err,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8'))
        put(STAGE / (phase + '_launch.json'), {
            'phase': phase, 'started_utc': started, 'command': command,
            'capture_pid': os.getpid(), 'launcher_pid': process.pid,
            'worker_pid_may_differ_from_launcher': True, 'input': binding(target), 'entry': before})
        for line in iter(process.stdout.readline, b''):
            out.write(line)
            out.flush()
            print(line.decode('utf-8', errors='replace').rstrip(), flush=True)
        process.stdout.close()
        code = process.wait()
    outputs, worker = {}, None
    if code == 0:
        outputs['checkpoint_validation'] = binding(STAGE / 'checkpoint_validation.json')
        worker = json.loads((STAGE / 'checkpoint_validation.json').read_text('utf-8'))['worker_pid']
    after = binding(entry)
    put(receipt_path, {
        'version': 'TCN-ED4-actual-command-receipt-1.0', 'phase': phase,
        'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': time.perf_counter() - start, 'exit_code': code,
        'capture_pid': os.getpid(), 'launcher_pid': process.pid, 'actual_worker_pid': worker,
        'command': command, 'input': binding(target), 'entry_before': before, 'entry_after': after,
        'entry_unchanged': before == after, 'stdout': binding(out_path), 'stderr': binding(err_path),
        'outputs': outputs, 'capture_source': binding(Path(__file__)), 'automatic_retry': False})
    print(json.dumps({'phase': phase, 'actual_exit_code': code,
                      'receipt': binding(receipt_path)}, ensure_ascii=True), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
