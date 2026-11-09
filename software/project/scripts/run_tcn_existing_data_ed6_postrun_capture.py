"""Root-only ED6 synthetic-helper, CPU-container, audit and publication capture.

The already frozen ED6 primary training capture is a separate source. Every
phase here requires an explicit input SHA and separately reviewed entry SHA.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_tcn_existing_data_ed6_capture import ROOT, STAGE, binding, local, put, read


def phase_paths(phase, slot=None):
    if phase in ('checkpoint_validation', 'audit'):
        if slot not in ('control43', 'candidate43'):
            raise ValueError('Per-trial postrun phase requires explicit paired slot')
        phase_root = STAGE / 'trials' / slot
    elif phase in ('postrun_helpers_tests', 'finalizer') and slot is None:
        phase_root = STAGE
    else:
        raise ValueError('Unknown phase or global phase supplied with a trial')
    receipt_name = 'checkpoint_execution_receipt.json' if phase == 'checkpoint_validation' else phase + '_execution_receipt.json'
    return phase_root, phase_root / receipt_name


def phase_spec(phase, expected_sha256, entry_sha256, slot=None):
    phase_root, _ = phase_paths(phase, slot)
    outputs = {}
    if phase == 'postrun_helpers_tests':
        entry = STAGE / 'test_postrun_helpers.py'
        target = entry
        command = [sys.executable, '-u', '-B', str(entry)]
    elif phase == 'checkpoint_validation':
        entry = STAGE / 'checkpoint_validation.py'
        target = phase_root / 'checkpoint_validation_registration.json'
        reg = read(target)
        if reg['source_binding']['sha256'] != entry_sha256:
            raise ValueError('Bounded CPU registration names another helper source')
        actual = read(local(reg['training_receipt_binding']['path']))
        if (binding(local(reg['training_receipt_binding']['path']))['sha256']
                != reg['training_receipt_binding']['sha256'] or actual['phase'] != 'training'
                or actual['exit_code'] != 0 or actual['entry_unchanged'] is not True):
            raise ValueError('Actual successful original training terminal required')
        command = [sys.executable, '-u', '-B', str(entry), '--slot', slot, '--registration-sha256', expected_sha256]
        outputs['checkpoint_validation'] = phase_root / 'checkpoint_validation.json'
    elif phase == 'audit':
        entry = STAGE / 'independent_audit.py'
        target = phase_root / 'training_registration.json'
        actual = read(phase_root / 'training_execution_receipt.json')
        cp = read(phase_root / 'checkpoint_execution_receipt.json')
        if actual['exit_code'] != 0 or cp['exit_code'] != 0:
            raise ValueError('Actual original training and bounded CPU helper must finish first')
        command = [sys.executable, '-u', '-B', str(entry), '--slot', slot, '--registration-sha256', expected_sha256]
        outputs['audit'] = phase_root / 'readonly_audit.json'
    else:
        if phase != 'finalizer':
            raise ValueError('Unknown postrun phase')
        entry = ROOT / 'scripts/finalize_tcn_existing_data_ed6.py'
        target = STAGE / 'selection_lock.json'
        for trial_slot in ('control43', 'candidate43'):
            trial = STAGE / 'trials' / trial_slot
            for name in ('training_execution_receipt.json', 'checkpoint_execution_receipt.json', 'audit_execution_receipt.json'):
                actual = read(trial / name)
                if actual['exit_code'] != 0 or actual['entry_unchanged'] is not True:
                    raise ValueError('Both trials require all actual successful predecessor CLIs')
        command = [sys.executable, '-u', '-B', str(entry), '--selection-lock-sha256', expected_sha256]
        outputs = {'completion': STAGE / 'completion_acceptance.json',
                   'finalization_receipt': STAGE / 'finalization_receipt.json'}
    if binding(target)['sha256'] != expected_sha256 or binding(entry)['sha256'] != entry_sha256:
        raise ValueError('Explicit postrun input or reviewed entry SHA differs')
    return entry, target, command, outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['postrun_helpers_tests', 'checkpoint_validation', 'audit', 'finalizer'])
    parser.add_argument('--slot', choices=['control43', 'candidate43'])
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--entry-sha256', required=True)
    args = parser.parse_args(argv)
    phase = args.phase; slot = args.slot
    phase_root, receipt_path = phase_paths(phase, slot)
    if receipt_path.exists() or (phase_root / (phase + '_capture_reservation.json')).exists():
        raise ValueError('This postrun phase is already reserved; no automatic retry')
    entry, target, command, outputs = phase_spec(phase, args.expected_sha256, args.entry_sha256, slot)
    sources = [entry]
    if phase == 'postrun_helpers_tests':
        sources += [STAGE / 'independent_audit.py', STAGE / 'checkpoint_validation.py', Path(__file__)]
    else:
        sources.append(Path(__file__))
    # Binding/path/write primitives are imported from the immutable primary
    # capture; explicitly preserve that direct dependency in every real receipt.
    sources.append(ROOT / 'scripts/run_tcn_existing_data_ed6_capture.py')
    source_before = [binding(p) for p in sources]
    before = binding(entry)
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    put(phase_root / (phase + '_capture_reservation.json'), {
        'phase': phase, 'started_utc': started, 'input': binding(target), 'entry': before,
        'sources_before': source_before, 'permanent_single_use': True, 'automatic_retry': False})
    out_path, err_path = phase_root / (phase + '.stdout.log'), phase_root / (phase + '.stderr.log')
    with out_path.open('xb') as out, err_path.open('xb') as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=err,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1'))
        put(phase_root / (phase + '_launch.json'), {
            'phase': phase, 'started_utc': started, 'command': command,
            'capture_pid': os.getpid(), 'launcher_pid': process.pid,
            'worker_pid_may_differ_from_launcher': True, 'input': binding(target), 'entry': before})
        for line in iter(process.stdout.readline, b''):
            out.write(line); out.flush()
            print(line.decode('utf-8', errors='replace').rstrip(), flush=True)
        process.stdout.close()
        code = process.wait()
    refs, worker = {}, None
    if code == 0:
        refs = {name: binding(p) for name, p in outputs.items()}
        if phase in ('checkpoint_validation', 'audit'):
            worker = read(next(iter(outputs.values()))).get('worker_pid')
        elif phase == 'finalizer':
            worker = read(outputs['finalization_receipt']).get('worker_pid')
    after = binding(entry)
    sources_after = [binding(p) for p in sources]
    receipt = {
        'version': 'TCN-ED6-actual-command-receipt-1.0', 'phase': phase, 'slot_id': slot,
        'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': time.perf_counter() - start, 'exit_code': code,
        'capture_pid': os.getpid(), 'launcher_pid': process.pid, 'actual_worker_pid': worker,
        'command': command, 'input': binding(target), 'entry_before': before, 'entry_after': after,
        'entry_unchanged': before == after, 'sources_before': source_before, 'sources_after': sources_after,
        'sources_unchanged': source_before == sources_after,
        'stdout': binding(out_path), 'stderr': binding(err_path), 'outputs': refs,
        'capture_source': binding(Path(__file__)), 'automatic_retry': False}
    if phase == 'postrun_helpers_tests':
        receipt.update(actual_model_or_optimizer_operations=0, real_records_read=False)
    put(receipt_path, receipt)
    print(json.dumps({'phase': phase, 'actual_exit_code': code, 'receipt': binding(receipt_path)},
                     ensure_ascii=True), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
