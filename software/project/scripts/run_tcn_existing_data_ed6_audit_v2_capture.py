"""Root-only one-use ED6 audit-v2 capture; no training or checkpoint load.

The original failed audit receipt/source/reservation remain immutable. Both
trial audits use the complete corrected audit and independent new reservations.
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


def source_matches(ref, expected):
    path = local(ref['path']); actual = binding(path)
    return (path.resolve() == local(expected).resolve()
            and actual['sha256'] == ref['sha256'] and actual['bytes'] == ref['bytes'])


def run_audit(args):
    trial = STAGE / 'trials' / args.slot
    entry = STAGE / 'independent_audit_v2.py'; target = trial / 'training_registration.json'
    repair_path = trial / 'audit_v2_registration.json'; repair = read(repair_path)
    if (binding(target)['sha256'] != args.expected_sha256 or binding(repair_path)['sha256'] != args.repair_registration_sha256
            or binding(entry)['sha256'] != args.entry_sha256
            or repair['v2_audit_source_binding']['sha256'] != args.entry_sha256
            or not source_matches(repair['v2_capture_source_binding'], Path(__file__))):
        raise ValueError('Explicit audit-v2 source or registration binding differs')
    receipt_path = trial / 'audit_v2_execution_receipt.json'
    if receipt_path.exists() or (trial / 'audit_v2_capture_reservation.json').exists():
        raise ValueError('Audit-v2 already permanently reserved; no automatic retry')
    for name, phase in [('training_execution_receipt.json', 'training'),
                        ('checkpoint_execution_receipt.json', 'checkpoint_validation')]:
        actual = read(trial / name)
        if (actual['phase'] != phase or actual['slot_id'] != args.slot or actual['exit_code'] != 0
                or actual['entry_unchanged'] is not True or actual['entry_before'] != actual['entry_after']):
            raise ValueError('Actual successful same-trial training and CPU-container audit required')
    sources = [entry, Path(__file__), ROOT / 'scripts/run_tcn_existing_data_ed6_capture.py',
               STAGE / 'independent_audit.py']
    before = binding(entry); sources_before = [binding(p) for p in sources]
    command = [sys.executable, '-u', '-B', str(entry), '--slot', args.slot,
               '--registration-sha256', args.expected_sha256,
               '--repair-registration-sha256', args.repair_registration_sha256]
    started = datetime.now(timezone.utc).isoformat(); timer = time.perf_counter()
    put(trial / 'audit_v2_capture_reservation.json', {'phase': 'audit', 'revision': 'v2', 'slot_id': args.slot,
        'started_utc': started, 'input': binding(target), 'repair_registration_binding': binding(repair_path),
        'entry': before, 'sources_before': sources_before, 'permanent_single_use': True, 'automatic_retry': False})
    stdout = trial / 'audit_v2.stdout.log'; stderr = trial / 'audit_v2.stderr.log'
    with stdout.open('xb') as out, stderr.open('xb') as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=err,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1'))
        put(trial / 'audit_v2_launch.json', {'phase': 'audit', 'revision': 'v2', 'slot_id': args.slot,
            'started_utc': started, 'command': command, 'capture_pid': os.getpid(), 'launcher_pid': process.pid,
            'worker_pid_may_differ_from_launcher': True, 'entry': before, 'input': binding(target),
            'repair_registration_binding': binding(repair_path)})
        for line in iter(process.stdout.readline, b''):
            out.write(line); out.flush(); print(line.decode('utf-8', errors='replace').rstrip(), flush=True)
        process.stdout.close(); code = process.wait()
    outputs = {}; worker = None
    if code == 0:
        outputs['audit'] = binding(trial / 'readonly_audit_v2.json')
        worker = read(trial / 'readonly_audit_v2.json')['worker_pid']
    after = binding(entry); sources_after = [binding(p) for p in sources]
    put(receipt_path, {'version': 'TCN-ED6-audit-v2-actual-command-receipt-1.0', 'phase': 'audit', 'revision': 'v2',
        'slot_id': args.slot, 'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': time.perf_counter() - timer, 'exit_code': code, 'capture_pid': os.getpid(),
        'launcher_pid': process.pid, 'actual_worker_pid': worker, 'command': command,
        'input': binding(target), 'repair_registration_binding': binding(repair_path),
        'entry_before': before, 'entry_after': after, 'entry_unchanged': before == after,
        'sources_before': sources_before, 'sources_after': sources_after, 'sources_unchanged': sources_before == sources_after,
        'stdout': binding(stdout), 'stderr': binding(stderr), 'outputs': outputs,
        'capture_source': binding(Path(__file__)), 'automatic_retry': False})
    print(json.dumps({'phase': 'audit', 'revision': 'v2', 'slot_id': args.slot, 'actual_exit_code': code,
                     'receipt': binding(receipt_path)}, ensure_ascii=True), flush=True)
    return code


def run_global(args):
    phase = args.phase
    if args.slot is not None or args.repair_registration_sha256 is not None:
        raise ValueError('Global repair tests/publication may not select a trial')
    if phase == 'repair_tests':
        entry = STAGE / 'test_audit_v2.py'; target = entry
        command = [sys.executable, '-u', '-B', str(entry)]; outputs = {}
        sources = [entry, STAGE / 'independent_audit_v2.py', STAGE / 'independent_audit.py',
                   STAGE / 'checkpoint_validation.py', Path(__file__),
                   ROOT / 'scripts/run_tcn_existing_data_ed6_postrun_capture.py',
                   ROOT / 'scripts/run_tcn_existing_data_ed6_capture.py']
    elif phase == 'finalizer_v2':
        if args.global_repair_registration_sha256 is None:
            raise ValueError('Publication-v2 requires explicit global repair registration')
        global_repair = binding(STAGE / 'audit_path_repair_v2_registration.json')
        if global_repair['sha256'] != args.global_repair_registration_sha256:
            raise ValueError('Explicit global repair registration differs')
        entry = ROOT / 'scripts/finalize_tcn_existing_data_ed6_v2.py'; target = STAGE / 'selection_lock.json'
        command = [sys.executable, '-u', '-B', str(entry), '--selection-lock-sha256', args.expected_sha256]
        outputs = {'completion_acceptance': STAGE / 'completion_acceptance.json',
                   'finalization_receipt': STAGE / 'finalization_receipt.json'}
        sources = [entry, Path(__file__), ROOT / 'scripts/run_tcn_existing_data_ed6_capture.py']
        for slot in ('control43', 'candidate43'):
            for name in ('training_execution_receipt.json', 'checkpoint_execution_receipt.json', 'audit_v2_execution_receipt.json'):
                actual = read(STAGE / 'trials' / slot / name)
                if actual['exit_code'] != 0 or actual['entry_unchanged'] is not True:
                    raise ValueError('Both actual trial pipelines must finish successfully before publication')
    else:
        raise ValueError('Unknown global repair phase')
    if binding(entry)['sha256'] != args.entry_sha256 or binding(target)['sha256'] != args.expected_sha256:
        raise ValueError('Explicit repair test/publication source or input differs')
    receipt_path = STAGE / (phase + '_execution_receipt.json')
    if receipt_path.exists() or (STAGE / (phase + '_capture_reservation.json')).exists():
        raise ValueError('Repair phase already permanently reserved; no automatic retry')
    before = binding(entry); source_before = [binding(p) for p in sources]
    started = datetime.now(timezone.utc).isoformat(); timer = time.perf_counter()
    put(STAGE / (phase + '_capture_reservation.json'), {'phase': phase, 'started_utc': started,
        'input': binding(target), 'entry': before, 'sources_before': source_before,
        'permanent_single_use': True, 'automatic_retry': False})
    stdout = STAGE / (phase + '.stdout.log'); stderr = STAGE / (phase + '.stderr.log')
    with stdout.open('xb') as out, stderr.open('xb') as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=err,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1'))
        put(STAGE / (phase + '_launch.json'), {'phase': phase, 'started_utc': started, 'command': command,
            'capture_pid': os.getpid(), 'launcher_pid': process.pid, 'worker_pid_may_differ_from_launcher': True,
            'entry': before, 'input': binding(target)})
        for line in iter(process.stdout.readline, b''):
            out.write(line); out.flush(); print(line.decode('utf-8', errors='replace').rstrip(), flush=True)
        process.stdout.close(); code = process.wait()
    refs = {key: binding(path) for key, path in outputs.items()} if code == 0 else {}
    worker = read(outputs['finalization_receipt']).get('worker_pid') if code == 0 and phase == 'finalizer_v2' else None
    after = binding(entry); source_after = [binding(p) for p in sources]
    result = {'version': 'TCN-ED6-repair-actual-command-receipt-1.0', 'phase': phase,
        'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.perf_counter() - timer,
        'exit_code': code, 'capture_pid': os.getpid(), 'launcher_pid': process.pid, 'actual_worker_pid': worker,
        'command': command, 'input': binding(target), 'entry_before': before, 'entry_after': after,
        'entry_unchanged': before == after, 'sources_before': source_before, 'sources_after': source_after,
        'sources_unchanged': source_before == source_after, 'stdout': binding(stdout), 'stderr': binding(stderr),
        'outputs': refs, 'capture_source': binding(Path(__file__)), 'automatic_retry': False}
    if phase == 'repair_tests':
        result.update(actual_model_or_optimizer_operations=0, real_records_read=False)
    else:
        result['global_repair_registration_binding'] = global_repair
    put(receipt_path, result)
    print(json.dumps({'phase': phase, 'actual_exit_code': code, 'receipt': binding(receipt_path)}, ensure_ascii=True), flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['repair_tests', 'audit_v2', 'finalizer_v2'])
    parser.add_argument('--slot', choices=['control43', 'candidate43'])
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--repair-registration-sha256')
    parser.add_argument('--global-repair-registration-sha256')
    parser.add_argument('--entry-sha256', required=True)
    args = parser.parse_args(argv)
    if args.phase == 'audit_v2':
        if args.slot is None or args.repair_registration_sha256 is None:
            raise ValueError('Audit-v2 requires explicit trial and repair registration')
        return run_audit(args)
    return run_global(args)


if __name__ == '__main__':
    raise SystemExit(main())
