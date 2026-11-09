"""Actual single-use CLI receipts for the separate ED7 delivery."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import tcn_existing_data_ed7_contract as c

ENTRIES = {'tests': 'test_tcn_existing_data_ed7.py', 'primary': 'tcn_existing_data_ed7_export.py',
           'reload': 'tcn_existing_data_ed7_export.py', 'audit': 'audit_tcn_existing_data_ed7.py',
           'finalize': 'finalize_tcn_existing_data_ed7.py'}
SOURCE_NAMES = ['tcn_existing_data_ed7_contract.py', 'run_tcn_existing_data_ed7_capture.py',
    'tcn_existing_data_ed7_export.py', 'tcn_existing_data_ed7_runtime.py',
    'test_tcn_existing_data_ed7.py', 'audit_tcn_existing_data_ed7.py',
    'finalize_tcn_existing_data_ed7.py', 'tcn_research_p5_model.py', 'tcn_candidate.py',
    'tcn_existing_data_ed4_features.py']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=tuple(ENTRIES))
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--entry-sha256', required=True)
    args = parser.parse_args(argv); phase = args.phase
    entry = c.ROOT / 'scripts' / ENTRIES[phase]
    target = entry if phase == 'tests' else c.REGISTRATION
    c.need(c.sha(entry) == args.entry_sha256 and c.sha(target) == args.expected_sha256,
           'Explicit reviewed ED7 source/input differs')
    receipt_path = c.STAGE / (phase + '_execution_receipt.json')
    reservation_path = c.STAGE / (phase + '_capture_reservation.json')
    c.need(not receipt_path.exists() and not reservation_path.exists(),
           'Actual ED7 phase already permanently reserved; no retry')
    if phase != 'tests':
        c.admit(args.expected_sha256, phase)
    command = [sys.executable, '-u', '-B', str(entry)]
    if phase in ('primary', 'reload'):
        command += ['--phase', phase, '--registration-sha256', args.expected_sha256]
    elif phase in ('audit', 'finalize'):
        command += ['--registration-sha256', args.expected_sha256]
    sources = [c.ROOT / 'scripts' / name for name in SOURCE_NAMES]
    before = c.binding(entry); sources_before = [c.binding(p) for p in sources]
    started = datetime.now(timezone.utc).isoformat(); clock = time.perf_counter()
    c.write_new(reservation_path, {'phase': phase, 'started_utc': started, 'input': c.binding(target),
        'entry': before, 'sources_before': sources_before, 'permanent_single_use': True,
        'automatic_retry': False})
    stdout = c.STAGE / (phase + '.stdout.log'); stderr = c.STAGE / (phase + '.stderr.log')
    with stdout.open('xb') as out, stderr.open('xb') as err:
        child = subprocess.Popen(command, cwd=c.ROOT, stdout=subprocess.PIPE, stderr=err,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1'))
        c.write_new(c.STAGE / (phase + '_launch.json'), {'phase': phase, 'started_utc': started,
            'command': command, 'capture_pid': os.getpid(), 'launcher_pid': child.pid,
            'worker_pid_may_differ_from_launcher': True, 'input': c.binding(target), 'entry': before})
        for line in iter(child.stdout.readline, b''):
            out.write(line); out.flush(); print(line.decode('utf-8', errors='replace').rstrip(), flush=True)
        child.stdout.close(); code = child.wait()
    outputs = {}; worker = None
    if phase != 'tests' and code == 0:
        acceptance = c.STAGE / (phase + '_acceptance.json')
        if phase == 'finalize':
            acceptance = c.STAGE / 'completion_acceptance.json'
        outputs['acceptance'] = c.binding(acceptance)
        value = c.read(acceptance); worker = value['worker_pid']
        if phase == 'primary':
            outputs['package_manifest'] = c.binding(c.PACKAGE / 'package_manifest.json')
        elif phase == 'reload':
            outputs['reload_logits'] = c.binding(c.STAGE / 'reload_logits.f32')
        elif phase == 'finalize':
            for name in ('report', 'plan', 'state', 'delivery_zip', 'delivery_zip_verification'):
                outputs[name] = value[name]
    failure = c.STAGE / (phase + '_failure.json')
    if failure.exists():
        outputs['failure'] = c.binding(failure)
    after = c.binding(entry); sources_after = [c.binding(p) for p in sources]
    result = {'version': 'TCN-ED7-actual-command-receipt-1.0', 'phase': phase,
        'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': time.perf_counter() - clock, 'exit_code': code, 'capture_pid': os.getpid(),
        'launcher_pid': child.pid, 'actual_worker_pid': worker, 'command': command,
        'input': c.binding(target), 'entry_before': before, 'entry_after': after,
        'entry_unchanged': before == after, 'sources_before': sources_before, 'sources_after': sources_after,
        'sources_unchanged': sources_before == sources_after, 'stdout': c.binding(stdout),
        'stderr': c.binding(stderr), 'outputs': outputs, 'capture_source': c.binding(__file__),
        'automatic_retry': False, 'capture_model_or_optimizer_operations': 0}
    if phase == 'tests':
        result.update(real_model_operations=0, real_checkpoint_loads=0, real_market_records_read=False,
                      real_Windows_power_API_calls=0)
    c.write_new(receipt_path, result)
    print(json.dumps({'phase': phase, 'actual_exit_code': code, 'receipt': c.binding(receipt_path)}, ensure_ascii=True), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
