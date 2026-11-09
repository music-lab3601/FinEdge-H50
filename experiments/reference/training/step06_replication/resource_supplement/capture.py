"""Actual CLI capture for one ED6 resource-only guard or pure-mock suite.

This controller has no Windows API or model implementation. The independently
registered guard alone owns its process handle and temporary thread request.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[3]


def helpers():
    spec = importlib.util.spec_from_file_location('ed6_resource_guard_source', DIRECTORY / 'guard.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['tests', 'guard'])
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--entry-sha256', required=True)
    args = parser.parse_args(argv); helper = helpers(); phase = args.phase
    name = 'resource_tests' if phase == 'tests' else 'resource_guard'
    receipt_path = DIRECTORY / (name + '_execution_receipt.json')
    if receipt_path.exists() or (DIRECTORY / (name + '_capture_reservation.json')).exists():
        raise ValueError('Resource phase already permanently reserved; no retry')
    entry = DIRECTORY / ('test_guard.py' if phase == 'tests' else 'guard.py')
    target = entry if phase == 'tests' else DIRECTORY / 'resource_supplement_registration.json'
    if helper.binding(entry)['sha256'] != args.entry_sha256 or helper.binding(target)['sha256'] != args.expected_sha256:
        raise ValueError('Explicit resource input or reviewed source differs')
    if phase == 'guard':
        helper.admit(target, args.expected_sha256)
        command = [sys.executable, '-u', '-B', str(entry), '--registration-sha256', args.expected_sha256]
    else:
        command = [sys.executable, '-u', '-B', str(entry)]
    sources = [entry, Path(__file__)]
    if phase == 'tests':
        sources.append(DIRECTORY / 'guard.py')
    before = helper.binding(entry); sources_before = [helper.binding(p) for p in sources]
    started = datetime.now(timezone.utc).isoformat(); clock = time.monotonic()
    helper.put(DIRECTORY / (name + '_capture_reservation.json'), {'phase': name, 'started_utc': started,
        'input': helper.binding(target), 'entry': before, 'sources_before': sources_before,
        'permanent_single_use': True, 'automatic_retry': False, 'target_worker_pid': 20296 if phase == 'guard' else None})
    stdout = DIRECTORY / (name + '.stdout.log'); stderr = DIRECTORY / (name + '.stderr.log')
    with stdout.open('xb') as out, stderr.open('xb') as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=err,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1'))
        helper.put(DIRECTORY / (name + '_launch.json'), {'phase': name, 'started_utc': started, 'command': command,
            'capture_pid': os.getpid(), 'launcher_pid': process.pid, 'worker_pid_may_differ_from_launcher': True,
            'target_worker_pid': 20296 if phase == 'guard' else None, 'input': helper.binding(target), 'entry': before})
        for line in iter(process.stdout.readline, b''):
            out.write(line); out.flush(); print(line.decode('utf-8', errors='replace').rstrip(), flush=True)
        process.stdout.close(); code = process.wait()
    outputs = {}; worker = None
    if phase == 'guard' and (DIRECTORY / 'guard_result.json').is_file():
        outputs['guard_result'] = helper.binding(DIRECTORY / 'guard_result.json')
        outputs['guard_events'] = helper.binding(DIRECTORY / 'guard_events.jsonl')
        worker = helper.read(DIRECTORY / 'guard_result.json')['actual_worker_pid']
    after = helper.binding(entry); sources_after = [helper.binding(p) for p in sources]
    result = {'version': 'TCN-ED6-resource-actual-command-receipt-1.0', 'phase': name,
        'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.monotonic() - clock,
        'exit_code': code, 'capture_pid': os.getpid(), 'launcher_pid': process.pid, 'actual_worker_pid': worker,
        'target_worker_pid': 20296 if phase == 'guard' else None,
        'command': command, 'input': helper.binding(target), 'entry_before': before, 'entry_after': after,
        'entry_unchanged': before == after, 'sources_before': sources_before, 'sources_after': sources_after,
        'sources_unchanged': sources_before == sources_after, 'stdout': helper.binding(stdout), 'stderr': helper.binding(stderr),
        'outputs': outputs, 'capture_source': helper.binding(Path(__file__)), 'automatic_retry': False,
        'model_operations': 0, 'checkpoint_loads': 0, 'training_or_budget_mutations': 0,
        'permanent_power_or_display_settings_changed': False}
    if phase == 'tests':
        result.update(real_records_read=False, real_Windows_API_calls=0)
    helper.put(receipt_path, result)
    print(json.dumps({'phase': name, 'actual_exit_code': code, 'receipt': helper.binding(receipt_path)}, ensure_ascii=True), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
