"""One separately reserved recheck after the declaration-only AST false positive.

Runs the unchanged49 synthetic suite. Never reuses the original failed namespace
or starts a numerical engineering phase. Formal capture/source core stays10.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.reference.training.scripts import tcn_existing_data_ed7_contract as c
from experiments.reference.training.scripts.run_tcn_existing_data_ed7_capture import SOURCE_NAMES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--entry-sha256', required=True)
    args = parser.parse_args(argv)
    entry = c.ROOT / 'scripts/test_tcn_existing_data_ed7.py'
    c.need(c.sha(entry) == args.expected_sha256 == args.entry_sha256,
           'The unchanged reviewed49 test source differs')
    old = c.read(c.STAGE / 'tests_execution_receipt.json')
    c.need(old['exit_code'] == 1 and old['sources_unchanged'] is True
           and c.read(c.STAGE / 'initial_failed_synthetic_snapshot/preservation.json')['real_model_operations'] == 0,
           'Original pre-model failed evidence must be preserved')
    name = 'tests_v2'
    receipt_path = c.STAGE / (name + '_execution_receipt.json')
    reservation_path = c.STAGE / (name + '_capture_reservation.json')
    c.need(not receipt_path.exists() and not reservation_path.exists(),
           'Revised synthetic phase already permanently reserved; no retry')
    sources = [c.ROOT / 'scripts' / name for name in SOURCE_NAMES] + [Path(__file__)]
    before = c.binding(entry); sources_before = [c.binding(p) for p in sources]
    command = [sys.executable, '-u', '-B', str(entry)]
    started = datetime.now(timezone.utc).isoformat(); clock = time.perf_counter()
    c.write_new(reservation_path, {'phase': 'tests', 'revision': 'v2', 'started_utc': started,
        'input': before, 'entry': before, 'sources_before': sources_before,
        'permanent_single_use': True, 'automatic_retry': False,
        'original_failure': c.binding(c.STAGE / 'tests_execution_receipt.json')})
    stdout = c.STAGE / (name + '.stdout.log'); stderr = c.STAGE / (name + '.stderr.log')
    with stdout.open('xb') as out, stderr.open('xb') as err:
        child = subprocess.Popen(command, cwd=c.ROOT, stdout=subprocess.PIPE, stderr=err,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1'))
        c.write_new(c.STAGE / (name + '_launch.json'), {'phase': 'tests', 'revision': 'v2',
            'started_utc': started, 'capture_pid': os.getpid(), 'launcher_pid': child.pid,
            'command': command, 'input': before, 'entry': before,
            'worker_pid_may_differ_from_launcher': True})
        for line in iter(child.stdout.readline, b''):
            out.write(line); out.flush(); print(line.decode('utf-8', errors='replace').rstrip(), flush=True)
        child.stdout.close(); code = child.wait()
    after = c.binding(entry); sources_after = [c.binding(p) for p in sources]
    c.write_new(receipt_path, {'version': 'TCN-ED7-actual-command-receipt-1.0', 'phase': 'tests',
        'revision': 'v2', 'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': time.perf_counter() - clock, 'exit_code': code, 'capture_pid': os.getpid(),
        'launcher_pid': child.pid, 'actual_worker_pid': None, 'command': command,
        'input': before, 'entry_before': before, 'entry_after': after, 'entry_unchanged': before == after,
        'sources_before': sources_before, 'sources_after': sources_after,
        'sources_unchanged': sources_before == sources_after, 'stdout': c.binding(stdout),
        'stderr': c.binding(stderr), 'outputs': {}, 'capture_source': c.binding(__file__),
        'automatic_retry': False, 'real_model_operations': 0, 'real_checkpoint_loads': 0,
        'real_market_records_read': False, 'real_Windows_power_API_calls': 0,
        'capture_model_or_optimizer_operations': 0,
        'original_failed_receipt': c.binding(c.STAGE / 'tests_execution_receipt.json')})
    print(json.dumps({'phase': 'tests', 'revision': 'v2', 'actual_exit_code': code,
                      'receipt': c.binding(receipt_path)}, ensure_ascii=True), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
