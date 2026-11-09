"""Root-only ED6 resource audit and gated delegation to frozen publication v2.

No numerical implementation is changed or imported. Publication is delegated
only after the independent resource audit has actually exited successfully.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.reference.training.scripts.audit_tcn_existing_data_ed6_resource_supplement import ROOT, STAGE, RES, REGISTRATION, admit, binding, bound, checked_receipt, equivalent_ref, local, need, read, verify, write_new


def publication_admission(reg, registration_sha):
    actual = checked_receipt(RES / 'resource_audit_execution_receipt.json', 'resource_audit')
    need(equivalent_ref(actual['input'], binding(REGISTRATION))
         and actual['input']['sha256'] == registration_sha
         and equivalent_ref(actual['entry_before'], reg['resource_audit_source_binding'])
         and equivalent_ref(actual['capture_source'], reg['resource_publication_gate_source_binding']),
         'Independent actual resource audit source/input differs')
    result = bound(actual['outputs']['resource_audit'])
    need(result['passed'] is True and result['true_safe_failure'] is True
         and result['resource_outcome'] == 'failed_no_handle_no_request'
         and result['resource_support_failed'] is True and result['guard_passed'] is False
         and result['resource_cleanup_passed'] is False and result['resource_cleanup_not_required'] is True
         and result['unreleased_resources'] == 0
         and result['original_training_worker_terminal_verified'] is True
         and result['guard_terminal_verified'] is True
         and equivalent_ref(result['registration_binding'], binding(REGISTRATION)),
         'Resource supplement has not completed successfully')
    return actual, result


def phase_command(phase, reg, registration_sha):
    if phase == 'audit':
        entry = verify(reg['resource_audit_source_binding'])
        return entry, [sys.executable, '-u', '-B', str(entry),
                       '--registration-sha256', registration_sha]
    publication_admission(reg, registration_sha)
    for slot in ('control43', 'candidate43'):
        for filename, actual_phase in [('training_execution_receipt.json', 'training'),
                                       ('checkpoint_execution_receipt.json', 'checkpoint_validation'),
                                       ('audit_v2_execution_receipt.json', 'audit')]:
            actual = checked_receipt(STAGE / 'trials' / slot / filename, actual_phase)
            need(actual['slot_id'] == slot, 'The original per-trial postrun pipeline differs')
    entry = verify(reg['frozen_finalizer_capture_source_binding'])
    return entry, [sys.executable, '-u', '-B', str(entry), 'finalizer_v2',
                   '--expected-sha256', reg['selection_lock_binding']['sha256'],
                   '--entry-sha256', reg['frozen_finalizer_source_binding']['sha256'],
                   '--global-repair-registration-sha256', reg['global_repair_registration_binding']['sha256']]


def run(phase, registration_sha, entry_sha):
    reg = admit(registration_sha)
    need(equivalent_ref(reg['resource_publication_gate_source_binding'], binding(Path(__file__)))
         and binding(Path(__file__))['sha256'] == entry_sha, 'Explicit resource gate source differs')
    entry, command = phase_command(phase, reg, registration_sha)
    label = 'resource_audit' if phase == 'audit' else 'resource_publication_gate'
    receipt_path = RES / (label + '_execution_receipt.json')
    reservation_path = RES / (label + '_capture_reservation.json')
    need(not receipt_path.exists() and not reservation_path.exists(),
         'Resource phase is permanently reserved; no automatic retry')
    sources = [Path(__file__), local(reg['resource_audit_source_binding']['path']), entry]
    if phase == 'publish':
        sources += [local(reg['frozen_finalizer_source_binding']['path']),
                    ROOT / 'scripts/run_tcn_existing_data_ed6_capture.py']
    sources = list(dict.fromkeys(sources))
    before = binding(entry); sources_before = [binding(p) for p in sources]
    started = datetime.now(timezone.utc).isoformat(); timer = time.perf_counter()
    write_new(reservation_path, {'phase': label, 'started_utc': started, 'input': binding(REGISTRATION),
        'entry': before, 'sources_before': sources_before, 'permanent_single_use': True, 'automatic_retry': False})
    stdout = RES / (label + '.stdout.log'); stderr = RES / (label + '.stderr.log')
    with stdout.open('xb') as out, stderr.open('xb') as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=err,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1'))
        write_new(RES / (label + '_launch.json'), {'phase': label, 'started_utc': started,
            'command': command, 'capture_pid': os.getpid(), 'launcher_pid': process.pid,
            'worker_pid_may_differ_from_launcher': True, 'input': binding(REGISTRATION), 'entry': before})
        for line in iter(process.stdout.readline, b''):
            out.write(line); out.flush(); print(line.decode('utf-8', errors='replace').rstrip(), flush=True)
        process.stdout.close(); code = process.wait()
    outputs = {}; worker = None
    if code == 0 and phase == 'audit':
        outputs['resource_audit'] = binding(RES / 'resource_audit.json')
        worker = read(RES / 'resource_audit.json')['worker_pid']
    elif code == 0:
        frozen = checked_receipt(STAGE / 'finalizer_v2_execution_receipt.json', 'finalizer_v2')
        need(equivalent_ref(frozen['capture_source'], reg['frozen_finalizer_capture_source_binding'])
             and equivalent_ref(frozen['entry_before'], reg['frozen_finalizer_source_binding'])
             and equivalent_ref(frozen['input'], reg['selection_lock_binding']),
             'Frozen actual publication identity differs')
        finalization = bound(frozen['outputs']['finalization_receipt'])
        report_ref = finalization['report']; verify(report_ref)
        supplement_date = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        report_path = RES / ('TCN_ED6_资源守护失败附页_' + supplement_date + '.txt')
        with local(report_path).open('x', encoding='utf-8', newline='\n') as stream:
            stream.write('ED6 临时电池守护附项：失败事实与独立主任务验收\n\n'
                '独立电池守护实际 exit2：OpenProcess 返回失败，未取得进程句柄，未调用执行状态请求。'
                '原失败预约、源码、结果与原始日志全部保留，未重试或新开守护。\n'
                '后续只读探针对三种访问权限均观察到 access denied（WinError 5）。'
                '探针工具观察为执行后补存，不冒称当时独立捕获收据。\n'
                'guard_passed=false，resource_support_failed=true；clear 与 CloseHandle API 均未调用，'
                '属于没有资源可清理，不是清理成功。资源附项按 failed_no_handle_no_request 安全关闭。\n'
                '原训练自身的临时 AC-only 请求清理与本失败守护分列。模型训练、CPU 检查、完整 D 缓存审计和'
                '固定代表选择仍按原冻结流程独立验收；本附项不改变其结果与质量边界。\n'
                '未修改原完成件、报告、计划、状态或交接文件；本附页不声称电池守护生效、系统从未睡眠或新泛化提升。\n')
            stream.flush(); os.fsync(stream.fileno())
        supplemental = {'version': 'TCN-ED6-resource-publication-supplement-1.0',
            'completed_utc': datetime.now(timezone.utc).isoformat(),
            'resource_outcome': 'failed_no_handle_no_request', 'resource_support_failed': True,
            'passed': True, 'true_safe_failure': True, 'guard_passed': False,
            'temporary_DC_idle_request_effective': False,
            'cleanup_passed': False, 'cleanup_not_required': True, 'unreleased_resources': 0,
            'clear_API_calls': 0, 'CloseHandle_API_calls': 0, 'cleanup_not_applicable': True,
            'resource_audit_binding': binding(RES / 'resource_audit.json'),
            'resource_audit_execution_receipt_binding': binding(RES / 'resource_audit_execution_receipt.json'),
            'registration_binding': binding(REGISTRATION),
            'frozen_finalizer_execution_receipt_binding': binding(STAGE / 'finalizer_v2_execution_receipt.json'),
            'original_completion_acceptance_binding': frozen['outputs']['completion_acceptance'],
            'original_report_binding': report_ref, 'resource_report_binding': binding(report_path),
            'original_guard_failed_receipt_binding': reg['resource_guard_receipt_binding'],
            'model_quality_or_selection_changed_by_supplement': False,
            'claim': 'The auxiliary guard failed with access denied before acquiring a handle or issuing a state request. The failed guard is preserved and was not retried. Primary model and publication gates passed independently. No successful guard/clear/close or uninterrupted-awake claim is made.'}
        write_new(RES / 'resource_supplement_acceptance.json', supplemental)
        outputs = {'frozen_finalizer_execution_receipt': binding(STAGE / 'finalizer_v2_execution_receipt.json'),
                   **frozen['outputs'], 'resource_audit': binding(RES / 'resource_audit.json')}
        outputs['resource_supplement_acceptance'] = binding(RES / 'resource_supplement_acceptance.json')
        outputs['resource_report'] = binding(report_path)
        worker = frozen['actual_worker_pid']
    after = binding(entry); sources_after = [binding(p) for p in sources]
    write_new(receipt_path, {'version': 'TCN-ED6-resource-supplement-actual-command-receipt-1.0',
        'phase': label, 'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': time.perf_counter() - timer, 'exit_code': code, 'capture_pid': os.getpid(),
        'launcher_pid': process.pid, 'actual_worker_pid': worker, 'command': command,
        'input': binding(REGISTRATION), 'entry_before': before, 'entry_after': after,
        'entry_unchanged': before == after, 'sources_before': sources_before, 'sources_after': sources_after,
        'sources_unchanged': sources_before == sources_after, 'stdout': binding(stdout), 'stderr': binding(stderr),
        'outputs': outputs, 'capture_source': binding(Path(__file__)), 'automatic_retry': False,
        'model_or_optimizer_operations': 0, 'checkpoint_loads': 0, 'training_or_budget_mutations': 0})
    print(json.dumps({'phase': label, 'actual_exit_code': code, 'receipt': binding(receipt_path)}, ensure_ascii=True), flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['audit', 'publish'])
    parser.add_argument('--registration-sha256', required=True)
    parser.add_argument('--entry-sha256', required=True)
    args = parser.parse_args(argv)
    return run(args.phase, args.registration_sha256, args.entry_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
