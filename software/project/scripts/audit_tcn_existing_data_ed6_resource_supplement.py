"""Independent ED6 temporary-power-request audit; standard-library metadata only.

This supplemental audit never imports a numerical/model module, opens market
records or checkpoints, changes the training budget, or changes power settings.
The original numerical and publication implementations remain immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'artifacts/tcn_existing_data_optimization_2026-10-17/step06_replication'
RES = STAGE / 'resource_supplement'
GUARD_REGISTRATION = RES / 'resource_supplement_registration.json'
REGISTRATION = RES / 'resource_postrun_registration.json'
TRIAL = STAGE / 'trials/candidate43'
TARGET_WORKER_PID = 20296
FINALIZER_SHA = 'e02052afdb0ae8ff7e82d1999d4e304438b176b7d5cd78b2e1d1d5a28073d2a1'
FINALIZER_CAPTURE_SHA = 'f4a01b17025ef9a53ca823335b5d23095e73c4fb5a018d085993cd23efe9b747'
GUARD_SOURCE_SHA = '773757701ce1f8b2c6d21cd963fe3fbba96073be7910e2960210d77a074489cd'
GUARD_CAPTURE_SHA = '833322f73925aff5bf5688a1092c12926314a94b09af6cf78045ec7618c0dcb3'
FAILED_GUARD_RECEIPT_SHA = '10ddb35c4b50d60f1950f22975f31587db7dcd37aee1143358e4601058979666'
LIMITS = {'maximum_wall_seconds': 64800, 'poll_interval_ms': 10000,
          'minimum_battery_percent_exclusive': 20, 'model_operations': 0,
          'checkpoint_loads': 0, 'training_or_budget_mutations': 0}


def need(value, message):
    if not value:
        raise ValueError(message)


def local(value):
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    need('..' not in p.parts and p.is_relative_to(ROOT), 'Outside D workspace')
    for q in (p, *p.parents):
        if q.exists():
            need(not q.is_symlink() and not getattr(q.lstat(), 'st_file_attributes', 0) & 0x400,
                 'Reparse evidence/output forbidden')
        if q == ROOT:
            break
    need(p.resolve().is_relative_to(ROOT.resolve()), 'Resolved evidence escaped workspace')
    return p


def sha(path):
    h = hashlib.sha256()
    with local(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    p = local(path)
    return {'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}


def equivalent_ref(left, right):
    for ref in (left, right):
        need(isinstance(ref, dict) and {'path', 'sha256', 'bytes'} <= set(ref), 'Complete identity required')
        need(type(ref['bytes']) is int and ref['bytes'] >= 0 and isinstance(ref['sha256'], str)
             and len(ref['sha256']) == 64, 'Invalid identity')
    return (local(left['path']).resolve() == local(right['path']).resolve()
            and left['sha256'] == right['sha256'] and left['bytes'] == right['bytes'])


def verify(ref):
    p = local(ref['path'])
    need(equivalent_ref(binding(p), ref), 'Evidence changed: ' + str(p))
    return p


def read(path):
    p = local(path)
    need(p.stat().st_size <= 32 * 1024**2, 'Oversized metadata')
    return json.loads(p.read_text('utf-8-sig'),
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + s)))


def bound(ref):
    return read(verify(ref))


def write_new(path, value):
    p = local(path)
    need(p.is_relative_to(RES), 'Wrong resource output')
    with p.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def checked_receipt(path, phase, expected_exit=0):
    result = read(path)
    need(result['phase'] == phase and result['exit_code'] == expected_exit and result['entry_unchanged'] is True
         and result['entry_before'] == result['entry_after'], 'Actual CLI/source terminal differs: ' + phase)
    for name in ('entry_before', 'entry_after', 'capture_source', 'stdout', 'stderr', 'input'):
        verify(result[name])
    if 'sources_before' in result:
        need(result['sources_before'] == result['sources_after'] and result['sources_unchanged'] is True,
             'Actual dependency sources changed')
        for ref in result['sources_before']:
            verify(ref)
    for ref in result['outputs'].values():
        verify(ref)
    return result


def verify_test_evidence(ref, expected_count):
    evidence = bound(ref)
    need(evidence['passed'] is True and evidence['actual_exit_code'] == 0
         and type(evidence['separate_synthetic_test_count']) is int
         and evidence['separate_synthetic_test_count'] == expected_count and evidence['no_real_data_or_ML'] is True,
         'Actual resource synthetic tests missing')
    actual = bound(evidence['actual_receipt'])
    need(actual['exit_code'] == 0 and actual['entry_unchanged'] is True
         and actual['entry_before'] == actual['entry_after'], 'Actual resource test failed')
    need(evidence['sources_before'] == evidence['sources_after'] == actual['sources_before'] == actual['sources_after']
         and actual['sources_unchanged'] is True,
         'Resource test source evidence changed')
    for source in evidence['sources_before']:
        verify(source)
    for name in ('entry_before', 'entry_after', 'capture_source', 'stdout', 'stderr'):
        verify(actual[name])
    need(equivalent_ref(evidence['test_source'], actual['entry_before']), 'Resource test entry differs')
    return evidence


def read_events(path):
    p = local(path)
    need(p.stat().st_size <= 32 * 1024**2, 'Oversized resource event log')
    text = p.read_text('utf-8')
    need(text.endswith('\n'), 'Incomplete resource JSONL terminal line')
    events = [json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
              for line in text.splitlines() if line.strip()]
    need(events and all(isinstance(event, dict) for event in events), 'Missing resource events')
    return events


def expected_request(ac_status, battery_percent, read_succeeded=True):
    """Only a known, healthy battery on known DC allows the temporary request."""
    return (read_succeeded is True and type(ac_status) is int and ac_status == 0 and type(battery_percent) is int
            and 20 < battery_percent <= 100)


def successful_terminal(training, guard_actual, result):
    need(training['phase'] == 'training' and training['slot_id'] == 'candidate43'
         and training['actual_worker_pid'] == TARGET_WORKER_PID and training['exit_code'] == 0,
         'Original training worker has not actually ended successfully')
    need(guard_actual['phase'] == 'resource_guard' and guard_actual['exit_code'] == 0
         and guard_actual['target_worker_pid'] == TARGET_WORKER_PID
         and type(guard_actual['actual_worker_pid']) is int and guard_actual['actual_worker_pid'] > 0,
         'Guard has not actually ended successfully')
    need(result['passed'] is True and result['target_worker_pid'] == TARGET_WORKER_PID
         and result['actual_worker_pid'] == result['worker_pid'] == result['guard_worker_pid']
         == guard_actual['actual_worker_pid'] and result['actual_worker_pid'] != TARGET_WORKER_PID,
         'Guard and target PID roles differ from actual terminal evidence')
    need(result['target_process_handle_signaled'] is True and result['last_wait_result'] == 0
         and result['stop_reason'] == 'original_target_process_signaled' and result['error'] is None,
         'Guard did not observe the original process handle signal')
    for name in ('model_operations', 'checkpoint_loads', 'training_or_budget_mutations'):
        need(type(result[name]) is int and result[name] == 0
             and type(guard_actual[name]) is int and guard_actual[name] == 0, 'Resource scope exceeded')
    need(result['limits'] == LIMITS and result['display_request_used'] is False
         and result['permanent_power_or_display_settings_changed'] is False
         and guard_actual['permanent_power_or_display_settings_changed'] is False,
         'Resource limits or display/permanent-power scope changed')
    need(isinstance(result['seconds'], (int, float)) and not isinstance(result['seconds'], bool)
         and math.isfinite(result['seconds']) and result['seconds'] >= 0, 'Invalid actual resource duration')


def validate_events(result, events, interval):
    need(len(events) >= 2 and events[0]['event'] == 'guard_started'
         and events[-1]['event'] == 'guard_finally', 'Actual guard start/finally events missing')
    need(sum(e['event'] == 'guard_started' for e in events) == 1
         and sum(e['event'] == 'guard_finally' for e in events) == 1
         and all(e['event'] in ('guard_started', 'execution_state_changed', 'guard_finally') for e in events),
         'Repeated/failed/unrecognized guard event')
    moments = [datetime.fromisoformat(e['utc']) for e in events]
    need(all(moment.utcoffset() is not None for moment in moments) and moments == sorted(moments),
         'Resource event timestamps lack timezone or are unordered')
    first, last = events[0], events[-1]
    need(first['target_worker_pid'] == TARGET_WORKER_PID and first['target_handle_bound'] is True
         and first['guard_worker_pid'] == result['actual_worker_pid']
         and first['target_creation'] == result['target_creation']
         and first['creation_interval_unix'] == result['creation_interval_unix'] == list(interval),
         'Original target process-handle identity differs')
    creation = result['target_creation']
    need(type(creation['filetime_100ns']) is int and creation['filetime_100ns'] > 0
         and type(creation['unix_seconds']) in (int, float) and math.isfinite(creation['unix_seconds'])
         and interval[0] <= creation['unix_seconds'] <= interval[1]
         and abs(creation['unix_seconds'] - (creation['filetime_100ns'] / 10**7 - 11644473600)) <= 1e-5,
         'Process creation identity is outside the original launch/reservation interval')
    previous = None; requests = releases = 0
    for event in events[1:-1]:
        power = event['power']
        wanted = expected_request(power['ACLineStatus'], power['BatteryLifePercent'], power['read_succeeded'])
        need(type(event['battery_only_request']) is bool and event['battery_only_request'] == wanted
             and event['battery_only_request'] != previous, 'Battery request policy or transition differs')
        need(type(event['requested']) is int and event['requested'] == (0x80000000 | (1 if wanted else 0))
             and type(event['returned_previous']) is int and event['returned_previous'] != 0
             and event['actual_API_call_succeeded'] is True,
             'Actual temporary execution-state API evidence differs')
        previous = wanted; requests += int(wanted); releases += int(not wanted)
    need(last['target_worker_pid'] == TARGET_WORKER_PID and last['opened_process_handle'] is True
         and last['requested'] == 0x80000000 and type(last['returned_previous']) is int
         and last['returned_previous'] != 0 and last['clear_actual_succeeded'] is True
         and last['CloseHandle_actual_succeeded'] is True
         and last['clear_error'] is None and last['close_error'] is None,
         'Actual finally clear/CloseHandle failed or missing')
    need(result['opened_process_handle'] is True and result['clear_actual_succeeded'] is True
         and result['CloseHandle_actual_succeeded'] is True
         and result['clear_error'] is None and result['close_error'] is None
         and result['final_clear_returned_previous'] == last['returned_previous'],
         'Guard cleanup result differs from the actual finally event')
    return {'events': len(events), 'battery_request_API_calls': requests, 'release_API_calls_before_finally': releases,
            'finally_release_API_calls': 1, 'CloseHandle_API_calls': 1}


def failed_no_handle_no_request(training, guard, result, events, diagnosis):
    """Accept only this observed failure before a handle or state request existed."""
    need(training['phase'] == 'training' and training['slot_id'] == 'candidate43'
         and training['exit_code'] == 0 and training['actual_worker_pid'] == TARGET_WORKER_PID,
         'Original training must actually finish successfully before resource closure')
    need(guard['phase'] == 'resource_guard' and guard['exit_code'] == 2
         and guard['actual_worker_pid'] == result['actual_worker_pid'] == result['guard_worker_pid'] == result['worker_pid']
         and type(result['actual_worker_pid']) is int and result['actual_worker_pid'] > 0
         and result['actual_worker_pid'] != TARGET_WORKER_PID
         and guard['target_worker_pid'] == result['target_worker_pid'] == TARGET_WORKER_PID,
         'Failed guard actual PID/terminal roles differ')
    need(result['passed'] is False and result['stop_reason'] == 'guard_error'
         and result['error'] == {'type': 'ValueError', 'message': 'OpenProcess failed'}
         and result['target_process_handle_signaled'] is False and result['last_wait_result'] is None
         and result['target_creation'] is None and result['opened_process_handle'] is False,
         'Failure was not the observed pre-handle OpenProcess failure')
    need(result['final_clear_returned_previous'] is None and result['clear_actual_succeeded'] is False
         and result['CloseHandle_actual_succeeded'] is False
         and result['clear_error'] is None and result['close_error'] is None,
         'A failed/unreleased acquired resource may not use the no-handle exception')
    need(len(events) == 2 and events[0]['event'] == 'guard_error' and events[0]['error'] == result['error']
         and events[1]['event'] == 'guard_finally', 'Failed guard had a start, state change or other event')
    final = events[1]
    need(final['target_worker_pid'] == TARGET_WORKER_PID and final['opened_process_handle'] is False
         and final['requested'] == 0x80000000 and final['returned_previous'] is None
         and final['clear_actual_succeeded'] is False and final['CloseHandle_actual_succeeded'] is False
         and final['clear_error'] is None and final['close_error'] is None,
         'Failed guard finally evidence contradicts no acquired resources')
    moments = [datetime.fromisoformat(e['utc']) for e in events]
    need(all(moment.utcoffset() is not None for moment in moments) and moments == sorted(moments),
         'Failed guard event timestamps invalid')
    need(result['limits'] == LIMITS and result['display_request_used'] is False
         and result['permanent_power_or_display_settings_changed'] is False
         and guard['permanent_power_or_display_settings_changed'] is False,
         'Failed guard resource scope differs')
    for name in ('model_operations', 'checkpoint_loads', 'training_or_budget_mutations'):
        need(type(result[name]) is int and result[name] == 0
             and type(guard[name]) is int and guard[name] == 0, 'Failed guard exceeded zero numerical scope')
    need(diagnosis['version'] == 'TCN-ED6-readonly-target-access-diagnosis-1.0'
         and diagnosis['SetThreadExecutionState_calls'] == 0 and diagnosis['model_operations'] == 0
         and diagnosis['training_or_budget_mutations'] == 0 and diagnosis['new_guard_reservation'] is False,
         'Read-only access diagnosis scope differs')
    probes = diagnosis['results']
    need(len(probes) == 3 and [p['access_mask'] for p in probes] == [1052672, 4096, 1048576]
         and all(p['target_pid'] == TARGET_WORKER_PID and p['opened'] is False
                 and p['WinError'] == 5 and p['actual_closed_if_opened'] is None for p in probes),
         'The three read-only probes did not all observe access denied without handles')
    return {'events': 2, 'battery_request_API_calls': 0, 'release_API_calls_before_finally': 0,
            'finally_release_API_calls': 0, 'CloseHandle_API_calls': 0,
            'cleanup_API_calls_not_applicable': True, 'probe_OpenProcess_access_denied_observations': 3}


def admit(registration_sha):
    need(sha(REGISTRATION) == registration_sha, 'Explicit resource registration differs')
    reg = read(REGISTRATION)
    need(reg['version'] == 'TCN-ED6-resource-postrun-registration-1.0'
         and reg['target_worker_pid'] == TARGET_WORKER_PID and reg['slot_id'] == 'candidate43',
         'Resource target differs from the existing candidate worker')
    need(reg['resource_outcome'] == 'failed_no_handle_no_request'
         and reg['diagnosis_evidence_kind'] == 'root_inline_posthoc_observation',
         'Only the explicitly observed finite no-handle failure closure is registered')
    need(equivalent_ref(reg['resource_audit_source_binding'], binding(Path(__file__))),
         'Resource auditor source differs')
    for name in ('guard_registration_binding', 'resource_audit_source_binding', 'resource_publication_gate_source_binding',
                 'frozen_finalizer_source_binding', 'frozen_finalizer_capture_source_binding',
                 'selection_lock_binding', 'global_repair_registration_binding',
                 'training_registration_binding', 'training_phase_reservation_binding', 'training_launch_binding',
                 'training_execution_receipt_binding', 'resource_guard_receipt_binding',
                 'resource_guard_result_binding', 'resource_guard_events_binding', 'target_access_diagnosis_binding',
                 'root_tool_observation_binding', 'failed_guard_evidence_review_binding'):
        verify(reg[name])
    need(local(reg['guard_registration_binding']['path']) == GUARD_REGISTRATION,
         'Original guard registration was replaced')
    guard_reg = bound(reg['guard_registration_binding'])
    need(guard_reg['version'] == 'TCN-ED6-DC-resource-registration-1.0'
         and guard_reg['target_worker_pid'] == TARGET_WORKER_PID and guard_reg['limits'] == LIMITS,
         'Original guard target or limits differ')
    need(guard_reg['guard_source_binding']['sha256'] == GUARD_SOURCE_SHA
         and guard_reg['capture_source_binding']['sha256'] == GUARD_CAPTURE_SHA
         and reg['resource_guard_receipt_binding']['sha256'] == FAILED_GUARD_RECEIPT_SHA,
         'Original failed guard evidence or its source was replaced')
    need(bound(guard_reg['source_review_binding'])['passed'] is True, 'Original guard source review missing')
    for parent_name, child_name in [('training_registration_binding', 'target_training_registration_binding'),
                                   ('training_phase_reservation_binding', 'target_training_reservation_binding'),
                                   ('training_launch_binding', 'target_training_launch_binding')]:
        need(equivalent_ref(reg[parent_name], guard_reg[child_name]), 'Original target registration was substituted')
    need(local(reg['frozen_finalizer_source_binding']['path']) == ROOT / 'scripts/finalize_tcn_existing_data_ed6_v2.py'
         and reg['frozen_finalizer_source_binding']['sha256'] == FINALIZER_SHA
         and local(reg['frozen_finalizer_capture_source_binding']['path']) == ROOT / 'scripts/run_tcn_existing_data_ed6_audit_v2_capture.py'
         and reg['frozen_finalizer_capture_source_binding']['sha256'] == FINALIZER_CAPTURE_SHA,
         'Frozen publication implementation was changed')
    phase = bound(reg['training_phase_reservation_binding'])
    need(phase['worker_pid'] == TARGET_WORKER_PID and phase['no_automatic_retry'] is True
         and equivalent_ref(phase['registration_binding'], reg['training_registration_binding']),
         'Original training phase identity differs')
    launch = bound(reg['training_launch_binding'])
    need(equivalent_ref(launch['input'], reg['training_registration_binding']), 'Training launcher identity differs')
    review = bound(reg['source_review_binding'])
    need(review['passed'] is True, 'Independent resource source review missing')
    guard_tests = verify_test_evidence(guard_reg['synthetic_test_evidence_binding'], 26)
    audit_tests = verify_test_evidence(reg['resource_supplement_tests_evidence_binding'], 41)
    tested = {local(ref['path']) for evidence in (guard_tests, audit_tests) for ref in evidence['sources_before']}
    need({local(reg[name]['path']) for name in ('resource_audit_source_binding',
          'resource_publication_gate_source_binding')} <= tested, 'Resource audit/publication source was not tested')
    return reg


def audit(registration_sha):
    reg = admit(registration_sha)
    training_path = verify(reg['training_execution_receipt_binding'])
    need(training_path == TRIAL / 'training_execution_receipt.json', 'Original terminal training receipt was replaced')
    training = checked_receipt(training_path, 'training')
    training_reg = bound(reg['training_registration_binding'])
    need(equivalent_ref(training['input'], reg['training_registration_binding'])
         and equivalent_ref(training['capture_source'], training_reg['capture_source_binding'])
         and local(training['entry_before']['path']) == ROOT / 'scripts/tcn_existing_data_ed6_train.py'
         and training['entry_before']['sha256'] == '729e4db69898a11abf1ba114cf5a443fca42ccee6200e6b924151d47c5ea475c',
         'Original training terminal source/input differs')
    guard_path = verify(reg['resource_guard_receipt_binding'])
    need(guard_path == RES / 'resource_guard_execution_receipt.json', 'Guard terminal receipt was replaced')
    guard = checked_receipt(guard_path, 'resource_guard', expected_exit=2)
    need(guard['sources_before'] == guard['sources_after'] and guard['sources_unchanged'] is True,
         'Guard actual dependency chain missing or changed')
    guard_reg = bound(reg['guard_registration_binding'])
    need(equivalent_ref(guard['input'], reg['guard_registration_binding'])
         and equivalent_ref(guard['entry_before'], guard_reg['guard_source_binding'])
         and equivalent_ref(guard['capture_source'], guard_reg['capture_source_binding']), 'Actual guard source/input differs')
    need(equivalent_ref(guard['outputs']['guard_result'], reg['resource_guard_result_binding'])
         and equivalent_ref(guard['outputs']['guard_events'], reg['resource_guard_events_binding'])
         and local(reg['resource_guard_result_binding']['path']) == RES / 'guard_result.json'
         and local(reg['resource_guard_events_binding']['path']) == RES / 'guard_events.jsonl',
         'Actual guard output identity differs')
    result = bound(reg['resource_guard_result_binding']); events = read_events(verify(reg['resource_guard_events_binding']))
    need(equivalent_ref(result['source_binding'], guard_reg['guard_source_binding'])
         and equivalent_ref(result['registration_binding'], reg['guard_registration_binding'])
         and equivalent_ref(result['events_binding'], reg['resource_guard_events_binding']),
         'Guard result provenance differs')
    diagnosis_path = verify(reg['target_access_diagnosis_binding'])
    need(diagnosis_path == RES / 'target_access_diagnosis.json', 'Original read-only diagnosis was replaced')
    diagnosis = read(diagnosis_path)
    failure_review = bound(reg['failed_guard_evidence_review_binding'])
    need(failure_review['passed'] is True and failure_review['observed_branch'] == 'failed_no_handle_no_request'
         and failure_review['guard_execution_passed'] is False
         and failure_review['temporary_DC_idle_request_effective'] is False
         and equivalent_ref(failure_review['actual_failed_guard'], reg['resource_guard_receipt_binding'])
         and equivalent_ref(failure_review['guard_result'], reg['resource_guard_result_binding'])
         and equivalent_ref(failure_review['events'], reg['resource_guard_events_binding'])
         and equivalent_ref(failure_review['root_readonly_access_probe'], reg['target_access_diagnosis_binding']),
         'Independent finite-failure evidence review differs')
    observation = bound(reg['root_tool_observation_binding'])
    need(observation['version'] == 'TCN-ED6-root-tool-observation-1.0'
         and observation['recorded_after_execution'] is True and observation['not_an_original_capture_receipt'] is True
         and observation['actual_exit_code'] == 0 and observation['actual_tool_chunk_id'] == 'b9f05d'
         and observation['probe_pid_from_actual_JSON'] == diagnosis['probe_pid']
         and observation['probe_utc_from_actual_JSON'] == diagnosis['utc']
         and observation['no_new_guard_retry'] is True
         and observation['no_invented_capture_source_or_OS_parent_chain'] is True
         and equivalent_ref(observation['diagnosis_binding'], reg['target_access_diagnosis_binding']),
         'Posthoc root tool observation provenance differs')
    launch = bound(reg['training_launch_binding']); phase = bound(reg['training_phase_reservation_binding'])
    interval = (datetime.fromisoformat(launch['started_utc']).timestamp(),
                datetime.fromisoformat(phase['reserved_utc']).timestamp())
    need(interval[0] <= interval[1], 'Original target creation interval is inverted')
    need(result['creation_interval_unix'] == list(interval), 'Original training creation interval metadata differs')
    counts = failed_no_handle_no_request(training, guard, result, events, diagnosis)
    # The event file must be the events actually captured from this guard worker.
    lines = verify(guard['stdout']).read_text('utf-8').splitlines()
    need(len(lines) == len(events) + 1 and [json.loads(line) for line in lines[:-1]] == events,
         'Resource JSONL differs from actual guard stdout')
    summary = json.loads(lines[-1])
    need(summary['passed'] is False and summary['worker_pid'] == result['actual_worker_pid']
         and equivalent_ref(summary['output'], reg['resource_guard_result_binding']),
         'Actual guard stdout terminal summary differs')
    reservation = read(RES / 'guard_phase_reservation.json')
    need(reservation['target_worker_pid'] == TARGET_WORKER_PID
         and reservation['guard_worker_pid'] == result['actual_worker_pid']
         and reservation['permanent_single_use'] is True and reservation['limits'] == LIMITS
         and equivalent_ref(reservation['registration_binding'], reg['guard_registration_binding']),
         'One-use guard reservation differs')
    receipt = {'version': 'TCN-ED6-resource-supplement-audit-1.0', 'passed': True,
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'worker_pid': os.getpid(),
        'registration_binding': binding(REGISTRATION), 'guard_registration_binding': reg['guard_registration_binding'],
        'original_training_worker_terminal_verified': True, 'guard_terminal_verified': True,
        'resource_outcome': 'failed_no_handle_no_request', 'resource_support_failed': True,
        'true_safe_failure': True, 'guard_passed': False,
        'resource_cleanup_passed': False, 'resource_cleanup_not_required': True,
        'unreleased_resources': 0, 'target_worker_pid': TARGET_WORKER_PID,
        'guard_actual_worker_pid': result['actual_worker_pid'], 'event_observations': counts,
        'guard_receipt_binding': reg['resource_guard_receipt_binding'],
        'guard_result_binding': reg['resource_guard_result_binding'],
        'guard_events_binding': reg['resource_guard_events_binding'],
        'training_receipt_binding': reg['training_execution_receipt_binding'],
        'guard_phase_reservation_binding': binding(RES / 'guard_phase_reservation.json'),
        'target_access_diagnosis_binding': reg['target_access_diagnosis_binding'],
        'root_tool_observation_binding': reg['root_tool_observation_binding'],
        'failed_guard_evidence_review_binding': reg['failed_guard_evidence_review_binding'],
        'diagnosis_provenance': 'Root inline read-only probe, documented after execution; no contemporaneous capture source/receipt is claimed.',
        'source_binding': binding(Path(__file__)), 'model_or_optimizer_operations': 0, 'checkpoint_loads': 0,
        'training_or_budget_mutations': 0, 'market_records_or_predictions_read': False,
        'permanent_power_or_display_settings_changed': False,
        'interpretation': 'The temporary resource guard failed before acquiring a handle or making an execution-state request; no cleanup API was applicable. ML completion is verified independently from its original terminal receipt. No guard success, successful clear/CloseHandle, uninterrupted-awake, OS parent-chain, or model-quality claim.'}
    write_new(RES / 'resource_audit.json', receipt)
    print(json.dumps({'passed': True, 'worker_pid': os.getpid(), 'output': binding(RES / 'resource_audit.json')}, ensure_ascii=True), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args(argv)
    return audit(args.registration_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
