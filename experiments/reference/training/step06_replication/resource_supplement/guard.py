"""One bounded battery-only idle request for the original ED6 candidate worker.

No model, checkpoint, data, budget, process termination, display request or
permanent power setting is accessed. All real Windows calls occur only in main.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as w
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[4]
DIRECTORY = Path(__file__).resolve().parent
STAGE = DIRECTORY.parent
TRIAL = STAGE / 'trials/candidate43'
TARGET = 20296
ES_CONTINUOUS, ES_SYSTEM_REQUIRED, WAIT_TIMEOUT = 0x80000000, 1, 258
LIMITS = {'maximum_wall_seconds': 64800, 'poll_interval_ms': 10000,
          'minimum_battery_percent_exclusive': 20, 'model_operations': 0,
          'checkpoint_loads': 0, 'training_or_budget_mutations': 0}


def need(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def local(value):
    path = Path(value); path = path if path.is_absolute() else ROOT / path
    need('..' not in path.parts and path.is_relative_to(ROOT), 'Resource metadata outside workspace')
    for ancestor in (path, *path.parents):
        if ancestor.exists():
            need(not ancestor.is_symlink() and not getattr(ancestor.lstat(), 'st_file_attributes', 0) & 0x400,
                 'Reparse resource metadata forbidden')
        if ancestor == ROOT:
            break
    need(path.resolve().is_relative_to(ROOT.resolve()), 'Resource metadata escaped workspace')
    return path


def binding(value):
    path = local(value); need(path.stat().st_size <= 16 * 1024**2, 'Oversized resource metadata')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


def checked(ref):
    path = local(ref['path']); actual = binding(path)
    need(actual['sha256'] == ref['sha256'] and actual['bytes'] == ref['bytes'], 'Resource binding changed')
    return path


def read(path):
    path = local(path); need(path.stat().st_size <= 16 * 1024**2, 'Oversized resource JSON')
    return json.loads(path.read_text('utf-8-sig'))


def put(path, value):
    path = local(path); need(path.is_relative_to(DIRECTORY), 'Resource writes restricted to supplement')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def wanted(power):
    return (power.get('read_succeeded') is True and power.get('ACLineStatus') == 0
            and type(power.get('BatteryLifePercent')) is int and 20 < power['BatteryLifePercent'] <= 100)


def validate_training_metadata(reservation, launch, training_ref):
    need(reservation['version'] == 'TCN-ED6-permanent-training-launch-1.0'
         and reservation['worker_pid'] == TARGET
         and reservation['reservation_id'] == 'TCN_EDO_20261017_ED6_candidate43'
         and reservation['registration_binding']['sha256'] == training_ref['sha256'], 'Original actual worker differs')
    need(launch['phase'] == 'training' and launch['requested_phase'] == 'training_candidate43'
         and launch['slot_id'] == 'candidate43' and launch['launcher_pid'] == 46972 and launch['capture_pid'] == 4252
         and launch['input']['sha256'] == training_ref['sha256']
         and launch['entry']['sha256'] == '729e4db69898a11abf1ba114cf5a443fca42ccee6200e6b924151d47c5ea475c',
         'Original launch roles/source differ; no parent-chain claim is inferred')
    lower = datetime.fromisoformat(launch['started_utc']).timestamp()
    upper = datetime.fromisoformat(reservation['reserved_utc']).timestamp()
    need(lower <= upper, 'Original worker creation interval inverted')
    return lower, upper


def admit(path, expected_sha):
    need(binding(path)['sha256'] == expected_sha, 'Explicit resource registration differs')
    reg = read(path)
    need(reg['version'] == 'TCN-ED6-DC-resource-registration-1.0' and reg['slot_id'] == 'candidate43'
         and reg['target_worker_pid'] == TARGET and reg['limits'] == LIMITS, 'Fixed resource scope differs')
    need(checked(reg['guard_source_binding']) == Path(__file__)
         and checked(reg['capture_source_binding']) == DIRECTORY / 'capture.py', 'Resource source identity differs')
    reservation_path = checked(reg['target_training_reservation_binding'])
    launch_path = checked(reg['target_training_launch_binding'])
    need(reservation_path == TRIAL / 'training_phase_reservation.json'
         and launch_path == TRIAL / 'training_launch.json'
         and checked(reg['target_training_registration_binding']) == TRIAL / 'training_registration.json',
         'Only original candidate43 training metadata may be used')
    interval = validate_training_metadata(read(reservation_path), read(launch_path), reg['target_training_registration_binding'])
    tests = read(checked(reg['synthetic_test_evidence_binding'])); review = read(checked(reg['source_review_binding']))
    need(tests['passed'] is True and tests['actual_exit_code'] == 0 and tests['no_real_data_or_ML'] is True
         and tests['no_real_Windows_API'] is True
         and review['passed'] is True, 'Actual pure-mock tests and independent source review required')
    need(not (TRIAL / 'training_execution_receipt.json').exists(), 'Original training already terminal; late guard forbidden')
    return reg, interval


class PowerStatus(ctypes.Structure):
    _fields_ = [('ACLineStatus', ctypes.c_ubyte), ('BatteryFlag', ctypes.c_ubyte),
                ('BatteryLifePercent', ctypes.c_ubyte), ('SystemStatusFlag', ctypes.c_ubyte),
                ('BatteryLifeTime', w.DWORD), ('BatteryFullLifeTime', w.DWORD)]


class CtypesKernel:
    def __init__(self):
        need(os.name == 'nt', 'Windows API guard only')
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {'OpenProcess': ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
                      'GetProcessTimes': ([w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4, w.BOOL),
                      'WaitForSingleObject': ([w.HANDLE, w.DWORD], w.DWORD),
                      'GetSystemPowerStatus': ([ctypes.POINTER(PowerStatus)], w.BOOL),
                      'SetThreadExecutionState': ([w.DWORD], w.DWORD), 'CloseHandle': ([w.HANDLE], w.BOOL)}
        for name, (args, result) in signatures.items():
            fn = getattr(self.api, name); fn.argtypes = args; fn.restype = result

    def open(self, pid):
        return self.api.OpenProcess(0x00100000 | 0x00001000, False, pid)

    def creation(self, handle):
        created, exited, kernel, user = (w.FILETIME() for _ in range(4))
        if not self.api.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)):
            raise ctypes.WinError(ctypes.get_last_error())
        ticks = (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
        return {'filetime_100ns': ticks, 'unix_seconds': ticks / 10**7 - 11644473600}

    def wait(self, handle, milliseconds):
        return int(self.api.WaitForSingleObject(handle, milliseconds))

    def power(self):
        value = PowerStatus(); ok = bool(self.api.GetSystemPowerStatus(ctypes.byref(value)))
        return {'read_succeeded': ok, 'ACLineStatus': int(value.ACLineStatus) if ok else None,
                'BatteryLifePercent': int(value.BatteryLifePercent) if ok else None}

    def state(self, value):
        return int(self.api.SetThreadExecutionState(value))

    def close(self, handle):
        return bool(self.api.CloseHandle(handle))


def run_guard(kernel, emit, interval, clock=time.monotonic, maximum_seconds=64800, poll_ms=10000):
    """Injectable OS boundary. Tests use mocks and never instantiate CtypesKernel."""
    need(maximum_seconds == LIMITS['maximum_wall_seconds'] and poll_ms == LIMITS['poll_interval_ms'], 'Guard limits changed')
    start = clock(); handle = None; last = None; signaled = False; last_wait = None; reason = None
    creation = None; error = None; clear = None; close = None; clear_error = None; close_error = None
    try:
        handle = kernel.open(TARGET); need(bool(handle), 'OpenProcess failed')
        creation = kernel.creation(handle)
        need(interval[0] <= creation['unix_seconds'] <= interval[1], 'Worker creation time outside original launch/reservation interval')
        emit({'event': 'guard_started', 'guard_worker_pid': os.getpid(), 'target_worker_pid': TARGET,
              'target_handle_bound': True, 'target_creation': creation, 'creation_interval_unix': list(interval)})
        while True:
            last_wait = kernel.wait(handle, 0)
            if last_wait == 0:
                signaled = True; reason = 'original_target_process_signaled'; break
            need(last_wait == WAIT_TIMEOUT, 'Unexpected WaitForSingleObject result')
            remaining = maximum_seconds - (clock() - start)
            if remaining <= 0:
                reason = 'maximum_18h_reached_target_not_signaled'; break
            power = kernel.power(); request = wanted(power)
            if request != last:
                flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if request else 0)
                returned = kernel.state(flags)
                need(returned != 0, 'SetThreadExecutionState failed')
                emit({'event': 'execution_state_changed', 'power': power, 'battery_only_request': request,
                      'requested': flags, 'returned_previous': returned, 'actual_API_call_succeeded': True})
                last = request
            last_wait = kernel.wait(handle, min(poll_ms, max(1, int(remaining * 1000))))
            if last_wait == 0:
                signaled = True; reason = 'original_target_process_signaled'; break
            need(last_wait == WAIT_TIMEOUT, 'Unexpected blocking WaitForSingleObject result')
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}; reason = 'guard_error'
        emit({'event': 'guard_error', 'error': error})
    finally:
        if handle:
            try:
                clear = kernel.state(ES_CONTINUOUS)
            except BaseException as exc:
                clear_error = {'type': type(exc).__name__, 'message': str(exc)}
            try:
                close = kernel.close(handle)
            except BaseException as exc:
                close_error = {'type': type(exc).__name__, 'message': str(exc)}
        emit({'event': 'guard_finally', 'target_worker_pid': TARGET, 'opened_process_handle': bool(handle),
              'requested': ES_CONTINUOUS, 'returned_previous': clear,
              'clear_actual_succeeded': bool(clear), 'CloseHandle_actual_succeeded': close is True,
              'clear_error': clear_error, 'close_error': close_error})
    passed = signaled and error is None and bool(clear) and close is True and clear_error is None and close_error is None
    return {'passed': passed, 'guard_worker_pid': os.getpid(), 'actual_worker_pid': os.getpid(), 'worker_pid': os.getpid(),
            'target_worker_pid': TARGET,
            'target_process_handle_signaled': signaled, 'last_wait_result': last_wait, 'stop_reason': reason,
            'target_creation': creation, 'creation_interval_unix': list(interval), 'error': error,
            'opened_process_handle': bool(handle), 'final_clear_returned_previous': clear,
            'clear_actual_succeeded': bool(clear), 'CloseHandle_actual_succeeded': close is True,
            'clear_error': clear_error, 'close_error': close_error, 'seconds': clock() - start,
            'limits': LIMITS, 'permanent_power_or_display_settings_changed': False, 'display_request_used': False,
            'model_operations': 0, 'checkpoint_loads': 0, 'training_or_budget_mutations': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args(argv)
    path = DIRECTORY / 'resource_supplement_registration.json'; reg, interval = admit(path, args.registration_sha256)
    need(not (DIRECTORY / 'guard_phase_reservation.json').exists() and not (DIRECTORY / 'guard_result.json').exists(),
         'Resource guard already permanently reserved; no retry')
    put(DIRECTORY / 'guard_phase_reservation.json', {'registration_binding': binding(path), 'guard_worker_pid': os.getpid(),
        'target_worker_pid': TARGET, 'started_utc': now(), 'permanent_single_use': True, 'limits': LIMITS})
    with (DIRECTORY / 'guard_events.jsonl').open('x', encoding='utf-8', newline='\n') as log:
        def emit(value):
            line = json.dumps({'utc': now(), **value}, ensure_ascii=True)
            log.write(line + '\n'); log.flush(); print(line, flush=True)
        result = run_guard(CtypesKernel(), emit, interval)
    result.update(version='TCN-ED6-DC-resource-guard-result-1.0', completed_utc=now(), registration_binding=binding(path),
                  source_binding=binding(__file__), events_binding=binding(DIRECTORY / 'guard_events.jsonl'))
    put(DIRECTORY / 'guard_result.json', result)
    print(json.dumps({'passed': result['passed'], 'worker_pid': os.getpid(), 'output': binding(DIRECTORY / 'guard_result.json')},
                     ensure_ascii=True), flush=True)
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
