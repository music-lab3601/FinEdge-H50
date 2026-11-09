"""Pure synthetic metadata/policy gates; no Windows API, data or model."""
import copy
import unittest
from unittest.mock import patch

import audit_tcn_existing_data_ed6_resource_supplement as audit
import run_tcn_existing_data_ed6_resource_publication_gate as gate


def fixtures():
    created = {'filetime_100ns': (11644473600 + 10) * 10**7, 'unix_seconds': 10.0}
    result = {'passed': True, 'target_worker_pid': 20296, 'actual_worker_pid': 1234,
              'worker_pid': 1234, 'guard_worker_pid': 1234,
              'target_process_handle_signaled': True, 'last_wait_result': 0,
              'stop_reason': 'original_target_process_signaled', 'error': None,
              'model_operations': 0, 'checkpoint_loads': 0, 'training_or_budget_mutations': 0,
              'limits': copy.deepcopy(audit.LIMITS), 'display_request_used': False,
              'permanent_power_or_display_settings_changed': False, 'seconds': 1.0,
              'target_creation': created, 'creation_interval_unix': [9.0, 11.0],
              'opened_process_handle': True, 'clear_actual_succeeded': True,
              'CloseHandle_actual_succeeded': True, 'clear_error': None, 'close_error': None,
              'final_clear_returned_previous': 0x80000000}
    training = {'phase': 'training', 'slot_id': 'candidate43', 'actual_worker_pid': 20296, 'exit_code': 0}
    actual = {'phase': 'resource_guard', 'exit_code': 0, 'target_worker_pid': 20296,
              'actual_worker_pid': 1234, 'model_operations': 0, 'checkpoint_loads': 0,
              'training_or_budget_mutations': 0, 'permanent_power_or_display_settings_changed': False}
    events = [
        {'event': 'guard_started', 'utc': '2026-10-18T00:00:00+00:00',
         'target_worker_pid': 20296, 'target_handle_bound': True, 'guard_worker_pid': 1234,
         'target_creation': copy.deepcopy(created), 'creation_interval_unix': [9.0, 11.0]},
        {'event': 'execution_state_changed', 'utc': '2026-10-18T00:00:01+00:00',
         'power': {'read_succeeded': True, 'ACLineStatus': 0, 'BatteryLifePercent': 100},
         'battery_only_request': True, 'requested': 0x80000001, 'returned_previous': 0x80000000,
         'actual_API_call_succeeded': True},
        {'event': 'execution_state_changed', 'utc': '2026-10-18T00:00:02+00:00',
         'power': {'read_succeeded': True, 'ACLineStatus': 1, 'BatteryLifePercent': 100},
         'battery_only_request': False, 'requested': 0x80000000, 'returned_previous': 0x80000001,
         'actual_API_call_succeeded': True},
        {'event': 'guard_finally', 'utc': '2026-10-18T00:00:03+00:00',
         'target_worker_pid': 20296, 'opened_process_handle': True, 'requested': 0x80000000,
         'returned_previous': 0x80000000, 'clear_actual_succeeded': True,
         'CloseHandle_actual_succeeded': True, 'clear_error': None, 'close_error': None}]
    return training, actual, result, events


def failed_fixtures():
    training, actual, result, _ = fixtures()
    actual['exit_code'] = 2
    result.update(passed=False, stop_reason='guard_error', target_process_handle_signaled=False,
                  last_wait_result=None, target_creation=None, opened_process_handle=False,
                  final_clear_returned_previous=None, clear_actual_succeeded=False,
                  CloseHandle_actual_succeeded=False,
                  error={'type': 'ValueError', 'message': 'OpenProcess failed'})
    events = [{'event': 'guard_error', 'utc': '2026-10-18T00:00:00+00:00', 'error': copy.deepcopy(result['error'])},
              {'event': 'guard_finally', 'utc': '2026-10-18T00:00:01+00:00',
               'target_worker_pid': 20296, 'opened_process_handle': False, 'requested': 0x80000000,
               'returned_previous': None, 'clear_actual_succeeded': False,
               'CloseHandle_actual_succeeded': False, 'clear_error': None, 'close_error': None}]
    diagnosis = {'version': 'TCN-ED6-readonly-target-access-diagnosis-1.0',
                 'SetThreadExecutionState_calls': 0, 'model_operations': 0,
                 'training_or_budget_mutations': 0, 'new_guard_reservation': False,
                 'results': [{'target_pid': 20296, 'access_mask': mask, 'opened': False,
                              'WinError': 5, 'actual_closed_if_opened': None}
                             for mask in [1052672, 4096, 1048576]]}
    return training, actual, result, events, diagnosis


class ResourceGates(unittest.TestCase):
    def test_01_known_DC100_requests(self):
        self.assertTrue(audit.expected_request(0, 100))

    def test_02_known_DC21_requests(self):
        self.assertTrue(audit.expected_request(0, 21))

    def test_03_DC20_releases(self):
        self.assertFalse(audit.expected_request(0, 20))

    def test_04_DC0_releases(self):
        self.assertFalse(audit.expected_request(0, 0))

    def test_05_AC_releases(self):
        self.assertFalse(audit.expected_request(1, 100))

    def test_06_unknown_AC_releases(self):
        self.assertFalse(audit.expected_request(255, 100))

    def test_07_unknown_battery_releases(self):
        self.assertFalse(audit.expected_request(0, 255))

    def test_08_failed_power_read_releases(self):
        self.assertFalse(audit.expected_request(0, 100, False))

    def test_09_boolean_battery_rejected(self):
        self.assertFalse(audit.expected_request(0, True))

    def test_10_float_battery_rejected(self):
        self.assertFalse(audit.expected_request(0, 99.0))

    def test_11_both_actual_terminals_pass(self):
        t, a, r, _ = fixtures(); audit.successful_terminal(t, a, r)

    def test_12_training_nonzero_fails(self):
        t, a, r, _ = fixtures(); t['exit_code'] = 1
        with self.assertRaises(ValueError): audit.successful_terminal(t, a, r)

    def test_13_wrong_training_worker_fails(self):
        t, a, r, _ = fixtures(); t['actual_worker_pid'] = 9999
        with self.assertRaises(ValueError): audit.successful_terminal(t, a, r)

    def test_14_guard_nonzero_fails(self):
        t, a, r, _ = fixtures(); a['exit_code'] = 2
        with self.assertRaises(ValueError): audit.successful_terminal(t, a, r)

    def test_15_time_cap_without_target_exit_fails(self):
        t, a, r, _ = fixtures(); r['stop_reason'] = 'maximum_18h_reached_target_not_signaled'
        with self.assertRaises(ValueError): audit.successful_terminal(t, a, r)

    def test_16_guard_target_PID_conflation_fails(self):
        t, a, r, _ = fixtures(); a['actual_worker_pid'] = 20296
        r['actual_worker_pid'] = r['worker_pid'] = r['guard_worker_pid'] = 20296
        with self.assertRaises(ValueError): audit.successful_terminal(t, a, r)

    def test_17_additional_model_operation_fails(self):
        t, a, r, _ = fixtures(); r['model_operations'] = 1
        with self.assertRaises(ValueError): audit.successful_terminal(t, a, r)

    def test_18_nonfinite_duration_fails(self):
        t, a, r, _ = fixtures(); r['seconds'] = float('nan')
        with self.assertRaises(ValueError): audit.successful_terminal(t, a, r)

    def test_19_actual_request_release_cleanup_pass(self):
        _, _, r, e = fixtures(); counts = audit.validate_events(r, e, (9.0, 11.0))
        self.assertEqual(counts['battery_request_API_calls'], 1)
        self.assertEqual(counts['CloseHandle_API_calls'], 1)

    def test_20_AC_request_fails(self):
        _, _, r, e = fixtures(); e[1]['power']['ACLineStatus'] = 1
        with self.assertRaises(ValueError): audit.validate_events(r, e, (9.0, 11.0))

    def test_21_low_battery_request_fails(self):
        _, _, r, e = fixtures(); e[1]['power']['BatteryLifePercent'] = 20
        with self.assertRaises(ValueError): audit.validate_events(r, e, (9.0, 11.0))

    def test_22_unknown_power_request_fails(self):
        _, _, r, e = fixtures(); e[1]['power']['read_succeeded'] = False
        with self.assertRaises(ValueError): audit.validate_events(r, e, (9.0, 11.0))

    def test_23_repeated_state_without_transition_fails(self):
        _, _, r, e = fixtures(); e[2] = copy.deepcopy(e[1])
        with self.assertRaises(ValueError): audit.validate_events(r, e, (9.0, 11.0))

    def test_24_missing_finally_fails(self):
        _, _, r, e = fixtures()
        with self.assertRaises(ValueError): audit.validate_events(r, e[:-1], (9.0, 11.0))

    def test_25_close_failure_fails(self):
        _, _, r, e = fixtures(); e[-1]['CloseHandle_actual_succeeded'] = False
        with self.assertRaises(ValueError): audit.validate_events(r, e, (9.0, 11.0))

    def test_26_creation_outside_original_interval_fails(self):
        _, _, r, e = fixtures()
        with self.assertRaises(ValueError): audit.validate_events(r, e, (10.5, 11.0))

    def test_27_display_flag_fails(self):
        _, _, r, e = fixtures(); e[1]['requested'] |= 2
        with self.assertRaises(ValueError): audit.validate_events(r, e, (9.0, 11.0))

    def test_28_zero_actual_API_return_fails(self):
        _, _, r, e = fixtures(); e[1]['returned_previous'] = 0
        with self.assertRaises(ValueError): audit.validate_events(r, e, (9.0, 11.0))

    def test_29_publication_missing_actual_audit_never_delegates(self):
        with patch.object(gate, 'checked_receipt', side_effect=ValueError('No actual receipt')):
            with self.assertRaises(ValueError): gate.phase_command('publish', {}, 'x')

    def test_30_publication_failed_cleanup_fails(self):
        actual = {'input': {'sha256': 'x'}, 'entry_before': {}, 'capture_source': {},
                  'outputs': {'resource_audit': {}}}
        result = {'passed': True, 'true_safe_failure': True, 'resource_outcome': 'failed_no_handle_no_request',
                  'resource_support_failed': True, 'guard_passed': False,
                  'resource_cleanup_passed': False, 'resource_cleanup_not_required': False}
        with patch.object(gate, 'checked_receipt', return_value=actual), \
                patch.object(gate, 'equivalent_ref', return_value=True), \
                patch.object(gate, 'binding', return_value={}), patch.object(gate, 'bound', return_value=result):
            with self.assertRaises(ValueError):
                gate.publication_admission({'resource_audit_source_binding': {},
                                            'resource_publication_gate_source_binding': {}}, 'x')

    def test_31_original_primary_receipt_without_dependency_array_supported(self):
        actual = {'phase': 'training', 'exit_code': 0, 'entry_unchanged': True,
                  'entry_before': {}, 'entry_after': {}, 'capture_source': {},
                  'stdout': {}, 'stderr': {}, 'input': {}, 'outputs': {}}
        with patch.object(audit, 'read', return_value=actual), patch.object(audit, 'verify', return_value=None):
            self.assertIs(audit.checked_receipt('unused', 'training'), actual)

    def test_32_exact_failed_no_handle_branch_closes_only_resource_item(self):
        counts = audit.failed_no_handle_no_request(*failed_fixtures())
        self.assertEqual(counts['finally_release_API_calls'], 0)
        self.assertTrue(counts['cleanup_API_calls_not_applicable'])

    def test_33_acquired_handle_cannot_use_exception(self):
        values = failed_fixtures(); values[2]['opened_process_handle'] = True
        with self.assertRaises(ValueError): audit.failed_no_handle_no_request(*values)

    def test_34_any_state_event_cannot_use_exception(self):
        values = failed_fixtures(); values[3].insert(0, {'event': 'execution_state_changed'})
        with self.assertRaises(ValueError): audit.failed_no_handle_no_request(*values)

    def test_35_actual_clear_call_cannot_use_no_request_exception(self):
        values = failed_fixtures(); values[2]['final_clear_returned_previous'] = 0x80000000
        with self.assertRaises(ValueError): audit.failed_no_handle_no_request(*values)

    def test_36_different_probe_error_rejected(self):
        values = failed_fixtures(); values[4]['results'][0]['WinError'] = 87
        with self.assertRaises(ValueError): audit.failed_no_handle_no_request(*values)

    def test_37_probe_opened_handle_rejected(self):
        values = failed_fixtures(); values[4]['results'][0]['opened'] = True
        with self.assertRaises(ValueError): audit.failed_no_handle_no_request(*values)

    def test_38_probe_execution_state_call_rejected(self):
        values = failed_fixtures(); values[4]['SetThreadExecutionState_calls'] = 1
        with self.assertRaises(ValueError): audit.failed_no_handle_no_request(*values)

    def test_39_resource_failure_does_not_bypass_training_terminal(self):
        values = failed_fixtures(); values[0]['exit_code'] = None
        with self.assertRaises(ValueError): audit.failed_no_handle_no_request(*values)

    def test_40_wrong_guard_terminal_cannot_use_exception(self):
        values = failed_fixtures(); values[1]['exit_code'] = 0
        with self.assertRaises(ValueError): audit.failed_no_handle_no_request(*values)

    def test_41_publication_allows_only_audited_zero_resource_failure(self):
        actual = {'input': {'sha256': 'x'}, 'entry_before': {}, 'capture_source': {},
                  'outputs': {'resource_audit': {}}}
        result = {'passed': True, 'true_safe_failure': True, 'resource_outcome': 'failed_no_handle_no_request',
                  'resource_support_failed': True, 'guard_passed': False,
                  'resource_cleanup_passed': False, 'resource_cleanup_not_required': True,
                  'unreleased_resources': 0, 'original_training_worker_terminal_verified': True,
                  'guard_terminal_verified': True, 'registration_binding': {}}
        with patch.object(gate, 'checked_receipt', return_value=actual), \
                patch.object(gate, 'equivalent_ref', return_value=True), \
                patch.object(gate, 'binding', return_value={}), patch.object(gate, 'bound', return_value=result):
            self.assertIs(gate.publication_admission({'resource_audit_source_binding': {},
                'resource_publication_gate_source_binding': {}}, 'x')[1], result)


if __name__ == '__main__':
    unittest.main()
