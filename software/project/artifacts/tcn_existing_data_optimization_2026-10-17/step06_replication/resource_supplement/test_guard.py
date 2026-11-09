"""Pure mock resource tests: no Windows API, real process, model or data."""
import ast
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('synthetic_ed6_guard', HERE / 'guard.py')
guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)


def power(ac=0, percent=100, ok=True):
    return {'read_succeeded': ok, 'ACLineStatus': ac, 'BatteryLifePercent': percent}


class Kernel:
    def __init__(self, powers=None):
        self.powers = list(powers or [power()]); self.index = 0; self.elapsed = 0.
        self.states = []; self.close_calls = []; self.open_pid = None
        self.creation_seconds = 15.; self.handle = 123; self.bad_wait = None
        self.state_fail_at = None; self.state_raise_at = None; self.close_value = True; self.close_raises = False

    def open(self, pid):
        self.open_pid = pid; return self.handle

    def creation(self, handle):
        return {'unix_seconds': self.creation_seconds, 'filetime_100ns': 1}

    def wait(self, handle, milliseconds):
        if self.bad_wait is not None: return self.bad_wait
        if milliseconds:
            self.elapsed += milliseconds / 1000.; self.index += 1
        return 0 if self.index >= len(self.powers) else guard.WAIT_TIMEOUT

    def power(self):
        return self.powers[self.index]

    def state(self, value):
        self.states.append(value)
        if len(self.states) == self.state_raise_at: raise OSError('synthetic state error')
        return 0 if len(self.states) == self.state_fail_at else guard.ES_CONTINUOUS

    def close(self, handle):
        self.close_calls.append(handle)
        if self.close_raises: raise OSError('synthetic close error')
        return self.close_value

    def clock(self):
        return self.elapsed


class PolicyTests(unittest.TestCase):
    def test_DC_known_above20_only(self):
        for percent in (21, 50, 100): self.assertTrue(guard.wanted(power(percent=percent)))

    def test_boundary20_and_lower_release(self):
        for percent in (0, 1, 19, 20): self.assertFalse(guard.wanted(power(percent=percent)))

    def test_AC_unknown_or_read_failure_release(self):
        for p in (power(ac=1), power(ac=255), power(percent=255), power(percent=None), power(ok=False), power(percent=True)):
            self.assertFalse(guard.wanted(p))

    def test_fixed_worker_and_resource_limits(self):
        self.assertEqual(guard.TARGET, 20296)
        self.assertEqual(guard.LIMITS['maximum_wall_seconds'], 18 * 3600)
        self.assertEqual(guard.LIMITS['model_operations'], 0)


class LifetimeTests(unittest.TestCase):
    def run_fake(self, kernel, **kwargs):
        events = []; result = guard.run_guard(kernel, events.append, (10., 20.), clock=kernel.clock, **kwargs)
        return result, events

    def test_DC_request_then_target_signaled_finally_clear_close(self):
        k = Kernel(); r, events = self.run_fake(k)
        self.assertTrue(r['passed']); self.assertTrue(r['target_process_handle_signaled'])
        self.assertEqual(r['last_wait_result'], 0); self.assertEqual(k.open_pid, 20296)
        self.assertEqual(k.states, [guard.ES_CONTINUOUS | 1, guard.ES_CONTINUOUS])
        self.assertEqual(k.close_calls, [123]); self.assertEqual(events[-1]['event'], 'guard_finally')

    def test_AC_return_clears_request_before_finally(self):
        k = Kernel([power(), power(ac=1)]); r, events = self.run_fake(k)
        self.assertTrue(r['passed']); self.assertEqual(k.states, [guard.ES_CONTINUOUS | 1, guard.ES_CONTINUOUS, guard.ES_CONTINUOUS])
        self.assertEqual([e['battery_only_request'] for e in events if e['event'] == 'execution_state_changed'], [True, False])

    def test_low_or_unknown_charge_releases_active_request(self):
        for p in (power(percent=20), power(percent=255), power(ok=False), power(ac=255)):
            k = Kernel([power(), p]); r, _ = self.run_fake(k)
            self.assertTrue(r['passed']); self.assertEqual(k.states[1], guard.ES_CONTINUOUS)

    def test_repeated_same_power_does_not_add_requests(self):
        k = Kernel([power(), power(), power()]); r, _ = self.run_fake(k)
        self.assertTrue(r['passed']); self.assertEqual(len(k.states), 2)

    def test_start_on_AC_no_system_required(self):
        k = Kernel([power(ac=1)]); r, _ = self.run_fake(k)
        self.assertTrue(r['passed']); self.assertEqual(k.states, [guard.ES_CONTINUOUS, guard.ES_CONTINUOUS])

    def test_PID_reuse_creation_outside_interval_rejected_and_closed(self):
        k = Kernel(); k.creation_seconds = 21.; r, _ = self.run_fake(k)
        self.assertFalse(r['passed']); self.assertEqual(k.states, [guard.ES_CONTINUOUS]); self.assertEqual(k.close_calls, [123])

    def test_creation_interval_inclusive_edges(self):
        for value in (10., 20.):
            k = Kernel(); k.creation_seconds = value; r, _ = self.run_fake(k); self.assertTrue(r['passed'])

    def test_invalid_wait_releases_and_closes(self):
        k = Kernel(); k.bad_wait = 0xffffffff; r, _ = self.run_fake(k)
        self.assertFalse(r['passed']); self.assertTrue(r['clear_actual_succeeded']); self.assertTrue(r['CloseHandle_actual_succeeded'])

    def test_open_failure_never_sets_request_or_closes_nonhandle(self):
        k = Kernel(); k.handle = 0; r, _ = self.run_fake(k)
        self.assertFalse(r['passed']); self.assertFalse(r['opened_process_handle']); self.assertEqual(k.states, []); self.assertEqual(k.close_calls, [])

    def test_state_zero_causes_error_and_final_cleanup(self):
        k = Kernel(); k.state_fail_at = 1; r, _ = self.run_fake(k)
        self.assertFalse(r['passed']); self.assertTrue(r['clear_actual_succeeded']); self.assertEqual(k.close_calls, [123])

    def test_state_exception_still_finally_clears_and_closes(self):
        k = Kernel(); k.state_raise_at = 1; r, _ = self.run_fake(k)
        self.assertFalse(r['passed']); self.assertTrue(r['clear_actual_succeeded']); self.assertEqual(k.close_calls, [123])

    def test_final_clear_failure_still_closes_handle(self):
        k = Kernel(); k.state_fail_at = 2; r, _ = self.run_fake(k)
        self.assertFalse(r['passed']); self.assertFalse(r['clear_actual_succeeded']); self.assertTrue(r['CloseHandle_actual_succeeded'])

    def test_final_clear_exception_still_closes_handle(self):
        k = Kernel(); k.state_raise_at = 2; r, _ = self.run_fake(k)
        self.assertFalse(r['passed']); self.assertIsNotNone(r['clear_error']); self.assertTrue(r['CloseHandle_actual_succeeded'])

    def test_close_false_or_exception_never_passes(self):
        for exception in (False, True):
            k = Kernel(); k.close_value = False; k.close_raises = exception; r, _ = self.run_fake(k)
            self.assertFalse(r['passed']); self.assertTrue(r['clear_actual_succeeded']); self.assertFalse(r['CloseHandle_actual_succeeded'])

    def test_18h_limit_is_not_target_exit_and_no_kill(self):
        k = Kernel(); calls = iter([0., 64801., 64801.])
        events = []; r = guard.run_guard(k, events.append, (10., 20.), clock=lambda: next(calls))
        self.assertFalse(r['passed']); self.assertFalse(r['target_process_handle_signaled'])
        self.assertEqual(r['stop_reason'], 'maximum_18h_reached_target_not_signaled')
        self.assertEqual(k.states, [guard.ES_CONTINUOUS]); self.assertEqual(k.close_calls, [123])

    def test_limits_cannot_be_increased(self):
        with self.assertRaises(ValueError): self.run_fake(Kernel(), maximum_seconds=64801)

    def test_no_display_or_model_flags(self):
        k = Kernel(); r, _ = self.run_fake(k)
        self.assertTrue(all(x in (guard.ES_CONTINUOUS, guard.ES_CONTINUOUS | 1) for x in k.states))
        for key in ('model_operations', 'checkpoint_loads', 'training_or_budget_mutations'): self.assertEqual(r[key], 0)
        self.assertFalse(r['display_request_used']); self.assertFalse(r['permanent_power_or_display_settings_changed'])


class AdmissionTests(unittest.TestCase):
    def fixture(self):
        source = {'sha256': '729e4db69898a11abf1ba114cf5a443fca42ccee6200e6b924151d47c5ea475c'}
        ref = {'sha256': 'synthetic_train'}
        reservation = {'version': 'TCN-ED6-permanent-training-launch-1.0', 'worker_pid': 20296,
                       'reservation_id': 'TCN_EDO_20261017_ED6_candidate43', 'registration_binding': ref,
                       'reserved_utc': '2026-10-18T06:41:31.586371+00:00'}
        launch = {'phase': 'training', 'requested_phase': 'training_candidate43', 'slot_id': 'candidate43',
                  'launcher_pid': 46972, 'capture_pid': 4252, 'input': ref, 'entry': source,
                  'started_utc': '2026-10-18T06:41:31.018262+00:00'}
        return reservation, launch, ref

    def test_original_worker_and_role_metadata(self):
        r, l, ref = self.fixture(); lower, upper = guard.validate_training_metadata(r, l, ref)
        self.assertLess(lower, upper); self.assertNotEqual(r['worker_pid'], l['launcher_pid'])

    def test_launcher_cannot_replace_actual_worker(self):
        r, l, ref = self.fixture(); r['worker_pid'] = l['launcher_pid']
        with self.assertRaises(ValueError): guard.validate_training_metadata(r, l, ref)

    def test_wrong_source_or_registration_rejected(self):
        for field in ('source', 'registration'):
            r, l, ref = self.fixture()
            if field == 'source': l['entry']['sha256'] = 'changed'
            else: ref = {'sha256': 'different'}
            with self.assertRaises(ValueError): guard.validate_training_metadata(r, l, ref)

    def test_workspace_escape_rejected(self):
        with self.assertRaises(ValueError): guard.local('@EXTERNAL_ROOT@/forbidden.bin')

    def test_import_performs_no_API_model_or_process_call(self):
        tree = ast.parse((HERE / 'guard.py').read_text('utf-8'))
        imports = [a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(set(imports) & {'torch', 'numpy', 'subprocess'})
        self.assertFalse(any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) for n in tree.body))


if __name__ == '__main__':
    unittest.main()
