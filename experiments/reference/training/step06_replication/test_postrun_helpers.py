"""Pure synthetic tests: no existing market stream, torch, model or optimizer."""
from shared.paths import resolve_path
import ast
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location('synthetic_' + name, HERE / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


audit = module('independent_audit')
cp = module('checkpoint_validation')


def metric(multiplier=1):
    ce = (3 * math.log1p(2 * math.exp(-2)) + math.log(3)) / 4
    return {'samples': 4 * multiplier, 'accuracy': .75, 'macro_f1': 7 / 9,
            'precision': [.5, 1., 1.], 'recall': [1., .5, 1.], 'f1': [2 / 3, 2 / 3, 1.],
            'support': [multiplier, 2 * multiplier, multiplier],
            'confusion_true_rows_predicted_columns': [[multiplier, 0, 0], [multiplier, multiplier, 0], [0, 0, multiplier]],
            'loss': ce}


class ArithmeticTests(unittest.TestCase):
    def test_known_metrics_and_class_imbalance(self):
        expected = metric()
        got, rank = audit.metrics(expected['confusion_true_rows_predicted_columns'], expected['loss'] * 4)
        audit.compare_metrics(expected, got, 'synthetic')
        self.assertEqual(rank, (Fraction(3, 4), Fraction(7, 9)))

    def test_empty_class(self):
        got, exact = audit.metrics([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(exact, (Fraction(1), Fraction(1, 3)))
        self.assertEqual(got['recall'], [1., 0., 0.])

    def test_integer_fraction_avoids_float_tie(self):
        n = 10**18
        _, a = audit.metrics([[n, 0, 0], [0, n, 0], [0, 0, n]])
        _, b = audit.metrics([[n - 1, 1, 0], [0, n, 0], [0, 0, n]])
        self.assertGreater(a, b)
        self.assertEqual(float(a[0]), float(b[0]))

    def test_malformed_or_empty_matrix_rejected(self):
        for matrix in ([[0] * 3] * 3, [[1]], [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]):
            with self.assertRaises(ValueError):
                audit.metrics(matrix)

    def test_no_recall_gate_in_either_helper(self):
        a = [[100, 0, 0], [100, 0, 0], [0, 0, 100]]
        b = [[50, 25, 25], [25, 50, 25], [25, 25, 50]]
        self.assertGreater(audit.metrics(a)[1], audit.metrics(b)[1])
        self.assertGreater(cp.rank(a), cp.rank(b))

    def test_checkpoint_replay_tie_and_patience(self):
        m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        h = [{'epoch': i, 'optimizer_updates': 48362, 'improved': i == 1, 'wait': i - 1,
              'development_validation': {'confusion_true_rows_predicted_columns': m}} for i in range(1, 5)]
        self.assertEqual(cp.replay(h)[:2], (1, 3))
        h.append({'epoch': 5, 'optimizer_updates': 48362, 'improved': False, 'wait': 4,
                  'development_validation': {'confusion_true_rows_predicted_columns': m}})
        with self.assertRaises(ValueError):
            cp.replay(h)

    def test_partial_checkpoint_rejected(self):
        item = {'epoch': 1, 'optimizer_updates': 48361, 'improved': True, 'wait': 0,
                'development_validation': {'confusion_true_rows_predicted_columns': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}
        with self.assertRaises(ValueError):
            cp.replay([item])

    def test_module_import_has_no_torch_or_training_source_import(self):
        for name in ('independent_audit', 'checkpoint_validation'):
            tree = ast.parse((HERE / (name + '.py')).read_text('utf-8'))
            imports = []
            for node in tree.body:
                if isinstance(node, ast.Import): imports.extend(a.name for a in node.names)
                if isinstance(node, ast.ImportFrom): imports.append(node.module)
            self.assertNotIn('torch', imports)
            self.assertFalse(any(x and x.startswith('tcn_') for x in imports))

    def test_outside_workspace_rejected_before_data_read(self):
        for f in (audit.path_in_workspace, cp.local):
            with self.assertRaises(ValueError): f(str(resolve_path('data://research/synthetic.bin')))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic_postrun_', dir=HERE)
        folder = Path(self.temp.name)
        self.old_trial = audit.ACTIVE_TRIAL
        audit.ACTIVE_TRIAL = folder
        self.data = {'ranges': [], 'day_label_reports': {}}
        self.baseline = {'per_day': {}}
        cache = {'version': 'TCN-ED6-epoch-DEV-cache-1.0', 'epoch': 1,
                 'decision': 'original_argmax', 'class_order': ['down', 'stationary', 'up'], 'per_day': {}}
        for day in audit.DEV:
            d = folder / day; d.mkdir()
            arrays = {'logits': (np.array([[2, 0, 0], [0, 0, 0], [0, 2, 0], [0, 0, 2]], dtype='<f4'), 'logits.f32', '<f4'),
                      'labels': (np.array([0, 1, 1, 2], dtype='u1'), 'labels.u8', 'u1'),
                      'anchors': (np.arange(104, 108, dtype='<u8'), 'anchors.u64', '<u8')}
            refs = {}
            for kind, (value, name, dtype) in arrays.items():
                p = d / name; value.tofile(p)
                refs[kind] = {**audit.identity(p), 'dtype': dtype, 'shape': list(value.shape)}
            cache['per_day'][day] = {'samples': 4, 'metrics': metric(), 'files': refs}
            self.data['ranges'].append({'date': day, 'endpoint_start': 104})
            self.data['day_label_reports'][day] = {'window_count': 4, 'label_sha256': refs['labels']['sha256']}
            self.baseline['per_day'][day] = {'files': {'anchors': refs['anchors']}}
        p = folder / 'cache_manifest.json'; p.write_text(json.dumps(cache), encoding='utf-8')
        self.cache = cache; self.manifest_path = p
        self.row = {'epoch': 1, 'cache_binding': audit.identity(p), 'development_validation': {
                    **metric(2), 'per_day': {day: metric() for day in audit.DEV}}}

    def tearDown(self):
        audit.ACTIVE_TRIAL = self.old_trial
        self.temp.cleanup()

    def test_full_independent_stream_arithmetic_and_tie(self):
        refs = []
        got, exact = audit.inspect_epoch(self.row, self.data, self.baseline, refs)
        audit.compare_metrics(metric(2), got, 'synthetic full stream')
        self.assertEqual(exact, (Fraction(3, 4), Fraction(7, 9)))
        self.assertEqual(len(refs), 6)

    def test_original_anchor_identity_required(self):
        self.baseline['per_day'][audit.DEV[0]]['files']['anchors'] = {'sha256': '0' * 64}
        with self.assertRaises(ValueError):
            audit.inspect_epoch(self.row, self.data, self.baseline, [])

    def test_payload_tamper_rejected(self):
        p = Path(self.cache['per_day'][audit.DEV[0]]['files']['labels']['path'])
        p.write_bytes(bytes([2, 1, 1, 2]))
        with self.assertRaises(ValueError):
            audit.inspect_epoch(self.row, self.data, self.baseline, [])

    def test_extra_candidate_payload_rejected(self):
        self.cache['per_day'][audit.DEV[0]]['files']['threshold'] = {}
        self.manifest_path.write_text(json.dumps(self.cache), encoding='utf-8')
        self.row['cache_binding'] = audit.identity(self.manifest_path)
        with self.assertRaises(ValueError):
            audit.inspect_epoch(self.row, self.data, self.baseline, [])


class BudgetTests(unittest.TestCase):
    def fixtures(self, slot='control43'):
        import copy
        rows = [{'slot_id': name, 'seed': 42 if i < 2 else 43, 'status': 'completed',
                 'reservation_id': 'synthetic_' + name, 'consumed_epochs': epochs,
                 'optimizer_updates': 48362 * epochs,
                 'actual_optimizer_steps_completed_last_snapshot': 48362 * epochs,
                 'every_historical_field': {'unchanged': True}}
                for i, (name, epochs) in enumerate([('feature42', 6), ('strategy42', 5), ('control43', 4), ('candidate43', 5)])]
        def pack(rs):
            return {'reserved_runs': copy.deepcopy(rs), 'consumed_epochs': sum(r['consumed_epochs'] for r in rs),
                    'optimizer_updates': sum(r['optimizer_updates'] for r in rs), 'maximum_runs': 4,
                    'maximum_epochs_per_run': 10, 'maximum_complete_epochs': 40,
                    'maximum_optimizer_updates': 1934480, 'maximum_optimizer_updates_per_run': 483620}
        count = 3 if slot == 'control43' else 4
        return pack(rows[:count - 1]), pack(rows[:count])

    def test_control_prefix_two_completed_rows(self):
        prior, budget = self.fixtures()
        got = audit.inspect_budget(budget, prior, 'synthetic_control43', 4, 'control43')
        self.assertEqual(got['global_optimizer_updates'], 725430)
        self.assertTrue(got['all_prior_records_unchanged'])

    def test_candidate_prefix_three_completed_rows(self):
        prior, budget = self.fixtures('candidate43')
        got = audit.inspect_budget(budget, prior, 'synthetic_candidate43', 5, 'candidate43')
        self.assertEqual(got['prior_runs'], 3)
        self.assertEqual(got['global_optimizer_updates'], 967240)

    def test_any_historical_field_change_rejected(self):
        prior, budget = self.fixtures()
        budget['reserved_runs'][0]['every_historical_field']['unchanged'] = False
        with self.assertRaises(ValueError): audit.inspect_budget(budget, prior, 'synthetic_control43', 4, 'control43')

    def test_wrong_reservation_order_rejected(self):
        prior, budget = self.fixtures()
        budget['reserved_runs'][0], budget['reserved_runs'][1] = budget['reserved_runs'][1], budget['reserved_runs'][0]
        with self.assertRaises(ValueError): audit.inspect_budget(budget, prior, 'synthetic_control43', 4, 'control43')

    def test_no_fifth_or_duplicate_run(self):
        prior, budget = self.fixtures('candidate43')
        budget['reserved_runs'].append(dict(budget['reserved_runs'][-1]))
        with self.assertRaises(ValueError): audit.inspect_budget(budget, prior, 'synthetic_candidate43', 5, 'candidate43')

    def test_global_counts_include_all_history(self):
        prior, budget = self.fixtures()
        budget['optimizer_updates'] = 193448
        with self.assertRaises(ValueError): audit.inspect_budget(budget, prior, 'synthetic_control43', 4, 'control43')

    def test_terminal_actual_steps_checked(self):
        prior, budget = self.fixtures()
        budget['reserved_runs'][-1]['actual_optimizer_steps_completed_last_snapshot'] -= 1
        with self.assertRaises(ValueError): audit.inspect_budget(budget, prior, 'synthetic_control43', 4, 'control43')

    def test_seed_or_caps_changed_rejected(self):
        for key in ('seed', 'maximum_runs'):
            prior, budget = self.fixtures()
            if key == 'seed': budget['reserved_runs'][-1]['seed'] = 42
            else: budget[key] += 1
            with self.assertRaises(ValueError): audit.inspect_budget(budget, prior, 'synthetic_control43', 4, 'control43')

    def test_noninteger_partial_excess_epochs_rejected(self):
        prior, budget = self.fixtures()
        for value in (0, True, 4., 11):
            with self.assertRaises(ValueError): audit.inspect_budget(budget, prior, 'synthetic_control43', value, 'control43')


class IdentityTests(unittest.TestCase):
    def fixture(self, slot='control43'):
        import copy
        reg = {'slot_id': slot, 'selection_lock_binding': {'sha256': 'lock'},
               'budget_prefix_snapshot_binding': {'sha256': 'prefix'}}
        a = copy.deepcopy(reg); a['normalization_binding'] = {'sha256': 'norm'}
        a['identity'] = copy.deepcopy(a); a['identity']['seed'] = 43
        a['paired_seed43_initialization_matches_control'] = True if slot == 'candidate43' else None
        return a, reg

    def test_control_none_and_candidate_true_producer_schema(self):
        for slot in audit.SLOTS:
            a, r = self.fixture(slot); audit.validate_trial_identity(a, r, slot)

    def test_slot_mismatch_rejected(self):
        a, r = self.fixture(); a['identity']['slot_id'] = 'candidate43'
        with self.assertRaises(ValueError): audit.validate_trial_identity(a, r, 'control43')

    def test_lock_or_prefix_mismatch_rejected(self):
        for key in ('selection_lock_binding', 'budget_prefix_snapshot_binding'):
            a, r = self.fixture(); a['identity'][key]['sha256'] = 'wrong'
            with self.assertRaises(ValueError): audit.validate_trial_identity(a, r, 'control43')

    def test_seed42_cannot_masquerade_as_seed43(self):
        a, r = self.fixture(); a['identity']['seed'] = 42
        with self.assertRaises(ValueError): audit.validate_trial_identity(a, r, 'control43')

    def test_terminal_immutable_snapshot_allows_stale_live_path(self):
        from unittest.mock import patch
        a, r = self.fixture(); a['budget_completion_snapshot_binding'] = {'path': 'immutable.json', 'sha256': 'terminal'}
        a['budget_binding'] = {'path': 'do_not_open_live.json', 'sha256': 'terminal'}
        refs = []
        def bound(ref):
            refs.append(ref['path'] if 'path' in ref else 'prefix'); return {'checked': True}
        with patch.object(audit, 'bound_json', side_effect=bound): audit.immutable_budget(a, r)
        self.assertEqual(refs, ['immutable.json', 'prefix'])

    def test_terminal_snapshot_sha_mismatch_rejected_without_read(self):
        a, r = self.fixture(); a['budget_completion_snapshot_binding'] = {'sha256': 'other'}
        a['budget_binding'] = {'sha256': 'terminal'}
        with self.assertRaises(ValueError): audit.immutable_budget(a, r)


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        scripts = HERE.parents[2] / 'scripts'
        sys.path.insert(0, str(scripts))
        spec = importlib.util.spec_from_file_location('synthetic_postcapture', scripts / 'run_tcn_existing_data_ed6_postrun_capture.py')
        cls.capture = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.capture)

    def test_checkpoint_canonical_filename(self):
        for slot in audit.SLOTS:
            root, path = self.capture.phase_paths('checkpoint_validation', slot)
            self.assertEqual(root.name, slot); self.assertEqual(path.name, 'checkpoint_execution_receipt.json')

    def test_audit_canonical_filename(self):
        root, path = self.capture.phase_paths('audit', 'control43')
        self.assertEqual(path.name, 'audit_execution_receipt.json')
        self.assertEqual(root.parent.name, 'trials')

    def test_global_phases_and_invalid_slots(self):
        self.assertEqual(self.capture.phase_paths('postrun_helpers_tests')[0], self.capture.STAGE)
        for phase, slot in [('audit', None), ('checkpoint_validation', 'feature42'), ('finalizer', 'control43'), ('unknown', None)]:
            with self.assertRaises(ValueError): self.capture.phase_paths(phase, slot)

    def test_primary_capture_unchanged_separate_import(self):
        text = Path(self.capture.__file__).read_text('utf-8')
        self.assertIn('from run_tcn_existing_data_ed6_capture import', text)
        self.assertIn("'--selection-lock-sha256'", text)
        self.assertIn("phase_root / 'readonly_audit.json'", text)


if __name__ == '__main__':
    unittest.main()
