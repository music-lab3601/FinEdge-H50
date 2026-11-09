"""Synthetic stdlib-only ED6 selection and permanent-budget boundary tests."""
from __future__ import annotations
import copy
from fractions import Fraction
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import tcn_existing_data_ed6_train as train


def ledger(slot='control43'):
    rows = [
        {'slot_id': 'feature42', 'reservation_id': 'parent_feature', 'seed': 42,
         'optimizer_updates': 290172, 'consumed_epochs': 6, 'status': 'completed'},
        {'slot_id': 'strategy42', 'reservation_id': 'parent_strategy', 'seed': 42,
         'optimizer_updates': 241810, 'consumed_epochs': 5, 'status': 'completed'}]
    if slot == 'candidate43':
        rows.append({'slot_id': 'control43', 'reservation_id': 'control', 'seed': 43,
                     'optimizer_updates': 193448, 'consumed_epochs': 4, 'status': 'completed'})
    rows.append({'slot_id': slot, 'reservation_id': 'synthetic', 'seed': 43,
                 'optimizer_updates': 0, 'consumed_epochs': 0, 'status': 'reserved'})
    return {'version': 'TCN-existing-data-permanent-budget-1.0', 'maximum_runs': 4,
            'maximum_epochs_per_run': 10, 'maximum_complete_epochs': 40,
            'maximum_optimizer_updates': 1934480, 'maximum_optimizer_updates_per_run': 483620,
            'reserved_runs': rows, 'optimizer_updates': sum(r['optimizer_updates'] for r in rows),
            'consumed_epochs': sum(r['consumed_epochs'] for r in rows)}


def row(epoch, matrix, improved, wait):
    return {'epoch': epoch, 'optimizer_updates': 48362, 'improved': improved, 'wait': wait,
            'development_validation': {'confusion_true_rows_predicted_columns': matrix}}


class RankTests(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(train.exact_rank([[3, 0, 0], [0, 3, 0], [0, 0, 3]]), (Fraction(1), Fraction(1)))

    def test_exact_without_float_rounding(self):
        n = 10**18
        a = [[n, 0, 0], [0, n, 0], [0, 0, n]]
        b = [[n - 1, 1, 0], [0, n, 0], [0, 0, n]]
        self.assertGreater(train.exact_rank(a), train.exact_rank(b))
        self.assertEqual(float(train.exact_rank(a)[0]), float(train.exact_rank(b)[0]))

    def test_no_flat_recall_gate(self):
        a = [[100, 0, 0], [100, 0, 0], [0, 0, 100]]
        b = [[50, 25, 25], [25, 50, 25], [25, 25, 50]]
        self.assertGreater(train.exact_rank(a), train.exact_rank(b))

    def test_f1_breaks_exact_accuracy_tie(self):
        a = [[0, 2, 0], [0, 2, 0], [0, 0, 2]]
        b = [[1, 1, 0], [1, 1, 0], [0, 0, 2]]
        self.assertEqual(train.exact_rank(a)[0], train.exact_rank(b)[0])
        self.assertNotEqual(train.exact_rank(a)[1], train.exact_rank(b)[1])

    def test_empty_class_zero_f1(self):
        self.assertEqual(train.exact_rank([[1, 0, 0], [0, 0, 0], [0, 0, 0]])[1], Fraction(1, 3))

    def test_malformed_counts_rejected(self):
        for m in ([], [[1]], [[True, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]):
            with self.assertRaises(ValueError): train.exact_rank(m)

    def test_all_zero_rejected(self):
        with self.assertRaises(ValueError): train.exact_rank([[0] * 3 for _ in range(3)])

    def test_tie_keeps_earlier(self):
        m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        selected, wait, _ = train.recompute_history([row(1, m, True, 0), row(2, m, False, 1)])
        self.assertEqual((selected, wait), (1, 1))

    def test_patience_three_stops(self):
        m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        h = [row(1, m, True, 0)] + [row(i, m, False, i - 1) for i in range(2, 5)]
        self.assertEqual(train.recompute_history(h)[:2], (1, 3))
        with self.assertRaises(ValueError): train.recompute_history(h + [row(5, m, False, 4)])

    def test_partial_epoch_never_selected(self):
        m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        r = row(1, m, True, 0); r['optimizer_updates'] = 48361
        with self.assertRaises(ValueError): train.recompute_history([r])

    def test_forged_improvement_rejected(self):
        m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        with self.assertRaises(ValueError): train.recompute_history([row(1, m, False, 0)])


class ConfigTests(unittest.TestCase):
    def setUp(self):
        train.configure('control43')

    def tearDown(self):
        train.configure('control43')

    def test_defaults(self):
        self.assertEqual(train.validate_config(copy.deepcopy(train.SETTINGS)), train.SETTINGS)

    def test_bias_rejected(self):
        c = copy.deepcopy(train.SETTINGS); c['bias'] = .1
        with self.assertRaises(ValueError): train.validate_config(c)

    def test_changed_strategy_rejected(self):
        for key, value in [('learning_rate', .0003), ('window_length', 200), ('weight_norm', False), ('seed', 42)]:
            c = copy.deepcopy(train.SETTINGS); c[key] = value
            with self.assertRaises(ValueError): train.validate_config(c)

    def test_bool_is_not_integer_setting(self):
        c = copy.deepcopy(train.SETTINGS); c['dropout'] = False
        with self.assertRaises(ValueError): train.validate_config(c)

    def test_pair_only_changes_feature_family_and_slot(self):
        a, b = train.settings_for('control43'), train.settings_for('candidate43')
        self.assertEqual({k for k in a if a[k] != b[k]}, {'slot_id', 'feature_family'})
        self.assertEqual(a['seed'], b['seed']); self.assertEqual(a['seed'], 43)
        self.assertEqual(a['learning_rate'], b['learning_rate']); self.assertEqual(a['learning_rate'], .001)
        train.validate_config(a, 'control43'); train.validate_config(b, 'candidate43')

    def test_unknown_slot_rejected(self):
        with self.assertRaises(ValueError): train.configure('candidate44')

    def test_candidate_cannot_silently_use_raw_control_features(self):
        c = train.settings_for('control43')
        with self.assertRaises(ValueError): train.validate_config(c, 'candidate43')


class BudgetTests(unittest.TestCase):
    def setUp(self):
        train.configure('control43')

    def tearDown(self):
        train.configure('control43')

    def test_valid_new_slot(self):
        self.assertEqual(train.validate_budget(ledger(), 'synthetic')['seed'], 43)

    def test_missing_wrong_reservation(self):
        with self.assertRaises(ValueError): train.validate_budget(ledger(), 'other')

    def test_fifth_slot_rejected(self):
        b = ledger(); b['reserved_runs'] *= 5
        with self.assertRaises(ValueError): train.validate_budget(b, 'synthetic')

    def test_count_desync_rejected(self):
        b = ledger(); b['optimizer_updates'] = 1
        with self.assertRaises(ValueError): train.validate_budget(b, 'synthetic')

    def test_cap_overrun_rejected(self):
        b = ledger(); b['reserved_runs'][0]['optimizer_updates'] = 483621; b['optimizer_updates'] = 483621
        with self.assertRaises(ValueError): train.validate_budget(b, 'synthetic')

    def test_precharge_persists_without_optimizer_success(self):
        with tempfile.TemporaryDirectory(dir=train.ROOT / 'artifacts') as td:
            p = Path(td); path = p / 'budget.json'
            with patch.object(train, 'STAGE', p), patch.object(train, 'BUDGET', path):
                train.atomic_json(path, ledger())
                writer = train.BudgetWriter('synthetic', train.binding(path))
                writer.change(debit=True)
                self.assertEqual(writer.attempts, 1); self.assertEqual(writer.completed, 0)
                self.assertEqual(train.read_json(path)['optimizer_updates'], 531983)
                writer.change(status='failed_no_automatic_retry')
                self.assertEqual(train.read_json(path)['optimizer_updates'], 531983)
                with self.assertRaises(ValueError): train.BudgetWriter('synthetic', train.binding(path))

    def test_external_ledger_change_stops(self):
        with tempfile.TemporaryDirectory(dir=train.ROOT / 'artifacts') as td:
            p = Path(td); path = p / 'budget.json'
            with patch.object(train, 'STAGE', p), patch.object(train, 'BUDGET', path):
                train.atomic_json(path, ledger()); w = train.BudgetWriter('synthetic', train.binding(path))
                value = train.read_json(path); value['extra'] = 'external'; train.atomic_json(path, value)
                with self.assertRaises(ValueError): w.change(debit=True)

    def test_epoch_charge_separate(self):
        with tempfile.TemporaryDirectory(dir=train.ROOT / 'artifacts') as td:
            p = Path(td); path = p / 'budget.json'
            with patch.object(train, 'STAGE', p), patch.object(train, 'BUDGET', path):
                train.atomic_json(path, ledger()); w = train.BudgetWriter('synthetic', train.binding(path))
                w.change(debit=True); w.completed += 1; w.change(complete_epoch=True)
                self.assertEqual((w.attempts, w.completed, w.epochs), (1, 1, 1))
                self.assertEqual(train.read_json(path)['consumed_epochs'], 12)

    def test_no_old_budget_write(self):
        with self.assertRaises(ValueError): train.atomic_json(train.ROOT / 'not_ED6.json', {})

    def test_completed_prefix_must_be_exact(self):
        b = ledger(); snapshot = copy.deepcopy(b)
        snapshot['reserved_runs'] = snapshot['reserved_runs'][:-1]
        self.assertEqual(train.validate_prefix_snapshot(b, snapshot), snapshot['reserved_runs'])
        b['reserved_runs'][0]['unrelated_parent_note'] = 'changed'
        with self.assertRaises(ValueError): train.validate_prefix_snapshot(b, snapshot)

    def test_candidate_requires_completed_control_prefix(self):
        train.configure('candidate43'); b = ledger('candidate43')
        self.assertEqual(train.validate_budget(b, 'synthetic')['slot_id'], 'candidate43')
        b['reserved_runs'][2]['status'] = 'running'
        with self.assertRaises(ValueError): train.validate_budget(b, 'synthetic')

    def test_no_skipping_control_reservation(self):
        train.configure('candidate43'); b = ledger('candidate43'); del b['reserved_runs'][2]
        with self.assertRaises(ValueError): train.validate_budget(b, 'synthetic')

    def test_current_update_preserves_whole_three_item_prefix(self):
        train.configure('candidate43')
        with tempfile.TemporaryDirectory(dir=train.ROOT / 'artifacts') as td:
            p = Path(td); path = p / 'budget.json'
            with patch.object(train, 'STAGE', p), patch.object(train, 'BUDGET', path):
                b = ledger('candidate43'); original = copy.deepcopy(b['reserved_runs'][:-1])
                train.atomic_json(path, b); writer = train.BudgetWriter('synthetic', train.binding(path))
                writer.change(debit=True)
                now = train.read_json(path)
                self.assertEqual(now['reserved_runs'][:-1], original)
                self.assertEqual(now['reserved_runs'][-1]['optimizer_updates'], 1)
                self.assertEqual(now['optimizer_updates'], 725431)

    def test_completed_prefix_counters_must_cover_complete_epochs(self):
        b = ledger(); b['reserved_runs'][0]['optimizer_updates'] -= 1; b['optimizer_updates'] -= 1
        with self.assertRaises(ValueError): train.validate_budget(b, 'synthetic')

    def test_import_is_ML_free(self):
        code = Path(train.__file__).read_text(encoding='utf-8')
        import ast
        tree = ast.parse(code)
        names = []
        for n in tree.body:
            if isinstance(n, ast.Import): names.extend(a.name for a in n.names)
            if isinstance(n, ast.ImportFrom): names.append(n.module)
        self.assertNotIn('torch', names); self.assertNotIn('numpy', names)


class OrderGateTests(unittest.TestCase):
    def setUp(self):
        train.configure('candidate43')
        base = train.ROOT_STAGE / 'trials/control43'
        def ref(path, letter): return {'path': str(path), 'sha256': letter * 64}
        self.lock = ref(train.ROOT_STAGE / 'selection_lock.json', 'a')
        identity = ref(base / 'training_registration.json', 'b')
        self.reg = {
            'control_training_receipt_binding': ref(base / 'training_execution_receipt.json', 'c'),
            'control_acceptance_binding': ref(base / 'run/training_acceptance.json', 'd'),
            'control_audit_receipt_binding': ref(base / 'audit_execution_receipt.json', 'e'),
            'control_audit_binding': ref(base / 'independent_audit.json', 'f'),
            'control_checkpoint_receipt_binding': ref(base / 'checkpoint_validation_execution_receipt.json', '1'),
            'control_checkpoint_validation_binding': ref(base / 'checkpoint_validation.json', '2')}
        cpreg = ref(base / 'checkpoint_validation_registration.json', '3')
        self.accepted = {'status': 'completed', 'slot_id': 'control43',
                         'identity': {'selection_lock_binding': self.lock, 'registration_binding': identity}}
        self.audit = {'passed': True, 'training_receipt_binding': self.reg['control_training_receipt_binding'],
                      'registration_binding': identity}
        self.cp = {'passed': True, 'actual_new_checkpoint_CPU_loads': 2, 'registration_binding': cpreg}
        self.cp_registration = {'training_receipt_binding': self.reg['control_training_receipt_binding'],
                                'training_registration_binding': identity}
        self.actuals = {
            str(self.reg['control_training_receipt_binding']['path']):
                {'outputs': {'training_acceptance': self.reg['control_acceptance_binding']}},
            str(self.reg['control_audit_receipt_binding']['path']):
                {'outputs': {'audit': self.reg['control_audit_binding']}},
            str(self.reg['control_checkpoint_receipt_binding']['path']):
                {'outputs': {'checkpoint_validation': self.reg['control_checkpoint_validation_binding']}}}
        self.payloads = {
            str(self.reg['control_acceptance_binding']['path']): self.accepted,
            str(self.reg['control_audit_binding']['path']): self.audit,
            str(self.reg['control_checkpoint_validation_binding']['path']): self.cp,
            str(cpreg['path']): self.cp_registration}

    def tearDown(self):
        train.configure('control43')

    def gate(self):
        with patch.object(train, 'successful_receipt', side_effect=lambda ref: self.actuals[str(ref['path'])]), \
             patch.object(train, 'read_binding', side_effect=lambda ref: self.payloads[str(ref['path'])]):
            return train.candidate_order_gate(self.reg, self.lock)

    def test_completed_original_control_gates_pass(self):
        self.assertEqual(self.gate(), self.accepted)

    def test_control_from_changed_family_lock_rejected(self):
        self.accepted['identity']['selection_lock_binding'] = {'path': self.lock['path'], 'sha256': '9' * 64}
        with self.assertRaises(ValueError): self.gate()

    def test_missing_bounded_cpu_success_rejected(self):
        self.cp['actual_new_checkpoint_CPU_loads'] = 0
        with self.assertRaises(ValueError): self.gate()

    def test_audit_for_another_registration_rejected(self):
        self.audit['registration_binding'] = {'path': self.audit['registration_binding']['path'], 'sha256': '8' * 64}
        with self.assertRaises(ValueError): self.gate()

    def test_candidate_cannot_precede_itself(self):
        self.accepted['slot_id'] = 'candidate43'
        with self.assertRaises(ValueError): self.gate()

    def test_cpu_inspection_of_another_training_receipt_rejected(self):
        self.cp_registration['training_receipt_binding'] = {
            'path': self.reg['control_training_receipt_binding']['path'], 'sha256': '7' * 64}
        with self.assertRaises(ValueError): self.gate()


if __name__ == '__main__':
    unittest.main()
