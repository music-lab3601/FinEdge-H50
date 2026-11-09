"""Synthetic stdlib-only ED4 selection and permanent-budget boundary tests."""
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

from software.training import tcn_existing_data_ed4_train as train


def ledger():
    return {'version': 'TCN-existing-data-permanent-budget-1.0', 'maximum_runs': 4,
            'maximum_epochs_per_run': 10, 'maximum_complete_epochs': 40,
            'maximum_optimizer_updates': 1934480, 'maximum_optimizer_updates_per_run': 483620,
            'reserved_runs': [{'slot_id': 'feature42', 'reservation_id': 'synthetic', 'seed': 42,
                               'optimizer_updates': 0, 'consumed_epochs': 0, 'status': 'reserved'}],
            'optimizer_updates': 0, 'consumed_epochs': 0}


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
    def test_defaults(self):
        self.assertEqual(train.validate_config(copy.deepcopy(train.SETTINGS)), train.SETTINGS)

    def test_bias_rejected(self):
        c = copy.deepcopy(train.SETTINGS); c['bias'] = .1
        with self.assertRaises(ValueError): train.validate_config(c)

    def test_changed_strategy_rejected(self):
        for key, value in [('learning_rate', .0003), ('window_length', 200), ('weight_norm', False), ('seed', 43)]:
            c = copy.deepcopy(train.SETTINGS); c[key] = value
            with self.assertRaises(ValueError): train.validate_config(c)

    def test_bool_is_not_integer_setting(self):
        c = copy.deepcopy(train.SETTINGS); c['dropout'] = False
        with self.assertRaises(ValueError): train.validate_config(c)


class BudgetTests(unittest.TestCase):
    def test_valid_new_slot(self):
        self.assertEqual(train.validate_budget(ledger(), 'synthetic')['seed'], 42)

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
                self.assertEqual(train.read_json(path)['optimizer_updates'], 1)
                writer.change(status='failed_no_automatic_retry')
                self.assertEqual(train.read_json(path)['optimizer_updates'], 1)
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
                self.assertEqual(train.read_json(path)['consumed_epochs'], 1)

    def test_no_old_budget_write(self):
        with self.assertRaises(ValueError): train.atomic_json(train.ROOT / 'not_ED4.json', {})

    def test_import_is_ML_free(self):
        code = Path(train.__file__).read_text(encoding='utf-8')
        import ast
        tree = ast.parse(code)
        names = []
        for n in tree.body:
            if isinstance(n, ast.Import): names.extend(a.name for a in n.names)
            if isinstance(n, ast.ImportFrom): names.append(n.module)
        self.assertNotIn('torch', names); self.assertNotIn('numpy', names)


if __name__ == '__main__':
    unittest.main()
