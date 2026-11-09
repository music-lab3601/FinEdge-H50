"""Pure stdlib decision/prefix boundaries; no files, model or real data are used."""
import copy
from fractions import Fraction
import unittest

import finalize_tcn_existing_data_ed6 as finalizer


C42 = [[1, 1, 0], [1, 1, 0], [1, 0, 1]]
K42 = [[1, 1, 0], [1, 1, 0], [0, 0, 2]]
C43 = [[1, 1, 0], [1, 1, 0], [0, 0, 2]]
LOWER_F1_SAME_ACCURACY = [[0, 2, 0], [0, 2, 0], [0, 0, 2]]


def snapshot(count):
    names = ['feature42', 'strategy42', 'control43', 'candidate43'][:count]
    rows = [{'slot_id': name, 'seed': 42 if i < 2 else 43, 'status': 'completed',
             'optimizer_updates': 48362 * (i + 1), 'consumed_epochs': i + 1,
             'unrelated_identity': {'preserve': [name]}} for i, name in enumerate(names)]
    return {'reserved_runs': rows, 'optimizer_updates': sum(r['optimizer_updates'] for r in rows),
            'consumed_epochs': sum(r['consumed_epochs'] for r in rows), 'status': 'completed_at_' + str(count)}


class AdoptionTests(unittest.TestCase):
    def test_accuracy_tie_with_lower_f1_still_passes(self):
        self.assertLess(finalizer.rank(LOWER_F1_SAME_ACCURACY)[1], finalizer.rank(C43)[1])
        stable, delta42, delta43 = finalizer.pair_rule(C42, K42, C43, LOWER_F1_SAME_ACCURACY)
        self.assertTrue(stable); self.assertGreater(delta42, 0); self.assertEqual(delta43, 0)

    def test_negative43_falls_back_despite_positive42(self):
        stable, delta42, delta43 = finalizer.pair_rule(C42, K42, C43, C42)
        self.assertFalse(stable); self.assertGreater(delta42, 0); self.assertLess(delta43, 0)

    def test_both_ties_are_not_an_improvement(self):
        self.assertFalse(finalizer.pair_rule(C42, C42, C43, C43)[0])

    def test_positive43_and_tied42_pass(self):
        self.assertTrue(finalizer.pair_rule(C42, C42, C42, K42)[0])

    def test_negative_count_rejected(self):
        with self.assertRaises(ValueError): finalizer.exact_accuracy([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_exact_negative_not_hidden_by_float_tie(self):
        n = 10**18
        control = [[n, 0, 0], [0, n, 0], [0, 0, n]]
        candidate = [[n - 1, 1, 0], [0, n, 0], [0, 0, n]]
        self.assertEqual(float(finalizer.exact_accuracy(control)), float(finalizer.exact_accuracy(candidate)))
        stable, _, delta43 = finalizer.pair_rule(C42, K42, control, candidate)
        self.assertFalse(stable); self.assertEqual(delta43, -Fraction(1, 3 * n))

    def test_empty_matrix_rejected(self):
        with self.assertRaises(ValueError): finalizer.exact_accuracy([[0] * 3 for _ in range(3)])


class FixedRepresentativeTests(unittest.TestCase):
    def fixture(self):
        parent = {'candidate_original_argmax_DEV_metrics': {'accuracy': .46},
                  'selected_checkpoint': {'path': 'candidate_seed42.pt'},
                  'normalization': {'path': 'candidate_norm.json'}}
        protocol = {'baseline': {'checkpoint': {'path': 'control_seed42.pt'}},
                    'data': {'baseline_normalization': {'path': 'control_norm.json'}}}
        return parent, protocol, {'accuracy': .45}

    def test_stable_representative_is_original_candidate_seed42(self):
        parent, protocol, baseline = self.fixture()
        chosen = finalizer.choose_representative(True, parent, protocol, baseline)
        self.assertEqual(chosen['seed'], 42); self.assertEqual(chosen['checkpoint']['path'], 'candidate_seed42.pt')
        self.assertIs(chosen['metrics'], parent['candidate_original_argmax_DEV_metrics'])

    def test_failed_pair_uses_original_wn42(self):
        parent, protocol, baseline = self.fixture()
        chosen = finalizer.choose_representative(False, parent, protocol, baseline)
        self.assertEqual(chosen['seed'], 42); self.assertEqual(chosen['checkpoint']['path'], 'control_seed42.pt')
        self.assertIs(chosen['metrics'], baseline)

    def test_extra_higher_seed43_artifact_cannot_be_chosen(self):
        parent, protocol, baseline = self.fixture()
        parent['highest_scoring_seed43_checkpoint'] = {'path': 'seed43_highest.pt'}
        parent['highest_scoring_seed43_accuracy'] = .999
        self.assertEqual(finalizer.choose_representative(True, parent, protocol, baseline)['checkpoint']['path'],
                         'candidate_seed42.pt')


class ImmutableBudgetTests(unittest.TestCase):
    def chain(self):
        a, b, d = snapshot(2), snapshot(3), snapshot(4)
        return a, b, copy.deepcopy(b), d, copy.deepcopy(d)

    def test_sequential_immutable_snapshots_pass_without_old_live_ref(self):
        chain = self.chain()
        # The function needs only immutable budget values and the final current
        # ledger; no stale control live-budget SHA or file is consulted.
        self.assertTrue(finalizer.validate_budget_chain(*chain))

    def test_modified_old_completed_prefix_rejected(self):
        a, b, c, d, current = self.chain()
        b['reserved_runs'][0]['unrelated_identity'] = {'preserve': ['modified']}
        with self.assertRaises(ValueError): finalizer.validate_budget_chain(a, b, c, d, current)

    def test_candidate_snapshot_must_match_control_terminal(self):
        a, b, c, d, current = self.chain(); c['status'] = 'reconstructed_with_different_identity'
        with self.assertRaises(ValueError): finalizer.validate_budget_chain(a, b, c, d, current)

    def test_current_must_match_candidate_terminal(self):
        a, b, c, d, current = self.chain(); current['reserved_runs'][-1]['optimizer_updates'] += 1
        with self.assertRaises(ValueError): finalizer.validate_budget_chain(a, b, c, d, current)

    def test_partial_or_failed_completed_epoch_rejected(self):
        a, b, c, d, current = self.chain()
        for value in (d, current): value['reserved_runs'][-1]['status'] = 'failed'
        with self.assertRaises(ValueError): finalizer.validate_budget_chain(a, b, c, d, current)

    def test_counter_sum_mismatch_rejected(self):
        a, b, c, d, current = self.chain()
        for value in (d, current): value['optimizer_updates'] += 1
        with self.assertRaises(ValueError): finalizer.validate_budget_chain(a, b, c, d, current)

    def test_fifth_reservation_rejected(self):
        a, b, c, d, current = self.chain()
        for value in (d, current): value['reserved_runs'].append(copy.deepcopy(value['reserved_runs'][-1]))
        with self.assertRaises(ValueError): finalizer.validate_budget_chain(a, b, c, d, current)


if __name__ == '__main__':
    unittest.main()
