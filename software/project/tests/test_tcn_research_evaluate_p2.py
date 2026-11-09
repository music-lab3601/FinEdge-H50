import unittest
import sys
from pathlib import Path
from fractions import Fraction
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import tcn_research_evaluate_p2 as e

class MetricsTests(unittest.TestCase):
    def test_perfect(self):
        r=e.metrics([[10,0,0],[0,20,0],[0,0,30]])
        self.assertEqual(r['accuracy'],1); self.assertEqual(r['macro_f1'],1)
    def test_unequal_support(self):
        r=e.metrics([[9,1,0],[5,5,0],[1,2,7]])
        self.assertEqual(r['support'],[10,10,10]); self.assertEqual(r['accuracy'],.7)
        self.assertAlmostEqual(r['macro_f1'],(18/25+10/18+14/17)/3)
    def test_constant_majority(self):
        r=e.majority_reference([3,6,9],0)
        self.assertEqual(r['accuracy'],1/6); self.assertAlmostEqual(r['macro_f1'],2/21)
    def test_train_majority_does_not_follow_tune(self):
        r=e.majority_reference([30,15,45],0)
        self.assertEqual(r['confusion_true_rows_predicted_columns'],[[30,0,0],[15,0,0],[45,0,0]])
    def test_valid_gate(self):
        self.assertTrue(e.quality([[9,1,0],[0,8,2],[1,1,8]],0)['eligible_on_tuning_only'])
    def test_strict_tie_rejected(self):
        self.assertFalse(e.quality([[10,0,0],[10,0,0],[10,0,0]],0)['eligible_on_tuning_only'])
    def test_flat_recall_gate(self):
        r=e.quality([[80,0,0],[10,0,0],[0,0,80]],1)
        self.assertTrue(r['accuracy_strictly_above_own_TRAIN_majority'])
        self.assertFalse(r['each_class_recall_at_least_point_two'])
    def test_exact_recall_boundary(self):
        self.assertTrue(e.quality([[1,2,2],[2,1,2],[2,2,1]],0)['each_class_recall_at_least_point_two'])
        self.assertFalse(e.quality([[1,2,3],[2,1,3],[2,3,1]],0)['each_class_recall_at_least_point_two'])
    def test_missing_class(self):
        r=e.quality([[4,0,0],[0,0,0],[0,0,4]],0)
        self.assertFalse(r['eligible_on_tuning_only'])
    def test_invalid_matrices(self):
        for matrix in [[],[[0]*3]*3,[[1]*2]*3,[[True,0,0],[0,1,0],[0,0,1]],[[1.,0,0],[0,1,0],[0,0,1]],[[-1,0,0],[0,1,0],[0,0,1]]]:
            with self.subTest(matrix=matrix),self.assertRaises(ValueError):e.metrics(matrix)
    def test_invalid_majority(self):
        for value in [True,-1,3,None]:
            with self.subTest(value=value),self.assertRaises(ValueError):e.majority_reference([1,1,1],value)
    def test_external_output_denied(self):
        with self.assertRaises(ValueError): e.checked_output('@EXTERNAL_ROOT@/blocked_output')
    def test_relative_output_denied(self):
        with self.assertRaises(ValueError): e.checked_output('artifacts/bad')
    def test_canonical_nonfinite(self):
        with self.assertRaises(ValueError): e.canonical({'value':float('nan')})
    def test_external_data_denied_before_model_runtime(self):
        with mock.patch.object(e,'baseline_metadata') as old,mock.patch.object(e,'runtime') as runtime:
            with self.assertRaises(ValueError): e.preflight('@EXTERNAL_ROOT@/manifest.json',str(e.STAGE/'unused'),'0'*64)
            old.assert_not_called(); runtime.assert_not_called()
    def test_evaluate_external_data_denied_before_model_runtime(self):
        with mock.patch.object(e,'baseline_metadata') as old,mock.patch.object(e,'runtime') as runtime:
            with self.assertRaises(ValueError): e.evaluate('@EXTERNAL_ROOT@/manifest.json',str(e.STAGE/'unused'),'0'*64)
            old.assert_not_called(); runtime.assert_not_called()

if __name__=='__main__':unittest.main()
