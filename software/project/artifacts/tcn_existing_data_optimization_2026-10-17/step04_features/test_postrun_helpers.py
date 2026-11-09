"""Pure synthetic tests: no existing market stream, torch, model or optimizer."""
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
            with self.assertRaises(ValueError): f('@EXTERNAL_ROOT@/synthetic.bin')


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic_postrun_', dir=HERE)
        folder = Path(self.temp.name)
        self.data = {'ranges': [], 'day_label_reports': {}}
        self.baseline = {'per_day': {}}
        cache = {'epoch': 1, 'decision': 'original_argmax', 'class_order': ['down', 'stationary', 'up'], 'per_day': {}}
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


if __name__ == '__main__':
    unittest.main()
