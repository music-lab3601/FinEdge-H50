"""Small stdlib adversarial checks for the independent P2 artifact auditor."""
from shared.paths import resolve_path
import importlib.util
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]/'software'
spec = importlib.util.spec_from_file_location('audit',ROOT/'training/audit_tcn_research_p2.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class Tests(unittest.TestCase):
    def test_e_refused_before_resolve(self):
        with patch.object(Path,'resolve',side_effect=AssertionError('must not resolve outside')):
            with self.assertRaises(ValueError): audit.safe(ROOT.parent/'outside_model.pt')

    def test_lexical_escape_refused_before_resolve(self):
        with patch.object(Path,'resolve',side_effect=AssertionError('must not resolve outside')):
            with self.assertRaises(ValueError): audit.safe(ROOT/'../outside.json')

    def test_sealed_label_stream_refused(self):
        with self.assertRaises(ValueError): audit.safe(audit.STAGE/'data_test/2025-12-17/labels.u8',binary=True)

    def test_original_market_record_binary_refused(self):
        with self.assertRaises(ValueError): audit.safe(audit.STAGE/'data_test/2025-12-15/records.bin',binary=True)

    def test_nonbinary_source_file_allowlist(self):
        self.assertEqual(audit.safe(ROOT/'training/audit_tcn_research_p2.py'),(ROOT/'training/audit_tcn_research_p2.py').resolve())

    def test_source_binding_checks_sha_without_json_parsing(self):
        p = ROOT/'training/tcn_research_data_p2.py'
        binding = {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        with patch.object(audit,'read',side_effect=AssertionError('Never parse source as JSON')):
            self.assertIsNone(audit.Audit().binding(binding,'source',metadata=False))

    def test_noninteger_confusion_refused(self):
        for v in (True,-1,.5):
            with self.assertRaises(ValueError): audit.metrics([[v,0,0],[0,1,0],[0,0,1]])

    def test_fraction_quality_pass(self):
        q=audit.exact_quality([[2,1,0],[0,3,1],[1,0,2]],0)
        self.assertTrue(q['eligible_on_tuning_only'])

    def test_equality_and_zero_recall_fail(self):
        q=audit.exact_quality([[2,0,0],[3,0,0],[4,0,0]],0)
        self.assertFalse(q['accuracy_strictly_above_own_TRAIN_majority'])
        self.assertFalse(q['macro_f1_strictly_above_own_TRAIN_majority'])
        self.assertFalse(q['each_class_recall_at_least_point_two'])

    def test_zero_denominator_metrics(self):
        m=audit.metrics([[0,0,0],[0,2,0],[0,0,0]])
        self.assertEqual(m['macro_f1'],1/3)
        self.assertEqual(m['recall'],[0,1,0])

    def test_duplicate_and_nan_json_refused(self):
        with tempfile.TemporaryDirectory(dir=None,prefix='p2_audit_test_') as d:
            p=Path(d)/'test.json'
            for body in ('{"a":1,"a":2}','{"a":NaN}','{"a":1e999}'):
                p.write_text(body,encoding='utf-8')
                with self.assertRaises(ValueError): audit.read(p)


if __name__=='__main__': unittest.main()
