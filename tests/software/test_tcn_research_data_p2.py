"""P2 data mathematics/geometry/admission tests; only temporary D synthetic records."""
from shared.paths import resolve_path
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]/'software'

import numpy as np
from software.training import tcn_research_data_p2 as d


def fixture(folder, rows=251):
    sources = {}; ranges = []; reports = {}; role_counts = {'train': np.zeros(3, np.int64), 'tuning_validation': np.zeros(3, np.int64)}
    origin_sources = {}; origin_ranges = []
    for day in d.TRAIN + d.TUNE:
        a = np.zeros(rows, dtype=np.dtype(d.RECORD_FIELDS)); a['sequence'] = np.arange(rows) + 1
        a['flags'] = 64 if day in d.TUNE else 0
        for j in range(40):
            a['features'][:, j] = (2_800_000 + j * 100 + (np.arange(rows) // 17 % 3) * 10) if j % 2 == 0 else (j + np.arange(rows) % 11)
        p = folder / (day + '.bin'); p.write_bytes(a.tobytes())
        labels, oracle = d.label_pair(a, np.arange(99, rows - 50))
        labelp = folder / (day + '.u8'); labelp.write_bytes(labels.tobytes())
        c = np.bincount(labels, minlength=3).tolist(); role = d.ROLES[day]
        s = {'records_path': str(p), 'records_sha256': d.sha(p), 'rows': rows, 'role': role,
             'day': day, 'source_control_verified': day in d.TRAIN, 'coverage_unknown': day in d.TUNE}
        r = {'date': day, 'role': role, 'start_row': 0, 'stop_row': rows, 'endpoint_start': 99,
             'endpoint_stop_exclusive': rows - 50, 'window_count': rows - 149, 'record_count': rows,
             'records_path': str(p), 'records_sha256': d.sha(p), 'source_control_verified': day in d.TRAIN,
             'coverage_unknown': day in d.TUNE, 'protocol_sha256': d.PIN_PROTOCOL, 'research_segment_id': 1, 'label_offset': 0}
        sources[day] = s; ranges.append(r)
        reports[day] = {'date': day, 'role': role, 'window_count': len(labels), 'class_counts': c,
                        'label_path': str(labelp), 'label_sha256': d.sha(labelp), 'label_bytes': len(labels)}
        role_counts[role] += c
    counts = role_counts['train'].tolist()
    norm = {'mean': [0.0] * 40, 'std': [1.0] * 40, 'record_count': rows * 5, 'fit_dates': list(d.TRAIN)}
    identity = {day: s['records_sha256'] for day, s in sources.items()}
    m = {'version': 'TCN-research-B-P2-data-1.0', 'status': 'prepared_real_data_passed', 'horizon': 50, 'threshold': 0,
         'original_flag64_cleared': False, 'sealed_labels_or_scores_opened': False, 'sealed_records_opened': False,
         'sealed_dates': list(d.SEALED), 'source_control_verified_for_new_days': False, 'coverage_unknown_for_new_days': True,
         'sources': sources, 'ranges': ranges, 'day_label_reports': reports, 'normalization': norm,
         'source_records_unchanged': True, 'source_sha256_before': identity, 'source_sha256_after': identity.copy(),
         'sample_counts': {'train': 5 * (rows - 149), 'tuning_validation': 2 * (rows - 149)},
         'train_class_counts': counts, 'tuning_class_counts': role_counts['tuning_validation'].tolist(),
         'majority_class_from_train': max(range(3), key=lambda k: counts[k])}
    return m


class P2DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / d.STAGE).mkdir(exist_ok=True, parents=True)
        cls.temp = tempfile.TemporaryDirectory(prefix='synthetic_data_test_', dir=ROOT / d.STAGE)
        cls.folder = Path(cls.temp.name); cls.data = fixture(cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @unittest.skipUnless((d.ROOT/d.P1/'completion_acceptance.json').is_file(), 'External historical P1 admission receipts are required')
    def test_admission_real_metadata_no_source_read(self):
        with mock.patch.object(d, 'verify_real_sources', side_effect=AssertionError('No records allowed')):
            m = d.load_admitted_manifest()
        self.assertEqual(m['_p1_manifest_binding']['sha256'], d.PIN_MANIFEST)

    def test_outside_path_rejected_before_resolve(self):
        with mock.patch.object(Path, 'resolve', side_effect=AssertionError('Must reject before resolve')):
            with self.assertRaises(ValueError): d.bounded(ROOT.parent/'outside.txt', (ROOT,))

    def test_role_comparison_rejected(self):
        with self.assertRaises(ValueError): d.admitted_days('comparison_validation')

    def test_role_final_rejected(self):
        with self.assertRaises(ValueError): d.ResearchWindows(self.data, 'final_test')

    def test_fixture_prepared_contract(self):
        self.assertIs(d.validate_prepared(self.data), self.data)

    def test_unique_stats_matches_concatenated_population(self):
        raw = np.arange(2800, dtype=np.uint64).reshape(70, 40) + 1
        raw[:, 7] = 2
        norm, review = d.fit_unique_statistics(lambda: iter((raw[:21], raw[21:45], raw[45:])))
        physical = raw.astype(np.float64); physical[:, 0::2] /= 10000
        self.assertTrue(np.allclose(norm['mean'], physical.mean(axis=0)))
        expected = physical.std(axis=0); expected[7] = 1
        self.assertTrue(np.allclose(norm['std'], expected))
        self.assertEqual(norm['record_count'], 70); self.assertEqual(norm['zero_variance_columns'], [7])
        self.assertTrue(review['passed'])

    def test_physical_float64_preserved(self):
        x = np.array([[10000001, 9] * 20], np.uint64)
        got = d.physical(x)
        self.assertEqual(got.dtype, np.float64); self.assertEqual(got[0, 0], 1000.0001); self.assertEqual(got[0, 1], 9)
        self.assertEqual(x[0, 0], 10000001)

    def test_labels_all_three_exact_integer(self):
        a = np.zeros(153, dtype=np.dtype(d.RECORD_FIELDS)); a['features'][:, 0] = 100; a['features'][:, 2] = 100
        a['features'][50, 0] = 99; a['features'][52, 2] = 101
        y, ref = d.label_pair(a, [0, 1, 2]); self.assertEqual(y.tolist(), [0, 1, 2]); self.assertTrue(np.array_equal(y, ref))

    def test_label_endpoint_rejection(self):
        a = np.zeros(100, dtype=np.dtype(d.RECORD_FIELDS))
        with self.assertRaises(ValueError): d.label_pair(a, [50])

    def test_float64_label_domain_rejection(self):
        a = np.zeros(100, dtype=np.dtype(d.RECORD_FIELDS)); a['features'][:, [0, 2]] = 2**51
        with self.assertRaises(ValueError): d.label_pair(a, [0])

    def test_stats_empty_rejection(self):
        with self.assertRaises(ValueError): d.fit_unique_statistics(lambda: iter([]))

    def test_zero_std_one_for_constant_features(self):
        raw = np.ones((30, 40), np.uint64) * 13
        norm, _ = d.fit_unique_statistics(lambda: iter((raw[:11], raw[11:])))
        self.assertEqual(norm['std'], [1.0] * 40); self.assertEqual(norm['zero_variance_columns'], list(range(40)))

    def test_dataset_reference_all_boundaries_and_mixed_dates(self):
        ds = d.ResearchWindows(self.data, 'train')
        ids = [0, 1, 101, 102, 509, 40, 0]
        try:
            result = ds.batch_details(ids)
            for k, idx in enumerate(ids):
                group = idx // 102; r = ds.entries[group]; end = 99 + idx % 102
                a = np.memmap(r['records_path'], dtype=np.dtype(d.RECORD_FIELDS), mode='r')
                try:
                    raw = a['features'][end-99:end+1].astype(np.float64); raw[:, 0::2] /= 10000
                    self.assertTrue(np.array_equal(result['X'][k], raw.T.astype(np.float32)))
                    self.assertEqual(result['y'][k], d.label_pair(a, [end])[0][0])
                    self.assertEqual(result['endpoints'][k], end); self.assertEqual(result['date'][k], r['date'])
                finally: a._mmap.close()
        finally: ds.close()

    def test_iter_sequential_equals_gather_exact(self):
        ds = d.ResearchWindows(self.data, 'tuning_validation')
        try:
            got = list(d.iter_batches(self.data, 'tuning_validation', batch_size=17))
            x = np.concatenate([g['X'] for g in got]); y = np.concatenate([g['y'] for g in got])
            refx, refy = ds.batch(np.arange(len(ds)))
            self.assertTrue(np.array_equal(x, refx)); self.assertTrue(np.array_equal(y, refy))
            self.assertEqual(np.concatenate([g['global_indices'] for g in got]).tolist(), list(range(len(ds))))
        finally: ds.close()

    def test_iter_shuffle_complete_repeatable(self):
        first = list(d.iter_batches(self.data, 'train', batch_size=17, shuffle_seed=43))
        second = list(d.iter_batches(self.data, 'train', batch_size=17, shuffle_seed=43))
        ids = np.concatenate([g['global_indices'] for g in first])
        self.assertEqual(sorted(ids.tolist()), list(range(510)))
        self.assertTrue(np.array_equal(ids, np.concatenate([g['global_indices'] for g in second])))

    def test_iter_tail_with_nontrivial_normalization_exact(self):
        norm = {'mean': (np.arange(40) * 0.13).tolist(), 'std': (0.3 + np.arange(40) * 0.07).tolist()}
        ds = d.ResearchWindows(self.data, 'tuning_validation', normalization=norm)
        try:
            batches = list(d.iter_batches(self.data, 'tuning_validation', norm, batch_size=19))
            self.assertEqual([len(g['y']) for g in batches], [19, 19, 19, 19, 19, 7] * 2)
            x = np.concatenate([g['X'] for g in batches]); y = np.concatenate([g['y'] for g in batches])
            refx, refy = ds.batch(np.arange(len(ds)))
            self.assertTrue(np.array_equal(x, refx)); self.assertTrue(np.array_equal(y, refy))
            r = ds.entries[0]; a = np.memmap(r['records_path'], dtype=np.dtype(d.RECORD_FIELDS), mode='r')
            try:
                raw = a['features'][0:100].astype(np.float64); raw[:, 0::2] /= 10000.
                reference = ((raw - np.asarray(norm['mean'])) / np.asarray(norm['std'])).T.astype(np.float32)
                self.assertTrue(np.array_equal(x[0], reference))
            finally: a._mmap.close()
        finally: ds.close()

    def test_shuffle_tail_no_drop(self):
        batches = list(d.iter_batches(self.data, 'train', batch_size=19, shuffle_seed=0))
        self.assertEqual(len(batches[-1]['y']), 16)
        ids = np.concatenate([g['global_indices'] for g in batches])
        self.assertEqual(sorted(ids.tolist()), list(range(510)))

    def test_minimal150_footprint_one_window_per_day(self):
        with tempfile.TemporaryDirectory(prefix='minimal_', dir=self.folder) as temp:
            data = fixture(Path(temp), rows=150)
            batches = list(d.iter_batches(data, 'train', batch_size=19))
            self.assertEqual([len(g['y']) for g in batches], [1] * 5)
            self.assertEqual([g['endpoints'].tolist() for g in batches], [[99]] * 5)

    def test_invalid_batch_empty(self):
        ds = d.ResearchWindows(self.data, 'train')
        try:
            with self.assertRaises(ValueError): ds.batch([])
        finally: ds.close()

    def test_invalid_batch_negative(self):
        ds = d.ResearchWindows(self.data, 'train')
        try:
            with self.assertRaises(ValueError): ds.batch([-1])
        finally: ds.close()

    def test_invalid_batch_float(self):
        ds = d.ResearchWindows(self.data, 'train')
        try:
            with self.assertRaises(ValueError): ds.batch([1.2])
        finally: ds.close()

    def test_invalid_batch_overflow(self):
        ds = d.ResearchWindows(self.data, 'train')
        try:
            with self.assertRaises(ValueError): ds.batch([510])
        finally: ds.close()

    def test_nan_norm_refused_before_records(self):
        m = copy.deepcopy(self.data); m['normalization']['mean'][0] = float('nan')
        with self.assertRaises(ValueError): d.validate_prepared(m)

    def test_zero_std_refused(self):
        m = copy.deepcopy(self.data); m['normalization']['std'][0] = 0
        with self.assertRaises(ValueError): d.validate_prepared(m)

    def test_tune_norm_fit_refused(self):
        m = copy.deepcopy(self.data); m['normalization']['fit_dates'].append('2025-12-15')
        with self.assertRaises(ValueError): d.validate_prepared(m)

    def test_window_footprint_refused(self):
        m = copy.deepcopy(self.data); m['ranges'][0]['endpoint_stop_exclusive'] += 1
        with self.assertRaises(ValueError): d.validate_prepared(m)

    def test_label_offset_refused(self):
        m = copy.deepcopy(self.data); m['ranges'][0]['label_offset'] = 1
        with self.assertRaises(ValueError): d.validate_prepared(m)

    def test_count_tamper_refused(self):
        m = copy.deepcopy(self.data); m['train_class_counts'][0] += 1
        with self.assertRaises(ValueError): d.validate_prepared(m)

    def test_majority_not_tune_refused(self):
        m = copy.deepcopy(self.data); m['majority_class_from_train'] = (m['majority_class_from_train'] + 1) % 3
        with self.assertRaises(ValueError): d.validate_prepared(m)

    def test_source_mutation_hash_refused(self):
        m = copy.deepcopy(self.data); m['sources'][d.TRAIN[0]]['records_sha256'] = '0' * 64
        with self.assertRaises(ValueError): d.verify_real_sources(m)

    def test_sealed_sources_refused(self):
        m = copy.deepcopy(self.data); m['sources']['2025-12-17'] = {}
        with self.assertRaises(ValueError): d.validate_prepared(m)

    def test_flag64_claim_refused(self):
        m = copy.deepcopy(self.data); m['original_flag64_cleared'] = True
        with self.assertRaises(ValueError): d.validate_prepared(m)

    def test_complete_synthetic_prepare_seven_files(self):
        origin = copy.deepcopy(self.data)
        origin.update({'unique_eligible_records_by_role': {'train': 1255},
                       '_p1_manifest_binding': {'path': 'synthetic', 'sha256': d.PIN_MANIFEST},
                       '_p1_completion_binding': {'path': 'synthetic', 'sha256': d.PIN_COMPLETION},
                       '_p1_audit_binding': {'path': 'synthetic', 'sha256': d.PIN_AUDIT}, 'protocol_sha256': d.PIN_PROTOCOL})
        out = self.folder / 'prepared_synthetic'
        with mock.patch.object(d, 'load_admitted_manifest', return_value=origin):
            prepared = d.prepare(outdir=out)
        self.assertTrue(prepared['integer_oracle_all_equal']); self.assertEqual(prepared['integer_oracle_checked_labels'], 714)
        self.assertEqual(prepared['normalization']['record_count'], 1255)
        self.assertFalse(prepared['sealed_records_opened']); self.assertEqual(prepared['source_sha256_before'], prepared['source_sha256_after'])
        self.assertEqual(set(prepared['sources']), set(d.TRAIN + d.TUNE)); d.validate_prepared(prepared)


if __name__ == '__main__':
    unittest.main(verbosity=2)
