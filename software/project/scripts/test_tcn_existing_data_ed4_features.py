"""Synthetic-only numerical and geometry checks for ED4 feature adaptation."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import tcn_existing_data_ed4_features as feature
from tcn_research_data_p2 import TRAIN, TUNE, ROLES, SEALED, RECORD_FIELDS


def raw_rows(count=155, shift=0):
    rows = np.empty((count, 40), dtype=np.uint64)
    for j in range(count):
        mid = 10000 + 4 * (j + shift)
        for level in range(10):
            rows[j, level * 4:level * 4 + 4] = [mid - 2 - level * 2, 10 + j + level,
                                               mid + 2 + level * 2, 20 + j + level]
    return rows


def fitted(rows):
    return feature.fit_unique_statistics(lambda: iter([rows[:71], rows[71:]]))[0]


def fixture(folder):
    """Seven tiny fake days; no existing market files or labels are opened."""
    count = 155
    rows = raw_rows(count)
    norm = fitted(rows)
    data = {'version': 'TCN-research-B-P2-data-1.0', 'status': 'prepared_real_data_passed',
            'horizon': 50, 'threshold': 0, 'original_flag64_cleared': False,
            'sealed_records_opened': False, 'sealed_labels_or_scores_opened': False,
            'sealed_dates': list(SEALED), 'source_control_verified_for_new_days': False,
            'coverage_unknown_for_new_days': True, 'source_records_unchanged': True,
            'normalization': {'fit_dates': list(TRAIN), 'record_count': count * 5,
                              'mean': [0.0] * 40, 'std': [1.0] * 40},
            'sources': {}, 'ranges': [], 'day_label_reports': {},
            'sample_counts': {'train': 30, 'tuning_validation': 12},
            'train_class_counts': [10, 10, 10], 'tuning_class_counts': [4, 4, 4],
            'majority_class_from_train': 0}
    cache_files = {}
    cached_ranges = []
    for day in TRAIN + TUNE:
        day_folder = folder / day
        day_folder.mkdir()
        record = np.zeros(165, dtype=np.dtype(RECORD_FIELDS))
        record['features'][5:160] = rows
        record['features'][:5] = rows[0]
        record['features'][160:] = rows[-1]
        record_path = day_folder / 'synthetic_records.bin'
        record.tofile(record_path)
        lab = day_folder / 'synthetic_labels.u8'
        np.asarray([0, 1, 2, 0, 1, 2], dtype='u1').tofile(lab)
        source_sha = feature.sha(record_path)
        data['sources'][day] = {'day': day, 'role': ROLES[day], 'rows': 165,
                                'records_path': str(record_path), 'records_sha256': source_sha}
        entry = {'date': day, 'role': ROLES[day], 'records_path': str(record_path),
                 'records_sha256': source_sha, 'start_row': 5, 'stop_row': 160,
                 'endpoint_start': 104, 'endpoint_stop_exclusive': 110,
                 'window_count': 6, 'label_offset': 0}
        data['ranges'].append(entry)
        data['day_label_reports'][day] = {'date': day, 'role': ROLES[day],
                 'window_count': 6, 'class_counts': [2, 2, 2], 'label_path': str(lab),
                 'label_bytes': 6, 'label_sha256': feature.sha(lab)}
        cache_path = day_folder / 'synthetic_normalized_rows.f32'
        values = feature.normalized_rows(rows, norm)
        values.tofile(cache_path)
        cache_files[day] = {**feature.binding(cache_path), 'dtype': '<f4',
                            'shape': [count, 40], 'role': ROLES[day]}
        cached_ranges.append({**entry, 'cache_offset': 0})
    source_hashes = {d: data['sources'][d]['records_sha256'] for d in TRAIN + TUNE}
    labels_hashes = {d: data['day_label_reports'][d]['label_sha256'] for d in TRAIN + TUNE}
    data['source_sha256_before'] = source_hashes
    data['source_sha256_after'] = source_hashes
    prepared = {'version': 'TCN-ED4-relative-row-cache-1.0', 'complete': True,
                'feature_id': feature.FEATURE_ID, 'normalization': norm,
                'sample_counts': data['sample_counts'], 'day_label_reports': data['day_label_reports'],
                'ranges': cached_ranges, 'cache_files': cache_files,
                'source_records_unchanged': True, 'source_sha256_before': source_hashes,
                'source_sha256_after': source_hashes, 'label_streams_unchanged': True,
                'labels_sha256_before': labels_hashes, 'labels_sha256_after': labels_hashes}
    return data, prepared


class TransformTests(unittest.TestCase):
    def test_best_bid_and_ask_reference(self):
        raw = raw_rows(3)
        x = feature.transform(raw)
        mid = (raw[:, 0].astype(np.float64) + raw[:, 2]) / 2
        np.testing.assert_array_equal(x[:, 0], -2 / mid)
        np.testing.assert_array_equal(x[:, 2], 2 / mid)

    def test_all_ten_levels(self):
        raw = raw_rows(7)
        x = feature.transform(raw)
        for i in range(len(raw)):
            mid = (int(raw[i, 0]) + int(raw[i, 2])) / 2
            for column in range(0, 40, 2):
                self.assertEqual(x[i, column], (int(raw[i, column]) - mid) / mid)

    def test_raw_quantities_unchanged(self):
        raw = raw_rows(10)
        np.testing.assert_array_equal(feature.transform(raw)[:, 1::2], raw[:, 1::2])

    def test_input_never_mutated(self):
        raw = raw_rows()
        original = raw.copy()
        feature.transform(raw)
        np.testing.assert_array_equal(raw, original)

    def test_batched_window_shape(self):
        raw = raw_rows(200).reshape(2, 100, 40)
        self.assertEqual(feature.transform(raw).shape, (2, 100, 40))

    def test_single_record(self):
        self.assertEqual(feature.transform(raw_rows(1)[0]).shape, (40,))

    def test_price_scale_invariance(self):
        raw = raw_rows(10)
        scaled = raw.copy()
        scaled[:, 0::2] *= 100
        np.testing.assert_array_equal(feature.transform(raw), feature.transform(scaled))

    def test_no_other_record_dependency(self):
        raw = raw_rows(10)
        before = feature.transform(raw)
        raw[9] = raw_rows(1, 1000)[0]
        np.testing.assert_array_equal(feature.transform(raw)[:9], before[:9])

    def test_large_raw_price_precision_retained(self):
        raw = raw_rows(1)
        raw[:, 0::2] += 10**12
        x = feature.transform(raw)
        self.assertNotEqual(x[0, 0], 0.0)
        self.assertEqual(x[0, 0], -2 / (10**12 + 10000))

    def test_invalid_shape_rejected(self):
        with self.assertRaises(ValueError):
            feature.transform(np.ones((3, 39)))

    def test_zero_price_rejected(self):
        raw = raw_rows(1)
        raw[0, 0] = 0
        with self.assertRaises(ValueError):
            feature.transform(raw)

    def test_nonfinite_rejected(self):
        raw = raw_rows(1).astype(np.float64)
        raw[0, 5] = np.inf
        with self.assertRaises(ValueError):
            feature.transform(raw)

    def test_complex_rejected(self):
        with self.assertRaises(ValueError):
            feature.transform(np.ones((2, 40), complex))

    def test_overprecision_price_rejected(self):
        raw = raw_rows(1)
        raw[0, 0] = 2**51
        with self.assertRaises(ValueError):
            feature.transform(raw)


class StatisticsTests(unittest.TestCase):
    def test_statistics_match_direct(self):
        rows = raw_rows()
        norm, review = feature.fit_unique_statistics(lambda: iter([rows[:17], rows[17:]]))
        x = feature.transform(rows)
        np.testing.assert_allclose(norm['mean'], x.mean(axis=0), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(norm['std'], x.std(axis=0), rtol=1e-10, atol=1e-12)
        self.assertTrue(review['passed'])

    def test_unique_row_count(self):
        rows = raw_rows(157)
        norm = fitted(rows)
        self.assertEqual(norm['record_count'], 157)
        self.assertEqual(norm['fit_dates'], list(TRAIN))
        self.assertFalse(norm['TUNE_fitted'])
        self.assertFalse(norm['sealed_fitted'])

    def test_constant_population_variance_std_one(self):
        raw = np.repeat(raw_rows(1), 8, axis=0)
        norm = fitted(raw)
        self.assertEqual(norm['zero_variance_columns'], list(range(40)))
        self.assertEqual(norm['std'], [1.0] * 40)

    def test_empty_train_rejected(self):
        with self.assertRaises(ValueError):
            feature.fit_unique_statistics(lambda: iter([]))

    def test_nonrepeatable_pass_count_rejected(self):
        calls = [0]
        def chunks():
            calls[0] += 1
            return iter([raw_rows(3 if calls[0] == 1 else 4)])
        with self.assertRaises(ValueError):
            feature.fit_unique_statistics(chunks)

    def test_frozen_norm_float32(self):
        raw = raw_rows()
        norm = fitted(raw)
        expected = ((feature.transform(raw) - np.array(norm['mean'])) / np.array(norm['std'])).astype('f4')
        actual = feature.normalized_rows(raw, norm)
        self.assertEqual(actual.dtype, np.dtype('<f4'))
        np.testing.assert_array_equal(actual, expected)

    def test_dev_norm_scope_rejected(self):
        norm = fitted(raw_rows())
        norm['fit_dates'].append('2025-12-15')
        with self.assertRaises(ValueError):
            feature.normalized_rows(raw_rows(1), norm)

    def test_nonpositive_norm_std_rejected(self):
        norm = fitted(raw_rows())
        norm['std'][0] = 0
        with self.assertRaises(ValueError):
            feature.validate_normalization(norm)


class WindowTests(unittest.TestCase):
    def setUp(self):
        feature.STAGE.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic_feature_test_', dir=feature.STAGE)
        self.data, self.prepared = fixture(Path(self.temp.name))
        self.datasets = []

    def tearDown(self):
        for ds in self.datasets:
            ds.close()
        self.temp.cleanup()

    def cached(self, role='train'):
        ds = feature.CachedRelativeWindows(self.data, self.prepared, role)
        self.datasets.append(ds)
        return ds

    def test_cached_vs_reference_byte_identical(self):
        actual = self.cached()
        ref = feature.RelativeResearchWindows(self.data, 'train', self.prepared['normalization'])
        self.datasets.append(ref)
        ids = np.array([0, 5, 6, 17, 29, 0, 8])
        a = actual.batch_details(ids)
        b = ref.batch_details(ids)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_no_E_records_mapped_by_cached_adapter(self):
        ds = self.cached()
        original = np.memmap
        touched = []
        def mapper(path, *args, **kwargs):
            touched.append(str(path))
            return original(path, *args, **kwargs)
        with mock.patch.object(np, 'memmap', side_effect=mapper):
            ds.batch(np.array([0, 29]))
        self.assertTrue(touched)
        self.assertTrue(all('synthetic_records.bin' not in p for p in touched))
        self.assertTrue(all('normalized_rows' in p or 'labels' in p for p in touched))

    def test_anchor_first_last_and_cross_day_order(self):
        ds = self.cached()
        item = ds.batch_details(np.array([0, 5, 6, 29]))
        np.testing.assert_array_equal(item['endpoints'], [104, 109, 104, 109])
        np.testing.assert_array_equal(item['date'], [TRAIN[0], TRAIN[0], TRAIN[1], TRAIN[-1]])
        self.assertEqual(item['X'].shape, (4, 40, 100))

    def test_labels_are_original_P2_order(self):
        ds = self.cached('tuning_validation')
        x, labels = ds.batch(np.arange(len(ds)))
        np.testing.assert_array_equal(labels, [0, 1, 2, 0, 1, 2] * 2)
        self.assertEqual(x.dtype, np.float32)

    def test_negative_or_oob_index_rejected(self):
        ds = self.cached()
        for ids in [np.array([-1]), np.array([len(ds)])]:
            with self.assertRaises(ValueError):
                ds.batch(ids)

    def test_float_and_empty_indices_rejected(self):
        ds = self.cached()
        for ids in [np.array([1.0]), np.array([], dtype=np.int64)]:
            with self.assertRaises(ValueError):
                ds.batch(ids)

    def test_sealed_role_rejected(self):
        with self.assertRaises(ValueError):
            self.cached('comparison_validation')

    def test_cache_offset_or_scope_tamper_rejected(self):
        for field in ['cache_offset', 'endpoint_start']:
            changed = copy.deepcopy(self.prepared)
            changed['ranges'][0][field] += 1
            with self.assertRaises(ValueError):
                feature.validate_cache(self.data, changed)

    def test_cache_size_tamper_rejected_before_mapping(self):
        ds = self.cached()
        with Path(self.prepared['cache_files'][TRAIN[0]]['path']).open('ab') as stream:
            stream.write(b'x')
        with self.assertRaises(ValueError):
            ds.batch(np.array([0]))

    def test_cache_path_outside_stage_rejected(self):
        changed = copy.deepcopy(self.prepared)
        changed['cache_files'][TRAIN[0]]['path'] = '@EXTERNAL_ROOT@/cache.f32'
        with self.assertRaises(ValueError):
            feature.validate_cache(self.data, changed)

    def test_batch_duplicate_indices_preserved(self):
        ds = self.cached()
        item = ds.batch_details(np.array([2, 2, 1]))
        np.testing.assert_array_equal(item['X'][0], item['X'][1])
        np.testing.assert_array_equal(item['y'], [2, 2, 1])

    def test_close_releases_all_mappings(self):
        ds = self.cached()
        ds.batch(np.array([0, 29]))
        self.assertTrue(ds.maps)
        ds.close()
        self.assertEqual(ds.maps, {})
        self.assertEqual(ds.label_maps, {})


if __name__ == '__main__':
    unittest.main()
