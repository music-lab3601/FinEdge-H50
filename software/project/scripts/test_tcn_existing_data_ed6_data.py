"""Pure synthetic ED6 cache tests: no existing records/labels/cache or ML.

Temporary path and norm-pin patches only describe generated fake fixtures in
this isolated test process. No old source, real stage, or production file is
modified, and the formal entry never patches inherited module globals.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import tcn_existing_data_ed6_data as data6
import tcn_existing_data_ed4_features as relative
from tcn_research_data_p2 import ResearchWindows, TRAIN, TUNE, ROLES, SEALED, RECORD_FIELDS, physical


def raw_rows(n=155):
    a = np.empty((n, 40), np.uint64)
    for i in range(n):
        mid = 1000000 + i * 4
        for level in range(10):
            a[i, level * 4:level * 4 + 4] = [mid - level * 3 - 2, 10 + i + level,
                                            mid + level * 3 + 2, 20 + i + level]
    return a


def norm_for(rows):
    x = physical(rows)
    std = x.std(0);std[std == 0] = 1.
    return {'fit_dates': list(TRAIN), 'record_count': 5 * len(rows),
            'mean': x.mean(0).tolist(), 'std': std.tolist()}


def fixture(folder, control_dir, parent_dir):
    rows = raw_rows();norm = norm_for(rows)
    norm_path = folder / 'synthetic_original_normalization.json'
    norm_path.write_text(json.dumps(norm), encoding='utf-8')
    original_norm_ref = data6.binding(norm_path)
    result = {'version': 'TCN-research-B-P2-data-1.0', 'status': 'prepared_real_data_passed',
       'horizon': 50, 'threshold': 0, 'original_flag64_cleared': False,
       'sealed_records_opened': False, 'sealed_labels_or_scores_opened': False,
       'sealed_dates': list(SEALED), 'source_control_verified_for_new_days': False,
       'coverage_unknown_for_new_days': True, 'source_records_unchanged': True,
       'normalization': norm, 'normalization_binding': original_norm_ref,
       'sources': {}, 'ranges': [], 'day_label_reports': {},
       'sample_counts': {'train': 30, 'tuning_validation': 12},
       'train_class_counts': [10, 10, 10], 'tuning_class_counts': [4, 4, 4],
       'majority_class_from_train': 0}
    control = {'version': data6.CONTROL_VERSION, 'feature_id': data6.CONTROL_FEATURE_ID,
       'complete': True, 'normalization_refitted': False, 'original_P2_normalization_reused': True,
       'normalization': norm, 'normalization_binding': original_norm_ref,
       'ranges': [], 'cache_files': {}, 'boundary_examples': []}
    relative_norm, _ = relative.fit_unique_statistics(lambda: iter([rows] * 5))
    candidate = {'version': 'TCN-ED4-relative-row-cache-1.0', 'feature_id': relative.FEATURE_ID,
                 'complete': True, 'normalization': relative_norm, 'ranges': [], 'cache_files': {}}
    for day in TRAIN + TUNE:
        f = folder / 'generated_sources' / day;f.mkdir(parents=True)
        records = np.zeros(165, np.dtype(RECORD_FIELDS))
        records['features'][5:160] = rows;records['features'][:5] = rows[0];records['features'][160:] = rows[-1]
        source_path = f / 'synthetic_records.bin';records.tofile(source_path)
        label_path = f / 'synthetic_labels.u8';np.array([0, 1, 2, 0, 1, 2], 'u1').tofile(label_path)
        source_sha = data6.sha(source_path)
        result['sources'][day] = {'day': day, 'role': ROLES[day], 'rows': 165,
                                  'records_path': str(source_path), 'records_sha256': source_sha}
        r = {'date': day, 'role': ROLES[day], 'records_path': str(source_path), 'records_sha256': source_sha,
             'start_row': 5, 'stop_row': 160, 'endpoint_start': 104,
             'endpoint_stop_exclusive': 110, 'window_count': 6, 'label_offset': 0}
        result['ranges'].append(r)
        examples = []
        for end in (104, 105, 109):
            x = data6.original_normalized_rows(records['features'][end - 99:end + 1], norm).T.copy()
            examples.append({'endpoint': end, 'input_float32_sha256': hashlib.sha256(x.tobytes()).hexdigest()})
        result['day_label_reports'][day] = {'date': day, 'role': ROLES[day], 'window_count': 6,
             'class_counts': [2, 2, 2], 'label_path': str(label_path), 'label_bytes': 6,
             'label_sha256': data6.sha(label_path), 'first_second_last_window_examples': examples}
        for prepared, pathroot, values in [
            (control, control_dir, data6.original_normalized_rows(rows, norm)),
            (candidate, parent_dir / 'prepared', relative.normalized_rows(rows, relative_norm))]:
            d = pathroot / day;d.mkdir(parents=True)
            path = d / 'synthetic_normalized_rows.f32';values.tofile(path)
            prepared['ranges'].append({**r, 'cache_offset': 0})
            prepared['cache_files'][day] = {**data6.binding(path), 'dtype': '<f4',
                                            'shape': [155, 40], 'role': ROLES[day]}
    sources = {d: result['sources'][d]['records_sha256'] for d in TRAIN + TUNE}
    labels = {d: result['day_label_reports'][d]['label_sha256'] for d in TRAIN + TUNE}
    result.update(source_sha256_before=sources, source_sha256_after=sources)
    for prepared in (control, candidate):
        prepared.update(sample_counts=result['sample_counts'], day_label_reports=result['day_label_reports'],
                        source_records_unchanged=True, source_sha256_before=sources, source_sha256_after=sources,
                        label_streams_unchanged=True, labels_sha256_before=labels, labels_sha256_after=labels)
    return result, control, candidate, original_norm_ref['sha256']


class PhysicalConversionTests(unittest.TestCase):
    def test_matches_original_P2_float64_formula(self):
        rows = raw_rows();norm = norm_for(rows)
        expected = ((physical(rows) - np.array(norm['mean'])) / np.array(norm['std'])).astype('f4')
        np.testing.assert_array_equal(data6.original_normalized_rows(rows, norm), expected)

    def test_quantities_not_logged_or_rescaled(self):
        rows = raw_rows(3);norm = {'mean': [0.] * 40, 'std': [1.] * 40}
        np.testing.assert_array_equal(data6.original_normalized_rows(rows, norm)[:, 1::2], rows[:, 1::2])

    def test_prices_divided_by_10000(self):
        rows = raw_rows(1);norm = {'mean': [0.] * 40, 'std': [1.] * 40}
        expected = (rows[:, 0::2].astype(np.float64) / 10000.).astype('f4')
        np.testing.assert_array_equal(data6.original_normalized_rows(rows, norm)[:, 0::2], expected)

    def test_final_F32_after_F64_arithmetic(self):
        rows = raw_rows(3);rows[:, 0::2] += 10**12
        norm = norm_for(rows)
        actual = data6.original_normalized_rows(rows, norm)
        expected = ((physical(rows) - np.array(norm['mean'])) / np.array(norm['std'])).astype('f4')
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_does_not_mutate_raw_or_norm(self):
        rows = raw_rows();old = rows.copy();norm = norm_for(rows);oldnorm = copy.deepcopy(norm)
        data6.original_normalized_rows(rows, norm)
        np.testing.assert_array_equal(rows, old);self.assertEqual(norm, oldnorm)

    def test_zero_norm_scale_rejected(self):
        norm = norm_for(raw_rows());norm['std'][3] = 0
        with self.assertRaises(ValueError):data6.original_normalized_rows(raw_rows(1), norm)

    def test_nonfinite_input_rejected(self):
        raw = raw_rows(1).astype(float);raw[0, 2] = np.nan
        with self.assertRaises(ValueError):data6.original_normalized_rows(raw, norm_for(raw_rows()))

    def test_bad_feature_shape_rejected(self):
        with self.assertRaises(ValueError):data6.original_normalized_rows(np.ones((4, 39)), norm_for(raw_rows()))

    def test_production_original_norm_pin(self):
        self.assertEqual(data6.P2_NORM_SHA, '60130df9c2ed5673395e0247dcc8279f297a288719ec9187097ad4823a0a8c0b')


class CacheWindowTests(unittest.TestCase):
    def setUp(self):
        data6.STAGE.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='synthetic_ed6_', dir=data6.STAGE)
        folder = Path(self.tmp.name);self.control_dir = folder / 'prepared_control';self.parent_dir = folder / 'fake_parent'
        self.data, self.control, self.candidate, pin = fixture(folder, self.control_dir, self.parent_dir)
        self.patches = [mock.patch.object(data6, 'CONTROL_DIR', self.control_dir),
                        mock.patch.object(data6, 'PARENT', self.parent_dir),
                        mock.patch.object(data6, 'P2_NORM_SHA', pin),
                        mock.patch.object(relative, 'STAGE', self.parent_dir)]
        for patch in self.patches:patch.start()
        self.datasets = []

    def tearDown(self):
        for dataset in self.datasets:dataset.close()
        for patch in reversed(self.patches):patch.stop()
        self.tmp.cleanup()

    def ds(self, slot='control43', role='train'):
        prepared = self.control if slot == 'control43' else self.candidate
        dataset = data6.CachedWindows(self.data, prepared, role, slot);self.datasets.append(dataset);return dataset

    def test_control_full_batch_matches_original_ResearchWindows_bytes(self):
        cached = self.ds();original = ResearchWindows(self.data, 'train');self.datasets.append(original)
        ids = np.array([0, 5, 6, 17, 29, 0, 8])
        a = cached.batch_details(ids);b = original.batch_details(ids)
        for key in a:np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(a['X'].tobytes(), b['X'].tobytes())

    def test_control_original_first_second_last_golden_inputs(self):
        cached = self.ds()
        for day_index, day in enumerate(TRAIN):
            item = cached.batch_details(np.array([6 * day_index, 6 * day_index + 1, 6 * day_index + 5]))
            for i, ex in enumerate(self.data['day_label_reports'][day]['first_second_last_window_examples']):
                self.assertEqual(hashlib.sha256(item['X'][i].tobytes()).hexdigest(), ex['input_float32_sha256'])

    def test_candidate_matches_frozen_relative_adapter(self):
        cached = self.ds('candidate43');old = relative.CachedRelativeWindows(self.data, self.candidate, 'train');self.datasets.append(old)
        ids = np.array([0, 5, 6, 17, 29, 0, 8])
        a = cached.batch_details(ids);b = old.batch_details(ids)
        for key in a:np.testing.assert_array_equal(a[key], b[key])

    def test_both_slots_have_same_labels_anchors_dates(self):
        ids = np.arange(12)
        a = self.ds('control43', 'tuning_validation').batch_details(ids)
        b = self.ds('candidate43', 'tuning_validation').batch_details(ids)
        for key in ('y', 'date', 'endpoints', 'global_indices'):np.testing.assert_array_equal(a[key], b[key])

    def test_formal_cached_adapter_never_maps_original_records(self):
        ds = self.ds();original = np.memmap;touched = []
        def mapper(path, *args, **kwargs):touched.append(str(path));return original(path, *args, **kwargs)
        with mock.patch.object(np, 'memmap', side_effect=mapper):ds.batch(np.array([0, 29]))
        self.assertTrue(touched);self.assertTrue(all('synthetic_records.bin' not in p for p in touched))

    def test_shape_dtype_and_duplicate_order(self):
        item = self.ds().batch_details(np.array([5, 5, 0]))
        self.assertEqual(item['X'].shape, (3, 40, 100));self.assertEqual(item['X'].dtype, np.float32)
        np.testing.assert_array_equal(item['X'][0], item['X'][1]);np.testing.assert_array_equal(item['y'], [2, 2, 0])

    def test_negative_out_of_bounds_empty_and_float_indices(self):
        ds = self.ds()
        for ids in (np.array([-1]), np.array([len(ds)]), np.array([], int), np.array([1.])):
            with self.assertRaises(ValueError):ds.batch(ids)

    def test_unknown_slot_or_sealed_role_rejected(self):
        with self.assertRaises(ValueError):data6.validate_cache(self.data, self.control, 'unknown')
        with self.assertRaises(ValueError):data6.CachedWindows(self.data, self.control, 'comparison_validation', 'control43')

    def test_variant_cache_mix_rejected(self):
        with self.assertRaises(ValueError):data6.validate_cache(self.data, self.candidate, 'control43')
        with self.assertRaises(ValueError):data6.validate_cache(self.data, self.control, 'candidate43')

    def test_control_norm_payload_or_pin_change_rejected(self):
        for field in ('payload', 'pin'):
            prepared = copy.deepcopy(self.control)
            if field == 'payload':prepared['normalization']['mean'][0] += 1
            else:prepared['normalization_binding']['sha256'] = '0' * 64
            with self.assertRaises(ValueError):data6.validate_cache(self.data, prepared, 'control43')

    def test_control_refit_claim_rejected(self):
        prepared = copy.deepcopy(self.control);prepared['normalization_refitted'] = True
        with self.assertRaises(ValueError):data6.validate_cache(self.data, prepared, 'control43')

    def test_cache_range_or_offset_change_rejected(self):
        for field in ('cache_offset', 'endpoint_start'):
            prepared = copy.deepcopy(self.control);prepared['ranges'][0][field] += 1
            with self.assertRaises(ValueError):data6.validate_cache(self.data, prepared, 'control43')

    def test_control_payload_path_outside_stage_rejected(self):
        prepared = copy.deepcopy(self.control);prepared['cache_files'][TRAIN[0]]['path'] = '@EXTERNAL_ROOT@/file.f32'
        with self.assertRaises(ValueError):data6.validate_cache(self.data, prepared, 'control43')

    def test_mismatched_label_integrity_declaration_rejected(self):
        prepared = copy.deepcopy(self.control);prepared['labels_sha256_after'][TRAIN[0]] = '0' * 64
        with self.assertRaises(ValueError):data6.validate_cache(self.data, prepared, 'control43')

    def test_cache_size_checked_before_mapping(self):
        ds = self.ds()
        with Path(self.control['cache_files'][TRAIN[0]]['path']).open('ab') as stream:stream.write(b'x')
        with self.assertRaises(ValueError):ds.batch(np.array([0]))

    def test_close_releases_all_maps(self):
        ds = self.ds();ds.batch(np.array([0, 29]));self.assertTrue(ds.maps)
        ds.close();self.assertEqual(ds.maps, {});self.assertEqual(ds.label_maps, {})

    def test_existing_prepare_target_rejected_before_source_read(self):
        with mock.patch.object(data6, 'verify_sources', side_effect=AssertionError('No source access allowed')):
            with self.assertRaises(ValueError):data6.prepare_control(self.data, self.control_dir)

    def test_wrong_prepare_target_rejected_before_source_read(self):
        with mock.patch.object(data6, 'verify_sources', side_effect=AssertionError('No source access allowed')):
            with self.assertRaises(ValueError):data6.prepare_control(self.data, self.control_dir.parent / 'other')


if __name__ == '__main__':
    unittest.main()
