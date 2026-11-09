"""ED7 compatibility-v3 NumPy/stdlib fixtures; no real Torch/ML or data.

The historical49 tests are preserved separately. ScriptModule API regressions
below use fake objects and AST inspection, never a genuine model or CP loader.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import audit_tcn_existing_data_ed7_compat as audit
import tcn_existing_data_ed7_compat_contract as contract
import tcn_existing_data_ed7_compat_runtime as runtime
import tcn_existing_data_ed4_features as old_features


def normalization():
    return {'feature_id': runtime.FEATURE_ID, 'fit_dates': list(runtime.TRAIN_DATES), 'TUNE_fitted': False,
            'sealed_fitted': False, 'record_count': 1200, 'mean': np.linspace(-.2, .8, 40).tolist(),
            'std': np.linspace(.5, 2.5, 40).tolist(),
            'price_formula': '(raw_price - (raw_bid1+raw_ask1)/2) / ((raw_bid1+raw_ask1)/2)',
            'quantity_formula': 'raw_quantity_unchanged_no_log',
            'transform_and_normalization_precision': 'float64; final stored rows float32'}


def raw_batch(batch=2):
    result = np.zeros((batch, 100, 40), dtype='<u8')
    for row in range(100):
        result[:, row, 0::2] = 1_000_001 + row * 7 + np.arange(20)
        result[:, row, 1::2] = np.arange(20) * 10 + row
    return result


class PreprocessingTests(unittest.TestCase):
    def test_exact_old_ED4_F64_transform_and_final_F32(self):
        raw = raw_batch(3); norm = normalization()
        expected = old_features.normalized_rows(raw, norm).transpose(0, 2, 1).copy()
        actual = runtime.preprocess_raw(raw, norm)
        self.assertEqual(actual.dtype, np.dtype('<f4')); self.assertTrue(actual.flags.c_contiguous)
        np.testing.assert_array_equal(actual, expected)

    def test_each_row_uses_own_midpoint(self):
        raw = raw_batch(1); norm = normalization(); norm['mean'] = [0.] * 40; norm['std'] = [1.] * 40
        got = runtime.preprocess_raw(raw, norm)
        for row in (0, 1, 99):
            x = raw[0, row].astype(np.float64); mid = (x[0] + x[2]) / 2
            self.assertEqual(got[0, 0, row], np.float32((x[0] - mid) / mid))

    def test_raw_quantity_unchanged_no_log_and_zero_allowed(self):
        raw = raw_batch(1); norm = normalization(); norm['mean'] = [0.] * 40; norm['std'] = [1.] * 40
        raw[0, 99, 1] = np.iinfo(np.uint64).max
        got = runtime.preprocess_raw(raw, norm)
        np.testing.assert_array_equal(got[:, 1::2, :], raw[:, :, 1::2].transpose(0, 2, 1).astype('<f4'))
        self.assertEqual(got[0, 1, 0], 0.)

    def test_exact_price_domain_upper_included_in_rejection(self):
        for value in (0, 2**51, 2**64 - 1):
            raw = raw_batch(1); raw[0, 0, 0] = value
            with self.assertRaises(ValueError): runtime.preprocess_raw(raw, normalization())

    def test_price_last_valid_value(self):
        raw = raw_batch(1); raw[..., 0::2] = 2**51 - 1
        self.assertTrue(np.isfinite(runtime.preprocess_raw(raw, normalization())).all())

    def test_negative_float_complex_signed_or_bool_raw_rejected(self):
        for dtype in (np.int64, np.float64, np.complex128, np.bool_):
            value = raw_batch(1).astype(dtype)
            if dtype == np.int64: value[0, 0, 1] = -1
            with self.assertRaises(ValueError): runtime.preprocess_raw(value, normalization())

    def test_wrong_raw_shape_and_batch_bounds(self):
        for value in (np.ones((100, 40), '<u8'), np.ones((1, 40, 100), '<u8'),
                      np.ones((0, 100, 40), '<u8'), np.ones((33, 100, 40), '<u8')):
            with self.assertRaises(ValueError): runtime.preprocess_raw(value, normalization())

    def test_all_admitted_batches_one_to32(self):
        for count in (1, 2, 16, 32):
            self.assertEqual(runtime.preprocess_raw(raw_batch(count), normalization()).shape, (count, 40, 100))

    def test_normalized_strict_F32_shape_and_finite(self):
        for value in (np.zeros((1, 40, 100), '<f8'), np.zeros((1, 100, 40), '<f4'),
                      np.zeros((0, 40, 100), '<f4'), np.zeros((33, 40, 100), '<f4'),
                      np.full((1, 40, 100), np.nan, '<f4'), np.full((1, 40, 100), np.inf, '<f4')):
            with self.assertRaises(ValueError): runtime.validate_normalized(value)

    def test_normalization_wrong_feature_or_training_scope(self):
        for key, value in [('feature_id', 'physical'), ('fit_dates', ['2025-12-15']),
                           ('TUNE_fitted', True), ('sealed_fitted', True)]:
            norm = normalization(); norm[key] = value
            with self.assertRaises(ValueError): runtime.validate_normalization(norm)

    def test_normalization_wrong_semantics(self):
        for key, value in [('price_formula', 'window_endpoint_mid'), ('quantity_formula', 'log1p'),
                           ('transform_and_normalization_precision', 'float32')]:
            norm = normalization(); norm[key] = value
            with self.assertRaises(ValueError): runtime.validate_normalization(norm)

    def test_normalization_shape_nonfinite_nonpositive(self):
        for key, value in [('mean', [0.] * 39), ('std', [1.] * 39), ('mean', [np.nan] * 40),
                           ('std', [np.inf] * 40), ('std', [0.] * 40), ('std', [-1.] * 40)]:
            norm = normalization(); norm[key] = value
            with self.assertRaises(ValueError): runtime.validate_normalization(norm)

    def test_normalization_boolean_values_rejected(self):
        norm = normalization(); norm['mean'][0] = True
        with self.assertRaises(ValueError): runtime.validate_normalization(norm)

    def test_F32_overflow_fails_closed(self):
        norm = normalization(); norm['std'] = [1e-300] * 40
        with self.assertRaises(ValueError): runtime.preprocess_raw(raw_batch(1), norm)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic_ED7_package_')
        self.package = Path(self.temp.name)
        required = {'runtime.py': Path(runtime.__file__).read_bytes(),
                    'normalization.json': json.dumps(normalization()).encode(), 'model_cpu.ts': b'synthetic_script_not_loaded',
                    'native/best.pt': b'synthetic_CP_not_deserialized', 'golden/selection.json': b'{}',
                    'golden/inputs.f32': np.zeros((32, 40, 100), '<f4').tobytes()}
        for kind in ('native', 'effective', 'script'):
            for batch in (32, 1): required[f'golden/{kind}_batch{batch}.f32'] = np.zeros((batch, 3), '<f4').tobytes()
        for name, value in required.items():
            path = self.package / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
        self.manifest = {'version': 'TCN-ED7-package-manifest-1.0', 'selected_family': 'ED4_feature42_lr0.001_seed42',
            'epoch': 3, 'seed': 42, 'model_format': 'TorchScript', 'execution_device': 'cpu',
            'semantic_contract': copy.deepcopy(runtime.SEMANTIC_CONTRACT),
            'files': {name: {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)} for name, value in required.items()}}
        self.manifest['normalization_sha256'] = self.manifest['files']['normalization.json']['sha256']
        self.manifest['checkpoint_sha256'] = self.manifest['files']['native/best.pt']['sha256']
        self.pin_norm = patch.object(runtime, 'FROZEN_NORMALIZATION_SHA', self.manifest['files']['normalization.json']['sha256'])
        self.pin_cp = patch.object(runtime, 'FROZEN_CHECKPOINT_SHA', self.manifest['files']['native/best.pt']['sha256'])
        self.pin_norm.start(); self.pin_cp.start(); self.rewrite()

    def tearDown(self):
        self.pin_cp.stop(); self.pin_norm.stop(); self.temp.cleanup()

    def rewrite(self):
        path = self.package / 'package_manifest.json'; path.write_text(json.dumps(self.manifest), 'utf-8')
        self.manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    def validate(self):
        return runtime.validate_package(self.package, self.manifest_sha)

    def test_complete_synthetic_package_without_any_model_load(self):
        with patch.object(runtime, '_new_runner', side_effect=AssertionError('must not construct runner')):
            manifest, norm = self.validate()
        self.assertEqual(manifest['seed'], 42); self.assertEqual(norm['fit_dates'], runtime.TRAIN_DATES)

    def test_external_manifest_hash_required_and_checked(self):
        for value in ('0' * 64, '', None, 'Z' * 64):
            with self.assertRaises(ValueError): runtime.validate_package(self.package, value)

    def test_missing_file_rejected_before_loader(self):
        (self.package / 'model_cpu.ts').unlink()
        with patch.object(runtime, '_new_runner') as loader:
            with self.assertRaises(ValueError): runtime.Predictor(self.package, self.manifest_sha)
            loader.assert_not_called()

    def test_payload_or_byte_count_tamper_rejected(self):
        for mode in ('payload', 'bytes'):
            if mode == 'payload':
                (self.package / 'model_cpu.ts').write_bytes(b'tampered')
            else:
                (self.package / 'model_cpu.ts').write_bytes(b'synthetic_script_not_loaded')
                self.manifest['files']['model_cpu.ts']['bytes'] += 1; self.rewrite()
            with self.assertRaises(ValueError): self.validate()

    def test_package_paths_absolute_parent_backslash_or_drive_rejected(self):
        for value in ('../escape', '/absolute', '@EXTERNAL_ROOT@/absolute', 'a\\b', 'a//b', './x'):
            with self.assertRaises(ValueError): runtime._file(self.package, value)

    def test_reparse_payload_rejected(self):
        with patch.object(Path, 'is_symlink', return_value=True):
            with self.assertRaises(ValueError): self.validate()

    def test_wrong_model_format_seed_family_epoch_or_device(self):
        for key, value in [('model_format', 'pickle'), ('seed', 43), ('epoch', 4),
                           ('selected_family', 'WN43'), ('execution_device', 'cuda')]:
            original = self.manifest[key]; self.manifest[key] = value; self.rewrite()
            with self.assertRaises(ValueError): self.validate()
            self.manifest[key] = original

    def test_bias_or_contract_type_change_rejected(self):
        for value in (.1, False):
            self.manifest['semantic_contract']['stationary_logit_bias'] = value; self.rewrite()
            with self.assertRaises(ValueError): self.validate()

    def test_wrong_normalization_hash_even_when_manifest_rebuilt(self):
        path = self.package / 'normalization.json'; data = normalization(); data['mean'][0] += 1
        path.write_text(json.dumps(data), 'utf-8')
        self.manifest['files']['normalization.json'] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
        self.manifest['normalization_sha256'] = self.manifest['files']['normalization.json']['sha256']
        self.rewrite()
        with self.assertRaises(ValueError): self.validate()

    def test_wrong_native_CP_hash_even_when_manifest_rebuilt(self):
        path = self.package / 'native/best.pt'; path.write_bytes(b'another_synthetic_CP')
        self.manifest['files']['native/best.pt'] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
        self.manifest['checkpoint_sha256'] = self.manifest['files']['native/best.pt']['sha256']
        self.rewrite()
        with self.assertRaises(ValueError): self.validate()

    def test_executing_runtime_source_must_match_package(self):
        path = self.package / 'runtime.py'; path.write_bytes(b'other_runtime_source')
        self.manifest['files']['runtime.py'] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
        self.rewrite()
        with self.assertRaises(ValueError): self.validate()

    def test_circular_manifest_entry_rejected(self):
        self.manifest['files']['package_manifest.json'] = {'sha256': '0' * 64, 'bytes': 0}; self.rewrite()
        with self.assertRaises(ValueError): self.validate()

    def test_predictor_uses_mock_only_and_first_class_tie(self):
        class Runner:
            calls = 0
            def run(self, values):
                self.calls += 1; return np.tile(np.array([[1, 1, 0]], '<f4'), (len(values), 1))
        runner = Runner()
        with patch.object(runtime, '_new_runner', return_value=runner) as loader:
            predictor = runtime.Predictor(self.package, self.manifest_sha)
            result = predictor.predict_normalized(np.zeros((2, 40, 100), '<f4'))
        self.assertEqual(loader.call_count, 1); self.assertEqual(runner.calls, 1)
        self.assertEqual(result['class_indices'].tolist(), [0, 0]); self.assertEqual(result['class_names'], ['down', 'down'])

    def test_invalid_normalized_input_does_not_call_mock_runner(self):
        with patch.object(runtime, '_new_runner') as loader:
            predictor = runtime.Predictor(self.package, self.manifest_sha)
            with self.assertRaises(ValueError): predictor.predict_normalized(np.zeros((33, 40, 100), '<f4'))
            loader.return_value.run.assert_not_called()

    def test_bad_mock_runner_output_fails_closed(self):
        for value in (np.zeros((1, 3), '<f8'), np.zeros((1, 4), '<f4'), np.full((1, 3), np.nan, '<f4')):
            with patch.object(runtime, '_new_runner') as loader:
                loader.return_value.run.return_value = value; predictor = runtime.Predictor(self.package, self.manifest_sha)
                with self.assertRaises(ValueError): predictor.predict_normalized(np.zeros((1, 40, 100), '<f4'))


def synthetic_cache(campaign):
    result = {'ranges': [], 'cache_files': {}, 'boundary_examples': []}
    for index, day in enumerate(contract.TRAIN):
        result['ranges'].append({'date': day, 'role': 'train', 'start_row': 10, 'stop_row': 1000,
            'cache_offset': 0, 'endpoint_start': 109, 'endpoint_stop_exclusive': 950, 'research_segment_id': index + 1})
        result['cache_files'][day] = {'path': str(campaign / 'step04_features/prepared' / day / 'normalized_rows.f32'),
            'role': 'train', 'dtype': '<f4', 'shape': [990, 40], 'bytes': 990 * 160, 'sha256': 'a' * 64}
        for endpoint in (109, 110, 949):
            result['boundary_examples'].append({'day': day, 'endpoint': endpoint, 'normalized_input_float32_sha256': 'b' * 64})
    return result


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic_ED7_contract_'); self.root = Path(self.temp.name)
        self.campaign = self.root / 'artifacts/campaign'; self.stage = self.campaign / 'step07'; self.stage.mkdir(parents=True)
        self.patches = [patch.object(contract, 'ROOT', self.root), patch.object(contract, 'CAMPAIGN', self.campaign),
                        patch.object(contract, 'STAGE', self.stage), patch.object(contract, 'PACKAGE', self.stage / 'package')]
        for item in self.patches: item.start()
        self.cache = synthetic_cache(self.campaign)

    def tearDown(self):
        for item in reversed(self.patches): item.stop()
        self.temp.cleanup()

    def test_expected32_plan_is_metadata_only(self):
        with patch('builtins.open', side_effect=AssertionError('no payload reads')):
            plan = contract.expected_window_plan(self.cache); contract.validate_window_plan(plan, self.cache)
        self.assertEqual(len(plan), 32); self.assertEqual(sum(w['expected_input_sha256'] is not None for w in plan), 15)
        audit.validate_window_plan(plan)

    def test_DEV_or_historical_window_rejected(self):
        for date in ('2025-12-15', '2025-12-16', '2025-12-17', '2025-12-18', '2025-12-19'):
            plan = contract.expected_window_plan(self.cache); plan[0]['date'] = date
            with self.assertRaises(ValueError): contract.validate_window_plan(plan, self.cache)
            with self.assertRaises(ValueError): audit.validate_window_plan(plan)

    def test_window_range_gap_or_H50_change_rejected(self):
        for key, value in [('cache_row_start', -1), ('cache_row_stop_exclusive', 99), ('endpoint', 999),
                           ('cache_range_start', 1), ('start_row', 109)]:
            plan = contract.expected_window_plan(self.cache); plan[0][key] = value
            with self.assertRaises(ValueError): contract.validate_window_plan(plan, self.cache)
            with self.assertRaises(ValueError): audit.validate_window_plan(plan)

    def test_duplicate_or_nonTRAIN_cache_rejected(self):
        plan = contract.expected_window_plan(self.cache); plan[-1] = copy.deepcopy(plan[0])
        with self.assertRaises(ValueError): contract.validate_window_plan(plan, self.cache)
        plan = contract.expected_window_plan(self.cache); plan[0]['source_cache']['role'] = 'tuning_validation'
        with self.assertRaises(ValueError): contract.validate_window_plan(plan, self.cache)

    def test_exact_no_training_finite_operation_budgets(self):
        self.assertEqual(len(contract.OPERATION_KEYS), 25)
        self.assertEqual(contract.PRIMARY_BUDGET['forward_calls'], 6); self.assertEqual(contract.PRIMARY_BUDGET['forward_samples'], 99)
        self.assertEqual(contract.RELOAD_BUDGET['forward_calls'], 1); self.assertEqual(contract.RELOAD_BUDGET['forward_samples'], 32)
        self.assertEqual(contract.PRIMARY_BUDGET['source_cache_window_reads'], 0)
        self.assertEqual(contract.PRIMARY_BUDGET['source_cache_payload_bytes_read'], 0)
        self.assertEqual(contract.PRIMARY_BUDGET['saved_golden_read_bytes'], 512000)
        self.assertEqual(contract.RELOAD_BUDGET['saved_golden_read_bytes'], 0)
        for key in ('training_runs', 'optimizer_steps', 'backward_calls', 'E_record_reads', 'label_reads', 'DEV_cache_reads', 'sealed_stream_reads'):
            self.assertEqual(contract.PRIMARY_BUDGET[key], 0); self.assertEqual(contract.RELOAD_BUDGET[key], 0)

    def test_permanent_phase_reservation_cannot_reopen(self):
        reg = {'_registration_binding': {'path': 'synthetic', 'sha256': 'c' * 64, 'bytes': 0},
               'primary_budget': contract.PRIMARY_BUDGET}
        reservation = contract.reserve(reg, 'primary'); self.assertTrue(reservation['permanent_single_use'])
        with self.assertRaises(ValueError): contract.reserve(reg, 'primary')

    def test_existing_acceptance_blocks_reservation(self):
        (self.stage / 'reload_acceptance.json').write_text('{}', 'utf-8')
        with self.assertRaises(ValueError): contract.reserve({'_registration_binding': {}}, 'reload')

    def test_workspace_escape_rejected(self):
        for path in (self.root.parent / 'escape', '../escape'):
            with self.assertRaises(ValueError): contract.safe(path)

    def test_import_does_not_execute_model_or_data_operations(self):
        for module in (runtime, contract, audit, old_features):
            tree = ast.parse(Path(module.__file__).read_text('utf-8'))
            imports = [a.name for node in tree.body if isinstance(node, ast.Import) for a in node.names]
            self.assertNotIn('torch', imports)
            self.assertFalse(any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) for n in tree.body))

    def admitted_fixture(self):
        def ref(name, sha='c' * 64):
            return {'path': str(self.stage / name), 'sha256': sha, 'bytes': 1}
        source = ref('tests.py'); review_ref = ref('review.json'); tests_ref = ref('tests.json')
        complete_ref = ref('completion.json'); cache_ref = ref('cache.json'); actual_ref = ref('actual.json')
        cp_ref = ref('native.pt', contract.CP_SHA); norm_ref = ref('normalization.json', contract.NORM_SHA)
        reg = {'version': 'TCN-ED7-delivery-registration-1.0', 'selected_family': contract.FAMILY,
            'compatibility_revision': 'torchscript_compatibility_v3',
            'failed_primary_operations': copy.deepcopy(contract.FAILED_PRIMARY_OPERATIONS),
            'cumulative_budget': copy.deepcopy(contract.CUMULATIVE_BUDGET),
            'representative_seed': 42, 'primary_budget': copy.deepcopy(contract.PRIMARY_BUDGET),
            'reload_budget': copy.deepcopy(contract.RELOAD_BUDGET), 'tolerance': copy.deepcopy(contract.TOLERANCE),
            'training_runs': 0, 'new_independent_test': False, 'deployment_authorized': False,
            'historical17_19_reads': False, 'source_bindings': {'tests': source},
            'evidence_bindings': {'ED6_completion': complete_ref}, 'source_review_binding': review_ref,
            'test_evidence_binding': tests_ref, 'checkpoint_binding': cp_ref, 'normalization_binding': norm_ref,
            'initial_failed_primary_binding': ref('initial_failure.json'),
            'initial_primary_receipt_binding': ref('initial_actual.json'),
            'original_registration_binding': ref('original_registration.json'),
            'saved_golden_input_binding': {**ref('saved_inputs.f32'), 'bytes': 512000},
            'saved_golden_selection_binding': ref('saved_selection.json'),
            'config_binding': ref('config.json'), 'protocol_binding': ref('protocol.json', contract.PROTOCOL_SHA),
            'cache_manifest_binding': cache_ref, 'window_plan': [], 'package_copies': []}
        docs = {str(self.stage / 'registration.json'): reg, review_ref['path']: {'passed': True, 'source_bindings': {'tests': source}},
            tests_ref['path']: {'passed': True, 'actual_exit_code': 0, 'actual_receipt': actual_ref},
            complete_ref['path']: {'passed': True, 'stable_development_adoption': True, 'representative_seed': 42,
                'retained_family': contract.FAMILY, 'historical_final_quality_changed': False,
                'retained_checkpoint': cp_ref, 'retained_normalization': norm_ref}, cache_ref['path']: {}}
        actual = {'entry_before': source, 'sources_before': [source]}
        return reg, docs, actual

    def mocked_admit(self, mutate=None):
        from contextlib import ExitStack
        reg, docs, actual = self.admitted_fixture()
        if mutate: mutate(reg)
        with ExitStack() as stack:
            stack.enter_context(patch.object(contract, 'REGISTRATION', self.stage / 'registration.json'))
            stack.enter_context(patch.object(contract, 'sha', return_value='registration'))
            stack.enter_context(patch.object(contract, 'read', side_effect=lambda p: copy.deepcopy(docs[str(p)])))
            stack.enter_context(patch.object(contract, 'verify', side_effect=lambda r: Path(r['path'])))
            stack.enter_context(patch.object(contract, 'checked_receipt', return_value=actual))
            stack.enter_context(patch.object(contract, 'validate_window_plan', return_value=True))
            stack.enter_context(patch.object(contract, 'verify_protected', return_value=0))
            stack.enter_context(patch.object(contract, 'verify_failed_attempt', return_value=contract.FAILED_PRIMARY_OPERATIONS))
            stack.enter_context(patch.object(contract, 'binding', return_value={'path': str(self.stage / 'registration.json'), 'sha256': 'registration', 'bytes': 1}))
            return contract.admit('registration', 'primary')

    def test_mocked_registration_admission_no_real_files_or_ML(self):
        self.assertEqual(self.mocked_admit()['representative_seed'], 42)

    def test_registration_wrong_family_or_seed_rejected(self):
        for key, value in [('selected_family', 'seed43'), ('representative_seed', 43)]:
            with self.assertRaises(ValueError): self.mocked_admit(lambda reg: reg.__setitem__(key, value))

    def test_registration_scope_cannot_expand_training_or_deployment(self):
        for key, value in [('training_runs', 1), ('new_independent_test', True), ('deployment_authorized', True),
                           ('historical17_19_reads', True)]:
            with self.assertRaises(ValueError): self.mocked_admit(lambda reg: reg.__setitem__(key, value))

    def test_registration_budget_extra_forward_or_reload_load_rejected(self):
        for budget, key in [('primary_budget', 'forward_calls'), ('reload_budget', 'torchscript_loads')]:
            def mutate(reg): reg[budget][key] += 1
            with self.assertRaises(ValueError): self.mocked_admit(mutate)

    def test_registration_tolerance_cannot_loosen(self):
        def mutate(reg): reg['tolerance']['atol'] = 1.
        with self.assertRaises(ValueError): self.mocked_admit(mutate)


class IndependentArithmeticTests(unittest.TestCase):
    def test_tolerance_and_exact_class_tie(self):
        reference = np.array([[1, 1, 0]], '<f4'); actual = reference.copy()
        measured = audit.compare_outputs(reference, actual, 1e-5, 1e-4)
        self.assertEqual(measured['reference_classes'], [0]); self.assertEqual(measured['class_agreement'], 1.)

    def test_small_numeric_error_with_same_class_passes(self):
        reference = np.array([[2, 1, 0]], '<f4'); actual = reference + np.float32(1e-6)
        self.assertTrue(audit.compare_outputs(reference, actual, 1e-5, 1e-4)['passed'])

    def test_tolerance_alone_does_not_allow_class_flip(self):
        reference = np.array([[1, 1 - 1e-6, 0]], '<f4'); actual = np.array([[1 - 1e-6, 1, 0]], '<f4')
        with self.assertRaises(ValueError): audit.compare_outputs(reference, actual, 1e-5, 1e-4)

    def test_out_of_tolerance_nonfinite_or_wrong_dtype_rejected(self):
        reference = np.array([[2, 1, 0]], '<f4')
        for actual in (reference + 1., np.full((1, 3), np.inf, '<f4'), reference.astype('<f8')):
            with self.assertRaises(ValueError): audit.compare_outputs(reference, actual, 1e-5, 1e-4)

    def test_negative_or_nonfinite_tolerance_rejected(self):
        values = np.array([[2, 1, 0]], '<f4')
        for atol, rtol in ((-1., 0.), (0., -1.), (np.nan, 0.), (0., np.inf)):
            with self.assertRaises(ValueError): audit.compare_outputs(values, values, atol, rtol)

    def test_reference_scale_controls_asymmetric_tolerance(self):
        reference = np.array([[1., 0., 0.]], '<f4'); observed = np.array([[2., 0., 0.]], '<f4')
        # Relative-to-observed would pass at .6; the fixed reference rule fails.
        with self.assertRaises(ValueError): audit.compare_outputs(reference, observed, 0., .6)


def synthetic_saved_fixture():
    plan = contract.expected_window_plan(synthetic_cache(Path(contract.CAMPAIGN)))
    inputs = np.arange(32 * 40 * 100, dtype='<f4').reshape(32, 40, 100) / np.float32(10000.)
    entries = []
    for index, item in enumerate(plan):
        value = hashlib.sha256(np.ascontiguousarray(inputs[index]).tobytes()).hexdigest()
        if item['expected_input_sha256'] is not None:
            item['expected_input_sha256'] = value
        entries.append({'index': index, **copy.deepcopy(item), 'input_sha256': value,
                        'boundary_match': True if item['expected_input_sha256'] is not None else None})
    selection = {'version': 'TCN-ED7-golden-selection-1.0', 'windows': 32,
                 'input_sha256': hashlib.sha256(inputs.tobytes()).hexdigest(), 'per_input': entries}
    return inputs, selection, plan


class CompatibilityRegressionTests(unittest.TestCase):
    def failed_attempt_fixture(self):
        base = Path('@EXTERNAL_ROOT@/synthetic_ED7_compat_fixture')
        def ref(name, size=1): return {'path': str(base / name), 'sha256': 'd' * 64, 'bytes': size}
        reg = {'initial_failed_primary_binding': ref('primary_failure.json'),
               'initial_primary_receipt_binding': ref('primary_execution_receipt.json'),
               'original_registration_binding': ref('registration.json'),
               'saved_golden_input_binding': ref('package/golden/inputs.f32', 512000),
               'saved_golden_selection_binding': ref('package/golden/selection.json')}
        failure = {'phase': 'primary', 'error_type': 'RuntimeError',
                   'error': 'requires_grad_ is not supported on ScriptModules',
                   'reservation_remains_consumed': True,
                   'worker_pid': 4242, 'process_provenance': {'pid': 4242},
                   'reservation': {'worker_pid': 4242, 'registration': reg['original_registration_binding']},
                   'operations_observed': {k: v for k, v in contract.FAILED_PRIMARY_OPERATIONS.items()
                                           if k != 'saved_golden_read_bytes'}}
        entry = {'path': str(contract.ROOT / 'scripts/tcn_existing_data_ed7_export.py'),
                 'sha256': 'd' * 64, 'bytes': 1}
        capture = {'path': str(contract.ROOT / 'scripts/run_tcn_existing_data_ed7_capture.py'),
                   'sha256': 'd' * 64, 'bytes': 1}
        actual = {'phase': 'primary', 'exit_code': 1, 'entry_unchanged': True, 'sources_unchanged': True,
                  'sources_before': [], 'sources_after': [], 'input': reg['original_registration_binding'],
                  'outputs': {'failure': reg['initial_failed_primary_binding']},
                  'entry_before': entry, 'entry_after': copy.deepcopy(entry), 'capture_source': capture,
                  'stdout': ref('stdout'), 'stderr': ref('stderr')}
        docs = {reg['initial_failed_primary_binding']['path']: failure,
                reg['initial_primary_receipt_binding']['path']: actual}
        return base, reg, failure, actual, docs

    def mocked_failed_attempt(self, mutate=None):
        base, reg, failure, actual, docs = self.failed_attempt_fixture()
        if mutate: mutate(reg, failure, actual)
        with patch.object(contract, 'BASE_STAGE', base), patch.object(contract, 'safe', side_effect=Path), \
             patch.object(contract, 'verify', side_effect=lambda ref: Path(ref['path'])), \
             patch.object(contract, 'binding', side_effect=lambda p: {'path': str(p), 'sha256': 'd' * 64, 'bytes': 1}), \
             patch.object(contract, 'read', side_effect=lambda path: copy.deepcopy(docs[str(path)])), \
             patch.object(contract, 'equivalent', side_effect=lambda a, b: a == b):
            return contract.verify_failed_attempt(reg)

    def test_original_failure_contract_uses_mock_metadata_only(self):
        self.assertEqual(self.mocked_failed_attempt(), contract.FAILED_PRIMARY_OPERATIONS)

    def test_original_failure_wrong_error_exit_or_observed_forward_rejected(self):
        mutations = [lambda _r, f, _a: f.__setitem__('error', 'another_error'),
                     lambda _r, _f, a: a.__setitem__('exit_code', 0),
                     lambda _r, f, _a: f['operations_observed'].__setitem__('forward_calls', 1)]
        for mutate in mutations:
            with self.assertRaises(ValueError): self.mocked_failed_attempt(mutate)

    def test_saved_input_wrong_original_path_or_byte_count_rejected(self):
        for key, value in [('bytes', 511999), ('path', '@EXTERNAL_ROOT@/other.f32')]:
            def mutate(reg, _f, _a): reg['saved_golden_input_binding'][key] = value
            with self.assertRaises(ValueError): self.mocked_failed_attempt(mutate)

    def test_loaded_script_mock_rejects_module_API_but_freezes_each_tensor(self):
        dtype = object()
        class Parameter:
            device = types.SimpleNamespace(type='cpu'); grad = None
            def __init__(self): self.dtype = dtype; self.requires_grad = True; self.calls = []
            def numel(self): return 33219
            def requires_grad_(self, value): self.calls.append(value); self.requires_grad = value; return self
        class ScriptModule:
            def __init__(self): self.parameter = Parameter(); self.eval_calls = 0; self.module_calls = 0
            def parameters(self): return iter([self.parameter])
            def eval(self): self.eval_calls += 1; return self
            def requires_grad_(self, _value):
                self.module_calls += 1; raise RuntimeError('requires_grad_ is not supported on ScriptModules')
        model = ScriptModule(); load_calls = []
        def fake_load(path, map_location): load_calls.append((path, map_location)); return model
        fake_torch = types.SimpleNamespace(float32=dtype, jit=types.SimpleNamespace(load=fake_load))
        with patch.dict('sys.modules', {'torch': fake_torch}):
            runner = runtime._TorchRunner('synthetic_script_not_read.ts')
        self.assertIs(runner.model, model); self.assertEqual(model.eval_calls, 1)
        self.assertEqual(model.module_calls, 0); self.assertEqual(model.parameter.calls, [False])
        self.assertFalse(model.parameter.requires_grad); self.assertEqual(len(load_calls), 1)
        self.assertEqual(load_calls[0][1], 'cpu'); self.assertEqual(runner.forward_calls, 0)

    def test_compiled_script_source_has_no_module_requires_grad_call(self):
        tree = ast.parse(Path(__file__).with_name('tcn_existing_data_ed7_compat_export.py').read_text('utf-8'))
        primary = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_primary')
        calls = [n for n in ast.walk(primary) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertFalse(any(n.func.attr == 'requires_grad_' for n in calls))
        self.assertTrue(any(n.func.attr == 'eval' and isinstance(n.func.value, ast.Name)
                            and n.func.value.id == 'scripted' for n in calls))
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'model_ready'
                            and isinstance(n.args[0], ast.Name) and n.args[0].id == 'scripted' for n in ast.walk(primary)))

    def test_loaded_script_source_requires_grad_is_tensor_only(self):
        tree = ast.parse(Path(runtime.__file__).read_text('utf-8'))
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_TorchRunner')
        init = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        calls = [n for n in ast.walk(init) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == 'requires_grad_']
        self.assertEqual(len(calls), 1); self.assertIsInstance(calls[0].func.value, ast.Name)
        self.assertEqual(calls[0].func.value.id, 'parameter')
        self.assertEqual(ast.literal_eval(calls[0].args[0]), False)

    def test_saved32_inputs_and15_original_hashes_match_without_source_reads(self):
        inputs, selection, plan = synthetic_saved_fixture()
        with patch('builtins.open', side_effect=AssertionError('no real cache/source access')):
            values, count = audit.validate_saved_selection(selection, inputs, plan)
        self.assertEqual(count, 15); self.assertEqual(len(values), 32)

    def test_saved_input_aggregate_hash_tamper_rejected(self):
        inputs, selection, plan = synthetic_saved_fixture(); inputs[31, 0, 0] += 1
        with self.assertRaises(ValueError): audit.validate_saved_selection(selection, inputs, plan)

    def test_saved_per_window_hash_tamper_rejected(self):
        inputs, selection, plan = synthetic_saved_fixture(); selection['per_input'][31]['input_sha256'] = '0' * 64
        with self.assertRaises(ValueError): audit.validate_saved_selection(selection, inputs, plan)

    def test_saved_original_boundary_hash_tamper_rejected(self):
        inputs, selection, plan = synthetic_saved_fixture()
        plan[0]['expected_input_sha256'] = '0' * 64; selection['per_input'][0]['expected_input_sha256'] = '0' * 64
        with self.assertRaises(ValueError): audit.validate_saved_selection(selection, inputs, plan)

    def test_saved_plan_endpoint_or_boundary_count_change_rejected(self):
        for mode in ('endpoint', 'boundary_count', 'boundary_flag', 'length'):
            inputs, selection, plan = synthetic_saved_fixture()
            if mode == 'endpoint': selection['per_input'][31]['endpoint'] += 1
            elif mode == 'boundary_count': plan[0]['expected_input_sha256'] = None
            elif mode == 'boundary_flag': selection['per_input'][0]['boundary_match'] = None
            else: selection['per_input'].pop()
            with self.assertRaises(ValueError): audit.validate_saved_selection(selection, inputs, plan)

    def test_saved_input_shape_dtype_or_nonfinite_rejected(self):
        inputs, selection, plan = synthetic_saved_fixture()
        for value in (inputs[:31], inputs.astype('<f8'), np.full(inputs.shape, np.nan, '<f4')):
            with self.assertRaises(ValueError): audit.validate_saved_selection(selection, value, plan)

    def test_exact_cumulative_failed_plus_primary_plus_reload_budget(self):
        observed = {k: v for k, v in contract.FAILED_PRIMARY_OPERATIONS.items() if k != 'saved_golden_read_bytes'}
        prior, mid, total = audit.cumulative_accounting(observed, contract.PRIMARY_BUDGET,
                                                     contract.RELOAD_BUDGET, contract.OPERATION_KEYS)
        self.assertEqual(prior, contract.FAILED_PRIMARY_OPERATIONS); self.assertEqual(total, contract.CUMULATIVE_BUDGET)
        for key, expected in [('eager_model_constructions', 4), ('checkpoint_deserializations', 2),
                              ('weight_norm_folds', 20), ('script_compilations', 2), ('torchscript_saves', 1),
                              ('torchscript_loads', 2), ('forward_calls', 7), ('forward_samples', 131),
                              ('source_cache_window_reads', 32), ('source_cache_payload_bytes_read', 512000),
                              ('saved_golden_read_bytes', 512000)]:
            self.assertEqual(total[key], expected)
        self.assertEqual(mid['forward_calls'], 6)

    def test_omitted_failed_cost_or_old_forward_misstatement_rejected(self):
        for key in ('eager_model_constructions', 'forward_calls', 'source_cache_window_reads'):
            observed = {k: v for k, v in contract.FAILED_PRIMARY_OPERATIONS.items() if k != 'saved_golden_read_bytes'}
            observed[key] += 1
            with self.assertRaises(ValueError): audit.cumulative_accounting(observed, contract.PRIMARY_BUDGET,
                                                                          contract.RELOAD_BUDGET, contract.OPERATION_KEYS)

    def test_new_source_cache_read_extra_forward_or_saved_decode_rejected(self):
        observed = {k: v for k, v in contract.FAILED_PRIMARY_OPERATIONS.items() if k != 'saved_golden_read_bytes'}
        for key in ('source_cache_window_reads', 'forward_calls', 'saved_golden_read_bytes'):
            primary = dict(contract.PRIMARY_BUDGET); primary[key] += 1
            with self.assertRaises(ValueError): audit.cumulative_accounting(observed, primary,
                                                                          contract.RELOAD_BUDGET, contract.OPERATION_KEYS)

    def test_boolean_operation_count_is_not_integer_budget(self):
        observed = {k: v for k, v in contract.FAILED_PRIMARY_OPERATIONS.items() if k != 'saved_golden_read_bytes'}
        primary = dict(contract.PRIMARY_BUDGET); primary['checkpoint_deserializations'] = True
        with self.assertRaises(ValueError): audit.cumulative_accounting(observed, primary,
                                                                      contract.RELOAD_BUDGET, contract.OPERATION_KEYS)


if __name__ == '__main__':
    unittest.main()
