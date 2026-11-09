"""Independent ED7 compatibility-v3 arithmetic: no torch, model or CP load.

Only the new delivery package, reload output and small registered evidence are
read. Original normalized caches, E records, labels and predictions are not read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_STAGE = ROOT / 'artifacts/tcn_existing_data_optimization_2026-10-17/step07_delivery'
STAGE = ORIGINAL_STAGE / 'torchscript_compatibility_v3'
PACKAGE = STAGE / 'package'
TRAIN = [f'2025-12-{i:02d}' for i in range(8, 13)]
FEATURE_ID = 'same_record_mid_relative_prices_raw_quantities_v1'


def need(value, message):
    if not value:
        raise ValueError(message)


def local(value):
    path = Path(value); path = path if path.is_absolute() else ROOT / path
    need('..' not in path.parts and path.is_relative_to(ROOT), 'Audit input outside workspace')
    for ancestor in (path, *path.parents):
        if ancestor.exists():
            need(not ancestor.is_symlink() and not getattr(ancestor.lstat(), 'st_file_attributes', 0) & 0x400,
                 'Reparse audit input forbidden')
        if ancestor == ROOT:
            break
    need(path.resolve().is_relative_to(ROOT.resolve()), 'Resolved audit input escaped workspace')
    return path


def digest(value):
    path = local(value); value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def binding(value):
    path = local(value)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path), 'bytes': path.stat().st_size}


def checked(ref):
    need(isinstance(ref, dict) and {'path', 'sha256', 'bytes'} <= set(ref), 'Complete binding required')
    path = local(ref['path'])
    need(path.is_file() and digest(path) == ref['sha256'] and path.stat().st_size == ref['bytes'],
         'Audit binding changed: ' + str(path))
    return path


def same(left, right):
    return (local(left['path']).resolve() == local(right['path']).resolve()
            and left['sha256'] == right['sha256'] and left['bytes'] == right['bytes'])


def pairs(values):
    result = {}
    for name, value in values:
        need(name not in result, 'Duplicate JSON key')
        result[name] = value
    return result


def read(value):
    path = local(value); need(path.stat().st_size <= 32 * 1024**2, 'Oversized audit metadata')
    return json.loads(path.read_text('utf-8-sig'), object_pairs_hook=pairs,
                      parse_constant=lambda name: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + name)))


def bound(ref):
    return read(checked(ref))


def put(path, value):
    path = local(path); need(path.is_relative_to(STAGE), 'Audit writes restricted to ED7')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def read_array(path, dtype, shape):
    path = local(path)
    need(path.is_relative_to(PACKAGE) or path == STAGE / 'reload_logits.f32', 'Only new ED7 numerical artifacts admitted')
    need(path.stat().st_size == math.prod(shape) * np.dtype(dtype).itemsize, 'Audit array shape/bytes differ')
    result = np.fromfile(path, dtype=dtype).reshape(shape)
    need(np.isfinite(result).all(), 'Nonfinite numerical artifact')
    return result


def compare_outputs(reference, actual, atol, rtol):
    need(reference.dtype == actual.dtype == np.dtype('<f4') and reference.shape == actual.shape
         and reference.ndim == 2 and reference.shape[1] == 3 and len(reference) > 0,
         'Output dtype/shape differs')
    need(np.isfinite(reference).all() and np.isfinite(actual).all(), 'Nonfinite golden output')
    need(type(atol) in (float, int) and type(rtol) in (float, int)
         and math.isfinite(atol) and math.isfinite(rtol) and atol >= 0 and rtol >= 0, 'Invalid comparison tolerance')
    a = reference.astype(np.float64); b = actual.astype(np.float64)
    delta = np.abs(a - b); allowed = float(atol) + float(rtol) * np.abs(a)
    need(np.all(delta <= allowed), 'Independent output tolerance failed')
    # Stable sorting reproduces lowest class index at ties independently of
    # the runtime's numpy argmax call.
    ra = np.argsort(-a, axis=1, kind='stable')[:, 0]
    rb = np.argsort(-b, axis=1, kind='stable')[:, 0]
    need(np.array_equal(ra, rb), 'Independent class identity failed')
    return {'passed': True, 'samples': len(a), 'max_absolute_difference': float(delta.max()),
            'class_agreement': 1.0, 'reference_classes': ra.tolist(), 'actual_classes': rb.tolist()}


def validate_window_plan(items):
    need(isinstance(items, list) and len(items) == 32, 'Exactly32 registered TRAIN inputs required')
    keys = []
    for item in items:
        need(item['date'] in TRAIN and item['source_cache']['role'] == 'train', 'Golden input outside TRAIN')
        need(type(item['endpoint']) is int and item['cache_row_stop_exclusive'] - item['cache_row_start'] == 100
             and item['start_row'] <= item['endpoint'] - 99 and item['endpoint'] + 50 < item['stop_row']
             and item['endpoint_start'] <= item['endpoint'] < item['endpoint_stop_exclusive'],
             'Golden input crosses its registered research range')
        need(item['source_cache']['dtype'] == '<f4' and item['source_cache']['shape'][1] == 40
             and item['cache_row_start'] >= 0
             and item['cache_row_stop_exclusive'] <= item['source_cache']['shape'][0], 'Golden cache geometry differs')
        need(item['cache_range_start'] <= item['cache_row_start']
             and item['cache_row_stop_exclusive'] <= item['cache_range_stop_exclusive']
             and item['cache_range_stop_exclusive'] - item['cache_range_start'] == item['stop_row'] - item['start_row']
             and item['cache_row_start'] == item['cache_range_start'] + item['endpoint'] - 99 - item['start_row'],
             'Window D-cache offset differs from registered range geometry')
        keys.append((item['date'], item['endpoint']))
    need(len(set(keys)) == 32, 'Duplicate golden endpoint')
    need(sum(item['expected_input_sha256'] is not None for item in items) == 15, 'Original15 TRAIN boundary references missing')
    return True


def expected_failed_operations(keys):
    need(len(keys) == len(set(keys)) == 25 and 'saved_golden_read_bytes' in keys,
         'Compatibility operation schema differs')
    result = dict.fromkeys(keys, 0)
    result.update(eager_model_constructions=2, checkpoint_deserializations=1,
        native_model_state_loads=1, folded_model_state_loads=1, weight_norm_folds=10,
        script_compilations=1, source_cache_window_reads=32, source_cache_payload_bytes_read=512000)
    return result


def cumulative_accounting(observed, primary, reloaded, keys):
    """Pure integer accounting, including consumed work from the failed phase."""
    need(set(observed) == set(keys) - {'saved_golden_read_bytes'}
         and all(type(v) is int and v >= 0 for v in observed.values()),
         'Original observed operation schema differs')
    prior = {k: observed.get(k, 0) for k in keys}
    need(prior == expected_failed_operations(keys), 'Original failed-primary work was omitted or changed')
    expected_primary = dict.fromkeys(keys, 0)
    expected_primary.update(eager_model_constructions=2, checkpoint_deserializations=1,
        native_model_state_loads=1, folded_model_state_loads=1, weight_norm_folds=10,
        script_compilations=1, torchscript_saves=1, torchscript_loads=1,
        forward_attempts=6, forward_calls=6, forward_samples=99, saved_golden_read_bytes=512000)
    expected_reload = dict.fromkeys(keys, 0)
    expected_reload.update(torchscript_loads=1, forward_attempts=1, forward_calls=1, forward_samples=32)
    need(primary == expected_primary and reloaded == expected_reload
         and all(type(v) is int and v >= 0 for v in (*primary.values(), *reloaded.values())),
         'Compatibility engineering budget changed')
    after_primary = {k: prior[k] + primary[k] for k in keys}
    total = {k: after_primary[k] + reloaded[k] for k in keys}
    return prior, after_primary, total


def validate_saved_selection(selection, inputs, plan):
    """Independent arithmetic over saved golden bytes, never source row caches."""
    validate_window_plan(plan)
    need(isinstance(inputs, np.ndarray) and inputs.dtype == np.dtype('<f4')
         and inputs.shape == (32, 40, 100) and np.isfinite(inputs).all(), 'Saved golden input domain differs')
    need(selection['version'] == 'TCN-ED7-golden-selection-1.0' and selection['windows'] == 32
         and len(selection['per_input']) == 32
         and selection['input_sha256'] == hashlib.sha256(np.ascontiguousarray(inputs).tobytes()).hexdigest(),
         'Saved golden aggregate identity differs')
    hashes = []; matches = 0
    for index, window in enumerate(plan):
        item = selection['per_input'][index]
        need(item['index'] == index and all(item[key] == value for key, value in window.items()),
             'Saved golden provenance differs')
        current = hashlib.sha256(np.ascontiguousarray(inputs[index]).tobytes()).hexdigest()
        need(item['input_sha256'] == current, 'Saved golden per-window hash differs')
        expected = window['expected_input_sha256']
        need(item['boundary_match'] is (True if expected is not None else None)
             and (expected is None or expected == current), 'Saved original TRAIN boundary differs')
        matches += expected is not None; hashes.append(current)
    need(matches == 15, 'Exactly15 original TRAIN boundaries required')
    return hashes, matches


def original_failure(reg, keys):
    failure_path = checked(reg['initial_failed_primary_binding'])
    old_cli_path = checked(reg['initial_primary_receipt_binding'])
    original_reg_path = checked(reg['original_registration_binding'])
    need(failure_path == ORIGINAL_STAGE / 'primary_failure.json'
         and old_cli_path == ORIGINAL_STAGE / 'primary_execution_receipt.json'
         and original_reg_path == ORIGINAL_STAGE / 'registration.json', 'Original failed phase identity differs')
    failure = read(failure_path); actual = read(old_cli_path); old_reg = read(original_reg_path)
    need(failure['phase'] == actual['phase'] == 'primary' and actual['exit_code'] == 1
         and actual['entry_unchanged'] is True and actual['entry_before'] == actual['entry_after']
         and actual['sources_before'] == actual['sources_after'] and actual['sources_unchanged'] is True
         and actual['automatic_retry'] is False and failure['automatic_retry'] is False
         and failure['reservation_remains_consumed'] is True
         and failure['error_type'] == 'RuntimeError'
         and failure['error'] == 'requires_grad_ is not supported on ScriptModules',
         'Original actual failure was replaced or its scope changed')
    need(same(actual['input'], reg['original_registration_binding'])
         and same(actual['outputs']['failure'], reg['initial_failed_primary_binding'])
         and same(failure['reservation']['registration'], reg['original_registration_binding'])
         and failure['reservation']['worker_pid'] == failure['worker_pid'] == failure['process_provenance']['pid']
         and failure['reservation']['permanent_single_use'] is True,
         'Original consumed worker/registration evidence differs')
    reservation = read(ORIGINAL_STAGE / 'primary_phase_reservation.json')
    need(failure['reservation'] == reservation and not (ORIGINAL_STAGE / 'primary_acceptance.json').exists(),
         'Original failed reservation was reopened or presented as success')
    for field in ('entry_before', 'entry_after', 'capture_source', 'stdout', 'stderr'):
        checked(actual[field])
    for ref in actual['sources_before']:
        checked(ref)
    need(checked(actual['entry_before']) == ROOT / 'scripts/tcn_existing_data_ed7_export.py'
         and checked(actual['capture_source']) == ROOT / 'scripts/run_tcn_existing_data_ed7_capture.py'
         and failure['process_provenance']['entry_sha256'] == actual['entry_before']['sha256']
         and failure['error'] in checked(actual['stderr']).read_text('utf-8-sig'),
         'Original implementation/error evidence differs')
    need(reg['window_plan'] == old_reg['window_plan'], 'The saved32 plan was reselected after failure')
    references = {'inputs': reg['saved_golden_input_binding'], 'selection': reg['saved_golden_selection_binding']}
    need(checked(references['inputs']) == ORIGINAL_STAGE / 'package/golden/inputs.f32'
         and checked(references['selection']) == ORIGINAL_STAGE / 'package/golden/selection.json'
         and references['inputs']['bytes'] == 512000, 'Saved inputs do not belong to the failed package')
    need(set(failure['operations_observed']) == set(keys) - {'saved_golden_read_bytes'}
         and {k: failure['operations_observed'].get(k, 0) for k in keys} == expected_failed_operations(keys),
         'Failed primary observed counts differ')
    return failure, actual


def package_path(relative):
    need(isinstance(relative, str) and relative and '\\' not in relative and ':' not in relative,
         'Package path must be relative POSIX')
    part = PurePosixPath(relative)
    need(not part.is_absolute() and '..' not in part.parts and part.as_posix() == relative, 'Unsafe package path')
    path = local(PACKAGE.joinpath(*part.parts)); need(path.is_relative_to(PACKAGE), 'Package path escaped')
    return path


def receipt(phase, reg):
    path = STAGE / (phase + '_execution_receipt.json'); value = read(path)
    need(value['phase'] == phase and value['exit_code'] == 0 and value['entry_unchanged'] is True
         and value['entry_before'] == value['entry_after'] and value['sources_before'] == value['sources_after']
         and value['sources_unchanged'] is True, 'Actual CLI terminal/source identity failed')
    for field in ('entry_before', 'entry_after', 'input', 'capture_source', 'stdout', 'stderr'):
        checked(value[field])
    need(same(value['input'], binding(STAGE / 'registration.json'))
         and checked(value['entry_before']) == ROOT / 'scripts/tcn_existing_data_ed7_compat_export.py'
         and checked(value['capture_source']) == ROOT / 'scripts/run_tcn_existing_data_ed7_compat_capture.py',
         'Actual original worker/capture/registration differs')
    registered = {local(ref['path']).resolve(): ref for ref in reg['source_bindings'].values()}
    observed = {checked(ref).resolve(): ref for ref in value['sources_before']}
    need(set(registered) == set(observed) and all(same(registered[p], observed[p]) for p in registered),
         'Actual source set differs from pre-registered sources')
    acceptance_path = checked(value['outputs']['acceptance'])
    need(acceptance_path == STAGE / (phase + '_acceptance.json'), 'Wrong actual acceptance output')
    accepted = read(acceptance_path)
    need(accepted['passed'] is True and accepted['status'] == 'completed' and accepted['phase'] == phase
         and accepted['worker_pid'] == value['actual_worker_pid'] == accepted['process_provenance']['pid']
         and same(accepted['registration_binding'], value['input'])
         and accepted['process_provenance']['entry_sha256'] == value['entry_before']['sha256'],
         'Actual worker provenance differs')
    reservation = read(STAGE / (phase + '_phase_reservation.json'))
    need(accepted['reservation'] == reservation and reservation['worker_pid'] == accepted['worker_pid']
         and reservation['phase'] == phase and reservation['permanent_single_use'] is True
         and reservation['automatic_retry'] is False and same(reservation['registration'], value['input']),
         'Original single-use reservation differs')
    for field in ('training_started', 'new_independent_test', 'labels_read', 'E_records_read', 'DEV_cache_read',
                  'historical17_19_stream_read', 'normalization_refitted', 'model_quality_changed',
                  'source_control_verified', 'real_raw_input_replay', 'original_flag64_cleared',
                  'NPU_verified', 'deployment_started'):
        need(accepted[field] is False, 'Delivery scope expanded: ' + field)
    need(accepted['coverage_unknown'] is True and accepted['real_golden_path'] == 'normalized_float32_only'
         and accepted['all_eval_CPU_no_gradients'] is True, 'CPU golden-only scope differs')
    return value, accepted


def validate_manifest(manifest, reg):
    need(manifest['version'] == 'TCN-ED7-package-manifest-1.0'
         and manifest['selected_family'] == 'ED4_feature42_lr0.001_seed42' and manifest['seed'] == 42
         and manifest['epoch'] == 3 and manifest['model_format'] == 'TorchScript'
         and manifest['execution_device'] == 'cpu' and manifest['training_started'] is False
         and manifest['new_independent_test'] is False and manifest['historical_final_quality_changed'] is False,
         'Portable model identity differs')
    need(manifest['compatibility_case'] == 'ScriptModule_requires_grad_Module_API_unsupported'
         and manifest['initial_failed_primary_sha256'] == reg['initial_failed_primary_binding']['sha256']
         and manifest['saved_golden_input_sha256'] == reg['saved_golden_input_binding']['sha256']
         and manifest['saved_golden_selection_sha256'] == reg['saved_golden_selection_binding']['sha256'],
         'Compatibility package lost its original-failure/saved-input lineage')
    semantics = manifest['semantic_contract']
    need(semantics['feature_id'] == FEATURE_ID and semantics['raw_shape'] == ['B', 100, 40]
         and semantics['normalized_shape'] == ['B', 40, 100] and semantics['raw_dtype'] == 'uint64'
         and semantics['normalized_dtype'] == 'float32' and semantics['maximum_batch_size'] == 32
         and semantics['raw_price_units'] == 'original_uint64_price_units_no_divide_10000'
         and semantics['quantity_formula'] == 'raw_quantity_unchanged_no_log'
         and semantics['transform_precision'] == 'float64_then_frozen_float64_normalization_then_float32'
         and semantics['feature_order'] == 'ask_price,ask_size,bid_price,bid_size for each level1..10'
         and semantics['class_order'] == ['down', 'stationary', 'up']
         and semantics['decision'] == 'original_argmax_first_index_tie'
         and type(semantics['stationary_logit_bias']) is int and semantics['stationary_logit_bias'] == 0
         and semantics['device'] == 'cpu' and semantics['training'] is False, 'Portable preprocessing semantics differ')
    need(semantics['price_formula'] == '(raw_price - (raw_ask1 + raw_bid1)/2) / ((raw_ask1 + raw_bid1)/2)',
         'Relative-current-record price formula differs')
    required = {'runtime.py', 'normalization.json', 'native/best.pt', 'model_cpu.ts', 'golden/inputs.f32',
                'golden/selection.json', *(f'golden/{kind}_batch{size}.f32'
                for kind in ('native', 'effective', 'script') for size in (32, 1))}
    files = manifest['files']; need(required <= set(files) and 'package_manifest.json' not in files, 'Missing/circular package files')
    observed = {p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob('*') if p.is_file()}
    need(observed == set(files) | {'package_manifest.json'}, 'Undeclared/missing package payload')
    for relative, ref in files.items():
        need(set(ref) == {'sha256', 'bytes'}, 'Invalid portable file identity')
        checked({'path': str(package_path(relative)), **ref})
    need(files['normalization.json']['sha256'] == manifest['normalization_sha256'] == reg['normalization_binding']['sha256']
         == '709114d09b2dd581bd9481e180973fdf1553452b644d24dcb211da7ea394caaf'
         and files['native/best.pt']['sha256'] == manifest['checkpoint_sha256'] == reg['checkpoint_binding']['sha256']
         == 'c480db50073dbb099c5374dc5df9f942044e2ab09f8d6bcff396b7bc4cb6dc60', 'Frozen native/normalization identity differs')
    # Explicitly permitted opaque native-CP verification; never deserialize it.
    checked(reg['checkpoint_binding'])
    norm = read(package_path('normalization.json'))
    need(norm['feature_id'] == FEATURE_ID and norm['fit_dates'] == TRAIN and norm['TUNE_fitted'] is False
         and norm['sealed_fitted'] is False and norm['record_count'] == 12381191, 'Frozen TRAIN normalization scope differs')
    mean, std = np.asarray(norm['mean'], np.float64), np.asarray(norm['std'], np.float64)
    need(mean.shape == std.shape == (40,) and np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std > 0),
         'Frozen normalization numerical domain differs')
    return files


def verify_protected(reg):
    inventory = bound(reg['protected_inventory_binding'])
    for ref in inventory['files']:
        # The new618-file inventory extends the original532 identities with
        # failed-phase evidence. Its sole admitted F32 is the saved golden
        # input named exactly by registration; it is hashed opaquely here.
        path = local(ref['path'])
        need(path.suffix.lower() not in ('.f32', '.u8', '.u64', '.bin', '.npy')
             or (path.suffix.lower() == '.f32' and same(ref, reg['saved_golden_input_binding'])),
             'Forbidden original stream in inventory')
        checked(ref)
    snapshot = checked(reg['budget_snapshot_binding'])
    need(digest(ORIGINAL_STAGE.parent / 'budget.json') == digest(snapshot), 'Closed campaign budget changed')
    baseline = ROOT / 'artifacts/tcn_frozen_baseline_wn42_2026-10-17'
    readonly = [p for p in (baseline / 'bundle').rglob('*') if p.is_file()]
    readonly.append(baseline / 'TCN_WN42_Frozen_Baseline_20261017.zip')
    need(len(readonly) == 45 and all(getattr(p.stat(), 'st_file_attributes', 0) & 1 for p in readonly),
         'Original baseline read-only identity differs')
    return len(inventory['files']), len(readonly)


def compare_stated(stated, actual):
    need(stated['passed'] is True and stated['class_agreement'] == actual['class_agreement'] == 1.0
         and math.isfinite(stated['max_absolute_difference'])
         and abs(stated['max_absolute_difference'] - actual['max_absolute_difference']) <= 1e-15,
         'Producer comparison differs from independent new-artifact arithmetic')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args(argv); started = time.perf_counter()
    # Shared code is used only for immutable authorization/admission/reservation;
    # all artifact parsing, SHA and numerical comparisons below are independent.
    import tcn_existing_data_ed7_compat_contract as contract
    reg = contract.admit(args.registration_sha256, 'audit'); reservation = contract.reserve(reg, 'audit')
    protected_before = verify_protected(reg)
    failed, failed_cli = original_failure(reg, contract.OPERATION_KEYS)
    primary_cli, primary = receipt('primary', reg); reload_cli, reload_value = receipt('reload', reg)
    need(datetime.fromisoformat(failed_cli['ended_utc']) <= datetime.fromisoformat(primary_cli['started_utc'])
         and datetime.fromisoformat(primary_cli['ended_utc']) <= datetime.fromisoformat(reload_cli['started_utc']),
         'Reload did not follow actual primary completion')
    need(primary['operations'] == reg['primary_budget'] == contract.PRIMARY_BUDGET
         and reload_value['operations'] == reg['reload_budget'] == contract.RELOAD_BUDGET
         and primary['tolerance'] == reload_value['tolerance'] == reg['tolerance'] == {'atol': 1e-5, 'rtol': 1e-4},
         'Exact finite operation budgets/tolerance differ')
    prior, after_primary, cumulative = cumulative_accounting(failed['operations_observed'],
        primary['operations'], reload_value['operations'], contract.OPERATION_KEYS)
    need(prior == contract.FAILED_PRIMARY_OPERATIONS == reg['failed_primary_operations']
         and cumulative == contract.CUMULATIVE_BUDGET == reg['cumulative_budget']
         and primary['initial_failed_operations'] == reload_value['initial_failed_operations'] == prior
         and primary['cumulative_operations'] == after_primary
         and reload_value['cumulative_operations'] == cumulative,
         'Old failure plus new operations do not equal the exact cumulative budget')
    for accepted in (primary, reload_value):
        need(accepted['compatibility_case'] == 'ScriptModule_requires_grad_Module_API_unsupported',
             'Unexpected engineering compatibility case')
        for key in ('initial_failed_primary_binding', 'initial_primary_receipt_binding', 'original_registration_binding'):
            need(same(accepted[key], reg[key]), 'Compatibility result used different original failure evidence')
        for key in ('inputs', 'selection'):
            need(same(accepted['saved_golden_bindings'][key], reg['saved_golden_' + ('input' if key == 'inputs' else key) + '_binding']),
                 'Saved golden result binding differs')
    manifest_ref = primary['package_manifest_binding']; need(same(manifest_ref, reload_value['package_manifest_binding'])
         and checked(manifest_ref) == PACKAGE / 'package_manifest.json', 'Primary/reload package identity differs')
    manifest = bound(manifest_ref); files = validate_manifest(manifest, reg)
    need(same(reload_value['primary_acceptance_binding'], binding(STAGE / 'primary_acceptance.json'))
         and same(reload_value['portable_module_source_binding'], binding(PACKAGE / 'runtime.py')),
         'Reload is not tied to this original package/primary result')
    plan = reg['window_plan']; validate_window_plan(plan)
    cache_metadata = bound(reg['cache_manifest_binding'])  # small JSON only
    original_boundaries = {(ex['day'], ex['endpoint']): ex['normalized_input_float32_sha256']
                          for ex in cache_metadata['boundary_examples'] if ex['day'] in TRAIN}
    need(len(original_boundaries) == 15, 'Old TRAIN boundary metadata changed')
    x = read_array(PACKAGE / 'golden/inputs.f32', '<f4', (32, 40, 100))
    selection = read(PACKAGE / 'golden/selection.json')
    need(selection['version'] == 'TCN-ED7-golden-selection-1.0' and selection['windows'] == 32
         and primary['golden']['windows'] == reload_value['golden_windows'] == 32
         and selection['per_input'] == primary['golden']['per_input']
         and selection['input_sha256'] == primary['golden']['input_sha256'] == digest(PACKAGE / 'golden/inputs.f32'),
         'Golden input/selection identity differs')
    hashes, matches = validate_saved_selection(selection, x, plan)
    for index, item in enumerate(plan):
        row = selection['per_input'][index]
        need(row['index'] == index and all(row[key] == value for key, value in item.items()), 'Golden registered provenance differs')
        value = hashlib.sha256(np.ascontiguousarray(x[index], dtype='<f4').tobytes()).hexdigest()
        need(row['input_sha256'] == value, 'Persisted normalized golden input SHA differs')
        expected = item['expected_input_sha256']
        if expected is not None:
            need(expected == original_boundaries[(item['date'], item['endpoint'])] == value
                 and row['boundary_match'] is True, 'Old TRAIN golden boundary differs')
        else:
            need(row['boundary_match'] is None, 'Unreferenced window falsely called old golden')
    need(matches == 15, 'Original TRAIN boundary count differs')
    need(primary['source_cache_access']['windows'] == primary['source_cache_access']['bytes'] == 0
         and primary['source_cache_access']['complete_files_hashed'] == 0
         and primary['source_cache_access']['reused_saved_normalized_inputs'] is True,
         'Source cache was reopened by the compatibility phase')
    stats = primary['saved_golden_file_stats_before']
    need(stats == primary['saved_golden_file_stats_after'] and set(stats) == {'inputs', 'selection'},
         'Saved golden stat declarations changed')
    saved_copies = primary['saved_golden_copy_receipts']
    need(len(saved_copies) == 2, 'Saved golden copy evidence missing')
    for key, copied, relative in zip(('inputs', 'selection'), saved_copies, ('golden/inputs.f32', 'golden/selection.json')):
        source = reg['saved_golden_' + ('input' if key == 'inputs' else key) + '_binding']
        info = checked(source).stat()
        actual_stat = {'path': str(checked(source)), 'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns,
                       'inode': info.st_ino, 'device': info.st_dev}
        need(stats[key] == actual_stat and copied['relative'] == relative
             and same(copied['source'], source) and checked(copied['target']) == package_path(relative)
             and source['sha256'] == copied['target']['sha256']
             and source['bytes'] == copied['target']['bytes'], 'Saved golden byte copy/stat identity differs')
    copies = primary['package_copy_receipts']; need(len(copies) == len(reg['package_copies']), 'Package copy receipt count differs')
    for requested, copied in zip(reg['package_copies'], copies):
        need(copied['relative'] == requested['target'] and same(copied['source'], requested['source'])
             and checked(copied['target']) == package_path(requested['target'])
             and copied['target']['sha256'] == requested['source']['sha256']
             and copied['target']['bytes'] == requested['source']['bytes'], 'Opaque package copy differs')
    native_hashes = [primary[key] for key in ('source_checkpoint_model_state_sha256', 'native_state_sha256_before', 'native_state_sha256_after')]
    effective_hashes = [primary[key] for key in ('effective_state_sha256_before', 'effective_state_sha256_after',
        'script_state_sha256_before', 'script_state_sha256_after', 'loaded_script_state_sha256_before', 'loaded_script_state_sha256_after')]
    effective_hashes += [reload_value[key] for key in ('loaded_script_state_sha256_before', 'loaded_script_state_sha256_after')]
    need(len(set(native_hashes)) == len(set(effective_hashes)) == 1
         and all(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
                 for value in native_hashes + effective_hashes), 'Model state fingerprint groups differ')
    need(primary['parameter_counts'] == {'native': 33539, 'effective': 33219, 'script': 33219}
         and primary['state_tensor_counts'] == {'native': 34, 'effective': 24, 'script': 24, 'loaded_script': 24}
         and len(primary['folded_weights']) == len({f['state_key'] for f in primary['folded_weights']}) == 10,
         'Native/folded architecture or exactly ten folds differ')
    need(primary['checkpoint_sha256_before'] == primary['checkpoint_sha256_after'] == reg['checkpoint_binding']['sha256']
         and primary['primary_script_sha256'] == files['model_cpu.ts']['sha256'], 'Opaque native/script identities differ')
    values = {}; comparisons = {}; tolerance = reg['tolerance']
    for batch in (32, 1):
        for kind in ('native', 'effective', 'script'):
            values[(kind, batch)] = read_array(PACKAGE / f'golden/{kind}_batch{batch}.f32', '<f4', (batch, 3))
        for name, left, right in (('native_effective', 'native', 'effective'), ('native_script', 'native', 'script'),
                                  ('effective_script', 'effective', 'script')):
            measured = compare_outputs(values[(left, batch)], values[(right, batch)], **tolerance)
            compare_stated(primary['comparisons']['batch' + str(batch)][name], measured)
            compare_stated(primary['golden']['batch' + str(batch)][name], measured)
            comparisons[f'primary_batch{batch}_{name}'] = measured
    need(checked(reload_value['output_binding']) == STAGE / 'reload_logits.f32', 'Reload output path differs')
    reloaded = read_array(STAGE / 'reload_logits.f32', '<f4', (32, 3))
    for name, kind in (('native_reloaded', 'native'), ('previous_script_reloaded', 'script')):
        measured = compare_outputs(values[(kind, 32)], reloaded, **tolerance)
        compare_stated(reload_value[name], measured); compare_stated(reload_value['comparisons'][name], measured)
        comparisons[name] = measured
    for kind in ('native', 'effective', 'script'):
        comparisons[kind + '_batch1_vs_batch32_first'] = compare_outputs(values[(kind, 32)][:1], values[(kind, 1)], **tolerance)
    # Re-hash only new package payloads after arithmetic; never the five source
    # row caches. Copies of native CP remain opaque throughout.
    validate_manifest(bound(manifest_ref), reg); checked(reload_value['output_binding'])
    protected_after = verify_protected(reg); need(protected_before == protected_after, 'Historical protection changed')
    result = {'version': 'TCN-ED7-compatibility-v3-independent-artifact-audit-1.0', 'passed': True, 'status': 'completed', 'phase': 'audit',
        'worker_pid': os.getpid(), 'completed_utc': datetime.now(timezone.utc).isoformat(), 'seconds': time.perf_counter() - started,
        'registration_binding': reg['_registration_binding'], 'reservation': reservation,
        'source_binding': binding(__file__), 'operations': dict.fromkeys(contract.OPERATION_KEYS, 0),
        'verified_primary_operations': primary['operations'], 'verified_reload_operations': reload_value['operations'],
        'verified_initial_failed_operations': prior, 'verified_cumulative_operations': cumulative,
        'cumulative_operation_accounting_passed': True, 'original_failure_preserved': True,
        'initial_failed_primary_binding': reg['initial_failed_primary_binding'],
        'initial_primary_receipt_binding': reg['initial_primary_receipt_binding'],
        'original_registration_binding': reg['original_registration_binding'],
        'saved_golden_bindings': primary['saved_golden_bindings'], 'saved_golden_input_reused': True,
        'saved_golden_decode_bytes_in_new_primary': 512000, 'source_cache_reopened': False,
        'verified_total_forward_calls': 7, 'verified_total_forward_samples': 131,
        'golden_input_windows': 32, 'old_TRAIN_boundary_matches': matches, 'per_input_sha256': hashes,
        'comparisons': comparisons, 'all_class_agreement': all(v['class_agreement'] == 1. for v in comparisons.values()),
        'maximum_absolute_difference': max(v['max_absolute_difference'] for v in comparisons.values()),
        'native_state_sha256': native_hashes[0], 'effective_script_state_sha256': effective_hashes[0],
        'state_fingerprints_unchanged': True, 'primary_receipt_binding': binding(STAGE / 'primary_execution_receipt.json'),
        'reload_receipt_binding': binding(STAGE / 'reload_execution_receipt.json'), 'package_manifest_binding': manifest_ref,
        'protected_files': protected_after[0], 'baseline_readonly_files': protected_after[1],
        'original_source_cache_payload_reads': 0, 'E_record_reads': 0, 'label_reads': 0, 'DEV_or_historical_stream_reads': 0,
        'model_constructions': 0, 'checkpoint_deserializations': 0, 'torchscript_loads': 0, 'forward_calls': 0,
        'new_market_accuracy': None, 'new_independent_test': False, 'raw_market_end_to_end_verified': False,
        'scope': 'Compatibility-only export using the previously saved32 normalized TRAIN inputs. The consumed original failed phase remains included in cumulative operations. No source cache reopened. Raw preprocessing has synthetic evidence only; historical model quality and data limitations remain.'}
    put(STAGE / 'audit_acceptance.json', result)
    print(json.dumps({'phase': 'audit', 'passed': True, 'worker_pid': os.getpid(), 'acceptance': binding(STAGE / 'audit_acceptance.json')},
                     ensure_ascii=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
