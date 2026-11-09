"""Independent ED6 per-trial D-only audit; no torch/model/raw-E access.

The root executes each slot exactly once after its original training and
bounded two-container CPU validation actually finish. Terminal budget evidence
is the immutable per-trial snapshot, never a legally stale live-budget SHA.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parent
CAMPAIGN = STAGE.parent
PARENT = CAMPAIGN / 'step04_features'
ACTIVE_TRIAL = STAGE
TRAIN = [f'2025-12-{i:02}' for i in range(8,13)]
DEV = ['2025-12-15','2025-12-16']
SLOTS = ('control43','candidate43')
PROTOCOL_SHA = '09654e6dfdb10b69feb6ee950c6668e71640bcea1d56fa9691a9564c8641fcf3'
P2_NORM_SHA = '60130df9c2ed5673395e0247dcc8279f297a288719ec9187097ad4823a0a8c0b'
CHUNK=131072



def require(value, message):
    if not value:
        raise ValueError(message)


def path_in_workspace(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require('..' not in path.parts and path.is_relative_to(ROOT), 'Outside D workspace')
    for part in (path, *path.parents):
        if part.exists():
            require(not part.is_symlink() and not getattr(part.lstat(), 'st_file_attributes', 0) & 0x400,
                    'Reparse audit input forbidden')
        if part == ROOT:
            break
    require(path.resolve().is_relative_to(ROOT.resolve()), 'Resolved audit input escaped D workspace')
    return path


def digest(path):
    path = path_in_workspace(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = path_in_workspace(path)
    return {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}


def checked(ref):
    require(isinstance(ref, dict) and {'path', 'sha256'} <= set(ref), 'Missing binding')
    path = path_in_workspace(ref['path'])
    require(path.is_file() and digest(path) == ref['sha256'], 'Audit SHA mismatch: ' + str(path))
    require('bytes' not in ref or path.stat().st_size == ref['bytes'], 'Audit size mismatch')
    return path


def read_json(path):
    path = path_in_workspace(path)
    require(path.stat().st_size <= 32 * 1024**2, 'Oversized audit metadata')
    return json.loads(path.read_text(encoding='utf-8-sig'),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))


def bound_json(ref):
    return read_json(checked(ref))


def metrics(matrix, ce_sum=None):
    a = np.asarray(matrix)
    require(a.shape == (3, 3) and a.dtype.kind in 'iu' and np.all(a >= 0), 'Invalid integer matrix')
    a = a.astype(np.int64)
    total = int(a.sum()); require(total > 0, 'Empty matrix')
    support = a.sum(1); predicted = a.sum(0); diagonal = a.diagonal()
    precision = [float(Fraction(int(diagonal[i]), int(predicted[i]))) if predicted[i] else 0. for i in range(3)]
    recall = [float(Fraction(int(diagonal[i]), int(support[i]))) if support[i] else 0. for i in range(3)]
    f1 = [Fraction(2 * int(diagonal[i]), int(support[i] + predicted[i]))
          if support[i] + predicted[i] else Fraction(0) for i in range(3)]
    rank = (Fraction(int(diagonal.sum()), total), sum(f1, Fraction(0)) / 3)
    result = {'samples': total, 'accuracy': float(rank[0]), 'macro_f1': float(rank[1]),
              'precision': precision, 'recall': recall, 'f1': [float(v) for v in f1],
              'support': support.tolist(), 'confusion_true_rows_predicted_columns': a.tolist()}
    if ce_sum is not None:
        result['loss'] = float(ce_sum) / total
    return result, rank


def compare_metrics(stated, actual, message, loss=True):
    require(stated['samples'] == actual['samples'] and stated['support'] == actual['support']
            and stated['confusion_true_rows_predicted_columns'] == actual['confusion_true_rows_predicted_columns'],
            message + ': integer counts')
    for key in ('accuracy', 'macro_f1', 'precision', 'recall', 'f1'):
        require(np.allclose(stated[key], actual[key], rtol=0, atol=3e-15), message + ': ' + key)
    if loss:
        require(math.isfinite(stated['loss']) and abs(stated['loss'] - actual['loss']) <= 2e-12,
                message + ': CE')


def read_array(ref, expected_dtype, expected_shape, admitted_root=None):
    require(ref.get('dtype') == expected_dtype and ref.get('shape') == list(expected_shape),
            'Cache dtype/shape differs')
    path = checked(ref)
    admitted_root = ACTIVE_TRIAL if admitted_root is None else admitted_root
    require(path.is_relative_to(admitted_root), 'Numerical cache outside its explicitly admitted stage')
    require(path.stat().st_size == int(np.prod(expected_shape)) * np.dtype(expected_dtype).itemsize,
            'Cache payload size/shape differs')
    return np.memmap(path, mode='r', dtype=expected_dtype, shape=expected_shape)


def verify_protected():
    inv = read_json(STAGE / 'protected_inventory.json')
    for ref in inv['files']:
        checked(ref)
    require(len(inv['files']) == inv['protected_files'], 'Protected inventory count differs')
    ro = []
    baseline = ROOT / 'artifacts/tcn_frozen_baseline_wn42_2026-10-17'
    candidates = list((baseline / 'bundle').rglob('*')) + [baseline / 'TCN_WN42_Frozen_Baseline_20261017.zip']
    for path in candidates:
        if path.is_file():
            require(getattr(path.stat(), 'st_file_attributes', 0) & 1, 'Baseline read-only flag lost')
            ro.append(path.relative_to(ROOT).as_posix())
    require(len(ro) == 45, 'Frozen baseline file count differs')
    return {'passed': True, 'protected_files': len(inv['files']), 'baseline_readonly_files': len(ro)}


def inspect_epoch(row, data, baseline_cache, payload_refs):
    cache = bound_json(row['cache_binding'])
    require(cache['version'] == 'TCN-ED6-epoch-DEV-cache-1.0'
            and cache['epoch'] == row['epoch'] and cache['decision'] == 'original_argmax'
            and cache['class_order'] == ['down', 'stationary', 'up'] and set(cache['per_day']) == set(DEV),
            'Epoch cache scope/decision differs')
    combined = np.zeros((3, 3), np.int64); total_ce = 0.; by_day = {}
    for day in DEV:
        item = cache['per_day'][day]; n = item['samples']; original = data['day_label_reports'][day]
        require(n == original['window_count'], 'Epoch denominator differs')
        refs = item['files']; require(set(refs) == {'logits', 'labels', 'anchors'}, 'Unexpected DEV payload set')
        require(refs['labels']['sha256'] == original['label_sha256']
                and refs['anchors']['sha256'] == baseline_cache['per_day'][day]['files']['anchors']['sha256'],
                'Candidate and original WN42 do not share identical ordered labels/anchors')
        z = y = anchors = None
        try:
            z = read_array(refs['logits'], '<f4', (n, 3)); y = read_array(refs['labels'], 'u1', (n,))
            anchors = read_array(refs['anchors'], '<u8', (n,))
            ranges = [r for r in data['ranges'] if r['date'] == day]
            require(len(ranges) == 1, 'Unexpected multi-range day')
            r = ranges[0]; c = np.zeros((3, 3), np.int64); ce = 0.
            for begin in range(0, n, CHUNK):
                stop = min(begin + CHUNK, n); a = z[begin:stop].astype(np.float64); lab = y[begin:stop].astype(np.int64)
                require(np.isfinite(a).all() and np.all((0 <= lab) & (lab <= 2)), 'Nonfinite/invalid DEV cache')
                require(np.array_equal(anchors[begin:stop], np.arange(r['endpoint_start'] + begin,
                        r['endpoint_start'] + stop, dtype='<u8')), 'DEV ordered anchor footprint differs')
                # Stable descending argsort gives the lowest class index at ties,
                # independently of the producer's numpy argmax implementation.
                pred = np.argsort(-a, axis=1, kind='stable')[:, 0]
                np.add.at(c, (lab, pred), 1)
                losses = np.logaddexp.reduce(a, axis=1) - a[np.arange(len(a)), lab]
                require(np.isfinite(losses).all() and np.all(losses >= -1e-14), 'Invalid independently reconstructed CE')
                ce += float(losses.sum(dtype=np.float64))
            m, _ = metrics(c, ce)
            compare_metrics(item['metrics'], m, f'Epoch{row["epoch"]}/{day} cache')
            compare_metrics(row['development_validation']['per_day'][day], m, f'Epoch{row["epoch"]}/{day} history')
            by_day[day] = m; combined += c; total_ce += ce
            payload_refs.extend(refs.values())
        finally:
            for value in (z, y, anchors):
                if value is not None:
                    value._mmap.close()
    pooled, rank = metrics(combined, total_ce)
    compare_metrics(row['development_validation'], pooled, f'Epoch{row["epoch"]} pooled')
    pooled['per_day'] = by_day
    return pooled, rank


def receipt(phase, trial):
    name = 'checkpoint_execution_receipt.json' if phase == 'checkpoint_validation' else phase + '_execution_receipt.json'
    result = read_json(trial / name)
    require(result.get('phase') == phase and result.get('slot_id') == trial.name
            and result.get('exit_code') == 0 and result.get('entry_unchanged') is True
            and result['entry_before'] == result['entry_after'], 'Actual successful trial CLI missing: ' + phase)
    for field in ('entry_before', 'input', 'stdout', 'stderr', 'capture_source'):
        checked(result[field])
    if 'sources_before' in result:
        require(result['sources_before'] == result['sources_after'], 'Postrun CLI sources changed')
        for ref in result['sources_before']:
            checked(ref)
    return result


def inspect_budget(budget, prefix, reservation_id, n_epoch, slot):
    require(slot in SLOTS and type(n_epoch) is int and 1 <= n_epoch <= 10, 'Invalid paired slot/epoch')
    names = ['feature42', 'strategy42', 'control43'] + (['candidate43'] if slot == 'candidate43' else [])
    rows = budget['reserved_runs']; old = prefix['reserved_runs']
    require([r.get('slot_id', r.get('id')) for r in rows] == names and rows[:-1] == old,
            'Complete immutable reservation prefix/order changed')
    for index, row in enumerate(rows):
        require(row['status'] == 'completed' and row['seed'] == (42 if index < 2 else 43)
                and type(row['consumed_epochs']) is int and 1 <= row['consumed_epochs'] <= 10
                and row['optimizer_updates'] == 48362 * row['consumed_epochs'], 'Prior or new terminal row invalid')
    require(old[0]['consumed_epochs'] == 6 and old[0]['optimizer_updates'] == 290172
            and old[1]['consumed_epochs'] == 5 and old[1]['optimizer_updates'] == 241810,
            'Completed seed42 budget prefix differs')
    updates = 48362 * n_epoch; current = rows[-1]
    require(current['reservation_id'] == reservation_id and current['consumed_epochs'] == n_epoch
            and current['optimizer_updates'] == updates
            and current['actual_optimizer_steps_completed_last_snapshot'] == updates,
            'Actual paired terminal reservation counters differ')
    for value in (budget, prefix):
        require(value['optimizer_updates'] == sum(r['optimizer_updates'] for r in value['reserved_runs'])
                and value['consumed_epochs'] == sum(r['consumed_epochs'] for r in value['reserved_runs']),
                'Global cumulative counters differ from complete rows')
        for field, expected in [('maximum_runs', 4), ('maximum_epochs_per_run', 10),
                                ('maximum_complete_epochs', 40), ('maximum_optimizer_updates', 1934480),
                                ('maximum_optimizer_updates_per_run', 483620)]:
            require(value[field] == expected, 'Permanent campaign cap changed')
        require(value['optimizer_updates'] <= 1934480 and value['consumed_epochs'] <= 40, 'Budget exceeded')
    return {'passed': True, 'all_prior_records_unchanged': True, 'new_slot': slot,
            'prior_runs': len(old), 'new_complete_epochs': n_epoch, 'new_optimizer_updates': updates,
            'global_complete_epochs': budget['consumed_epochs'], 'global_optimizer_updates': budget['optimizer_updates']}


def immutable_budget(acceptance, registration):
    ref = acceptance['budget_completion_snapshot_binding']
    require(ref['sha256'] == acceptance['budget_binding']['sha256'],
            'Terminal immutable budget does not match recorded terminal live identity')
    require(acceptance['budget_prefix_snapshot_binding'] == registration['budget_prefix_snapshot_binding']
            == acceptance['identity']['budget_prefix_snapshot_binding'], 'Budget prefix identity differs')
    # Do not open acceptance.budget_binding: its recorded live path legitimately
    # changes when the next pre-registered slot is appended.
    return bound_json(ref), bound_json(registration['budget_prefix_snapshot_binding'])


def inspect_prepared(data, prepared, slot):
    require(prepared['complete'] is True and set(prepared['cache_files']) == set(TRAIN + DEV)
            and prepared['day_label_reports'] == data['day_label_reports'], 'Expanded/incomplete row cache')
    require(prepared['source_records_unchanged'] is True and prepared['label_streams_unchanged'] is True
            and prepared['source_sha256_before'] == prepared['source_sha256_after']
            == {d: data['sources'][d]['records_sha256'] for d in TRAIN + DEV}
            and prepared['labels_sha256_before'] == prepared['labels_sha256_after']
            == {d: data['day_label_reports'][d]['label_sha256'] for d in TRAIN + DEV},
            'Preparation source/label declared identity differs')
    require(prepared['sealed_records_opened'] is False and prepared['sealed_labels_or_scores_opened'] is False
            and prepared['original_flag64_cleared'] is False and prepared['E_drive_writes'] == 0,
            'Preparation scope differs')
    norm = bound_json(prepared['normalization_binding'])
    require(norm == prepared['normalization'] and norm['fit_dates'] == TRAIN and norm['record_count'] == 12381191,
            'Normalization fit scope differs')
    if slot == 'control43':
        require(prepared['version'] == 'TCN-ED6-original-row-cache-1.0'
                and prepared['feature_id'] == 'original_physical_prices_raw_quantities_frozen_P2_norm_v1'
                and prepared['normalization_binding']['sha256'] == P2_NORM_SHA
                and norm == data['normalization'] and prepared['normalization_refitted'] is False
                and prepared['original_P2_normalization_reused'] is True
                and prepared['boundary_inputs_byte_equal_original_P2'] is True,
                'Original physical control/norm identity differs')
        base = STAGE / 'prepared_control'; source = ROOT / 'scripts/tcn_existing_data_ed6_data.py'
    else:
        require(prepared['version'] == 'TCN-ED4-relative-row-cache-1.0'
                and prepared['feature_id'] == 'same_record_mid_relative_prices_raw_quantities_v1'
                and prepared['normalization_binding']['sha256']
                == '709114d09b2dd581bd9481e180973fdf1553452b644d24dcb211da7ea394caaf',
                'Inherited relative candidate identity differs')
        require(norm['TUNE_fitted'] is False and norm['sealed_fitted'] is False, 'Relative normalization scope differs')
        base = PARENT / 'prepared'; source = ROOT / 'scripts/tcn_existing_data_ed4_features.py'
    require(checked(prepared['code_binding']) == source, 'Row-cache preparation source differs')
    require(bound_json(prepared['statistics_independent_review_binding']) == prepared['statistics_independent_review']
            and prepared['statistics_independent_review']['passed'] is True
            and prepared['statistics_independent_review']['TUNE_fitted'] is False
            and prepared['statistics_independent_review']['sealed_fitted'] is False, 'Inherited statistics acceptance missing')
    require(len(prepared['ranges']) == len(data['ranges']) == 7, 'Range count changed')
    for cached, original in zip(prepared['ranges'], data['ranges']):
        require({k: v for k, v in cached.items() if k != 'cache_offset'} == original
                and cached['cache_offset'] == 0, 'Row-cache range/anchor footprint differs')
    maps = {}; refs = []; labels = []
    try:
        for day in TRAIN + DEV:
            ref = prepared['cache_files'][day]
            require(ref['role'] == ('train' if day in TRAIN else 'tuning_validation'), 'Row-cache role differs')
            n = sum(r['stop_row'] - r['start_row'] for r in data['ranges'] if r['date'] == day)
            maps[day] = read_array(ref, '<f4', (n, 40), base); refs.append(ref)
            item = data['day_label_reports'][day]
            labels.append(identity(checked({'path': item['label_path'], 'sha256': item['label_sha256'],
                                            'bytes': item['label_bytes']})))
        examples = prepared['boundary_examples']; require(len(examples) == 21, 'Boundary reference count differs')
        for day in TRAIN + DEV:
            r = next(r for r in data['ranges'] if r['date'] == day)
            chosen = [x for x in examples if x['day'] == day]
            expected_ends = [r['endpoint_start'], r['endpoint_start'] + 1, r['endpoint_stop_exclusive'] - 1]
            require([x['endpoint'] for x in chosen] == expected_ends, 'Boundary endpoint set differs')
            for ex in chosen:
                start, stop = ex['cache_row_start'], ex['cache_row_stop_exclusive']
                require(start == ex['endpoint'] - 99 - r['start_row'] and stop == start + 100
                        and ex['input_within_same_registered_range'] is True, 'Boundary range geometry differs')
                x = np.ascontiguousarray(maps[day][start:stop].T)
                value = hashlib.sha256(x.tobytes()).hexdigest()
                require(value == ex['normalized_input_float32_sha256'], 'Persisted row-cache X differs')
                if slot == 'control43':
                    originals = [v for v in data['day_label_reports'][day]['first_second_last_window_examples']
                                 if v['endpoint'] == ex['endpoint']]
                    require(len(originals) == 1 and value == ex['original_P2_input_float32_sha256']
                            == originals[0]['input_float32_sha256'], 'Original P2 golden X differs')
        return {'passed': True, 'boundary_windows_reconstructed': 21, 'unique_TRAIN_rows': 12381191,
                'normalization_refitted': False, 'full_TRAIN_moment_recomputation_in_ED6': False,
                'row_cache_files': refs, 'original_labels': labels, 'norm_binding': prepared['normalization_binding']}
    finally:
        for value in maps.values():
            value._mmap.close()


def validate_trial_identity(acceptance, reg, slot):
    require(acceptance['slot_id'] == acceptance['identity']['slot_id'] == reg['slot_id'] == slot,
            'Acceptance trial slot differs')
    for field in ('selection_lock_binding', 'budget_prefix_snapshot_binding', 'normalization_binding'):
        expected = reg[field] if field != 'normalization_binding' else acceptance[field]
        require(acceptance[field] == acceptance['identity'][field] == expected, 'Trial identity differs: ' + field)
    require(acceptance['paired_seed43_initialization_matches_control'] is (True if slot == 'candidate43' else None),
            'Paired seed43 initialization declaration differs')
    require(acceptance['identity']['seed'] == 43, 'Trial seed differs')


def main(argv=None):
    global ACTIVE_TRIAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slot', choices=SLOTS, required=True)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args(argv); slot = args.slot; trial = STAGE / 'trials' / slot; ACTIVE_TRIAL = trial
    started = time.perf_counter(); reg_path = trial / 'training_registration.json'
    require(digest(reg_path) == args.registration_sha256, 'Explicit training registration differs')
    reg = read_json(reg_path)
    require(reg['step'] == 'ED6' and reg['slot_id'] == slot, 'Wrong paired audit phase')
    require(not (trial / 'audit_phase_reservation.json').exists() and not (trial / 'readonly_audit.json').exists(),
            'Trial audit already reserved; no retry')
    with (trial / 'audit_phase_reservation.json').open('x', encoding='utf-8') as stream:
        json.dump({'registration_binding': identity(reg_path), 'worker_pid': os.getpid(), 'slot_id': slot,
                   'permanent_single_use': True, 'scope': 'D-only arithmetic; no ML'}, stream, indent=2); stream.write('\n')
    protocol = bound_json(reg['protocol_binding']); lock = bound_json(reg['selection_lock_binding'])
    require(reg['protocol_binding']['sha256'] == PROTOCOL_SHA and protocol['campaign_id'] == 'TCN_EDO_20261017'
            and protocol['data']['train_dates'] == TRAIN and protocol['data']['development_dates'] == DEV,
            'Protocol or date scope differs')
    require(lock['version'] == 'TCN-ED6-selection-lock-1.0' and lock['branch'] == 'trained_candidate'
            and lock['candidate_family'] == 'ED4_feature42_lr0.001' and lock['replication_seed'] == 43
            and lock['representative_seed'] == 42 and lock['decision'] == 'original_argmax'
            and lock['stationary_logit_bias'] == 0 and lock['window_length'] == 100 and lock['horizon'] == 50
            and lock['common_DEV_samples'] == 6612655 and lock['source_bindings'] == reg['source_bindings']
            and lock['config_bindings'][slot] == reg['config_binding']
            and lock['feature_cache_bindings'][slot] == reg['feature_cache_binding']
            and lock['prepared_manifest_binding'] == reg['prepared_manifest_binding'] == protocol['data']['source_manifest'],
            'Frozen paired family/cache/config lock differs')
    require(len(reg['source_bindings']) == 7, 'Numerical source set expanded')
    for name, expected in reg['source_bindings'].items():
        require(digest(ROOT / name) == expected, 'Frozen numerical source changed')
    checked(reg['source_review_binding']); checked(reg['test_evidence_binding'])
    data = bound_json(reg['prepared_manifest_binding']); prepared = bound_json(reg['feature_cache_binding'])
    config = bound_json(reg['config_binding'])
    require(config['slot_id'] == slot and config['seed'] == 43 and config['learning_rate'] == .001
            and config['window_length'] == 100 and config['horizon'] == 50
            and config['decision'] == 'original_argmax' and config['weight_norm'] is True,
            'Paired training configuration differs')
    require(prepared['normalization_binding'] == lock['normalization_bindings'][slot], 'Locked norm differs')
    prep_cli = bound_json(reg['preparation_receipt_binding'])
    require(prep_cli['phase'] == 'preparation' and prep_cli['exit_code'] == 0 and prep_cli['entry_unchanged'] is True
            and prep_cli['entry_before'] == prep_cli['entry_after'], 'Actual accepted preparation missing')
    for item in ('entry_before', 'stdout', 'stderr', 'capture_source', 'input'):
        checked(prep_cli[item])
    require(prep_cli['outputs']['feature_cache_manifest'] == reg['feature_cache_binding'],
            'Actual preparation output is not this locked cache')
    prep_reg = bound_json(prep_cli['input'])
    require(prep_reg['prepared_manifest_binding'] == reg['prepared_manifest_binding']
            and prep_reg['protocol_binding']['sha256'] == PROTOCOL_SHA, 'Actual cache parent differs')
    before_protected = verify_protected(); train_cli = receipt('training', trial)
    require(train_cli['input']['sha256'] == args.registration_sha256, 'Training used another registration')
    acceptance = bound_json(train_cli['outputs']['training_acceptance']); history = bound_json(train_cli['outputs']['history'])
    require(acceptance['status'] == 'completed' and 1 <= len(history) <= 10
            and acceptance['epochs_completed'] == len(history), 'Actual training incomplete')
    require(acceptance['development_validation_only'] is True and acceptance['sealed_dates_opened'] is False
            and acceptance['original_flag64_cleared'] is False and acceptance['source_control_verified'] is False
            and acceptance['coverage_unknown'] is True and acceptance['original_argmax_without_ED3_bias'] is True
            and acceptance['automatic_next_action'] is False, 'Result scope differs')
    require(acceptance['worker_pid'] == acceptance['actual_worker_pid'] == train_cli['actual_worker_pid'], 'Worker roles differ')
    validate_trial_identity(acceptance, reg, slot)
    require(acceptance['identity']['registration_binding'] == identity(reg_path)
            and acceptance['normalization_binding'] == prepared['normalization_binding']
            and acceptance['identity']['source_bindings'] == reg['source_bindings'], 'Acceptance identity chain differs')
    initialization = read_json(trial / 'run/initialization.json')
    require(initialization['seed'] == 43 and initialization['fresh_cold_start'] is True
            and initialization['historical_checkpoint_loaded'] is False and initialization['all_gradients_none'] is True
            and initialization['parameter_count'] == 33539, 'Fresh seed43 initialization differs')
    if slot == 'candidate43':
        control_init = bound_json(reg['control_initialization_binding'])
        require(initialization['model_state_sha256'] == control_init['model_state_sha256']
                and control_init['seed'] == 43, 'Paired cold initialization differs')
    preparation_check = inspect_prepared(data, prepared, slot)
    before_payloads = bound_json(acceptance['cache_payloads_before_binding'])
    require(before_payloads == bound_json(acceptance['cache_payloads_after_binding']) and before_payloads['passed'] is True,
            'Trainer row-cache payload before/after differs')
    for day in TRAIN + DEV:
        require(before_payloads['per_day'][day] == {'cache_sha256': prepared['cache_files'][day]['sha256'],
                'label_sha256': data['day_label_reports'][day]['label_sha256']}, 'Trainer payload evidence differs')
    baseline_path = CAMPAIGN / 'step02_diagnosis/evaluation/cache_manifest.json'
    require(digest(baseline_path) == '1808492b6b2e118b09a88114bf7af4a837414e1e96a41427fef832e08407c106',
            'Frozen WN42 metadata differs')
    baseline_cache = read_json(baseline_path)

    payload_refs = []; observations = []; best = None; selected = None; wait = 0
    for index, row in enumerate(history, 1):
        require(row['epoch'] == index and row['optimizer_updates'] == 48362
                and row['train_forward_calls'] == 48362 and row['development_forward_calls'] == 25831,
                'Incomplete/nonconsecutive epoch or ML call count differs')
        online = row['train']; om, _ = metrics(online['confusion_true_rows_predicted_columns'])
        compare_metrics(online, om, 'Online TRAIN count arithmetic', loss=False)
        require(online['scope'] == 'online_training_before_each_update' and om['samples'] == 12380446,
                'Online TRAIN misrepresented as fixed-model evaluation')
        measured, rank = inspect_epoch(row, data, baseline_cache, payload_refs)
        improved = best is None or rank > best
        wait = 0 if improved else wait + 1
        if improved:
            best, selected = rank, index
        require(row['improved'] is improved and row['wait'] == wait, 'Exact rank/patience replay differs')
        require(index == len(history) or wait < 3, 'Training continued beyond patience boundary')
        observations.append({'epoch': index, 'metrics': measured, 'improved': improved, 'wait': wait,
                             'exact_rank': [[x.numerator, x.denominator] for x in rank]})
    require(selected == acceptance['selected_epoch']
            and acceptance['selected_cache_binding'] == history[selected - 1]['cache_binding'], 'Selected artifact differs')
    require((wait >= 3 and acceptance['stop_reason'] == 'patience3')
            or (len(history) == 10 and wait < 3 and acceptance['stop_reason'] == 'max10'), 'Stop boundary differs')
    compare_metrics(acceptance['selected_development_metrics'], observations[selected - 1]['metrics'], 'Selected acceptance')
    n_epoch = len(history); updates = 48362 * n_epoch
    counters = acceptance['counters']
    expected_counts = {'models_constructed': 1, 'historical_checkpoint_loads': 0, 'optimizers_constructed': 1,
                       'forward_calls': (48362 + 25831) * n_epoch, 'backward_calls': updates,
                       'optimizer_attempts_charged': updates, 'optimizer_steps_completed': updates,
                       'train_windows': 12380446 * n_epoch, 'development_windows': 6612655 * n_epoch}
    require(counters == expected_counts, 'Actual call/update/window counters differ')
    require(acceptance['optimizer_attempts_charged'] == acceptance['optimizer_steps_completed'] == updates,
            'Charged attempts differ from actual completed steps')
    budget, prefix = immutable_budget(acceptance, reg)
    budget_check = inspect_budget(budget, prefix, reg['reservation_id'], n_epoch, slot)
    if slot == 'candidate43':
        control_accepted = bound_json(reg['control_acceptance_binding'])
        require(reg['budget_prefix_snapshot_binding']['sha256']
                == control_accepted['budget_completion_snapshot_binding']['sha256'],
                'Candidate prefix does not match completed immutable control budget')
    cps = acceptance['epoch_checkpoint_bindings']
    require(set(cps) == {str(i) for i in range(1, n_epoch + 1)}, 'Complete epoch checkpoint missing')
    for ref in cps.values():
        checked(ref)  # Opaque new checkpoint hashing; no deserialization.
    for ref in acceptance['checkpoint_bindings'].values():
        checked(ref)
    cp_cli = receipt('checkpoint_validation', trial)
    cp_result = bound_json(cp_cli['outputs']['checkpoint_validation'])
    require(cp_result['slot_id'] == slot and cp_result['passed'] is True and cp_result['actual_new_checkpoint_CPU_loads'] == 2
            and cp_result['models_constructed'] == cp_result['optimizers_constructed']
            == cp_result['forwards'] == cp_result['backwards'] == cp_result['optimizer_updates'] == 0
            and cp_result['historical_checkpoint_loads'] == 0 and cp_result['RNG_restored'] is False,
            'Bounded checkpoint validation failed or exceeded load limits')
    checked(cp_result['source_binding']); cp_registration = bound_json(cp_result['registration_binding'])
    require(cp_registration['training_registration_binding']['sha256'] == args.registration_sha256
            and cp_registration['checkpoint_bindings'] == acceptance['checkpoint_bindings']
            and cp_registration['training_receipt_binding']['sha256'] == digest(trial / 'training_execution_receipt.json'),
            'CPU validation did not use the original successful training/checkpoint identities')
    cp_observations = cp_result['observations']
    require([v['checkpoint_role'] for v in cp_observations] == ['best.pt', 'latest.pt']
            and cp_observations[0]['epoch'] == selected and cp_observations[1]['epoch'] == n_epoch
            and all(v['passed'] is True and v['parameter_numel'] == 33539 and v['forwards'] == 0
                    for v in cp_observations), 'CPU checkpoint observation identity differs')
    cleanup = read_json(trial / 'run/resource_cleanup.json')
    require(cleanup['worker_pid'] == acceptance['worker_pid'] and cleanup['cleanup_succeeded'] is True
            and cleanup['request_active_after_finally'] is False
            and cleanup['permanent_power_or_display_settings_changed'] is False and cleanup['run_error'] is None
            and cleanup['counters'] == counters, 'Actual finally cleanup failed/missing')
    require(cleanup['events'] and cleanup['events'][-1].get('final_cleanup') is True
            and cleanup['events'][-1]['returned_previous'] != 0, 'Final request-clear API evidence missing')
    for event in cleanup['events'][:-1]:
        require(event['returned_previous'] != 0 and event['temporary_system_idle_request']
                is (event['ACLineStatus'] == 1), 'AC-only resource request evidence differs')
    for ref in payload_refs + preparation_check['row_cache_files'] + preparation_check['original_labels']:
        checked(ref)
    after_protected = verify_protected(); require(after_protected == before_protected, 'Protected evidence changed')
    baseline_matrix = np.zeros((3, 3), np.int64)
    for day in DEV:
        baseline_matrix += np.array(baseline_cache['per_day'][day]['metrics']['confusion_true_rows_predicted_columns'])
    baseline_metrics, baseline_rank = metrics(baseline_matrix)
    parent_completion = bound_json(lock['frozen_parent_completion_binding'])
    require(parent_completion['passed'] is True and parent_completion['phase_id'] == 'ED4'
            and parent_completion['selected_epoch'] == 3
            and parent_completion['one_common_complete_DEV_anchor_set'] is True, 'Locked seed42 parent differs')
    parent_metrics = parent_completion['candidate_original_argmax_DEV_metrics']
    parent_cache = bound_json(parent_completion['selected_cache'])
    for day in DEV:
        require(parent_cache['per_day'][day]['files']['anchors']['sha256']
                == baseline_cache['per_day'][day]['files']['anchors']['sha256']
                and parent_cache['per_day'][day]['files']['labels']['sha256']
                == data['day_label_reports'][day]['label_sha256'], 'Frozen seed42 DEV anchors differ')
    candidate_metrics = observations[selected - 1]['metrics']
    result = {'version': 'TCN-ED6-independent-D-only-audit-1.0', 'passed': True, 'slot_id': slot,
              'worker_pid': os.getpid(), 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'seconds': time.perf_counter() - started,
              'registration_binding': identity(reg_path), 'source_binding': identity(Path(__file__)),
              'selection_lock_binding': reg['selection_lock_binding'],
              'training_receipt_binding': identity(trial / 'training_execution_receipt.json'),
              'preparation_receipt_binding': reg['preparation_receipt_binding'],
              'initialization_binding': identity(trial / 'run/initialization.json'),
              'paired_seed43_initialization_matches_control': (True if slot == 'candidate43' else None),
              'protected_evidence': after_protected, 'prepared_cache_check': preparation_check,
              'epochs_recomputed': n_epoch, 'DEV_windows_recomputed': 6612655 * n_epoch,
              'DEV_payload_files_verified_before_after': len(payload_refs), 'epoch_observations': observations,
              'selected_epoch': selected, 'counters': counters,
              'budget_binding': acceptance['budget_completion_snapshot_binding'],
              'budget_completion_snapshot_binding': acceptance['budget_completion_snapshot_binding'],
              'budget_prefix_snapshot_binding': reg['budget_prefix_snapshot_binding'],
              'prefix_budget_check': budget_check,
              'original_WN42_original_argmax_DEV_metrics': baseline_metrics,
              'frozen_seed42_ED4_original_argmax_DEV_metrics': parent_metrics,
              'selected_original_argmax_DEV_metrics': candidate_metrics,
              'candidate_accuracy_minus_seed42_WN42_percentage_points': 100 * float(best[0] - baseline_rank[0]),
              'comparison_same_complete_DEV_anchors': True, 'no_ED3_bias_added': True,
              'resource_cleanup_binding': identity(trial / 'run/resource_cleanup.json'),
              'checkpoint_validation_binding': identity(trial / 'checkpoint_validation.json'),
              'checkpoint_validation_execution_receipt_binding': identity(trial / 'checkpoint_execution_receipt.json'),
              'separate_checkpoint_CPU_loads': 2,
              'checkpoint_validation_scope': 'Opaque new hashes only; no checkpoint deserialization in this audit',
              'extra_model_loads': 0, 'extra_models_constructed': 0, 'extra_forwards': 0,
              'extra_optimizers': 0, 'extra_training_runs_or_updates': 0, 'E_record_reads': 0,
              'historical17_19_stream_reads': 0, 'new_independent_test': False, 'next_step_started': False,
              'scope': 'Current observable data and registered screening rules; development validation only; no new independent test, significance, or generalization claim.'}
    out = trial / 'readonly_audit.json'
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'passed': True, 'slot_id': slot, 'audit': identity(out), 'epochs_recomputed': n_epoch,
                      'selected_epoch': selected, 'extra_forwards': 0}, ensure_ascii=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
