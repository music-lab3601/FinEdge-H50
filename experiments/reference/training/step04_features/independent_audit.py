"""Independent ED4 posterior audit; D workspace only, no torch/model/E records.

The root executes this exactly once after the original training CLI has actually
exited. This module imports no training/feature/evaluation implementation. It
rebuilds numerical results from each new DEV cache with independent arithmetic.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import stat
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parent
CAMPAIGN = STAGE.parent
TRAIN = [f'2025-12-{i:02}' for i in range(8, 13)]
DEV = ['2025-12-15', '2025-12-16']
PROTOCOL_SHA = '09654e6dfdb10b69feb6ee950c6668e71640bcea1d56fa9691a9564c8641fcf3'
CHUNK = 131072


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


def receipt(phase):
    path = STAGE / (phase + '_execution_receipt.json')
    result = read_json(path)
    require(result.get('phase') == phase and result.get('exit_code') == 0
            and result.get('entry_unchanged') is True, 'Actual CLI success missing: ' + phase)
    require(result['entry_before'] == result['entry_after'], 'Actual CLI source changed')
    checked(result['entry_before']); checked(result['stdout']); checked(result['stderr'])
    checked(result['input']); checked(result['capture_source'])
    return result


def read_array(ref, expected_dtype, expected_shape):
    require(ref.get('dtype') == expected_dtype and ref.get('shape') == list(expected_shape),
            'Cache dtype/shape differs')
    path = checked(ref)
    require(path.is_relative_to(STAGE), 'Numerical cache outside this new ED4 stage')
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


def inspect_prepared(data, prepared):
    require(prepared['complete'] is True and set(prepared['cache_files']) == set(TRAIN + DEV),
            'Incomplete/expanded feature cache')
    require(prepared['source_records_unchanged'] is True and prepared['label_streams_unchanged'] is True
            and prepared['source_sha256_before'] == prepared['source_sha256_after']
            == {d: data['sources'][d]['records_sha256'] for d in TRAIN + DEV}, 'Source identity evidence differs')
    require(prepared['labels_sha256_before'] == prepared['labels_sha256_after']
            == {d: data['day_label_reports'][d]['label_sha256'] for d in TRAIN + DEV},
            'Original label identity evidence differs')
    require(prepared['sealed_records_opened'] is False and prepared['sealed_labels_or_scores_opened'] is False
            and prepared['original_flag64_cleared'] is False and prepared['E_drive_writes'] == 0,
            'Preparation scope/flag evidence differs')
    norm = bound_json(prepared['normalization_binding'])
    require(norm == prepared['normalization'] and norm['record_count'] == 12381191
            and norm['fit_dates'] == TRAIN and norm['TUNE_fitted'] is False and norm['sealed_fitted'] is False,
            'TRAIN-only normalization scope differs')
    original = data['normalization']
    require(np.allclose(np.array(norm['mean'])[1::2], np.array(original['mean'])[1::2], rtol=1e-10, atol=1e-8)
            and np.allclose(np.array(norm['std'])[1::2], np.array(original['std'])[1::2], rtol=1e-8, atol=1e-8),
            'Original quantity normalization differs')
    require(bound_json(prepared['statistics_independent_review_binding']) == prepared['statistics_independent_review']
            and prepared['statistics_independent_review']['passed'] is True, 'Feature statistics review differs')
    require(len(prepared['ranges']) == len(data['ranges']) == 7, 'Feature ranges changed')
    for ref, original_range in zip(prepared['ranges'], data['ranges']):
        require({k: v for k, v in ref.items() if k != 'cache_offset'} == original_range,
                'Feature cache anchor/range changed')
        require(ref['cache_offset'] == 0, 'Expected exactly one zero-offset range per date')
    maps = {}; refs = []; labels = []
    try:
        for day in TRAIN + DEV:
            ref = prepared['cache_files'][day]
            require(ref['role'] == ('train' if day in TRAIN else 'tuning_validation'), 'Feature cache role differs')
            count = sum(r['stop_row'] - r['start_row'] for r in data['ranges'] if r['date'] == day)
            maps[day] = read_array(ref, '<f4', (count, 40)); refs.append(ref)
            original_label = data['day_label_reports'][day]
            path = checked({'path': original_label['label_path'], 'sha256': original_label['label_sha256'],
                            'bytes': original_label['label_bytes']})
            labels.append(identity(path))
        examples = prepared['boundary_examples']
        require(len(examples) == 21, 'Boundary reference count differs')
        for ex in examples:
            day = ex['day']; require(day in TRAIN + DEV, 'Forbidden boundary date')
            start, stop = ex['cache_row_start'], ex['cache_row_stop_exclusive']
            require(stop - start == 100, 'Boundary window is not L100')
            x = np.ascontiguousarray(maps[day][start:stop].T)
            require(hashlib.sha256(x.tobytes()).hexdigest() == ex['normalized_input_float32_sha256'],
                    'Written cache boundary does not reproduce F64 reference input')
        n = 0; sum_x = np.zeros(40); sum_x2 = np.zeros(40)
        for day in TRAIN:
            a = maps[day]
            for begin in range(0, len(a), CHUNK):
                x = a[begin:begin + CHUNK].astype(np.float64)
                require(np.isfinite(x).all(), 'Nonfinite normalized TRAIN row')
                n += len(x); sum_x += x.sum(0); sum_x2 += np.einsum('ij,ij->j', x, x)
        require(n == 12381191, 'Independent unique TRAIN row count differs')
        means = sum_x / n; variance = sum_x2 / n - means * means
        zero = set(norm['zero_variance_columns'])
        require(np.all(np.abs(means) <= 2e-6), 'Cached unique TRAIN normalized mean differs from zero')
        require(all(abs(float(variance[i]) - (0 if i in zero else 1)) <= 2e-6 for i in range(40)),
                'Cached unique TRAIN normalized population variance differs')
        return {'passed': True, 'boundary_windows_reconstructed': len(examples), 'unique_TRAIN_rows': n,
                'normalized_mean_max_abs': float(np.max(np.abs(means))),
                'normalized_variance_min': float(variance.min()), 'normalized_variance_max': float(variance.max()),
                'row_cache_files': refs, 'original_labels': labels, 'norm_binding': prepared['normalization_binding']}
    finally:
        for a in maps.values():
            a._mmap.close()


def inspect_epoch(row, data, baseline_cache, payload_refs):
    cache = bound_json(row['cache_binding'])
    require(cache['epoch'] == row['epoch'] and cache['decision'] == 'original_argmax'
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    reg_path = STAGE / 'training_registration.json'
    require(digest(reg_path) == args.registration_sha256, 'Explicit training registration differs')
    reg = read_json(reg_path)
    require(reg['step'] == 'ED4', 'Wrong independent audit phase')
    protocol = bound_json(reg['protocol_binding'])
    require(reg['protocol_binding']['sha256'] == PROTOCOL_SHA and protocol['campaign_id'] == 'TCN_EDO_20261017',
            'Audit protocol differs')
    require(protocol['data']['train_dates'] == TRAIN and protocol['data']['development_dates'] == DEV,
            'Audit dates differ')
    data = bound_json(reg['prepared_manifest_binding']); prepared = bound_json(reg['feature_cache_binding'])
    for name, expected in reg['source_bindings'].items():
        require(digest(ROOT / name) == expected, 'Reviewed numerical source changed')
    checked(reg['source_review_binding']); checked(reg['test_evidence_binding'])
    before_protected = verify_protected()
    prep_cli, train_cli = receipt('preparation'), receipt('training')
    actual_prepared_ref = prep_cli['outputs']['feature_cache_manifest']
    require(checked(actual_prepared_ref) == checked(reg['feature_cache_binding'])
            and actual_prepared_ref['sha256'] == reg['feature_cache_binding']['sha256'],
            'Actual preparation output is not the registered training feature cache')
    prep_registration = bound_json(prep_cli['input'])
    require(prep_registration['prepared_manifest_binding']['sha256'] == reg['prepared_manifest_binding']['sha256']
            and prep_registration['protocol_binding']['sha256'] == PROTOCOL_SHA,
            'Actual preparation used another registered parent/protocol')
    require(train_cli['input']['sha256'] == args.registration_sha256, 'Actual training used another registration')
    acceptance = bound_json(train_cli['outputs']['training_acceptance'])
    history = bound_json(train_cli['outputs']['history'])
    require(acceptance['status'] == 'completed' and 1 <= len(history) <= 10
            and acceptance['epochs_completed'] == len(history), 'Training not actually complete')
    require(acceptance['development_validation_only'] is True and acceptance['sealed_dates_opened'] is False
            and acceptance['original_flag64_cleared'] is False and acceptance['source_control_verified'] is False
            and acceptance['coverage_unknown'] is True and acceptance['original_argmax_without_ED3_bias'] is True
            and acceptance['automatic_next_action'] is False, 'Training result scope/limitations differ')
    require(acceptance['actual_worker_pid'] == train_cli['actual_worker_pid']
            and train_cli['actual_worker_pid'] == acceptance['worker_pid'], 'Worker roles disagree')
    preparation_check = inspect_prepared(data, prepared)
    before_payloads = bound_json(acceptance['cache_payloads_before_binding'])
    after_payloads = bound_json(acceptance['cache_payloads_after_binding'])
    require(before_payloads == after_payloads and before_payloads['passed'] is True,
            'Trainer cache payload before/after evidence differs')
    for day in TRAIN + DEV:
        require(before_payloads['per_day'][day] == {
            'cache_sha256': prepared['cache_files'][day]['sha256'],
            'label_sha256': data['day_label_reports'][day]['label_sha256']}, 'Trainer payload identity evidence differs')
    baseline_manifest_path = CAMPAIGN / 'step02_diagnosis/evaluation/cache_manifest.json'
    baseline_cache = read_json(baseline_manifest_path)
    require(digest(baseline_manifest_path) == '1808492b6b2e118b09a88114bf7af4a837414e1e96a41427fef832e08407c106',
            'Frozen WN42 cache manifest changed')
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
    budget = bound_json(acceptance['budget_binding'])
    require(len(budget['reserved_runs']) == 1 and budget['optimizer_updates'] == updates
            and budget['consumed_epochs'] == n_epoch and updates <= 483620, 'New budget aggregate differs')
    slot = budget['reserved_runs'][0]
    require(slot.get('slot_id', slot.get('id')) == 'feature42' and slot['seed'] == 42
            and slot['reservation_id'] == reg['reservation_id'] and slot['optimizer_updates'] == updates
            and slot['consumed_epochs'] == n_epoch and slot['status'] == 'completed', 'Permanent feature42 budget differs')
    require(slot['actual_optimizer_steps_completed_last_snapshot'] == updates, 'Completed-step terminal budget differs')
    cps = acceptance['epoch_checkpoint_bindings']
    require(set(cps) == {str(i) for i in range(1, n_epoch + 1)}, 'Complete epoch checkpoint missing')
    for ref in cps.values():
        checked(ref)  # Opaque new checkpoint hashing; no deserialization.
    for ref in acceptance['checkpoint_bindings'].values():
        checked(ref)
    cp_cli = receipt('checkpoint_validation')
    cp_result = bound_json(cp_cli['outputs']['checkpoint_validation'])
    require(cp_result['passed'] is True and cp_result['actual_new_checkpoint_CPU_loads'] == 2
            and cp_result['models_constructed'] == cp_result['optimizers_constructed']
            == cp_result['forwards'] == cp_result['backwards'] == cp_result['optimizer_updates'] == 0
            and cp_result['historical_checkpoint_loads'] == 0 and cp_result['RNG_restored'] is False,
            'Bounded checkpoint validation failed or exceeded load limits')
    checked(cp_result['source_binding']); cp_registration = bound_json(cp_result['registration_binding'])
    require(cp_registration['training_registration_binding']['sha256'] == args.registration_sha256
            and cp_registration['checkpoint_bindings'] == acceptance['checkpoint_bindings']
            and cp_registration['training_receipt_binding']['sha256'] == digest(STAGE / 'training_execution_receipt.json'),
            'CPU validation did not use the original successful training/checkpoint identities')
    cp_observations = cp_result['observations']
    require([v['checkpoint_role'] for v in cp_observations] == ['best.pt', 'latest.pt']
            and cp_observations[0]['epoch'] == selected and cp_observations[1]['epoch'] == n_epoch
            and all(v['passed'] is True and v['parameter_numel'] == 33539 and v['forwards'] == 0
                    for v in cp_observations), 'CPU checkpoint observation identity differs')
    cleanup = read_json(STAGE / 'run/resource_cleanup.json')
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
    candidate_metrics = observations[selected - 1]['metrics']; candidate_rank = best
    result = {'version': 'TCN-ED4-independent-D-only-audit-1.0', 'passed': True,
              'completed_utc': datetime.now(timezone.utc).isoformat(), 'seconds': time.perf_counter() - started,
              'registration_binding': identity(reg_path), 'source_binding': identity(Path(__file__)),
              'training_receipt_binding': identity(STAGE / 'training_execution_receipt.json'),
              'preparation_receipt_binding': identity(STAGE / 'preparation_execution_receipt.json'),
              'protected_evidence': after_protected, 'prepared_cache_check': preparation_check,
              'epochs_recomputed': n_epoch, 'DEV_windows_recomputed': 6612655 * n_epoch,
              'DEV_payload_files_verified_before_after': len(payload_refs), 'epoch_observations': observations,
              'selected_epoch': selected, 'counters': counters, 'budget_binding': identity(CAMPAIGN / 'budget.json'),
              'original_WN42_original_argmax_DEV_metrics': baseline_metrics,
              'selected_original_argmax_DEV_metrics': candidate_metrics,
              'candidate_accuracy_minus_original_WN42_percentage_points': 100 * float(candidate_rank[0] - baseline_rank[0]),
              'strict_accuracy_improvement_for_later_replication': candidate_rank[0] > baseline_rank[0],
              'comparison_same_complete_DEV_anchors': True, 'no_ED3_bias_added': True,
              'resource_cleanup_binding': identity(STAGE / 'run/resource_cleanup.json'),
              'checkpoint_validation_binding': identity(STAGE / 'checkpoint_validation.json'),
              'checkpoint_validation_execution_receipt_binding': identity(STAGE / 'checkpoint_validation_execution_receipt.json'),
              'separate_checkpoint_CPU_loads': 2,
              'checkpoint_validation_scope': 'Opaque hashes only; no checkpoint deserialization in this audit',
              'extra_model_loads': 0, 'extra_models_constructed': 0, 'extra_forwards': 0,
              'extra_optimizers': 0, 'extra_training_runs_or_updates': 0, 'E_record_reads': 0,
              'historical17_19_stream_reads': 0, 'new_independent_test': False, 'next_step_started': False,
              'scope': 'Based on current observable data and registered screening rules; development validation only; no statistical significance or new generalization claim.'}
    out = STAGE / 'independent_audit.json'
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'passed': True, 'audit': identity(out), 'epochs_recomputed': n_epoch,
                      'selected_epoch': selected, 'extra_forwards': 0}, ensure_ascii=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
