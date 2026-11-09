"""Independent stdlib audit of P2 saved metadata and new D-only label/prediction streams.

No NumPy/PyTorch imports, market records, old weights or E accesses. Source file
before/after hashes and numeric normalization checks are reviewed as receipts;
only new D label/prediction bytes are independently counted and joined here.
"""
from __future__ import annotations
from array import array
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import hashlib
import json
import math
import ntpath
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'artifacts/tcn_research_p2_2026-10-14'
TRAIN = [f'2025-12-{d:02d}' for d in range(8, 13)]
TUNE = ['2025-12-15', '2025-12-16']
SEALED = ['2025-12-17', '2025-12-18', '2025-12-19']
P1_SHA = '871d233407a81f5d61c9f3950362f9c2424bf5df225c9ef07d19d76579f3eeec'
PROTOCOL_SHA = '7003d63f5b6b49f7d6f757ebaee1128c06091f1d8a0cbd0c1abb53b044a0c9aa'
OLD_MANIFEST_SHA = 'e8402f0ea730c0dbdb1727f8c300e8b453ca68e3e8fd80fa9698fb7ffdf27bad'
OLD_CP_SHA = '8d36f62e8a6839acfd557a1d8178f6052b3aca76436f1a2c017f95bbee05d846'
OLD_CONFIG_SHA = 'f4fe394a5f81b1383727227d2fb9741a0c2e93d08d2e432a4bfa70ccc867c342'
OLD_NORMALIZATION_SHA = '3348edadbfc99583f1afdc22466b18f2c27ab88c9fba5b9b7b73419e7a3e0fd9'
MODEL_SHA = 'acc9e0b6edfc10793df25c69173d94cd9c6c76c74d73294e327dd91b7c3c45fc'
MAX_SMALL = 2 * 1024**2
SHA_RE = re.compile(r'[0-9a-f]{64}\Z')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def safe(path, binary=False):
    """Check lexical workspace containment before resolve/stat/open (reject E)."""
    raw = str(path)
    if not Path(raw).is_absolute():
        raw = str(ROOT / raw)
    norm = ntpath.normcase(ntpath.normpath(raw))
    base = ntpath.normcase(str(ROOT))
    if not norm.startswith(base + '\\'):
        raise ValueError('Audit refuses a non-workspace path: ' + raw)
    p = Path(raw).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('Audit refuses an escaped resolved path')
    if binary:
        label = (p.name == 'labels.u8' and p.parent.name in TRAIN + TUNE
                 and p.parent.parent.parent == STAGE and p.parent.parent.name.startswith('data_'))
        prediction = (re.fullmatch(r'2025-12-(?:15|16)\.predictions\.u8', p.name) is not None
                      and p.parent == STAGE / 'old_baseline')
        if not p.is_relative_to(STAGE) or not (label or prediction):
            raise ValueError('Only new P2 TRAIN/TUNE labels and TUNE prediction byte streams are allowed')
    elif p.suffix.lower() not in {'.json', '.py', '.txt', '.log', '.md', '.jsonl', '.lock'}:
        raise ValueError('Only bounded source/metadata files may be read')
    return p


def read(path):
    p = safe(path)
    if p.stat().st_size > MAX_SMALL:
        raise ValueError('Small metadata exceeds 2 MiB')
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('Duplicate JSON field: ' + k)
            result[k] = v
        return result
    def constant(v):
        raise ValueError('Nonfinite JSON constant: ' + v)
    def finite_float(v):
        n = float(v)
        if not math.isfinite(n):
            raise ValueError('Nonfinite JSON float')
        return n
    return json.loads(p.read_text(encoding='utf-8-sig'), object_pairs_hook=pairs,
                      parse_constant=constant, parse_float=finite_float)


def sha(path):
    p = safe(path)
    if p.stat().st_size > MAX_SMALL:
        raise ValueError('Only bounded small metadata/source may be hashed')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def metrics(matrix):
    if not isinstance(matrix, list) or len(matrix) != 3 or any(not isinstance(r, list) or len(r) != 3 for r in matrix):
        raise ValueError('3x3 confusion required')
    if any(type(v) is not int or v < 0 for row in matrix for v in row):
        raise ValueError('Nonnegative integer confusion counts required')
    support = [sum(r) for r in matrix]
    predicted = [sum(matrix[i][j] for i in range(3)) for j in range(3)]
    n = sum(support)
    if n <= 0:
        raise ValueError('Empty metrics')
    return {'samples': n, 'accuracy': sum(matrix[i][i] for i in range(3)) / n,
            'macro_f1': sum(2 * matrix[i][i] / (support[i] + predicted[i])
                            if support[i] + predicted[i] else 0 for i in range(3)) / 3,
            'recall': [matrix[i][i] / support[i] if support[i] else 0 for i in range(3)],
            'precision': [matrix[i][i] / predicted[i] if predicted[i] else 0 for i in range(3)],
            'f1': [2 * matrix[i][i] / (support[i] + predicted[i])
                   if support[i] + predicted[i] else 0 for i in range(3)],
            'support': support, 'confusion_true_rows_predicted_columns': matrix}


def majority_matrix(support, majority):
    if type(majority) is not int or not 0 <= majority < 3:
        raise ValueError('Invalid TRAIN majority')
    result = [[0] * 3 for _ in range(3)]
    for i, n in enumerate(support):
        result[i][majority] = n
    return result


def exact_quality(matrix, majority):
    s = [sum(r) for r in matrix]
    p = [sum(matrix[i][j] for i in range(3)) for j in range(3)]
    a = Fraction(sum(matrix[i][i] for i in range(3)), sum(s))
    f = sum((Fraction(2 * matrix[i][i], s[i] + p[i]) if s[i] + p[i] else Fraction(0)
             for i in range(3)), Fraction(0)) / 3
    ba = Fraction(s[majority], sum(s))
    bf = Fraction(2 * s[majority], sum(s) + s[majority]) / 3
    r = all(s[i] > 0 and Fraction(matrix[i][i], s[i]) >= Fraction(1, 5) for i in range(3))
    return {'eligible_on_tuning_only': a > ba and f > bf and r,
            'accuracy_strictly_above_own_TRAIN_majority': a > ba,
            'macro_f1_strictly_above_own_TRAIN_majority': f > bf,
            'each_class_recall_at_least_point_two': r,
            'majority_reference': metrics(majority_matrix(s, majority))}


class Audit:
    def __init__(self):
        self.checks = []
        self.binary_bytes = 0
        self.binary_reports = {}

    def check(self, value, label):
        ok = bool(value)
        self.checks.append({'check': label, 'passed': ok})
        if not ok:
            raise ValueError(label)

    def binding(self, binding, label, metadata=True):
        self.check(isinstance(binding, dict) and isinstance(binding.get('path'), str)
                   and isinstance(binding.get('sha256'), str)
                   and SHA_RE.fullmatch(binding['sha256']) is not None, label + ': valid binding')
        p = safe(binding['path'])
        self.check(sha(p) == binding['sha256'], label + ': current SHA')
        return read(p) if metadata else None

    def metric(self, value, matrix, label):
        expected = metrics(matrix)
        for key, item in expected.items():
            if isinstance(item, float):
                self.check(type(value.get(key)) in (float, int) and math.isfinite(value[key])
                           and abs(value[key] - item) <= 1e-12, label + ': ' + key)
            elif isinstance(item, list) and item and isinstance(item[0], float):
                self.check(isinstance(value.get(key), list) and len(value[key]) == len(item)
                           and all(type(a) in (float, int) and math.isfinite(a) and abs(a-b) <= 1e-12
                                   for a, b in zip(value[key], item)), label + ': ' + key)
            else:
                self.check(value.get(key) == item, label + ': ' + key)
        if 'loss' in value:
            self.check(type(value['loss']) in (float, int) and math.isfinite(value['loss'])
                       and value['loss'] >= 0, label + ': finite nonnegative CE')
        return expected

    def quality(self, value, matrix, majority, label):
        expected = exact_quality(matrix, majority)
        for key in expected:
            if key == 'majority_reference':
                self.metric(value[key], expected[key]['confusion_true_rows_predicted_columns'], label + ': own majority')
            else:
                self.check(type(value.get(key)) is bool and value[key] is expected[key], label + ': ' + key)

    def binary_labels(self, binding, expected_count, label):
        p = safe(binding['path'], binary=True)
        self.check(p.stat().st_size == expected_count, label + ': one byte per endpoint')
        h = hashlib.sha256()
        counts = Counter()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(262144), b''):
                h.update(block)
                counts.update(block)
                self.binary_bytes += len(block)
        self.check(set(counts) <= {0, 1, 2}, label + ': exact class domain')
        self.check(h.hexdigest() == binding['sha256'], label + ': complete saved stream SHA')
        self.check(sum(counts.values()) == expected_count, label + ': complete labels coverage')
        return [counts[i] for i in range(3)]

    def binary_matrix(self, labels_binding, pred_binding, count, label):
        lp, pp = safe(labels_binding['path'], binary=True), safe(pred_binding['path'], binary=True)
        self.check(lp.stat().st_size == pp.stat().st_size == count, label + ': joined stream lengths')
        lh, ph, matrix = hashlib.sha256(), hashlib.sha256(), [[0] * 3 for _ in range(3)]
        with lp.open('rb') as ls, pp.open('rb') as ps:
            while True:
                a, b = ls.read(262144), ps.read(262144)
                self.check(len(a) == len(b), label + ': chunk lengths')
                if not a:
                    break
                lh.update(a); ph.update(b)
                self.binary_bytes += len(a) + len(b)
                joined = Counter(zip(a, b))
                self.check(all(y in {0, 1, 2} and p in {0, 1, 2} for y, p in joined), label + ': byte class domain')
                for (y, p), n in joined.items():
                    matrix[y][p] += n
        self.check(lh.hexdigest() == labels_binding['sha256'], label + ': labels saved SHA')
        self.check(ph.hexdigest() == pred_binding['sha256'], label + ': prediction saved SHA')
        self.check(sum(map(sum, matrix)) == count, label + ': full joined count')
        return matrix

    def protected(self, path):
        old = read(path)
        self.check(isinstance(old, dict) and bool(old), 'Protected small-file set nonempty')
        for p, digest in old.items():
            self.check(sha(p) == digest, 'Protected old SHA retained: ' + p)
        return len(old)

    def run(self, bundle_path):
        b = read(bundle_path)
        data = self.binding(b['data_manifest'], 'P2 prepared data')
        self.check(data['version'] == 'TCN-research-B-P2-data-1.0'
                   and data['status'] == 'prepared_real_data_passed', 'Real data preparation status')
        self.check(data['p1_manifest_binding']['sha256'] == P1_SHA, 'Fixed P1 origin SHA')
        p1 = self.binding(data['p1_manifest_binding'], 'Accepted P1 manifest')
        self.binding(data['p1_completion_binding'], 'Accepted P1 completion')
        self.binding(data['p1_audit_binding'], 'Accepted P1 independent audit')
        self.check(data['protocol_sha256'] == p1['protocol_sha256'] == PROTOCOL_SHA, 'Research protocol fixed identity')
        self.check(set(data['sources']) == set(TRAIN + TUNE)
                   and set(data['day_label_reports']) == set(TRAIN + TUNE), 'Exactly seven unsealed source/label dates')
        self.check(data['sources'] == {d: p1['sources'][d] for d in TRAIN + TUNE}, 'All current sources exactly P1 accepted metadata')
        expected_ranges = [r for r in p1['ranges'] if r['date'] in TRAIN + TUNE]
        self.check([{k:v for k,v in r.items() if k != 'label_offset'} for r in data['ranges']] == expected_ranges,
                   'All current ranges exactly P1 accepted offsets')
        self.check(data['sample_counts'] == {'train': 12380446, 'tuning_validation': 6612655}, 'Fixed TRAIN/TUNE denominator')
        self.check(data['source_sha256_before'] == data['source_sha256_after']
                   == {d: p1['sources'][d]['records_sha256'] for d in TRAIN + TUNE}, 'Saved real records hashes before/after/accepted equal')
        self.check(data['source_records_unchanged'] is True, 'Real source hash receipt passed')
        self.check(data['horizon'] == 50 and data['threshold'] == 0, 'H50 zero threshold labels')
        self.check(data['source_control_verified_for_new_days'] is False and data['coverage_unknown_for_new_days'] is True
                   and data['original_flag64_cleared'] is False, 'Research uncertainty retained')
        for key in ('sealed_records_opened', 'sealed_labels_or_scores_opened'):
            self.check(data[key] is False, 'Data prep kept sealed scope: ' + key)
        for key in ('model_forwards', 'optimizer_updates', 'training_runs', 'new_training_reservations', 'E_drive_writes'):
            self.check(type(data[key]) is int and data[key] == 0, 'Data prep no model/training: ' + key)
        self.binding(data['code_binding'], 'Prepared real adapter source', metadata=False)
        norm = self.binding(data['normalization_binding'], 'Unique TRAIN normalizer')
        review = self.binding(data['statistics_independent_review_binding'], 'Independent numerical statistics receipt')
        self.check(norm == data['normalization'] and review == data['statistics_independent_review'], 'Bound stats payloads equal manifest')
        self.check(norm['fit_dates'] == TRAIN and norm['record_count'] == 12381191, 'Only unique five-TRAIN rows fitted')
        self.check(norm['physical_prices_divisor'] == 10000, 'Price physical units retained')
        self.check(len(norm['mean']) == len(norm['std']) == 40
                   and all(type(v) in (int, float) and math.isfinite(v) for v in norm['mean'])
                   and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in norm['std']), 'Forty finite means and positive std')
        zero = norm['zero_variance_columns']
        self.check(isinstance(zero, list) and zero == sorted(set(zero))
                   and all(type(i) is int and 0 <= i < 40 and norm['std'][i] == 1 for i in zero), 'Zero-variance replacement exactly one')
        self.check(review['passed'] is True and review['unique_rows_not_repeated_windows'] is True
                   and review['TUNE_fitted'] is False and review['sealed_fitted'] is False, 'Independent norm receipt accepted scope')
        self.check(review['methods'] == ['pooled Chan moments', 'independent sum mean and centered variance'], 'Distinct numerical statistics methods saved')
        for k in ('primary_record_count', 'independent_mean_record_count', 'independent_variance_record_count'):
            self.check(review[k] == norm['record_count'], 'Independent statistics unique count: ' + k)
        for name, series in [('mean', norm['mean']), ('std', norm['std'])]:
            diff = review['maximum_absolute_' + name + '_difference']
            self.check(type(diff) in (int, float) and math.isfinite(diff) and diff >= 0
                       and diff <= review[name + '_atol'] + review[name + '_rtol'] * max(abs(v) for v in series),
                       'Independent saved maximum numeric difference: ' + name)
        role_counts = {'train': [0,0,0], 'tuning_validation': [0,0,0]}
        day_counts, all_labels, examples = {}, 0, 0
        for day in TRAIN + TUNE:
            report = data['day_label_reports'][day]
            role = 'train' if day in TRAIN else 'tuning_validation'
            self.check(report['date'] == day and report['role'] == role, 'Label day and role: ' + day)
            rows = [r for r in data['ranges'] if r['date'] == day]
            count = sum(r['stop_row'] - r['start_row'] - 149 for r in rows)
            self.check(report['window_count'] == report['label_bytes'] == count, 'Label geometry/window/byte count: ' + day)
            pos = 0
            eh = hashlib.sha256()
            for r in rows:
                self.check(r['label_offset'] == pos and r['endpoint_start'] == r['start_row'] + 99
                           and r['endpoint_stop_exclusive'] == r['stop_row'] - 50,
                           'Original endpoint offsets and full footprint: ' + day)
                pos += r['window_count']
                for start in range(r['endpoint_start'], r['endpoint_stop_exclusive'], 100000):
                    values = array('Q', range(start, min(start + 100000, r['endpoint_stop_exclusive'])))
                    self.check(values.itemsize == 8, 'stdlib endpoint uint64 width: ' + day)
                    if sys.byteorder != 'little': values.byteswap()
                    eh.update(values.tobytes())
            self.check(eh.hexdigest() == report['endpoint_stream_sha256'], 'Independent original endpoint stream SHA: ' + day)
            self.check(report['integer_oracle_all_equal'] is True and report['integer_oracle_checked_labels'] == count,
                       'Saved entire-label integer oracle verification: ' + day)
            self.check(report['original_flags_unchanged'] is True, 'Source flags retained: ' + day)
            actual = self.binary_labels({'path': report['label_path'], 'sha256': report['label_sha256']}, count, 'D labels ' + day)
            self.check(actual == report['class_counts'], 'Independent class counts match real saved evidence: ' + day)
            for i in range(3): role_counts[role][i] += actual[i]
            for ex in report['first_second_last_window_examples']:
                self.check(ex['label'] == ex['integer_oracle_label'] and ex['full_footprint_inside_research_range'] is True
                           and SHA_RE.fullmatch(ex['input_float32_sha256']) is not None, 'Saved actual first/second/last reference: ' + day)
                examples += 1
            day_counts[day] = count; all_labels += count
        self.check(role_counts['train'] == data['train_class_counts']
                   and role_counts['tuning_validation'] == data['tuning_class_counts'], 'Independent TRAIN/TUNE aggregate class counts')
        majority = max(range(3), key=lambda i: role_counts['train'][i])
        self.check(data['majority_class_from_train'] == majority and data['majority_tie_policy'] == 'lowest class index', 'TRAIN-only majority and tie rule')
        self.check(data['integer_oracle_all_equal'] is True and data['integer_oracle_checked_labels'] == all_labels, 'Total saved actual integer label checks')
        runtime = self.binding(b['runtime_preflight'], 'Real-input CPU/GPU preflight')
        self.check(runtime['passed'] is True and runtime['runtime_preflight_passed'] is True, 'Real-input runtime accepted')
        self.check(runtime['data_manifest_sha256'] == b['data_manifest']['sha256'] and runtime['model_code_sha256'] == MODEL_SHA, 'Runtime actual input/model identities')
        expected_checks = {(role, normalizer, batch) for role in ('train','tuning_validation')
                           for normalizer in ('five_day','old_two_day') for batch in (1,16,256)}
        actual_checks = {(r['role'],r['normalization'],r['batch_size']) for r in runtime['checks']}
        self.check(actual_checks == expected_checks and len(runtime['checks']) == 12, 'Runtime all twelve real input/normalization/batch checks')
        for r in runtime['checks']:
            self.check(r['allclose'] is True and r['argmax_equal'] is True and math.isfinite(r['max_abs_error'])
                       and r['max_abs_error'] >= 0 and sum(r['label_counts']) == r['batch_size'], 'Saved CPU/GPU actual tensor check')
        self.check(runtime['max_cpu_gpu_abs_error'] == max(r['max_abs_error'] for r in runtime['checks']), 'Runtime maximum error receipt')
        self.check(runtime['model_constructions'] == 3 and runtime['forward_calls'] == 27 and runtime['backward_calls'] == 2, 'Preflight actual model/forward/backward activity')
        self.check(len(runtime['gradient_checks']) == 2 and {g['device'] for g in runtime['gradient_checks']} == {'cpu','cuda'}, 'Both bounded real TRAIN gradient checks')
        for g in runtime['gradient_checks']:
            self.check(g['gradient_tensors'] == 24 and math.isfinite(g['gradient_l2_norm']) and g['gradient_l2_norm'] > 0
                       and math.isfinite(g['loss']) and g['loss'] >= 0, 'Saved bounded differentiability check: ' + g['device'])
        self.check(runtime['model_state_unchanged'] is True and runtime['all_gradients_cleared'] is True, 'Preflight clears gradients without updating state')
        for key in ('optimizer_constructions','optimizer_updates','training_epochs','training_runs'):
            self.check(runtime[key] == 0, 'Preflight no training/optimizer: ' + key)
        self.check(runtime['sealed_labels_or_scores_opened'] is False and runtime['source_flag64_cleared'] is False, 'Runtime research sealed scope intact')
        self.check(runtime['old_checkpoint_sha256'] == OLD_CP_SHA and runtime['old_selected_epoch'] == 1, 'Runtime fixed old checkpoint identity')
        self.check(runtime['original_checkpoint_sha256_after'] == OLD_CP_SHA, 'Runtime old checkpoint file SHA retained')
        ref = runtime['independent_real_window_reference']
        self.check(ref['samples_checked'] == 21 and ref['boundary_middle_last_per_range'] is True
                   and ref['input_tensor_exact'] is True and ref['integer_H50_label_exact'] is True,
                   'Actual independent seven-range boundary/middle/last input reference')
        rt = runtime['runtime']
        self.check(rt['precision'] == 'float32' and rt['cpu_threads'] == 4 and rt['tf32'] is False
                   and rt['deterministic_algorithms'] is True, 'Runtime scoped FP32 deterministic settings')
        ev = self.binding(b['baseline_evaluation'], 'Real fixed old baseline evaluation')
        for k,v in {'checkpoint_sha256':OLD_CP_SHA,'selected_epoch':1,'old_model_config_sha256':OLD_CONFIG_SHA,
                    'old_origin_manifest_sha256':OLD_MANIFEST_SHA,'model_code_sha256':MODEL_SHA}.items():
            self.check(ev[k] == v, 'Fixed historical identity: ' + k)
        self.check(ev['data_manifest_sha256'] == b['data_manifest']['sha256'], 'Fixed evaluation input dataset identity')
        self.check(ev['normalization_fit_dates'] == TRAIN[:2] and ev['own_TRAIN_class_counts'] == [1448237,1379387,1427062]
                   and ev['own_TRAIN_majority'] == 0, 'Original two-day TRAIN majority and normalizer scope')
        self.check(ev['old_normalization_sha256'] == OLD_NORMALIZATION_SHA, 'Fixed independently identified original normalizer payload')
        self.check(ev['original_checkpoint_sha256_after'] == OLD_CP_SHA, 'Full evaluation old checkpoint file SHA retained')
        self.check(ev['protocol_sha256'] == PROTOCOL_SHA and ev['data_sources_identity_sha256'] == canonical(data['sources'])
                   and ev['tuning_ranges_sha256'] == canonical([r for r in data['ranges'] if r['role']=='tuning_validation']),
                   'Evaluation research protocol/source/range aliases')
        self.check(ev['source_records_sha256_before'] == ev['source_records_sha256_after']
                   == {d:data['sources'][d]['records_sha256'] for d in TUNE}, 'Full evaluation actual source SHA receipts')
        self.check(set(ev['per_day']) == set(TUNE), 'Only the two common TUNE dates evaluated')
        aggregate = [[0]*3 for _ in range(3)]
        loss_sum = 0.; fwd = 0
        for day in TUNE:
            r, labels = ev['per_day'][day], data['day_label_reports'][day]
            c = self.binary_matrix({'path':labels['label_path'],'sha256':labels['label_sha256']}, r['predictions'], day_counts[day], 'D fixed model join ' + day)
            self.metric(r, c, 'Recomputed fixed model metrics ' + day)
            self.check(r['support'] == labels['class_counts'] and r['ordered_labels_sha256'] == labels['label_sha256']
                       and r['ordered_endpoints_sha256'] == labels['endpoint_stream_sha256'], 'Identical raw window/endpoint/label evaluation coverage: ' + day)
            expected_batches = sum((rr['window_count'] + 255)//256 for rr in data['ranges'] if rr['date'] == day)
            self.check(r['batches'] == expected_batches, 'Tail batch included: ' + day)
            fwd += expected_batches; loss_sum += r['loss'] * day_counts[day]
            self.check(type(r['cross_entropy_sum']) in (float,int) and math.isfinite(r['cross_entropy_sum'])
                       and r['cross_entropy_sum'] >= 0
                       and abs(r['loss']-r['cross_entropy_sum']/day_counts[day]) <= 1e-12,
                       'Saved daily CE sum and denominator: ' + day)
            self.quality(r['quality'], c, 0, 'Fixed old own-majority gate ' + day)
            for i in range(3):
                for j in range(3): aggregate[i][j] += c[i][j]
        self.metric(ev['aggregate'], aggregate, 'Full common TUNE baseline')
        self.check(abs(ev['aggregate']['loss'] - loss_sum/sum(day_counts[d] for d in TUNE)) <= 1e-12, 'Weighted aggregate CE from saved daily losses')
        self.quality(ev['quality'], aggregate, 0, 'Exact aggregate old-majority quality gate')
        self.metric(ev['candidate_majority_reference'], majority_matrix(ev['aggregate']['support'], majority), 'Five-TRAIN majority on same common TUNE')
        self.check(ev['model_state_sha256_before'] == ev['model_state_sha256_after']
                   and SHA_RE.fullmatch(ev['model_state_sha256_before']) is not None
                   and ev['model_state_unchanged'] is True and ev['all_gradients_none'] is True, 'Fixed model state and gradients preserved')
        self.check(ev['forward_calls'] == fwd and ev['aggregate']['samples'] == 6612655, 'Exactly full TUNE forward batches/denominator')
        for key in ('backward_calls','optimizer_updates','training_epochs','training_runs','new_reservations'):
            self.check(ev[key] == 0, 'Full evaluation no training: ' + key)
        self.check(ev['sealed_labels_or_scores_opened'] is False and ev['source_control_verified_for_new_days'] is False
                   and ev['coverage_unknown_for_new_days'] is True and ev['original_flag64_cleared'] is False, 'Final evaluation research limitations retained')
        self.check(ev['device'] == 'cuda' and ev['precision'] == 'float32' and ev['batch_size'] == 256
                   and ev['result_wording'] == '基于当前可观察数据和既定筛选规则的表现', 'Final scoped evaluation/report semantics')
        cfg = self.binding(b['config'], 'Independent five-day training config')
        self.check(cfg['training_authorized_in_P2'] is False and cfg['formal_start_requires_step'] == 'P3'
                   and cfg['maximum_runs'] == 1 and cfg['seed'] == 42 and cfg['max_epochs'] == 10
                   and cfg['early_stop_patience'] == 3, 'Prepared future P3 finite budget, no P2 launch')
        self.check(cfg['learning_rate'] == .001 and cfg['model']['dropout'] == 0
                   and cfg['model']['weight_norm'] is False and cfg['fresh_initialization'] is True, 'Prepared unchanged architecture fresh training settings')
        launch = self.binding(b['entry_launch'], 'Accepted real entry launch binding')
        self.check(launch['config'] == b['config'] and launch['prepared_manifest'] == b['data_manifest']
                   and launch['runtime'] == b['runtime_preflight'], 'Real entry config/input/runtime binding links')
        for path,digest in launch['code_sha256'].items():
            self.check(sha(ROOT/path) == digest, 'Launch bound current source SHA: ' + path)
        completion = self.binding(launch['completion'], 'P2 completion input/entry acceptance')
        self.check(completion['entry_ready'] is True and completion['training_input_ready'] is True
                   and completion['runtime_preflight_passed'] is True and completion['formal_training_started'] is False
                   and completion['sealed_labels_or_scores_opened'] is False, 'Completion engineering ready without formal training')
        if 'daily_report' in b:
            daily = self.binding(b['daily_report'], 'Real P2 daily report')
            self.check(daily['version'] == 'TCN-research-real-daily-report-1.0' and daily['step'] == 'P2'
                       and daily['status'] == 'real_data_and_old_model_baseline_completed_new_model_not_trained', 'Daily report actual P2 scope')
            self.check(daily['bindings'] == {'data':b['data_manifest'],'evaluation':b['baseline_evaluation'],
                                            'runtime':b['runtime_preflight']}, 'Daily report three actual evidence bindings')
            self.check(daily['historical_TCN01'] == ev['aggregate'] and daily['historical_TCN01_quality'] == ev['quality']
                       and daily['real_input_runtime'] == runtime, 'Daily report exact real measured model/runtime evidence')
            self.check(daily['train_unique_records'] == norm['record_count'] and daily['train_windows'] == 12380446
                       and daily['tuning_windows'] == 6612655 and daily['five_TRAIN_class_counts'] == data['train_class_counts']
                       and daily['five_TRAIN_majority_class'] == majority, 'Daily report real denominators and TRAIN counts')
            self.check(daily['five_TRAIN_class_proportions'] == [v/12380446 for v in data['train_class_counts']], 'Daily report measured class proportions')
            def ref(value, support, chosen, label):
                adapted = dict(value)
                adapted['samples'] = value['total']
                adapted['confusion_true_rows_predicted_columns'] = value['confusion_matrix']
                adapted.pop('loss', None)
                self.metric(adapted, majority_matrix(support, chosen), label)
                self.check(value['majority_class'] == chosen and value['chosen_from'] == 'own TRAIN labels', label + ': TRAIN source')
            ref(daily['old_TRAIN_majority_reference'], data['tuning_class_counts'], 0, 'Daily old TRAIN reference')
            ref(daily['five_TRAIN_majority_reference'], data['tuning_class_counts'], majority, 'Daily five TRAIN reference')
            self.check(set(daily['per_day']) == set(TUNE), 'Daily report common TUNE date set')
            for day in TUNE:
                item = daily['per_day'][day]
                self.check(item['samples'] == day_counts[day] and item['historical_TCN01'] == ev['per_day'][day], 'Daily exact day model evidence: ' + day)
                support = data['day_label_reports'][day]['class_counts']
                ref(item['old_TRAIN_majority_reference'], support, 0, 'Daily old reference ' + day)
                ref(item['five_TRAIN_majority_reference'], support, majority, 'Daily five reference ' + day)
            new = daily['five_day_new_model']
            self.check(new['accuracy'] is None and new['macro_f1'] is None and new['recall'] is None
                       and new['training_epochs'] == 0, 'Daily new model scores unmeasured')
            self.check(daily['sealed_labels_or_scores_opened'] is False and daily['coverage_unknown_for_new_days'] is True
                       and daily['source_control_verified_for_new_days'] is False and daily['original_flag64_cleared'] is False
                       and daily['formal_training_started'] is False and daily['result_wording'] == ev['result_wording'], 'Daily result qualification retained')
            self.check(daily['metrics_bridge_source_sha256'] == sha(ROOT/'scripts/tcn_majority_contract.py')
                       and daily['report_source_sha256'] == sha(ROOT/'scripts/tcn_research_daily_report.py'), 'Daily preserved arithmetic kernel and report source SHA')
        tests = []
        unique_tests = 0
        for i, binding in enumerate(b.get('test_receipts', []), 1):
            t = self.binding(binding, 'P2 test execution receipt ' + str(i))
            self.check(t.get('exit_code') == 0 or t.get('returncode') == 0 or t.get('passed') is True,
                       'Final new tests pass receipt ' + str(i))
            for path,digest in t.get('bindings',{}).items():
                self.check(sha(ROOT/path) == digest, 'Tests bound actual final source/log SHA: ' + path)
            for path,digest in t.get('code_sha256',{}).items():
                self.check(sha(ROOT/path) == digest, 'Tests bound actual final code/config SHA: ' + path)
            if 'source_sha256' in t:
                source = ('tcn_research_daily_report.py' if Path(binding['path']).name.startswith('daily_report_tests')
                          else 'tcn_research_evaluate_p2.py')
                self.check(t['source_sha256'] == sha(ROOT/'scripts'/source), 'Tests bound final numerical/report source')
            if 'test_sha256' in t:
                test = ('test_tcn_research_daily_report.py' if Path(binding['path']).name.startswith('daily_report_tests')
                        else 'test_tcn_research_evaluate_p2.py')
                self.check(t['test_sha256'] == sha(ROOT/'tests'/test), 'Tests bound final test source')
            n = next((t[k] for k in ('unique_tests','test_count','tests') if k in t), None)
            self.check(type(n) is int and n > 0, 'Final receipt actual distinct test count ' + str(i))
            unique_tests += n
            tests.append(t)
        old_count = self.protected(b['protected_before'])
        return {'normalization_record_count': norm['record_count'], 'TRAIN_class_counts': role_counts['train'],
                'TUNE_class_counts': role_counts['tuning_validation'], 'TRAIN_majority': majority,
                'independently_counted_label_bytes': all_labels, 'independent_tuning_confusion': aggregate,
                'independent_tuning_metrics': metrics(aggregate), 'saved_reference_examples': examples,
                'preflight_model_constructions': 3, 'preflight_forward_calls': 27, 'preflight_backward_calls': 2,
                'full_evaluation_forward_calls': fwd, 'formal_training_runs': 0,
                'protected_small_files': old_count, 'protected_small_files_unchanged': True,
                'entry_ready': True, 'formal_P3_training_authorized': False,
                'source_records_hashes_remeasured_in_this_audit': False,
                'tests_receipt_count': len(tests), 'final_unique_tests':unique_tests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = safe(args.output)
    if not out.is_relative_to(STAGE) or out.exists():
        raise ValueError('Audit output must be a new P2 metadata file')
    started = time.perf_counter()
    auditor = Audit()
    try:
        result = auditor.run(args.bundle)
        result.update(version='TCN-research-P2-independent-stdlib-audit-1.0', passed=True,
                      checked_utc=stamp(), seconds=time.perf_counter()-started,
                      source_sha256=sha(__file__), bundle_sha256=sha(args.bundle),
                      check_count=len(auditor.checks), checks=auditor.checks,
                      binary_bytes_read=auditor.binary_bytes,
                      scope='D metadata/new label and prediction bytes; no E/records/weights/ML/model/statistic rerun',
                      E_accesses=0, market_record_reads=0, model_calls=0,
                      normalization_numerical_recomputation=False)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps({'passed': True, 'checks': len(auditor.checks), 'seconds': result['seconds']}))
    except Exception as exc:
        failure = {'passed': False, 'checked_utc': stamp(), 'error': str(exc),
                   'seconds': time.perf_counter()-started, 'checks': auditor.checks,
                   'check_count': len(auditor.checks), 'binary_bytes_read': auditor.binary_bytes}
        out.write_text(json.dumps(failure, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
