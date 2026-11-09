"""P2 real research-data preparation. Admission is stdlib-only; never opens sealed days."""
from shared.paths import resolve_path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'artifacts/tcn_research_p2_2026-10-14'
P1 = 'artifacts/tcn_research_screening_p1_2026-10-14'
TRAIN = tuple(f'2025-12-{i:02}' for i in range(8, 13))
TUNE = ('2025-12-15', '2025-12-16')
SEALED = ('2025-12-17', '2025-12-18', '2025-12-19')
ROLES = {**dict.fromkeys(TRAIN, 'train'), **dict.fromkeys(TUNE, 'tuning_validation'),
         '2025-12-17': 'comparison_validation', '2025-12-18': 'final_test', '2025-12-19': 'final_test'}
PIN_MANIFEST = '871d233407a81f5d61c9f3950362f9c2424bf5df225c9ef07d19d76579f3eeec'
PIN_COMPLETION = '37c6ba4700ad00783322d54e9f08f5a5238fa9f1fb8c68d97dc159e7188f019a'
PIN_AUDIT = '653edaccbc6fe60baded1f8eb59114e3e111f5d8a1a7ecc88077b8b36a660390'
PIN_PROTOCOL = '7003d63f5b6b49f7d6f757ebaee1128c06091f1d8a0cbd0c1abb53b044a0c9aa'
PIN_REGISTRY = '891ab6f049edce1e4817c399e7eae2cdf07a108e68c0c55e28ade23dbda9ced3'
PIN_RECEIPT = 'c065cf598b964cc19806332c88f207198f218cf28e2283029f42e3be7b269896'
RECORD_FIELDS = [('sequence', '<u8'), ('timestamp_ns', '<u8'), ('segment_id', '<u4'),
                 ('flags', '<u4'), ('bid_levels', '<u4'), ('ask_levels', '<u4'), ('features', '<u8', (40,))]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def bounded(path, roots):
    """Reject an outside lexical target before resolve/stat/open."""
    p = Path(os.path.abspath(os.fspath(path)))
    roots = tuple(Path(os.path.abspath(os.fspath(r))) for r in roots)
    need(any(p.is_relative_to(r) for r in roots), 'Outside admitted roots')
    result = p.resolve()
    need(any(result.is_relative_to(r.resolve()) for r in roots), 'Resolved path outside admitted roots')
    return result


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def metadata(path, expected, root):
    p = bounded(path, (root,))
    need(p.is_file() and p.stat().st_size <= 4 * 1024**2, 'Missing/oversize admitted metadata')
    need(sha(p) == expected, 'Admitted metadata changed: ' + str(p))
    return json_read(p)


def _integer(value, low=0):
    return type(value) is int and value >= low


def validate_ranges(manifest):
    """Structural admission without opening data files or importing numpy."""
    need(isinstance(manifest, dict) and manifest.get('date_roles') == ROLES, 'Date roles changed')
    need(isinstance(manifest.get('sources'), dict) and set(manifest['sources']) == set(ROLES), 'Sources missing')
    need(manifest.get('record_schema_bytes') == 352, 'Record geometry changed')
    need(manifest.get('engineering_screening_passed') is True and manifest.get('research_data_index_ready') is True,
         'Research screening not accepted')
    need(manifest.get('original_flag64_cleared') is False and
         manifest.get('source_control_verified_for_new_days') is False and
         manifest.get('coverage_unknown_for_new_days') is True, 'Research limitation changed')
    need(manifest.get('sealed_labels_or_scores_opened') is False, 'Sealed evidence already opened')
    ranges = manifest.get('ranges')
    need(isinstance(ranges, list) and len(ranges) > 0, 'Missing research ranges')
    previous = {}
    totals = dict.fromkeys(set(ROLES.values()), 0)
    unique = dict.fromkeys(set(ROLES.values()), 0)
    days_seen = set()
    for r in ranges:
        need(isinstance(r, dict) and r.get('date') in ROLES, 'Unknown range day')
        day = r['date']; source = manifest['sources'][day]; days_seen.add(day)
        need(r.get('role') == ROLES[day] == source.get('role') and source.get('day') == day, 'Range role mismatch')
        need(r.get('records_path') == source.get('records_path') and
             r.get('records_sha256') == source.get('records_sha256'), 'Range source mismatch')
        need(all(_integer(r.get(k)) for k in ('start_row', 'stop_row', 'endpoint_start', 'endpoint_stop_exclusive')),
             'Non-integer range bounds')
        start, stop = r['start_row'], r['stop_row']
        need(stop > start and _integer(source.get('rows'), 1) and stop <= source['rows'], 'Source bounds invalid')
        need(start >= previous.get(day, 0), 'Ranges overlap or are unordered')
        previous[day] = stop
        need(r['endpoint_start'] == start + 99 and r['endpoint_stop_exclusive'] == stop - 50,
             'Input100/H50 footprint changed')
        n = max(0, stop - start - 149)
        need(r.get('record_count') == stop - start and r.get('window_count') == n and n > 0, 'Counts/geometry mismatch')
        need(r.get('protocol_sha256') == manifest.get('protocol_sha256'), 'Range protocol mismatch')
        need(r.get('source_control_verified') == source.get('source_control_verified') and
             r.get('coverage_unknown') == source.get('coverage_unknown'), 'Control provenance mismatch')
        totals[r['role']] += n; unique[r['role']] += stop - start
    need(days_seen == set(ROLES), 'A registered day has no ranges')
    need(totals == manifest.get('sample_counts') and unique == manifest.get('unique_eligible_records_by_role'),
         'Role count mismatch')
    return manifest


def load_admitted_manifest(root=ROOT):
    """P1 hard bindings are checked before numpy imports or any source-record access."""
    root = Path(root).resolve()
    p1 = root / P1
    completion = metadata(p1 / 'completion_acceptance.json', PIN_COMPLETION, root)
    audit = metadata(p1 / 'readonly_audit_final.json', PIN_AUDIT, root)
    pointer = json_read(bounded(p1 / 'screened_latest.json', (root,)))
    need(pointer.get('manifest_sha256') == PIN_MANIFEST and pointer.get('engineering_screening_passed') is True,
         'P1 pointer does not name accepted manifest')
    m = metadata(pointer['manifest_path'], PIN_MANIFEST, root)
    protocol = metadata(m['protocol_path'], PIN_PROTOCOL, root)
    metadata(m['source_registry_path'], PIN_REGISTRY, root)
    need(pointer.get('execution_receipt_sha256') == PIN_RECEIPT, 'P1 receipt identity changed')
    receipt = metadata(pointer['execution_receipt_path'], PIN_RECEIPT, root)
    need(completion.get('all_ten_dates_passed') is True and completion.get('independent_audit_passed') is True and
         completion.get('source_records_sha_before_after_equal') is True, 'P1 acceptance missing')
    need(audit.get('status') == 'passed' and audit.get('checks_passed') == audit.get('checks_total') and
         audit.get('result', {}).get('manifest_sha256') == PIN_MANIFEST, 'Independent P1 audit missing')
    need(receipt.get('status') == 'screening_passed' and receipt.get('all_ten_sources_scanned') is True,
         'P1 screening receipt missing')
    need(protocol.get('date_roles') == ROLES and protocol.get('user_direction_accepted') is True,
         'Research protocol not adopted')
    validate_ranges(m)
    m['_p1_manifest_binding'] = {'path': str(Path(pointer['manifest_path'])), 'sha256': PIN_MANIFEST}
    m['_p1_completion_binding'] = {'path': str(p1 / 'completion_acceptance.json'), 'sha256': PIN_COMPLETION}
    m['_p1_audit_binding'] = {'path': str(p1 / 'readonly_audit_final.json'), 'sha256': PIN_AUDIT}
    return m


def admitted_days(role):
    need(role in ('train', 'tuning_validation'), 'P2 forbids comparison/final labels and inputs')
    return TRAIN if role == 'train' else TUNE


def verify_real_sources(data):
    """Only the seven TRAIN/TUNE files, never a sealed file."""
    sources = data.get('sources')
    need(isinstance(sources, dict), 'Missing sources')
    result = {}
    for day in TRAIN + TUNE:
        source = sources[day]
        need(source['role'] == ROLES[day], 'Source role changed')
        path = bounded(source['records_path'], (Path(str(resolve_path('data://research/FinEdge_LOB_Data'))), ROOT))
        need(path.is_file() and path.stat().st_size == source['rows'] * 352, 'Records missing/size changed')
        digest = sha(path)
        need(digest == source['records_sha256'], 'Source records changed: ' + day)
        result[day] = digest
    return result


def physical(features):
    import numpy as np
    x = np.asarray(features, dtype=np.float64).copy()
    x[..., 0::2] /= 10000.0
    return x


def label_pair(records, ends):
    """Original float64 H50/threshold0 production formula and independent integer oracle."""
    import numpy as np
    ends = np.asarray(ends, dtype=np.int64)
    need(ends.ndim == 1 and np.all(ends >= 0) and np.all(ends + 50 < len(records)), 'Label endpoint out of bounds')
    current_raw = records['features'][ends][:, [0, 2]]
    future_raw = records['features'][ends + 50][:, [0, 2]]
    need(np.all(current_raw > 0) and np.all(future_raw > 0), 'Invalid midpoint')
    need(np.max(current_raw, initial=0) < 2**51 and np.max(future_raw, initial=0) < 2**51,
         'Midpoint exceeds exact float64 integer domain')
    current = current_raw.astype(np.float64).sum(axis=1)
    future = future_raw.astype(np.float64).sum(axis=1)
    production = np.where(future > current, 2, np.where(future < current, 0, 1)).astype(np.uint8)
    delta = future_raw[:, 0].astype(np.int64) + future_raw[:, 1].astype(np.int64)
    delta -= current_raw[:, 0].astype(np.int64) + current_raw[:, 1].astype(np.int64)
    reference = (np.sign(delta) + 1).astype(np.uint8)
    need(np.array_equal(production, reference), 'H50 integer oracle mismatch')
    return production, reference


def fit_unique_statistics(chunks_factory):
    """Chan pooled moments; independently recompute sum/centered population variance."""
    import numpy as np
    n = 0; mean = np.zeros(40); m2 = np.zeros(40)
    low = np.full(40, np.inf); high = np.full(40, -np.inf)
    for raw in chunks_factory():
        x = physical(raw)
        need(x.ndim == 2 and x.shape[1] == 40 and np.isfinite(x).all(), 'Invalid physical features')
        if len(x) == 0:
            continue
        count = len(x); avg = x.mean(axis=0); delta = avg - mean
        m2 += ((x - avg)**2).sum(axis=0) + delta**2 * n * count / (n + count)
        mean += delta * count / (n + count); n += count
        low = np.minimum(low, x.min(axis=0)); high = np.maximum(high, x.max(axis=0))
    need(n > 0, 'No unique TRAIN rows')
    zero = low == high
    std = np.sqrt(np.maximum(m2 / n, 0)); std[zero] = 1.
    total = np.zeros(40); n2 = 0
    for raw in chunks_factory():
        x = physical(raw); total += x.sum(axis=0); n2 += len(x)
    mean2 = total / n2
    centered = np.zeros(40); n3 = 0
    for raw in chunks_factory():
        x = physical(raw); centered += ((x - mean2)**2).sum(axis=0); n3 += len(x)
    std2 = np.sqrt(np.maximum(centered / n3, 0)); std2[zero] = 1.
    need(n == n2 == n3 and np.allclose(mean, mean2, rtol=1e-10, atol=1e-8) and
         np.allclose(std, std2, rtol=1e-8, atol=1e-8), 'Independent TRAIN statistics mismatch')
    norm = {'record_count': n, 'mean': mean.tolist(), 'std': std.tolist(),
            'zero_variance_columns': np.flatnonzero(zero).tolist(),
            'fit_scope': 'unique research-eligible TRAIN records only; population std',
            'physical_prices_divisor': 10000, 'fit_dates': list(TRAIN)}
    review = {'passed': True, 'methods': ['pooled Chan moments', 'independent sum mean and centered variance'],
              'primary_record_count': n, 'independent_mean_record_count': n2,
              'independent_variance_record_count': n3,
              'maximum_absolute_mean_difference': float(np.max(np.abs(mean - mean2))),
              'maximum_absolute_std_difference': float(np.max(np.abs(std - std2))),
              'mean_rtol': 1e-10, 'mean_atol': 1e-8, 'std_rtol': 1e-8, 'std_atol': 1e-8,
              'unique_rows_not_repeated_windows': True, 'TUNE_fitted': False, 'sealed_fitted': False}
    return norm, review


def prepare(root=ROOT, outdir=None):
    started = time.perf_counter(); root = Path(root).resolve()
    source = load_admitted_manifest(root)
    stage = root / STAGE
    stage.mkdir(parents=True, exist_ok=True)
    outdir = bounded(outdir or stage / ('data_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')), (stage,))
    need(not outdir.exists(), 'Preparation output already exists')
    before = verify_real_sources(source)  # Admission before numpy import.
    import numpy as np
    dtype = np.dtype(RECORD_FIELDS)
    need(dtype.itemsize == 352, 'Wrong record dtype')
    maps = {}
    outdir.mkdir()
    entries = [dict(r) for r in source['ranges'] if r['date'] in TRAIN + TUNE]
    try:
        for day in TRAIN + TUNE:
            maps[day] = np.memmap(source['sources'][day]['records_path'], dtype=dtype, mode='r')
        def chunks():
            for r in entries:
                if r['role'] != 'train':
                    continue
                for begin in range(r['start_row'], r['stop_row'], 65536):
                    yield maps[r['date']]['features'][begin:min(begin + 65536, r['stop_row'])]
        normalization, review = fit_unique_statistics(chunks)
        need(normalization['record_count'] == source['unique_eligible_records_by_role']['train'], 'TRAIN unique count changed')
        json_write(outdir / 'normalization.json', normalization)
        json_write(outdir / 'statistics_independent_review.json', review)
        day_reports = {}; role_counts = {'train': np.zeros(3, np.int64), 'tuning_validation': np.zeros(3, np.int64)}
        label_checks = 0
        for day in TRAIN + TUNE:
            folder = outdir / day; folder.mkdir(); label_path = folder / 'labels.u8'
            label_count = 0; count = np.zeros(3, np.int64); identity = hashlib.sha256(); examples = []
            a = maps[day]
            with label_path.open('xb') as stream:
                for r in [e for e in entries if e['date'] == day]:
                    r['label_offset'] = label_count
                    for begin in range(r['endpoint_start'], r['endpoint_stop_exclusive'], 100000):
                        ends = np.arange(begin, min(begin + 100000, r['endpoint_stop_exclusive']), dtype=np.int64)
                        y, oracle = label_pair(a, ends)
                        count += np.bincount(y, minlength=3); label_count += len(y); label_checks += len(y)
                        stream.write(y.tobytes())
                        identity.update(ends.astype('<u8', copy=False).tobytes())
                    for end in sorted({r['endpoint_start'], r['endpoint_start'] + 1, r['endpoint_stop_exclusive'] - 1}):
                        raw = a['features'][end - 99:end + 1]
                        x = ((physical(raw) - np.asarray(normalization['mean'])) / np.asarray(normalization['std'])).T.astype(np.float32)
                        y, oracle = label_pair(a, [end])
                        examples.append({'endpoint': end, 'label': int(y[0]), 'integer_oracle_label': int(oracle[0]),
                                         'input_float32_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
                                         'full_footprint_inside_research_range': end - 99 >= r['start_row'] and end + 50 < r['stop_row']})
            expected = sum(r['window_count'] for r in entries if r['date'] == day)
            need(label_count == expected and int(count.sum()) == expected, 'Label stream count mismatch')
            role_counts[ROLES[day]] += count
            report = {'date': day, 'role': ROLES[day], 'label_path': str(label_path), 'label_sha256': sha(label_path),
                      'label_bytes': label_path.stat().st_size, 'window_count': label_count,
                      'class_counts': count.tolist(), 'endpoint_stream_dtype': '<u8',
                      'endpoint_stream_sha256': identity.hexdigest(), 'label_formula': 'endpoint midpoint H50 relative change; threshold 0; no smoothing',
                      'integer_oracle_checked_labels': label_count, 'integer_oracle_all_equal': True,
                      'first_second_last_window_examples': examples, 'original_flags_unchanged': True,
                      'source_control_verified': source['sources'][day]['source_control_verified'],
                      'coverage_unknown': source['sources'][day]['coverage_unknown']}
            json_write(folder / 'labels_report.json', report); day_reports[day] = report
            print('P2 LABELS ' + day + ' ' + str(label_count) + ' ' + str(count.tolist()), flush=True)
    finally:
        for a in maps.values():
            a._mmap.close()
    after = verify_real_sources(source)
    need(before == after, 'Source records changed during P2')
    counts = role_counts['train'].tolist(); majority = max(range(3), key=lambda k: counts[k])
    result = {'version': 'TCN-research-B-P2-data-1.0', 'completed_utc': utc(), 'status': 'prepared_real_data_passed',
              'code_binding': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
              'p1_manifest_binding': source['_p1_manifest_binding'], 'p1_completion_binding': source['_p1_completion_binding'],
              'p1_audit_binding': source['_p1_audit_binding'], 'protocol_sha256': source['protocol_sha256'],
              'normalization': normalization, 'normalization_binding': {'path': str(outdir / 'normalization.json'), 'sha256': sha(outdir / 'normalization.json')},
              'statistics_independent_review_binding': {'path': str(outdir / 'statistics_independent_review.json'), 'sha256': sha(outdir / 'statistics_independent_review.json')},
              'statistics_independent_review': review, 'ranges': entries,
              'sources': {day: source['sources'][day] for day in TRAIN + TUNE}, 'day_label_reports': day_reports,
              'sample_counts': {role: source['sample_counts'][role] for role in ('train', 'tuning_validation')},
              'train_class_counts': counts, 'majority_class_from_train': majority,
              'tuning_class_counts': role_counts['tuning_validation'].tolist(),
              'majority_tie_policy': 'lowest class index', 'horizon': 50, 'threshold': 0,
              'source_sha256_before': before, 'source_sha256_after': after, 'source_records_unchanged': before == after,
              'integer_oracle_checked_labels': label_checks, 'integer_oracle_all_equal': True,
              'sealed_dates': list(SEALED), 'sealed_records_opened': False, 'sealed_labels_or_scores_opened': False,
              'source_control_verified_for_new_days': False, 'coverage_unknown_for_new_days': True,
              'original_flag64_cleared': False, 'model_forwards': 0, 'optimizer_updates': 0, 'training_runs': 0,
              'new_training_reservations': 0, 'E_drive_writes': 0,
              'training_ready': False, 'remaining_before_training': 'real entry/runtime/evaluation acceptance and P3 trigger',
              'elapsed_seconds': time.perf_counter() - started}
    json_write(outdir / 'manifest.json', result)
    result['_manifest_path'] = str(outdir / 'manifest.json')
    result['_manifest_sha256'] = sha(outdir / 'manifest.json')
    return result


def validate_prepared(data):
    need(isinstance(data, dict) and data.get('version') == 'TCN-research-B-P2-data-1.0' and
         data.get('status') == 'prepared_real_data_passed', 'Not accepted P2 data')
    need(type(data.get('horizon')) is int and data['horizon'] == 50 and type(data.get('threshold')) is int and data['threshold'] == 0 and
         data.get('original_flag64_cleared') is False and data.get('sealed_labels_or_scores_opened') is False,
         'P2 research/label contract changed')
    need(data.get('sealed_records_opened') is False and data.get('sealed_dates') == list(SEALED) and
         data.get('source_control_verified_for_new_days') is False and data.get('coverage_unknown_for_new_days') is True,
         'P2 coverage/sealed limits changed')
    need(set(data.get('sources', {})) == set(TRAIN + TUNE), 'P2 must contain exactly seven unsealed sources')
    need(set(data.get('day_label_reports', {})) == set(TRAIN + TUNE), 'Label days missing')
    need(data.get('source_records_unchanged') is True and data.get('source_sha256_before') == data.get('source_sha256_after'),
         'P2 source integrity not accepted')
    n = data.get('normalization', {})
    need(n.get('fit_dates') == list(TRAIN) and n.get('record_count', 0) > 0 and
         len(n.get('mean', [])) == len(n.get('std', [])) == 40, 'TRAIN normalization invalid')
    need(all(type(v) in (int, float) and math.isfinite(v) for v in n['mean']) and
         all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in n['std']), 'Non-finite/invalid normalization')
    totals = {'train': 0, 'tuning_validation': 0}; previous = {}
    for r in data.get('ranges', []):
        day = r.get('date'); need(day in TRAIN + TUNE and r.get('role') == ROLES[day], 'Forbidden/unknown range')
        need(all(_integer(r.get(k)) for k in ('start_row', 'stop_row', 'endpoint_start', 'endpoint_stop_exclusive')) and
             r['start_row'] < r['stop_row'] <= data['sources'][day]['rows'], 'Prepared source bounds changed')
        need(r['endpoint_start'] == r['start_row'] + 99 and r['endpoint_stop_exclusive'] == r['stop_row'] - 50 and
             r['window_count'] == r['stop_row'] - r['start_row'] - 149 and r['window_count'] > 0, 'Window geometry changed')
        need(r.get('label_offset') == previous.get(day, 0), 'Label offset changed')
        previous[day] = previous.get(day, 0) + r['window_count']; totals[r['role']] += r['window_count']
        need(r['records_path'] == data['sources'][day]['records_path'] and
             r['records_sha256'] == data['sources'][day]['records_sha256'], 'Prepared range source changed')
    need(totals == data.get('sample_counts'), 'Prepared totals changed')
    train_counts = [0, 0, 0]; tune_counts = [0, 0, 0]
    for day, report in data['day_label_reports'].items():
        need(report['date'] == day and report['role'] == ROLES[day] and report['window_count'] == previous.get(day), 'Day labels changed')
        c = report['class_counts']; need(len(c) == 3 and all(_integer(v) for v in c) and sum(c) == report['window_count'], 'Class counts invalid')
        counts = train_counts if day in TRAIN else tune_counts
        for k in range(3): counts[k] += c[k]
    need(train_counts == data.get('train_class_counts') and tune_counts == data.get('tuning_class_counts'), 'Role counts changed')
    need(data.get('majority_class_from_train') == max(range(3), key=lambda k: train_counts[k]), 'Majority/tie policy changed')
    return data


def load_prepared(path, expected_sha256=None):
    p = bounded(path, (ROOT / STAGE,))
    need(p.is_file() and p.stat().st_size <= 4 * 1024**2, 'Prepared manifest missing')
    digest = sha(p)
    need(expected_sha256 is not None and digest == expected_sha256, 'Explicit prepared-manifest identity required')
    data = validate_prepared(json_read(p))
    origin = load_admitted_manifest(ROOT)  # Recheck original accepted research metadata, no real records.
    need(data.get('p1_manifest_binding', {}).get('sha256') == PIN_MANIFEST, 'P2 origin changed')
    need(data['sources'] == {day: origin['sources'][day] for day in TRAIN + TUNE}, 'P2 source identity differs from P1')
    expected_ranges = [r for r in origin['ranges'] if r['date'] in TRAIN + TUNE]
    need([{k: v for k, v in r.items() if k != 'label_offset'} for r in data['ranges']] == expected_ranges,
         'P2 ranges differ from accepted P1 ranges')
    for name in ('normalization_binding', 'statistics_independent_review_binding'):
        binding = data[name]; current = metadata(binding['path'], binding['sha256'], ROOT)
        if name == 'normalization_binding': need(current == data['normalization'], 'Normalization payload changed')
        else: need(current == data['statistics_independent_review'] and current.get('passed') is True,
                   'Independent statistics evidence changed')
    binding = data['code_binding']; code = bounded(binding['path'], (ROOT,))
    need(sha(code) == binding['sha256'] and code == Path(__file__).resolve(), 'Prepared adapter code identity changed')
    for report in data['day_label_reports'].values():
        label = bounded(report['label_path'], (p.parent,))
        need(label.is_file() and label.stat().st_size == report['label_bytes'] == report['window_count'] and
             sha(label) == report['label_sha256'], 'Label stream changed')
    data['_manifest_path'] = str(p); data['_manifest_sha256'] = digest
    return data


class ResearchWindows:
    """Read-only on-demand contiguous windows, with full shuffle batch support."""
    def __init__(self, data, role, normalization=None):
        admitted_days(role); validate_prepared(data)
        import numpy as np
        self.data = data; self.role = role
        self.entries = [r for r in data['ranges'] if r['role'] == role]
        self.cumulative = np.cumsum([r['window_count'] for r in self.entries]).tolist()
        norm = data['normalization'] if normalization is None else normalization
        self.mean = np.asarray(norm['mean'], dtype=np.float64); self.std = np.asarray(norm['std'], dtype=np.float64)
        need(self.mean.shape == self.std.shape == (40,) and np.isfinite(self.mean).all() and
             np.isfinite(self.std).all() and np.all(self.std > 0), 'Invalid supplied normalization')
        self.maps = {}; self.label_maps = {}
        need(len(self) > 0, 'Empty research split')

    def __len__(self):
        return self.cumulative[-1] if self.cumulative else 0

    def _maps_for(self, day):
        import numpy as np
        if day not in self.maps:
            s = self.data['sources'][day]
            need(s['role'] == self.role and day in admitted_days(self.role), 'Forbidden source read')
            self.maps[day] = np.memmap(s['records_path'], dtype=np.dtype(RECORD_FIELDS), mode='r')
            self.label_maps[day] = np.memmap(self.data['day_label_reports'][day]['label_path'], dtype='u1', mode='r')
        return self.maps[day], self.label_maps[day]

    def batch_details(self, indices):
        import numpy as np
        ids = np.asarray(indices)
        need(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer) and len(ids) > 0, 'Invalid batch indices')
        ids = ids.astype(np.int64, copy=False)
        need(np.all(ids >= 0) and np.all(ids < len(self)), 'Batch indices out of bounds')
        groups = np.searchsorted(self.cumulative, ids, side='right')
        x = np.empty((len(ids), 40, 100), np.float32); y = np.empty(len(ids), np.int64)
        ends_out = np.empty(len(ids), np.int64); dates = np.empty(len(ids), dtype='U10')
        for group in np.unique(groups):
            r = self.entries[group]; positions = np.flatnonzero(groups == group)
            relative = ids[positions] - (self.cumulative[group - 1] if group else 0)
            ends = r['endpoint_start'] + relative
            a, labels = self._maps_for(r['date'])
            raw = a['features'][ends[:, None] + np.arange(-99, 1)]
            values = (physical(raw) - self.mean) / self.std
            x[positions] = values.transpose(0, 2, 1)
            y[positions] = labels[r['label_offset'] + relative]
            ends_out[positions] = ends; dates[positions] = r['date']
        need(np.isfinite(x).all() and np.all((y >= 0) & (y <= 2)), 'Invalid batch numeric values')
        return {'X': x, 'y': y, 'date': dates, 'endpoints': ends_out, 'global_indices': ids}

    def batch(self, indices):
        result = self.batch_details(indices)
        return result['X'], result['y']

    def __getitem__(self, index):
        x, y = self.batch([index]); return x[0], y[0]

    def close(self):
        for a in list(self.maps.values()) + list(self.label_maps.values()): a._mmap.close()
        self.maps.clear(); self.label_maps.clear()


def iter_batches(data, role, norm=None, batch_size=256, shuffle_seed=None):
    """Bounded sequential physical-row reuse; shuffle uses identical arbitrary-index gather."""
    admitted_days(role); need(_integer(batch_size, 1), 'Invalid batch size')
    ds = ResearchWindows(data, role, norm)
    import numpy as np
    try:
        if shuffle_seed is not None:
            need(_integer(shuffle_seed), 'Invalid shuffle seed')
            order = np.random.default_rng(shuffle_seed).permutation(len(ds))
            for begin in range(0, len(order), batch_size):
                yield ds.batch_details(order[begin:begin + batch_size])
            return
        offset = 0
        for r in ds.entries:
            a, labels = ds._maps_for(r['date'])
            for begin in range(r['endpoint_start'], r['endpoint_stop_exclusive'], batch_size):
                stop = min(begin + batch_size, r['endpoint_stop_exclusive']); count = stop - begin
                values = (physical(a['features'][begin - 99:stop]) - ds.mean) / ds.std
                views = np.lib.stride_tricks.sliding_window_view(values, 100, axis=0)
                x = np.asarray(views, dtype=np.float32, order='C')
                relative = begin - r['endpoint_start']
                y = np.asarray(labels[r['label_offset'] + relative:r['label_offset'] + relative + count], dtype=np.int64)
                need(x.shape == (count, 40, 100) and np.isfinite(x).all(), 'Sequential window shape/numeric mismatch')
                yield {'X': x, 'y': y, 'date': np.full(count, r['date']),
                       'endpoints': np.arange(begin, stop, dtype=np.int64),
                       'global_indices': np.arange(offset + relative, offset + relative + count, dtype=np.int64)}
            offset += r['window_count']
    finally:
        ds.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['admit', 'prepare'])
    parser.add_argument('--outdir')
    args = parser.parse_args(argv)
    if args.command == 'admit':
        m = load_admitted_manifest()
        print(json.dumps({'admitted': True, 'manifest_sha256': PIN_MANIFEST,
                          'TRAIN_windows': m['sample_counts']['train'], 'TUNE_windows': m['sample_counts']['tuning_validation']}))
    else:
        m = prepare(outdir=args.outdir)
        print(json.dumps({'manifest_path': m['_manifest_path'], 'manifest_sha256': m['_manifest_sha256'],
                          'train_class_counts': m['train_class_counts'], 'tuning_class_counts': m['tuning_class_counts'],
                          'elapsed_seconds': m['elapsed_seconds']}))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as error:
        print('REFUSED: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
