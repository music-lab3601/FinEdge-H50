"""ED4 causal relative-mid prices and read-only normalized row cache.

This module has no torch dependency and performs no file or data access on import.
Only even price channels change. Odd raw quantities retain the P2 convention.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from software.training.tcn_research_data_p2 import ROOT, RECORD_FIELDS, TRAIN, TUNE, ROLES, validate_prepared, admitted_days

STAGE = ROOT / 'artifacts/tcn_existing_data_optimization_2026-10-17/step04_features'
PROTOCOL_SHA = '09654e6dfdb10b69feb6ee950c6668e71640bcea1d56fa9691a9564c8641fcf3'
FEATURE_ID = 'same_record_mid_relative_prices_raw_quantities_v1'
CHUNK_ROWS = 65536


def need(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def read_bound_metadata(ref):
    need(isinstance(ref, dict) and set(ref) >= {'path', 'sha256'}, 'Missing metadata binding')
    path = Path(ref['path'])
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    need(path.is_relative_to(ROOT.resolve()) and path.is_file()
         and path.stat().st_size <= 16 * 1024**2, 'Metadata outside workspace/missing/oversized')
    need(sha(path) == ref['sha256'] and ('bytes' not in ref or path.stat().st_size == ref['bytes']),
         'Metadata binding differs')
    return json.loads(path.read_text(encoding='utf-8'))


def transform(features):
    """Float64 (price-current-mid)/current-mid, with raw quantity odd columns.

    Each record uses only its own best bid/ask. There is no window endpoint
    recentering, future reference, smoothing, logarithm, or extra channel.
    """
    import numpy as np
    raw = np.asarray(features)
    need(raw.ndim >= 1 and raw.shape[-1] == 40, 'Expected raw (...,40) features')
    need(np.issubdtype(raw.dtype, np.number) and not np.issubdtype(raw.dtype, np.complexfloating),
         'Features must be real numeric')
    x = raw.astype(np.float64, copy=True)
    need(np.isfinite(x).all() and np.all(x >= 0), 'Invalid raw feature values')
    price = x[..., 0::2]
    need(np.all(price > 0) and np.all(price < 2**51), 'Prices outside admitted exact F64 integer domain')
    mid = (x[..., 0] + x[..., 2]) / 2.0
    need(np.isfinite(mid).all() and np.all(mid > 0), 'Invalid current-record midpoint')
    x[..., 0::2] = (price - mid[..., None]) / mid[..., None]
    need(np.isfinite(x).all(), 'Nonfinite transformed features')
    return x


def validate_normalization(norm):
    import numpy as np
    need(isinstance(norm, dict) and norm.get('feature_id') == FEATURE_ID,
         'Wrong relative-feature normalization')
    need(norm.get('fit_dates') == list(TRAIN) and norm.get('TUNE_fitted') is False
         and norm.get('sealed_fitted') is False and type(norm.get('record_count')) is int
         and norm['record_count'] > 0, 'Normalization scope differs')
    mean = np.asarray(norm.get('mean'), dtype=np.float64)
    std = np.asarray(norm.get('std'), dtype=np.float64)
    need(mean.shape == std.shape == (40,) and np.isfinite(mean).all()
         and np.isfinite(std).all() and np.all(std > 0), 'Invalid relative normalization')
    return mean, std


def normalized_rows(features, normalization):
    import numpy as np
    mean, std = validate_normalization(normalization)
    values = ((transform(features) - mean) / std).astype('<f4')
    need(np.isfinite(values).all(), 'Nonfinite normalized F32 rows')
    return values


def fit_unique_statistics(chunks_factory):
    """Three TRAIN-only passes; Chan and independent centered variance agree."""
    import numpy as np
    n = 0
    mean = np.zeros(40, np.float64)
    m2 = np.zeros(40, np.float64)
    low = np.full(40, np.inf)
    high = np.full(40, -np.inf)
    for raw in chunks_factory():
        x = transform(raw)
        need(x.ndim == 2, 'Statistics require record rows')
        if not len(x):
            continue
        count = len(x)
        average = x.mean(axis=0)
        delta = average - mean
        m2 += ((x - average)**2).sum(axis=0) + delta**2 * n * count / (n + count)
        mean += delta * count / (n + count)
        n += count
        low = np.minimum(low, x.min(axis=0))
        high = np.maximum(high, x.max(axis=0))
    need(n > 0, 'No unique TRAIN rows')
    zero = low == high
    std = np.sqrt(np.maximum(m2 / n, 0))
    std[zero] = 1.0
    total = np.zeros(40, np.float64)
    n2 = 0
    for raw in chunks_factory():
        x = transform(raw)
        need(x.ndim == 2, 'Statistics require record rows')
        total += x.sum(axis=0)
        n2 += len(x)
    need(n2 == n, 'TRAIN mean pass row count differs')
    mean2 = total / n2
    centered = np.zeros(40, np.float64)
    n3 = 0
    for raw in chunks_factory():
        x = transform(raw)
        need(x.ndim == 2, 'Statistics require record rows')
        centered += ((x - mean2)**2).sum(axis=0)
        n3 += len(x)
    need(n3 == n, 'TRAIN variance pass row count differs')
    std2 = np.sqrt(np.maximum(centered / n3, 0))
    std2[zero] = 1.0
    atol = np.full(40, 1e-8)
    atol[0::2] = 1e-14
    need(np.all(np.abs(mean - mean2) <= atol + 1e-10 * np.abs(mean2))
         and np.all(np.abs(std - std2) <= atol + 1e-8 * np.abs(std2)),
         'Independent relative TRAIN statistics mismatch')
    norm = {'feature_id': FEATURE_ID, 'record_count': n, 'mean': mean.tolist(),
            'std': std.tolist(), 'zero_variance_columns': np.flatnonzero(zero).tolist(),
            'fit_scope': 'unique P2 research-eligible TRAIN records only; population std',
            'fit_dates': list(TRAIN), 'TUNE_fitted': False, 'sealed_fitted': False,
            'price_formula': '(raw_price - (raw_bid1+raw_ask1)/2) / ((raw_bid1+raw_ask1)/2)',
            'quantity_formula': 'raw_quantity_unchanged_no_log',
            'transform_and_normalization_precision': 'float64; final stored rows float32',
            'zero_variance_policy': 'population zero variance uses std=1'}
    validate_normalization(norm)
    review = {'passed': True, 'methods': ['pooled Chan moments', 'independent sum and centered variance'],
              'primary_record_count': n, 'independent_mean_record_count': n2,
              'independent_variance_record_count': n3,
              'maximum_absolute_mean_difference': float(np.max(np.abs(mean - mean2))),
              'maximum_absolute_std_difference': float(np.max(np.abs(std - std2))),
              'relative_price_atol': 1e-14, 'quantity_atol': 1e-8,
              'mean_rtol': 1e-10, 'std_rtol': 1e-8,
              'unique_rows_not_repeated_windows': True, 'TUNE_fitted': False, 'sealed_fitted': False}
    return norm, review


def _source_hashes(data):
    result = {}
    for day in TRAIN + TUNE:
        source = data['sources'][day]
        path = Path(source['records_path'])
        need(source['day'] == day and source['role'] == ROLES[day], 'Source role differs')
        need(path.is_file() and path.stat().st_size == source['rows'] * 352,
             'Source records size differs: ' + day)
        digest = sha(path)
        need(digest == source['records_sha256'], 'Source records SHA differs: ' + day)
        result[day] = digest
    return result


def _label_hashes(data):
    result = {}
    for day in TRAIN + TUNE:
        report = data['day_label_reports'][day]
        path = Path(report['label_path'])
        need(path.is_file() and path.stat().st_size == report['label_bytes'] == report['window_count'],
             'P2 label size differs: ' + day)
        digest = sha(path)
        need(digest == report['label_sha256'], 'P2 labels SHA differs: ' + day)
        result[day] = digest
    return result


def prepare(data, outdir, *, cache_root=None):
    """Prepare a new row cache after the caller validates the data manifest."""
    started = time.perf_counter()
    validate_prepared(data)
    outdir = Path(outdir).resolve()
    root = Path(cache_root).resolve() if cache_root is not None else STAGE.resolve()
    need(outdir.is_relative_to(root) and outdir != root,
         'Feature cache must be a new ED4 child directory')
    need(not outdir.exists(), 'Feature cache already exists; no overwrite/retry')
    before = _source_hashes(data)
    labels_before = _label_hashes(data)
    import numpy as np
    maps = {}
    outdir.mkdir()
    try:
        for day in TRAIN + TUNE:
            maps[day] = np.memmap(data['sources'][day]['records_path'], dtype=np.dtype(RECORD_FIELDS), mode='r')

        def chunks():
            for r in data['ranges']:
                if r['role'] != 'train':
                    continue
                for begin in range(r['start_row'], r['stop_row'], CHUNK_ROWS):
                    yield maps[r['date']]['features'][begin:min(begin + CHUNK_ROWS, r['stop_row'])]

        norm, review = fit_unique_statistics(chunks)
        expected_train_rows = sum(r['stop_row'] - r['start_row'] for r in data['ranges'] if r['role'] == 'train')
        need(norm['record_count'] == expected_train_rows, 'Unique TRAIN normalization denominator differs')
        need(np.allclose(np.asarray(norm['mean'])[1::2], np.asarray(data['normalization']['mean'])[1::2],
                         rtol=1e-10, atol=1e-8)
             and np.allclose(np.asarray(norm['std'])[1::2], np.asarray(data['normalization']['std'])[1::2],
                             rtol=1e-8, atol=1e-8), 'Raw quantity TRAIN statistics changed')
        review['quantity_statistics_match_original_P2'] = True
        write_new(outdir / 'normalization.json', norm)
        write_new(outdir / 'statistics_independent_review.json', review)
        cached_ranges = []
        files = {}
        examples = []
        for day in TRAIN + TUNE:
            folder = outdir / day
            folder.mkdir()
            path = folder / 'normalized_rows.f32'
            offset = 0
            with path.open('xb') as stream:
                for r in [r for r in data['ranges'] if r['date'] == day]:
                    cached_ranges.append({**r, 'cache_offset': offset})
                    for begin in range(r['start_row'], r['stop_row'], CHUNK_ROWS):
                        raw = maps[day]['features'][begin:min(begin + CHUNK_ROWS, r['stop_row'])]
                        stream.write(normalized_rows(raw, norm).tobytes(order='C'))
                    for end in sorted({r['endpoint_start'], r['endpoint_start'] + 1,
                                       r['endpoint_stop_exclusive'] - 1}):
                        rows = normalized_rows(maps[day]['features'][end - 99:end + 1], norm)
                        x = np.ascontiguousarray(rows.T)
                        examples.append({'day': day, 'endpoint': end,
                                         'normalized_input_float32_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
                                         'cache_row_start': offset + end - 99 - r['start_row'],
                                         'cache_row_stop_exclusive': offset + end + 1 - r['start_row'],
                                         'input_within_same_registered_range': True})
                    offset += r['stop_row'] - r['start_row']
            need(path.stat().st_size == offset * 40 * 4, 'Normalized cache size differs')
            files[day] = {**binding(path), 'dtype': '<f4', 'shape': [offset, 40],
                          'role': ROLES[day]}
            print(json.dumps({'utc': utc(), 'phase': 'ED4_feature_cache', 'day': day,
                              'normalized_rows': offset, 'model_forwards': 0}, ensure_ascii=True), flush=True)
    finally:
        for a in maps.values():
            a._mmap.close()
    after = _source_hashes(data)
    labels_after = _label_hashes(data)
    need(before == after and labels_before == labels_after, 'Source/labels changed during cache preparation')
    result = {'version': 'TCN-ED4-relative-row-cache-1.0', 'feature_id': FEATURE_ID,
              'complete': True, 'completed_utc': utc(),
              'code_binding': binding(__file__),
              'p2_manifest_binding': {'path': data.get('_manifest_path'), 'sha256': data.get('_manifest_sha256')},
              'normalization': norm, 'normalization_binding': binding(outdir / 'normalization.json'),
              'statistics_independent_review': review,
              'statistics_independent_review_binding': binding(outdir / 'statistics_independent_review.json'),
              'ranges': cached_ranges, 'cache_files': files,
              'sample_counts': data['sample_counts'], 'day_label_reports': data['day_label_reports'],
              'boundary_examples': examples, 'source_sha256_before': before, 'source_sha256_after': after,
              'source_records_unchanged': True, 'labels_sha256_before': labels_before,
              'labels_sha256_after': labels_after, 'label_streams_unchanged': True,
              'source_control_verified_for_new_days': False, 'coverage_unknown_for_new_days': True,
              'original_flag64_cleared': False, 'sealed_records_opened': False,
              'sealed_labels_or_scores_opened': False, 'E_drive_writes': 0,
              'model_forwards': 0, 'training_runs': 0, 'optimizer_updates': 0,
              'elapsed_seconds': time.perf_counter() - started}
    write_new(outdir / 'cache_manifest.json', result)
    return result


def validate_cache(data, prepared, *, cache_root=None, data_root=None):
    """Metadata-only cache contract, before any numeric cache/label mapping."""
    validate_prepared(data)
    need(prepared.get('version') == 'TCN-ED4-relative-row-cache-1.0'
         and prepared.get('feature_id') == FEATURE_ID and prepared.get('complete') is True,
         'Incomplete/wrong relative cache')
    validate_normalization(prepared['normalization'])
    need(prepared.get('sample_counts') == data['sample_counts']
         and prepared.get('day_label_reports') == data['day_label_reports']
         and set(prepared.get('cache_files', {})) == set(TRAIN + TUNE), 'Cache source/sample scope differs')
    need(prepared.get('source_records_unchanged') is True
         and prepared.get('source_sha256_before') == prepared.get('source_sha256_after')
         == {day: data['sources'][day]['records_sha256'] for day in TRAIN + TUNE}
         and prepared.get('label_streams_unchanged') is True
         and prepared.get('labels_sha256_before') == prepared.get('labels_sha256_after')
         == {day: data['day_label_reports'][day]['label_sha256'] for day in TRAIN + TUNE},
         'Cache source/label integrity differs')
    need(len(prepared.get('ranges', [])) == len(data['ranges']), 'Cache ranges missing')
    offsets = dict.fromkeys(TRAIN + TUNE, 0)
    for original, cached in zip(data['ranges'], prepared['ranges']):
        need({k: v for k, v in cached.items() if k != 'cache_offset'} == original,
             'Cache ranges differ from registered P2 ranges')
        day = original['date']
        need(type(cached.get('cache_offset')) is int and cached['cache_offset'] == offsets[day],
             'Cache range offset differs')
        offsets[day] += original['stop_row'] - original['start_row']
    root = Path(cache_root).resolve() if cache_root is not None else STAGE.resolve()
    source_root = Path(data_root).resolve() if data_root is not None else ROOT.resolve()
    for day in TRAIN + TUNE:
        ref = prepared['cache_files'][day]
        need(ref.get('dtype') == '<f4' and ref.get('shape') == [offsets[day], 40]
             and ref.get('role') == ROLES[day] and ref.get('bytes') == offsets[day] * 160,
             'Cache array geometry differs')
        need(Path(ref['path']).resolve().is_relative_to(root),
             'Cache payload outside selected cache root')
        need(Path(data['day_label_reports'][day]['label_path']).resolve().is_relative_to(source_root),
             'Labels outside selected data root')
    return prepared


class RelativeResearchWindows:
    """Immediate F64 reference adapter; formal training uses cached adapter below."""
    def __init__(self, data, role, normalization):
        from software.training.tcn_research_data_p2 import ResearchWindows
        self._base = ResearchWindows(data, role, normalization)
        self.data = data
        self.role = role
        self.entries = self._base.entries
        self.cumulative = self._base.cumulative
        self.normalization = normalization
        validate_normalization(normalization)

    def __len__(self):
        return len(self._base)

    def batch_details(self, indices):
        import numpy as np
        ids = np.asarray(indices)
        need(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer) and len(ids) > 0,
             'Invalid batch indices')
        ids = ids.astype(np.int64, copy=False)
        need(np.all(ids >= 0) and np.all(ids < len(self)), 'Batch index out of bounds')
        groups = np.searchsorted(self.cumulative, ids, side='right')
        x = np.empty((len(ids), 40, 100), np.float32)
        y = np.empty(len(ids), np.int64)
        dates = np.empty(len(ids), dtype='U10')
        ends_out = np.empty(len(ids), np.int64)
        for group in np.unique(groups):
            r = self.entries[group]
            positions = np.flatnonzero(groups == group)
            relative = ids[positions] - (self.cumulative[group - 1] if group else 0)
            ends = r['endpoint_start'] + relative
            a, labels = self._base._maps_for(r['date'])
            raw = a['features'][ends[:, None] + np.arange(-99, 1)]
            x[positions] = normalized_rows(raw, self.normalization).transpose(0, 2, 1)
            y[positions] = labels[r['label_offset'] + relative]
            dates[positions] = r['date']
            ends_out[positions] = ends
        need(np.isfinite(x).all() and np.all((y >= 0) & (y <= 2)), 'Invalid batch numerics')
        return {'X': x, 'y': y, 'date': dates, 'endpoints': ends_out, 'global_indices': ids}

    def batch(self, indices):
        item = self.batch_details(indices)
        return item['X'], item['y']

    def close(self):
        self._base.close()


class CachedRelativeWindows:
    """Gather windows from normalized row caches and label streams."""
    def __init__(self, data, prepared, role, *, cache_root=None, data_root=None):
        import numpy as np
        admitted_days(role)
        validate_cache(data, prepared, cache_root=cache_root, data_root=data_root)
        self.data = data
        self.prepared = prepared
        self.role = role
        self.entries = [r for r in prepared['ranges'] if r['role'] == role]
        self.cumulative = np.cumsum([r['window_count'] for r in self.entries]).tolist()
        self.maps = {}
        self.label_maps = {}
        need(len(self) > 0, 'Empty relative-cache split')

    def __len__(self):
        return self.cumulative[-1] if self.cumulative else 0

    def _maps_for(self, day):
        import numpy as np
        need(day in admitted_days(self.role) and self.data['sources'][day]['role'] == self.role,
             'Forbidden cache day/role')
        if day not in self.maps:
            ref = self.prepared['cache_files'][day]
            path = Path(ref['path'])
            need(path.is_file() and path.stat().st_size == ref['bytes'], 'Cache size changed')
            self.maps[day] = np.memmap(path, dtype='<f4', shape=tuple(ref['shape']), mode='r')
            report = self.data['day_label_reports'][day]
            label_path = Path(report['label_path'])
            need(label_path.is_file() and label_path.stat().st_size == report['label_bytes'], 'Label size changed')
            self.label_maps[day] = np.memmap(label_path, dtype='u1', mode='r')
        return self.maps[day], self.label_maps[day]

    def batch_details(self, indices):
        import numpy as np
        ids = np.asarray(indices)
        need(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer) and len(ids) > 0,
             'Invalid batch indices')
        ids = ids.astype(np.int64, copy=False)
        need(np.all(ids >= 0) and np.all(ids < len(self)), 'Batch index out of bounds')
        groups = np.searchsorted(self.cumulative, ids, side='right')
        x = np.empty((len(ids), 40, 100), np.float32)
        y = np.empty(len(ids), np.int64)
        dates = np.empty(len(ids), dtype='U10')
        ends_out = np.empty(len(ids), np.int64)
        for group in np.unique(groups):
            r = self.entries[group]
            positions = np.flatnonzero(groups == group)
            relative = ids[positions] - (self.cumulative[group - 1] if group else 0)
            ends = r['endpoint_start'] + relative
            cache_ends = r['cache_offset'] + ends - r['start_row']
            rows, labels = self._maps_for(r['date'])
            x[positions] = rows[cache_ends[:, None] + np.arange(-99, 1)].transpose(0, 2, 1)
            y[positions] = labels[r['label_offset'] + relative]
            dates[positions] = r['date']
            ends_out[positions] = ends
        need(np.isfinite(x).all() and np.all((y >= 0) & (y <= 2)), 'Invalid cached batch numerics')
        return {'X': x, 'y': y, 'date': dates, 'endpoints': ends_out, 'global_indices': ids}

    def batch(self, indices):
        item = self.batch_details(indices)
        return item['X'], item['y']

    def close(self):
        for a in list(self.maps.values()) + list(self.label_maps.values()):
            a._mmap.close()
        self.maps.clear()
        self.label_maps.clear()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--registration', required=True)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args(argv)
    reg_path = Path(args.registration).resolve()
    need(reg_path.parent == STAGE.resolve() and sha(reg_path) == args.registration_sha256,
         'ED4 registration identity differs')
    reg = json.loads(reg_path.read_text(encoding='utf-8'))
    need(reg.get('step') == 'ED4' and reg.get('protocol_binding', {}).get('sha256') == PROTOCOL_SHA,
         'ED4 phase/protocol binding differs')
    protocol = read_bound_metadata(reg['protocol_binding'])
    need(protocol.get('campaign_id') == 'TCN_EDO_20261017'
         and protocol['data']['train_dates'] == list(TRAIN)
         and protocol['data']['development_dates'] == list(TUNE), 'Protocol date scope differs')
    authorization = read_bound_metadata(reg['authorization_binding'])
    need(authorization.get('human_authorization') == {
            'source': 'direct_current_human_user_message', 'quote': '帮我完成第4步', 'step': 'ED4'}
         and authorization.get('training_authorized') is True
         and authorization.get('train_dates') == list(TRAIN)
         and authorization.get('development_dates') == list(TUNE)
         and authorization.get('excluded_dates') == ['2025-12-17', '2025-12-18', '2025-12-19'],
         'ED4 authorization/date scope differs')
    need(reg.get('source_bindings', {}).get('scripts/tcn_existing_data_ed4_features.py') == sha(__file__),
         'Feature source differs from frozen registration')
    for name, digest in reg['source_bindings'].items():
        source_path = (ROOT / name).resolve()
        need(source_path.is_relative_to(ROOT.resolve()) and sha(source_path) == digest,
             'Frozen preparation source differs: ' + name)
    manifest_ref = reg['prepared_manifest_binding']
    need(Path(manifest_ref['path']).resolve() == Path(args.manifest).resolve()
         and manifest_ref['sha256'] == args.manifest_sha256
         and manifest_ref['sha256'] == protocol['data']['source_manifest']['sha256'],
         'P2 manifest differs from registration/protocol')
    read_bound_metadata(manifest_ref)
    need(Path(args.output).resolve() == (STAGE / 'prepared').resolve(),
         'Preparation output differs from registered ED4 target')
    from software.training.tcn_research_data_p2 import load_prepared
    data = load_prepared(args.manifest, args.manifest_sha256)
    result = prepare(data, args.output)
    print(json.dumps({'status': 'prepared', 'cache_manifest': str(Path(args.output) / 'cache_manifest.json'),
                      'rows': {day: ref['shape'][0] for day, ref in result['cache_files'].items()},
                      'model_forwards': 0, 'optimizer_updates': 0}, ensure_ascii=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
