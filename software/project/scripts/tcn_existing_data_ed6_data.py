"""ED6 shared D-row-cache interface and fixed-P2-normalization control cache.

No files or numeric/ML libraries are opened/imported merely by importing this
module. The control cache is built once by the root after registration; the
candidate inherits the already accepted ED4 relative-price cache unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from tcn_research_data_p2 import (ROOT, RECORD_FIELDS, TRAIN, TUNE, ROLES,
                                  validate_prepared, admitted_days)

STAGE = ROOT / 'artifacts/tcn_existing_data_optimization_2026-10-17/step06_replication'
PARENT = STAGE.parent / 'step04_features'
CONTROL_DIR = STAGE / 'prepared_control'
PROTOCOL_SHA = '09654e6dfdb10b69feb6ee950c6668e71640bcea1d56fa9691a9564c8641fcf3'
P2_NORM_SHA = '60130df9c2ed5673395e0247dcc8279f297a288719ec9187097ad4823a0a8c0b'
CONTROL_FEATURE_ID = 'original_physical_prices_raw_quantities_frozen_P2_norm_v1'
CONTROL_VERSION = 'TCN-ED6-original-row-cache-1.0'
CHUNK_ROWS = 65536
SLOTS = ('control43', 'candidate43')


def need(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def local(value, base=ROOT):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    need('..' not in p.parts and p.is_relative_to(base), 'Outside admitted D path')
    for q in (p, *p.parents):
        if q.exists():
            need(not q.is_symlink() and not getattr(q.lstat(), 'st_file_attributes', 0) & 0x400,
                 'Reparse D path forbidden')
        if q == base:
            break
    need(p.resolve().is_relative_to(base.resolve()), 'Resolved path escaped admitted D root')
    return p


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    p = local(path)
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def read_bound(ref):
    need(isinstance(ref, dict) and {'path', 'sha256'} <= set(ref), 'Missing metadata binding')
    p = local(ref['path'])
    need(p.is_file() and p.stat().st_size <= 16 * 1024**2 and sha(p) == ref['sha256']
         and ('bytes' not in ref or p.stat().st_size == ref['bytes']), 'Metadata identity differs')
    return json.loads(p.read_text('utf-8-sig'))


def write_new(path, value):
    p = local(path, STAGE)
    with p.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def original_normalized_rows(raw_features, normalization):
    """Exactly P2 physical F64 -> frozen mean/std F64 -> F32; no fit/log."""
    import numpy as np
    from tcn_research_data_p2 import physical
    raw = np.asarray(raw_features)
    need(raw.ndim >= 1 and raw.shape[-1] == 40, 'Raw feature geometry differs')
    x = physical(raw)
    mean = np.asarray(normalization['mean'], np.float64)
    std = np.asarray(normalization['std'], np.float64)
    need(mean.shape == std.shape == (40,) and np.isfinite(mean).all()
         and np.isfinite(std).all() and np.all(std > 0), 'Invalid frozen P2 normalization')
    need(np.isfinite(x).all(), 'Invalid original physical features')
    result = ((x - mean) / std).astype('<f4')
    need(np.isfinite(result).all(), 'Nonfinite original normalized F32 rows')
    return result


def verify_sources(data):
    result = {}
    for day in TRAIN + TUNE:
        source = data['sources'][day]
        path = Path(source['records_path'])
        need(source['day'] == day and source['role'] == ROLES[day]
             and path.is_file() and path.stat().st_size == source['rows'] * 352,
             'Source role/geometry differs: ' + day)
        result[day] = sha(path)
        need(result[day] == source['records_sha256'], 'Source SHA differs: ' + day)
    return result


def verify_labels(data):
    result = {}
    for day in TRAIN + TUNE:
        item = data['day_label_reports'][day]
        path = local(item['label_path'])
        need(path.is_file() and path.stat().st_size == item['label_bytes'] == item['window_count'],
             'Original label size differs: ' + day)
        result[day] = sha(path)
        need(result[day] == item['label_sha256'], 'Original label SHA differs: ' + day)
    return result


def prepare_control(data, outdir=CONTROL_DIR):
    """Root-only one-time original cache; keeps all old sources/norm/labels."""
    started = time.perf_counter()
    validate_prepared(data)
    outdir = local(outdir, STAGE)
    need(outdir == CONTROL_DIR and not outdir.exists(), 'Control cache output is fixed/new only')
    norm_ref = data['normalization_binding']
    need(norm_ref['sha256'] == P2_NORM_SHA and read_bound(norm_ref) == data['normalization'],
         'Original fixed P2 normalization identity differs')
    before = verify_sources(data)
    labels_before = verify_labels(data)
    import numpy as np
    dtype = np.dtype(RECORD_FIELDS)
    need(dtype.itemsize == 352, 'Original record geometry differs')
    outdir.mkdir()
    cached_ranges = []
    files = {}
    examples = []
    boundary_count = 0
    for day in TRAIN + TUNE:
        source = data['sources'][day]
        records = np.memmap(source['records_path'], dtype=dtype, mode='r')
        folder = outdir / day
        folder.mkdir()
        path = folder / 'normalized_rows.f32'
        offset = 0
        try:
            with path.open('xb') as stream:
                for r in [r for r in data['ranges'] if r['date'] == day]:
                    cached_ranges.append({**r, 'cache_offset': offset})
                    for begin in range(r['start_row'], r['stop_row'], CHUNK_ROWS):
                        raw = records['features'][begin:min(begin + CHUNK_ROWS, r['stop_row'])]
                        stream.write(original_normalized_rows(raw, data['normalization']).tobytes(order='C'))
                    refs = data['day_label_reports'][day]['first_second_last_window_examples']
                    selected = sorted({r['endpoint_start'], r['endpoint_start'] + 1,
                                       r['endpoint_stop_exclusive'] - 1})
                    for end in selected:
                        reference = [v for v in refs if v['endpoint'] == end]
                        need(len(reference) == 1, 'Original P2 boundary reference missing/duplicated')
                        rows = original_normalized_rows(records['features'][end - 99:end + 1], data['normalization'])
                        x = np.ascontiguousarray(rows.T)
                        digest = hashlib.sha256(x.tobytes()).hexdigest()
                        need(digest == reference[0]['input_float32_sha256'],
                             'Control normalized input is not byte-identical to original P2')
                        examples.append({'day': day, 'endpoint': end,
                            'normalized_input_float32_sha256': digest,
                            'original_P2_input_float32_sha256': reference[0]['input_float32_sha256'],
                            'cache_row_start': offset + end - 99 - r['start_row'],
                            'cache_row_stop_exclusive': offset + end + 1 - r['start_row'],
                            'input_within_same_registered_range': True})
                        boundary_count += 1
                    offset += r['stop_row'] - r['start_row']
        finally:
            records._mmap.close()
        need(path.stat().st_size == offset * 160, 'Control row-cache geometry differs')
        # Verify the actual persisted row cache against all three old P2 golden
        # windows; this is a data-byte check with zero model forwards.
        written = np.memmap(path, dtype='<f4', mode='r', shape=(offset, 40))
        try:
            for ex in [v for v in examples if v['day'] == day]:
                x = np.ascontiguousarray(written[ex['cache_row_start']:ex['cache_row_stop_exclusive']].T)
                need(hashlib.sha256(x.tobytes()).hexdigest() == ex['original_P2_input_float32_sha256'],
                     'Persisted control cache does not reproduce original P2 golden input')
        finally:
            written._mmap.close()
        files[day] = {**binding(path), 'dtype': '<f4', 'shape': [offset, 40], 'role': ROLES[day]}
        print(json.dumps({'utc': utc(), 'phase': 'ED6_control_cache', 'day': day,
                          'rows': offset, 'model_forwards': 0}, ensure_ascii=True), flush=True)
    need(boundary_count == 21 and sum(v['shape'][0] for v in files.values()) == 18994144,
         'Control cache finite row/boundary denominator differs')
    after = verify_sources(data)
    labels_after = verify_labels(data)
    need(before == after and labels_before == labels_after, 'Sources or original labels changed during cache preparation')
    result = {'version': CONTROL_VERSION, 'feature_id': CONTROL_FEATURE_ID, 'complete': True,
              'completed_utc': utc(), 'worker_pid': os.getpid(), 'code_binding': binding(__file__),
              'p2_manifest_binding': {'path': data.get('_manifest_path'), 'sha256': data.get('_manifest_sha256')},
              'normalization': data['normalization'], 'normalization_binding': binding(norm_ref['path']),
              'normalization_refitted': False, 'original_P2_normalization_reused': True,
              'statistics_independent_review': data['statistics_independent_review'],
              'statistics_independent_review_binding': data['statistics_independent_review_binding'],
              'ranges': cached_ranges, 'cache_files': files, 'sample_counts': data['sample_counts'],
              'day_label_reports': data['day_label_reports'], 'boundary_examples': examples,
              'boundary_inputs_byte_equal_original_P2': True, 'boundary_inputs_checked': boundary_count,
              'source_sha256_before': before, 'source_sha256_after': after,
              'source_records_unchanged': True, 'labels_sha256_before': labels_before,
              'labels_sha256_after': labels_after, 'label_streams_unchanged': True,
              'source_control_verified_for_new_days': False, 'coverage_unknown_for_new_days': True,
              'original_flag64_cleared': False, 'sealed_records_opened': False,
              'sealed_labels_or_scores_opened': False, 'E_drive_writes': 0,
              'model_forwards': 0, 'training_runs': 0, 'optimizer_updates': 0,
              'elapsed_seconds': time.perf_counter() - started}
    write_new(outdir / 'cache_manifest.json', result)
    return result


def validate_cache(data, prepared, slot):
    """Metadata-only variant/path/range admission; trainer hashes payloads."""
    need(slot in SLOTS, 'Unknown ED6 slot')
    validate_prepared(data)
    if slot == 'candidate43':
        from tcn_existing_data_ed4_features import validate_cache as validate_relative
        validate_relative(data, prepared)
        return prepared
    import numpy as np
    need(prepared.get('version') == CONTROL_VERSION and prepared.get('feature_id') == CONTROL_FEATURE_ID
         and prepared.get('complete') is True and prepared.get('normalization_refitted') is False
         and prepared.get('original_P2_normalization_reused') is True, 'Not accepted original control cache')
    need(prepared['normalization'] == data['normalization']
         and prepared['normalization_binding']['sha256'] == P2_NORM_SHA
         and prepared['sample_counts'] == data['sample_counts']
         and prepared['day_label_reports'] == data['day_label_reports'], 'Control normalization/data scope differs')
    mean = np.asarray(prepared['normalization']['mean']);std = np.asarray(prepared['normalization']['std'])
    need(mean.shape == std.shape == (40,) and np.isfinite(mean).all() and np.isfinite(std).all()
         and np.all(std > 0), 'Invalid original normalization')
    need(prepared.get('source_records_unchanged') is True
         and prepared['source_sha256_before'] == prepared['source_sha256_after']
         == {day: data['sources'][day]['records_sha256'] for day in TRAIN + TUNE}
         and prepared.get('label_streams_unchanged') is True
         and prepared['labels_sha256_before'] == prepared['labels_sha256_after']
         == {day: data['day_label_reports'][day]['label_sha256'] for day in TRAIN + TUNE},
         'Control source/label integrity evidence differs')
    need(len(prepared['ranges']) == len(data['ranges'])
         and set(prepared['cache_files']) == set(TRAIN + TUNE), 'Control date/range scope differs')
    offsets = dict.fromkeys(TRAIN + TUNE, 0)
    for original, cached in zip(data['ranges'], prepared['ranges']):
        need({k: v for k, v in cached.items() if k != 'cache_offset'} == original
             and type(cached.get('cache_offset')) is int and cached['cache_offset'] == offsets[original['date']],
             'Control cache range/offset differs')
        offsets[original['date']] += original['stop_row'] - original['start_row']
    for day in TRAIN + TUNE:
        item = prepared['cache_files'][day]
        need(item['dtype'] == '<f4' and item['shape'] == [offsets[day], 40]
             and item['bytes'] == offsets[day] * 160 and item['role'] == ROLES[day], 'Control cache geometry differs')
        local(item['path'], CONTROL_DIR)
        local(data['day_label_reports'][day]['label_path'])
    return prepared


class CachedWindows:
    """Unified arbitrary-index D-cache windows; all returned anchors remain P2."""
    def __init__(self, data, prepared, role, slot):
        import numpy as np
        admitted_days(role)
        validate_cache(data, prepared, slot)
        self.data = data;self.prepared = prepared;self.role = role;self.slot = slot
        self.entries = [r for r in prepared['ranges'] if r['role'] == role]
        self.cumulative = np.cumsum([r['window_count'] for r in self.entries]).tolist()
        self.maps = {};self.label_maps = {}
        need(len(self) > 0, 'Empty ED6 cache split')

    def __len__(self):
        return self.cumulative[-1] if self.cumulative else 0

    def _maps_for(self, day):
        import numpy as np
        need(day in admitted_days(self.role) and self.data['sources'][day]['role'] == self.role,
             'Forbidden cache day/role')
        if day not in self.maps:
            item = self.prepared['cache_files'][day]
            base = CONTROL_DIR if self.slot == 'control43' else PARENT / 'prepared'
            path = local(item['path'], base)
            need(path.is_file() and path.stat().st_size == item['bytes'], 'Cache size changed')
            self.maps[day] = np.memmap(path, dtype='<f4', shape=tuple(item['shape']), mode='r')
            lab = self.data['day_label_reports'][day]
            path = local(lab['label_path'])
            need(path.is_file() and path.stat().st_size == lab['label_bytes'], 'Label size changed')
            self.label_maps[day] = np.memmap(path, dtype='u1', mode='r')
        return self.maps[day], self.label_maps[day]

    def batch_details(self, indices):
        import numpy as np
        ids = np.asarray(indices)
        need(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer) and len(ids) > 0, 'Invalid indices')
        ids = ids.astype(np.int64, copy=False)
        need(np.all(ids >= 0) and np.all(ids < len(self)), 'Out-of-bounds indices')
        groups = np.searchsorted(self.cumulative, ids, side='right')
        x = np.empty((len(ids), 40, 100), np.float32);y = np.empty(len(ids), np.int64)
        dates = np.empty(len(ids), dtype='U10');ends_out = np.empty(len(ids), np.int64)
        for group in np.unique(groups):
            r = self.entries[group];positions = np.flatnonzero(groups == group)
            relative = ids[positions] - (self.cumulative[group - 1] if group else 0)
            ends = r['endpoint_start'] + relative
            cache_ends = r['cache_offset'] + ends - r['start_row']
            rows, labels = self._maps_for(r['date'])
            x[positions] = rows[cache_ends[:, None] + np.arange(-99, 1)].transpose(0, 2, 1)
            y[positions] = labels[r['label_offset'] + relative]
            dates[positions] = r['date'];ends_out[positions] = ends
        need(np.isfinite(x).all() and np.all((y >= 0) & (y <= 2)), 'Invalid cached input/labels')
        return {'X': x, 'y': y, 'date': dates, 'endpoints': ends_out, 'global_indices': ids}

    def batch(self, indices):
        item = self.batch_details(indices)
        return item['X'], item['y']

    def close(self):
        for a in list(self.maps.values()) + list(self.label_maps.values()):
            a._mmap.close()
        self.maps.clear();self.label_maps.clear()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare-control'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--registration', required=True)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args(argv)
    rp = local(args.registration, STAGE)
    need(rp.parent == STAGE and sha(rp) == args.registration_sha256, 'Wrong preparation registration')
    reg = json.loads(rp.read_text('utf-8-sig'))
    need(reg.get('step') == 'ED6', 'Wrong cache preparation phase')
    protocol = read_bound(reg['protocol_binding'])
    need(reg['protocol_binding']['sha256'] == PROTOCOL_SHA
         and protocol['campaign_id'] == 'TCN_EDO_20261017', 'Pinned campaign protocol differs')
    authorization = read_bound(reg['authorization_binding'])
    need(authorization['human_authorization'] == {'source': 'direct_current_human_user_message',
         'quote': '帮我完成第6步', 'step': 'ED6'} and authorization['training_authorized'] is True,
         'Current human ED6 authorization missing')
    source = reg['prepared_manifest_binding']
    need(local(source['path']) == local(args.manifest) and source['sha256'] == args.manifest_sha256
         == protocol['data']['source_manifest']['sha256'], 'P2 manifest identity differs')
    norm = read_bound(reg['normalization_binding'])
    need(reg['normalization_binding']['sha256'] == P2_NORM_SHA
         == protocol['data']['baseline_normalization']['sha256'], 'Original P2 norm identity differs')
    need(reg['source_bindings'].get('scripts/tcn_existing_data_ed6_data.py') == sha(__file__),
         'Registered ED6 adapter differs')
    for name, digest in reg['source_bindings'].items():
        need(sha(local(name)) == digest, 'Frozen numerical dependency differs: ' + name)
    from tcn_research_data_p2 import load_prepared
    data = load_prepared(args.manifest, args.manifest_sha256)
    need(norm == data['normalization'], 'No replacement or newly fitted normalization permitted')
    result = prepare_control(data, args.output)
    print(json.dumps({'status': 'prepared_control', 'cache_manifest': str(CONTROL_DIR / 'cache_manifest.json'),
                      'rows': sum(v['shape'][0] for v in result['cache_files'].values()),
                      'boundary_inputs_equal_original_P2': 21, 'model_forwards': 0,
                      'optimizer_updates': 0}, ensure_ascii=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
