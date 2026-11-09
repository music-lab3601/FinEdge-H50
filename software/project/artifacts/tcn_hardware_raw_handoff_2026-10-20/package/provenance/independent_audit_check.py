"""Independent package-only, no-model audit of the 2026-10-20 raw handoff."""
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
import numpy as np

STAGE = Path(__file__).resolve().parent
PKG = STAGE / 'package'
checks = []
read_paths = set()


def need(condition, description):
    if not condition:
        raise AssertionError(description)
    checks.append(description)


def rb(relative):
    p = (PKG / relative).resolve()
    if not p.is_relative_to(PKG.resolve()):
        raise ValueError('Read outside package refused')
    read_paths.add(relative)
    return p.read_bytes()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def js(relative):
    return json.loads(rb(relative).decode('utf-8-sig'))


def main():
    need('torch' not in sys.modules, 'No torch module loaded at audit start')
    m = js('manifest.json')
    initial = {p.relative_to(PKG).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
               for p in PKG.rglob('*') if p.is_file()}
    for rel, ref in m['files'].items():
        b = rb(rel)
        if len(b) != ref['bytes'] or sha(b) != ref['sha256']:
            raise AssertionError('Outer manifest identity mismatch: ' + rel)
    need(len(m['files']) == 53, 'All 53 outer payload sizes and SHA256 digests match')
    need(m['record_count'] == 1024 and m['raw_shape'] == [1024, 40]
         and m['raw_dtype'] == '<u8' and m['normalized_dtype'] == '<f4',
         '1024 by 40 little-endian uint64/raw and float32/reference contract')
    need(m['date'] == '2025-12-08' and m['symbol'] == 'AAPL' and m['role'] == 'train'
         and m['model_seed'] == 42 and m['model_epoch'] == 3,
         'Fixed AAPL TRAIN date and ED4 seed42 epoch3 identity')
    schema = js('provenance/source_schema.json')
    need(schema['row_bytes'] == 352 and schema['dtype'] == m['source_record_dtype']
         and schema['timestamp_unit'] == 'ns since exchange midnight',
         '352-byte source schema and timestamp semantics agree')
    source_bytes, raw_bytes = rb('source_slice.records.bin'), rb('raw_book.u64le.bin')
    normalized_bytes = rb('normalized_rows.f32le.bin')
    masks = np.frombuffer(rb('screening_reject_masks.u32le.bin'), dtype='<u4')
    n = 1024
    need(len(source_bytes) == n * 352 and len(raw_bytes) == n * 320
         and len(normalized_bytes) == n * 160 and masks.shape == (n,),
         'Binary geometries have no padding/header or truncated rows')
    raw = np.frombuffer(raw_bytes, dtype='<u8').reshape(n, 40)
    normalized = np.frombuffer(normalized_bytes, dtype='<f4').reshape(n, 40)
    rows = [json.loads(line) for line in rb('row_metadata.jsonl').decode('utf-8').splitlines()]
    need(len(rows) == n and not np.any(masks), '1024 metadata rows and all rejection masks zero')
    start, stop = m['source_rows']['start_inclusive'], m['source_rows']['stop_exclusive']
    need(start == 120855 and stop == start + n, 'Exact contiguous original record interval [120855,121879)')
    midnight_utc = datetime(2025, 12, 8, 5, tzinfo=timezone.utc)
    midnight_unix_ns = ((date(2025, 12, 8) - date(1970, 1, 1)).days * 86400 + 18000) * 10**9
    headers = []
    for i, row in enumerate(rows):
        rec = struct.unpack_from('<QQIIII40Q', source_bytes, i * 352)
        seq, ts, seg, flags, bid_levels, ask_levels = rec[:6]
        if source_bytes[i * 352 + 32:(i + 1) * 352] != raw[i].tobytes():
            raise AssertionError('Raw features not byte-identical to source row ' + str(i))
        sec, nano = divmod(ts, 10**9)
        expected_local = f'2025-12-08T{sec//3600:02}:{sec//60%60:02}:{sec%60:02}.{nano:09}-05:00'
        expected_utc = (midnight_utc + timedelta(seconds=sec)).strftime('%Y-%m-%dT%H:%M:%S') + f'.{nano:09}Z'
        expected = {'sample_row_0based': i, 'sample_order_1based': i + 1,
            'source_record_row_0based': start + i, 'date': m['date'], 'symbol': m['symbol'],
            'snippet_id': m['snippet_id'], 'research_segment_id': m['research_segment_id'],
            'native_segment_id': seg, 'source_message_position_1based': seq,
            'source_flags': flags, 'research_reject_mask': int(masks[i]),
            'bid_levels': bid_levels, 'ask_levels': ask_levels,
            'timestamp_ns_since_exchange_midnight': ts, 'exchange_datetime_est': expected_local,
            'derived_utc_datetime': expected_utc, 'derived_unix_timestamp_ns': midnight_unix_ns + ts}
        if row != expected:
            raise AssertionError('Metadata not exact source-derived row ' + str(i))
        if seg != m['native_segment_id'] or flags != 0 or min(bid_levels, ask_levels) < 10:
            raise AssertionError('Native eligibility differs at row ' + str(i))
        headers.append(rec[:6])
    need(True, 'Every raw row is original source features bytes; every metadata field matches all 1024 source headers')
    need(True, 'Every EST/UTC string and Unix nanosecond preserves all 9 fractional digits exactly')
    need(all(headers[i][0] > headers[i-1][0] and headers[i][1] >= headers[i-1][1]
             for i in range(1, n)), 'Original message order strictly increases; timestamps never regress')
    need(m['first_row'] == rows[0] and m['last_row'] == rows[-1], 'Manifest first/last metadata agree with sidecar')
    p1, p2 = js('provenance/P1_manifest.json'), js('provenance/P2_manifest.json')
    protocol, screen = js('provenance/screening_protocol.json'), js('provenance/day_screening.json')
    need(sha(rb('provenance/P1_manifest.json')) == p2['p1_manifest_binding']['sha256']
         and sha(rb('provenance/P2_manifest.json')) == '95b45a9faebf2dd5657901c919ac89aff82ef4a0b137c073b7865ec715493ba0'
         and sha(rb('provenance/screening_protocol.json')) == m['source_range']['protocol_sha256'],
         'Copied screening/P1/P2 provenance bindings match frozen identities')
    r1 = next(r for r in p1['ranges'] if r['date'] == m['date'])
    r2 = next(r for r in p2['ranges'] if r['date'] == m['date'])
    need(all(r2[k] == v for k, v in r1.items()) and r2 == m['source_range']
         and start == r2['start_row'] and stop <= r2['stop_row']
         and screen['screening_passed'] and screen['research_window_acceptance_passed']
         and screen['source_control_verified'] and not screen['coverage_unknown'],
         'Snippet is the first 1024 original rows inside accepted same research TRAIN range')
    segments = js('provenance/native_segments.json')
    need(any(s['segment_id'] == m['native_segment_id'] and s['start_row'] <= start
             and s['end_row'] + 1 >= stop for s in segments), 'Entire interval lies in one original native segment')
    need(protocol['timezone_profile']['offset_seconds'] == -18000
         and m['date'] in protocol['timezone_profile']['allowed_dates']
         and all(34500 * 10**9 <= h[1] < 57300 * 10**9 for h in headers),
         'Registered fixed EST date and all timestamps inside frozen 09:35–15:55 research clock')
    need(np.all(raw > 0) and np.all(raw < 2**53)
         and np.all(raw[:, 0] > raw[:, 2])
         and np.all(np.diff(raw[:, 0::4].astype(np.int64), axis=1) > 0)
         and np.all(np.diff(raw[:, 2::4].astype(np.int64), axis=1) < 0),
         'All values positive/exactly representable in float64; uncrossed sorted top10 prices')
    norm = js('ed4_model/normalization.json')
    transformed = raw.astype(np.float64)
    mid = (transformed[:, 0] + transformed[:, 2]) / 2.0
    transformed[:, 0::2] = (transformed[:, 0::2] - mid[:, None]) / mid[:, None]
    mean, std = np.asarray(norm['mean'], dtype=np.float64), np.asarray(norm['std'], dtype=np.float64)
    need(mean.shape == std.shape == (40,) and np.isfinite(mean).all()
         and np.isfinite(std).all() and np.all(std > 0)
         and np.array_equal(transformed[:, 1::2], raw[:, 1::2]),
         'Frozen 40-column normalization is valid and original quantities stay unchanged')
    actual = np.ascontiguousarray((transformed - mean) / std, dtype='<f4')
    need(actual.tobytes() == normalized_bytes and np.isfinite(actual).all()
         and sha(normalized_bytes) == m['ED4_cache_slice_sha256_recomputed'],
         'Independent float64 relative-price/normalization then float32 matches all 40960 stored values bit for bit')
    ws = js('windows.json')
    windows = ws['windows']
    inputs = np.frombuffer(rb('window_inputs.f32le.bin'), dtype='<f4').reshape(6, 40, 100)
    logits = np.frombuffer(rb('window_logits.f32le.bin'), dtype='<f4').reshape(6, 3)
    classes = np.frombuffer(rb('window_classes.u8.bin'), dtype='u1')
    starts = [0, 1, 2, 100, 500, 924]
    need(len(windows) == 6 and classes.shape == (6,) and np.isfinite(logits).all()
         and ws['class_order'] == m['class_order'] == ['down', 'stationary', 'up'],
         'Six explicit finite-logit windows use down/stationary/up class order')
    for i, w in enumerate(windows):
        a, b = starts[i], starts[i] + 100
        if not (w['window_index'] == i and w['window_id'] == f'W{i:02}'
                and w['sample_start_inclusive_0based'] == a and w['sample_stop_exclusive_0based'] == b
                and w['sample_endpoint_0based'] == b-1 and w['source_start_row_0based'] == start+a
                and w['source_endpoint_row_0based'] == start+b-1 and w['date'] == m['date']
                and w['symbol'] == m['symbol'] and w['research_segment_id'] == m['research_segment_id']
                and w['first_source_message_position_1based'] == headers[a][0]
                and w['last_source_message_position_1based'] == headers[b-1][0]
                and w['input_binary_offset_bytes'] == i*16000):
            raise AssertionError('Window mapping differs: ' + str(i))
        for key, index in [('first_time', a), ('last_time', b-1)]:
            expected = {k: rows[index][k] for k in ['timestamp_ns_since_exchange_midnight',
                'exchange_datetime_est', 'derived_utc_datetime', 'derived_unix_timestamp_ns']}
            if w[key] != expected:
                raise AssertionError('Window timestamp differs: ' + str(i))
        if not (inputs[i].tobytes() == np.ascontiguousarray(normalized[a:b].T).tobytes()
                and sha(inputs[i].tobytes()) == w['input_sha256']
                and sha(logits[i].tobytes()) == w['logits_sha256']
                and w['logits_down_stationary_up'] == logits[i].tolist()
                and int(classes[i]) == int(logits[i].argmax()) == w['predicted_class_index']
                and w['predicted_class_name'] == m['class_order'][int(classes[i])]
                and w['is_market_truth_label'] is False):
            raise AssertionError('Window tensor/logits/class identity differs: ' + str(i))
    need(True, 'All six windows map exactly from row references to contiguous channel-first tensors and times')
    need(True, 'Every per-window input/logits SHA, JSON logits and original argmax class matches stored binary outputs')
    mm = js('ed4_model/package_manifest.json')
    need(sha(rb('ed4_model/package_manifest.json')) == m['model_manifest_sha256']
         == 'bf8244237ac5fc4726b33fe2aee49a15b6f5ac2fdb46485bbd6289933fbdabee'
         and sha(rb('ed4_model/normalization.json')) == m['normalization_sha256']
         == mm['normalization_sha256'], 'Portable model/normalization manifest copies match original ED7 identities')
    for rel, ref in mm['files'].items():
        b = rb('ed4_model/' + rel)
        if len(b) != ref['bytes'] or sha(b) != ref['sha256']:
            raise AssertionError('Inner model manifest mismatch: ' + rel)
    need(mm['selected_family'] == m['model_family'] == 'ED4_feature42_lr0.001_seed42'
         and mm['seed'] == 42 and mm['epoch'] == 3
         and mm['semantic_contract']['stationary_logit_bias'] == 0,
         'All inner model payload identities verified with fixed family, seed, epoch and zero bias')
    need('torch' not in sys.modules, 'No torch/model deserialization, model construction or forward during audit')
    final = {p.relative_to(PKG).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
             for p in PKG.rglob('*') if p.is_file()}
    need(initial == final, 'Audit made no package mutations; all package file signatures unchanged')
    result = {'status': 'passed', 'finished_shanghai': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'audit_script_sha256': sha(Path(__file__).read_bytes()), 'checks_passed': len(checks), 'checks': checks,
        'raw_records_checked': n, 'normalized_values_bit_identical': n*40, 'explicit_windows_checked': 6,
        'classes': classes.tolist(), 'outer_payload_hashes_checked': len(m['files']),
        'inner_model_payload_hashes_checked': len(mm['files']), 'package_files_read': sorted(read_paths),
        'package_manifest_sha256': sha(rb('manifest.json')), 'model_forward_calls': 0,
        'model_deserializations': 0, 'training_or_optimizer_operations': 0,
        'E_drive_old_cache_labels_or_sealed_streams_read': False,
        'scope': 'Package-only NumPy/stdlib independent audit. Source_slice is internally tied to raw rows; original E source/cache were not reopened. Model outputs checked against delivered files and argmax, not recomputed by this auditor. Whole-file model assets hashed opaquely without deserialization.'}
    with (STAGE / 'independent_audit.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ['status','checks_passed','raw_records_checked',
        'normalized_values_bit_identical','explicit_windows_checked','classes','model_forward_calls']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
