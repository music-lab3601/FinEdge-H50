"""Validate a delivered, fixed-scope STATUS job; emit facts, never gate MBO rows.

No network, account, market-book, label, model, or optimizer access. Unknown
tri-state values remain unknown per record. No status action is interpreted as
native ITCH Q/M/H/h. This acceptance cannot make a dataset training-ready.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path, PurePosixPath
import re
import struct
import sys

import databento_dbn as dbn

VERSION = 'Databento-status-technical-check-1.0'
DATES = tuple(f'2025-12-{day:02d}' for day in range(15, 20))
START = '2025-12-15T00:00:00Z'
END = '2025-12-20T00:00:00Z'
DAY_NS = 86400 * 10**9
STATUS_ROW = struct.Struct('<BBHIQQHHH3s7s')
TRI_FIELDS = ('is_trading', 'is_quoting', 'is_short_sell_restricted')
ZSTD_SPEC = 'https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md'


def now():
    return datetime.now(timezone.utc).isoformat()


def midnight_ns(day):
    return int(datetime.combine(date.fromisoformat(day), datetime.min.time(),
                                timezone.utc).timestamp()) * 10**9


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024**2):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 4 * 1024**2:
        raise ValueError('Missing or oversized metadata JSON')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, obj):
    # Exclusive writes preserve earlier successful receipts and failed attempts.
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def relative_name(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Expected POSIX relative filename')
    name = PurePosixPath(value)
    if name.is_absolute() or '..' in name.parts or str(name) != value:
        raise ValueError('Unsafe relative filename')
    return value


def validate_receipt(receipt):
    expected = {'dataset': 'XNAS.ITCH', 'schema': 'status', 'symbols': ['AAPL'],
                'start_utc': START, 'end_utc': END}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('Receipt scope differs from approved STATUS request')
    if not re.fullmatch(r'XNAS-[A-Z0-9-]+', receipt.get('job_id', '')):
        raise ValueError('Missing or malformed job ID')
    entries = receipt.get('files')
    if not isinstance(entries, list) or not entries:
        raise ValueError('No receipt files')
    names, data = set(), {}
    for entry in entries:
        name = relative_name(entry.get('path'))
        if name in names:
            raise ValueError('Duplicate delivered file')
        names.add(name)
        if entry.get('role') not in ('data', 'metadata', 'manifest'):
            raise ValueError('Unexpected file role')
        if type(entry.get('bytes')) is not int or entry['bytes'] <= 0:
            raise ValueError('Missing positive publisher file size')
        if not re.fullmatch(r'[0-9a-f]{64}', entry.get('sha256', '')):
            raise ValueError('Missing normalized SHA-256')
        if entry['role'] == 'data':
            match = re.fullmatch(r'xnas-itch-(202512\d\d)\.status\.dbn\.zst', name)
            if match is None:
                raise ValueError('Unexpected status daily filename')
            day = date.fromisoformat(f'{match[1][:4]}-{match[1][4:6]}-{match[1][6:]}').isoformat()
            if day not in DATES or day in data:
                raise ValueError('Wrong or duplicated daily STATUS date')
            data[day] = entry
        elif not name.endswith('.json'):
            raise ValueError('Nondata receipt entry must be JSON metadata')
    if set(data) != set(DATES):
        raise ValueError('Exactly five approved STATUS day files required')
    if sum(entry['role'] == 'manifest' for entry in entries) != 1:
        raise ValueError('Exactly one publisher manifest required')
    if not any(entry['role'] == 'metadata' and entry['path'] == 'metadata.json' for entry in entries):
        raise ValueError('Publisher metadata.json required')
    return data


def local_path(root, name):
    path = root / relative_name(name)
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or not path.is_file():
        raise ValueError('Missing or redirected delivered file')
    return path


def verify_delivery(receipt):
    data = validate_receipt(receipt)
    root = Path(receipt['file_root']).resolve()
    if not root.is_dir():
        raise ValueError('Delivered file root missing')
    checked = {}
    for entry in receipt['files']:
        path = local_path(root, entry['path'])
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError('Publisher size or SHA mismatch: ' + entry['path'])
        checked[entry['path']] = {'path': str(path), 'bytes': entry['bytes'],
                                  'sha256': entry['sha256'], 'role': entry['role']}
    meta = read_json(root / 'metadata.json')
    query = meta.get('query', {})
    if meta.get('job_id') != receipt['job_id'] or any(query.get(k) != v for k, v in
            [('dataset', 'XNAS.ITCH'), ('schema', 'status'), ('symbols', ['AAPL']),
             ('start', midnight_ns(DATES[0])), ('end', midnight_ns('2025-12-20')),
             ('encoding', 'dbn'), ('compression', 'zstd')]):
        raise ValueError('Publisher metadata request mismatch')
    if 'limit' not in query or query['limit'] is not None:
        raise ValueError('STATUS request must have no record limit')
    if meta.get('customizations', {}).get('split_duration') != 'day':
        raise ValueError('STATUS delivery must be split by UTC day')
    manifest_entry = next(e for e in receipt['files'] if e['role'] == 'manifest')
    manifest = read_json(root / manifest_entry['path'])
    if manifest.get('job_id') != receipt['job_id']:
        raise ValueError('Publisher manifest job mismatch')
    publisher = {}
    for item in manifest.get('files', []):
        name = relative_name(item['filename'])
        if name in publisher:
            raise ValueError('Duplicate publisher manifest entry')
        publisher[name] = item
    expected = set(checked) - {manifest_entry['path']}
    if set(publisher) != expected:
        raise ValueError('Publisher manifest file set differs from receipt')
    for name in expected:
        item, entry = publisher[name], checked[name]
        if item.get('size') != entry['bytes'] or item.get('hash') != 'sha256:' + entry['sha256']:
            raise ValueError('Receipt does not match publisher manifest')
    return root, data, checked


def check_zstd_frames(stream):
    """Check frame termination, not decompression or checksum values.

    The official native decoder can return all records from a frame missing its
    last byte. Therefore buffer()==b'' is supplemented with frame/block boundary
    checks. Actual payload decoding remains exclusively in DBNDecoder.
    """
    stream.seek(0, 2)
    total = stream.tell()
    stream.seek(0)
    frames = skipped = blocks = checksums = 0

    def take(size):
        value = stream.read(size)
        if len(value) != size:
            raise ValueError('Truncated Zstd frame/header/checksum')
        return value

    def skip(size):
        if stream.tell() + size > total:
            raise ValueError('Truncated Zstd block/frame payload')
        stream.seek(size, 1)

    while stream.tell() < total:
        magic = int.from_bytes(take(4), 'little')
        if 0x184D2A50 <= magic <= 0x184D2A5F:
            skip(int.from_bytes(take(4), 'little'))
            skipped += 1
            continue
        if magic != 0xFD2FB528:
            raise ValueError('Unexpected compressed frame magic or trailing data')
        frames += 1
        descriptor = take(1)[0]
        if descriptor & 8:
            raise ValueError('Reserved Zstd descriptor bit')
        single = bool(descriptor & 32)
        content_flag = descriptor >> 6
        content_bytes = (1 if single else 0, 2, 4, 8)[content_flag]
        skip((0 if single else 1) + (0, 1, 2, 4)[descriptor & 3] + content_bytes)
        while True:
            header = int.from_bytes(take(3), 'little')
            block_type, block_size = (header >> 1) & 3, header >> 3
            if block_type == 3 or block_size > 128 * 1024:
                raise ValueError('Reserved or oversized Zstd block')
            skip(1 if block_type == 1 else block_size)
            blocks += 1
            if header & 1:
                break
        if descriptor & 4:
            take(4)
            checksums += 1
    if not frames:
        raise ValueError('No Zstd data frames')
    return {'bytes': total, 'frames': frames, 'skippable_frames': skipped,
            'blocks': blocks, 'checksum_fields': checksums,
            'frame_boundaries_complete': True, 'checksum_values_checked_by_this_parser': False,
            'format_reference': ZSTD_SPEC}


def validate_metadata(meta, day):
    if (meta.dataset != 'XNAS.ITCH' or str(meta.schema) != 'status' or
            meta.symbols != ['AAPL'] or meta.partial or meta.not_found or
            str(meta.stype_in) != 'raw_symbol' or str(meta.stype_out) != 'instrument_id' or
            meta.ts_out or meta.limit is not None or meta.version not in (1, 2, 3)):
        raise ValueError('DBN metadata scope/symbology/version mismatch')
    start = midnight_ns(day)
    if meta.start != start or meta.end != start + DAY_NS:
        raise ValueError('DBN metadata interval does not match daily filename')
    mappings = meta.mappings
    if set(mappings) != {'AAPL'}:
        raise ValueError('DBN AAPL symbol mappings mismatch')
    ids = set()
    intervals = []
    day_date = date.fromisoformat(day)
    for item in mappings['AAPL']:
        a, b, symbol = item['start_date'], item['end_date'], item['symbol']
        if not isinstance(a, date) or not isinstance(b, date) or a >= b or not re.fullmatch(r'[1-9][0-9]*', symbol):
            raise ValueError('Malformed DBN instrument mapping interval')
        instrument = int(symbol)
        if instrument >= 2**32:
            raise ValueError('DBN instrument ID exceeds uint32')
        intervals.append({'start_date': a.isoformat(), 'end_date': b.isoformat(), 'symbol': symbol})
        if a <= day_date < b:
            ids.add(instrument)
    if len(ids) != 1:
        raise ValueError('Missing or ambiguous AAPL instrument ID for day')
    return ids, {'version': meta.version, 'dataset': meta.dataset, 'schema': str(meta.schema),
                 'symbols': meta.symbols, 'start': meta.start, 'end': meta.end, 'limit': meta.limit,
                 'instrument_ids': sorted(ids), 'mappings': {'AAPL': intervals}}


def tri_state(value):
    # Strict bool identity: integer 0/1 and '~' are not accepted as implicit state.
    if value is True:
        return 'Y'
    if value is False:
        return 'N'
    if value is None:
        return '~'
    raise ValueError('Unexpected decoded tri-state value')


def validate_record(record, day, instrument_ids, previous_recv=None):
    if int(record.publisher_id) != 2 or int(record.instrument_id) not in instrument_ids:
        raise ValueError('Wrong publisher or daily AAPL instrument ID')
    recv, index = int(record.ts_recv), int(record.ts_index)
    start = midnight_ns(day)
    if index != recv or not start <= recv < start + DAY_NS:
        raise ValueError('STATUS primary/receive timestamp outside approved day')
    if previous_recv is not None and recv < previous_recv:
        raise ValueError('STATUS receive/index timestamp regression')
    raw = bytes(record)
    if len(raw) != STATUS_ROW.size:
        raise ValueError('Unexpected STATUS record byte size')
    fields = STATUS_ROW.unpack(raw)
    if fields[0] * 4 != len(raw) or fields[1] != 18:
        raise ValueError('Unexpected STATUS record header')
    raw_tri = fields[9]
    if any(character not in b'YN~' for character in raw_tri):
        raise ValueError('Invalid raw DBN tri-state encoding')
    states = {name: tri_state(getattr(record, name)) for name in TRI_FIELDS}
    if ''.join(states[name] for name in TRI_FIELDS).encode('ascii') != raw_tri:
        raise ValueError('Raw and decoded tri-state values differ')
    result = {'ts_index': index, 'ts_recv': recv, 'ts_event': int(record.ts_event),
              'publisher_id': int(record.publisher_id), 'instrument_id': int(record.instrument_id),
              **states}
    for name in ('action', 'reason', 'trading_event'):
        value = getattr(record, name)
        number = int(value)
        if not 0 <= number <= 65535:
            raise ValueError('STATUS enum value outside uint16')
        result[name] = number
        result[name + '_enum'] = str(value) if not isinstance(value, int) else 'unrecognized_numeric_value'
    if tuple(result[n] for n in ('action', 'reason', 'trading_event')) != fields[6:9]:
        raise ValueError('Raw and decoded status enums differ')
    return result


def decode_file(path, day, timeline_path):
    """Decode the entire delivered STATUS file once, recording a technical timeline."""
    with Path(path).open('rb') as source:
        frames = check_zstd_frames(source)
    decoder = dbn.DBNDecoder(compression=dbn.Compression.ZSTD)
    metadata, ids, previous = None, None, None
    count = 0
    action_counts, reason_counts, event_counts, combinations = (Counter() for _ in range(4))
    tri_counts = {name: Counter() for name in TRI_FIELDS}
    first = last = None
    event_min = event_max = None
    event_regressions = 0
    previous_event = None
    with Path(path).open('rb') as source, Path(timeline_path).open('x', encoding='utf-8') as timeline:
        def consume(records):
            nonlocal metadata, ids, previous, count, first, last
            nonlocal event_min, event_max, event_regressions, previous_event
            for record in records:
                if isinstance(record, dbn.Metadata):
                    if metadata is not None or count:
                        raise ValueError('Repeated or late DBN metadata')
                    ids, metadata = validate_metadata(record, day)
                    continue
                if metadata is None or not isinstance(record, dbn.StatusMsg):
                    raise ValueError('Non-STATUS record or STATUS before metadata')
                fact = validate_record(record, day, ids, previous)
                count += 1
                fact['record_ordinal'] = count
                timeline.write(json.dumps(fact, ensure_ascii=False, allow_nan=False) + '\n')
                previous = fact['ts_recv']
                first = fact['ts_recv'] if first is None else first
                last = fact['ts_recv']
                event = fact['ts_event']
                if event != dbn.UNDEF_TIMESTAMP:
                    event_min = event if event_min is None else min(event_min, event)
                    event_max = event if event_max is None else max(event_max, event)
                    event_regressions += int(previous_event is not None and event < previous_event)
                    previous_event = event
                action_counts[str(fact['action'])] += 1
                reason_counts[str(fact['reason'])] += 1
                event_counts[str(fact['trading_event'])] += 1
                for name in TRI_FIELDS:
                    tri_counts[name][fact[name]] += 1
                combination = '|'.join(str(fact[n]) for n in
                    ('action', 'reason', 'trading_event', *TRI_FIELDS))
                combinations[combination] += 1
        while block := source.read(1024**2):
            consume(decoder.write_and_decode(block))
        consume(decoder.decode())
        if decoder.buffer():
            raise ValueError('Undecoded DBN bytes at EOF')
    if metadata is None or count == 0:
        raise ValueError('No STATUS metadata or records for approved day')
    return {'date': day, 'records': count, 'metadata': metadata,
            'first_ts_recv': first, 'last_ts_recv': last,
            'ts_recv_and_index_monotonic': True, 'ts_event_min': event_min,
            'ts_event_max': event_max, 'ts_event_regressions_observed': event_regressions,
            'action_counts': dict(action_counts), 'reason_counts': dict(reason_counts),
            'trading_event_counts': dict(event_counts),
            'tri_state_counts': {k: dict(v) for k, v in tri_counts.items()},
            'field_combination_order': ['action', 'reason', 'trading_event', *TRI_FIELDS],
            'field_combination_counts': dict(combinations),
            'compressed_frame_check': frames, 'decoder_eof_buffer_empty': True,
            'native_sequence_available': False, 'unknown_state_carried_forward': False,
            'mapping_verified': False, 'training_ready': False,
            'timeline_path': str(Path(timeline_path).resolve()), 'timeline_sha256': sha(timeline_path)}


def run(receipt_path, output):
    if version('databento-dbn') != '0.71.0':
        raise ValueError('Pinned databento-dbn dependency version differs')
    receipt_path, output = Path(receipt_path).resolve(), Path(output).resolve()
    receipt = read_json(receipt_path)
    root, data, before = verify_delivery(receipt)
    if output.is_relative_to(root) or root.is_relative_to(output) or receipt_path.is_relative_to(root):
        raise ValueError('Output/receipt must be separate from delivered source tree')
    output.mkdir(parents=True, exist_ok=True)
    acceptance = output / 'status_technical_acceptance.json'
    if acceptance.exists() or any((output / f'status_{day}.json').exists() for day in DATES):
        raise ValueError('Existing technical acceptance: do not repeat full decoding')
    days = {}
    for day in DATES:
        source = root / data[day]['path']
        result = decode_file(source, day, output / f'status_timeline_{day}.jsonl')
        if sha(source) != data[day]['sha256'] or source.stat().st_size != data[day]['bytes']:
            raise ValueError('Source changed during STATUS decoding')
        result.update({'source_path': str(source), 'source_sha256': data[day]['sha256'],
                       'source_bytes': data[day]['bytes']})
        write_json(output / f'status_{day}.json', result)
        days[day] = result
        print(json.dumps({'date': day, 'status_records': result['records'],
                          'mapping_verified': False, 'training_ready': False}), flush=True)
    _, _, after = verify_delivery(receipt)
    if before != after:
        raise ValueError('Delivery identity changed during verification')
    report = {'version': VERSION, 'completed_utc': now(), 'job_id': receipt['job_id'],
              'status': 'status_download_integrity_and_technical_decode_passed',
              'receipt_path': str(receipt_path), 'receipt_sha256': sha(receipt_path),
              'checker_path': str(Path(__file__).resolve()), 'checker_sha256': sha(__file__),
              'dependency': {'databento-dbn': version('databento-dbn')},
              'files': before, 'days': days, 'total_records': sum(d['records'] for d in days.values()),
              'source_identity_before_after_equal': True, 'publisher_manifest_checked': True,
              'mapping_verified': False, 'training_ready': False,
              'native_ITCH_Q_M_H_h_mapping_verified': False,
              'status_to_mbo_join_performed': False, 'mbo_flag64_cleared': False,
              'unknown_state_carried_forward': False, 'old_raw_records_models_opened': False,
              'labels_computed': False, 'model_forward_count': 0, 'optimizer_updates': 0,
              'limitations': ['Technical STATUS acceptance does not prove the original ITCH Q/M/H/h gate mapping.',
                              'Status contains no native sequence; equal-timestamp join ordering is not proven.',
                              'Y/N/~ are recorded per message; unknown values do not inherit any previous state.']}
    write_json(acceptance, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run(args.receipt, args.output)
    print(json.dumps({'status': report['status'], 'job_id': report['job_id'],
                      'total_records': report['total_records'], 'mapping_verified': False,
                      'training_ready': False}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'status_technical_check_failed',
                          'error_type': type(error).__name__, 'error': str(error),
                          'mapping_verified': False, 'training_ready': False}), file=sys.stderr)
        raise SystemExit(1)
