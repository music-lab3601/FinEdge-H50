"""Synthetic STATUS contracts; never access existing market files or models."""
from datetime import date, timedelta
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import databento_status_check as check
import databento_dbn as dbn


DAY = check.DATES[0]
INSTRUMENT = 38


def meta(day=DAY, **overrides):
    day_date = date.fromisoformat(day)
    values = dict(dataset='XNAS.ITCH', start=check.midnight_ns(day),
                  end=check.midnight_ns(day) + check.DAY_NS,
                  stype_in=dbn.SType.RAW_SYMBOL, stype_out=dbn.SType.INSTRUMENT_ID,
                  schema=dbn.Schema.STATUS, symbols=['AAPL'],
                  mappings=[SimpleNamespace(raw_symbol='AAPL', intervals=[SimpleNamespace(
                      start_date=day_date, end_date=day_date + timedelta(days=1),
                      symbol=str(INSTRUMENT))])])
    values.update(overrides)
    return dbn.Metadata(**values)


def status(day=DAY, offset=2, **overrides):
    values = dict(publisher_id=2, instrument_id=INSTRUMENT,
                  ts_event=check.midnight_ns(day) + offset - 1,
                  ts_recv=check.midnight_ns(day) + offset,
                  action=dbn.StatusAction.TRADING, is_trading=dbn.TriState.YES,
                  is_quoting=dbn.TriState.YES, is_short_sell_restricted=dbn.TriState.NO)
    values.update(overrides)
    return dbn.StatusMsg(**values)


def compress(data):
    output = io.BytesIO()
    transcoder = dbn.Transcoder(output, dbn.Encoding.DBN, dbn.Compression.ZSTD)
    transcoder.write(data)
    transcoder.finish()
    return output.getvalue()


def payload(day=DAY, records=None, metadata=None):
    if records is None:
        records = [status(day, offset=2), status(day, offset=3, is_trading=dbn.TriState.NOT_AVAILABLE),
                   status(day, offset=4, action=dbn.StatusAction.HALT, is_trading=dbn.TriState.NO)]
    return compress(bytes(metadata or meta(day)) + b''.join(bytes(record) for record in records))


def entry(path, role):
    content = path.read_bytes()
    return dict(path=path.name, role=role, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def delivered_fixture(parent):
    source = parent / 'source'
    source.mkdir()
    files = []
    for day in check.DATES:
        path = source / f'xnas-itch-{day.replace("-", "")}.status.dbn.zst'
        path.write_bytes(payload(day))
        files.append(entry(path, 'data'))
    job = 'XNAS-TEST-STATUS'
    metadata = {'job_id': job, 'query': {'dataset': 'XNAS.ITCH', 'schema': 'status',
        'symbols': ['AAPL'], 'start': check.midnight_ns(check.DATES[0]),
        'end': check.midnight_ns('2025-12-20'), 'limit': None, 'encoding': 'dbn', 'compression': 'zstd'},
        'customizations': {'split_duration': 'day'}}
    (source / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    (source / 'condition.json').write_text('{"synthetic": true}', encoding='utf-8')
    files.extend([entry(source / 'metadata.json', 'metadata'), entry(source / 'condition.json', 'metadata')])
    manifest = {'job_id': job, 'files': [dict(filename=item['path'], size=item['bytes'],
                                            hash='sha256:' + item['sha256']) for item in files]}
    (source / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    files.append(entry(source / 'manifest.json', 'manifest'))
    return dict(job_id=job, dataset='XNAS.ITCH', schema='status', symbols=['AAPL'],
                start_utc=check.START, end_utc=check.END, file_root=str(source), files=files)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-status-check-')
        self.parent = Path(self.temp.name)
        self.receipt = delivered_fixture(self.parent)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_scope_and_file_set(self):
        self.assertEqual(set(check.validate_receipt(self.receipt)), set(check.DATES))

    def test_wrong_dataset_schema_symbol_or_interval_refused(self):
        for key, wrong in [('dataset', 'GLBX.MDP3'), ('schema', 'mbo'), ('symbols', ['MSFT']),
                           ('start_utc', '2025-12-14T00:00:00Z'), ('end_utc', '2025-12-19T00:00:00Z')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                check.validate_receipt({**self.receipt, key: wrong})

    def test_missing_day_refused(self):
        self.receipt['files'].pop(0)
        with self.assertRaises(ValueError):
            check.validate_receipt(self.receipt)

    def test_duplicate_file_refused(self):
        self.receipt['files'].append(self.receipt['files'][0].copy())
        with self.assertRaises(ValueError):
            check.validate_receipt(self.receipt)

    def test_nonstatus_day_filename_refused(self):
        self.receipt['files'][0]['path'] = 'xnas-itch-20251215.mbo.dbn.zst'
        with self.assertRaises(ValueError):
            check.validate_receipt(self.receipt)

    def test_unsafe_path_refused(self):
        for path in ('../outside.dbn.zst', '@EXTERNAL_ROOT@/outside.dbn.zst', '/outside', 'nested\\outside'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                check.relative_name(path)

    def test_missing_hash_or_size_refused(self):
        for key, value in [('sha256', ''), ('bytes', 0), ('bytes', True)]:
            item = self.receipt['files'][0]
            previous = item[key]
            item[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                check.validate_receipt(self.receipt)
            item[key] = previous

    def test_actual_hash_tampering_refused(self):
        path = Path(self.receipt['file_root']) / self.receipt['files'][0]['path']
        path.write_bytes(path.read_bytes() + b'bad')
        with self.assertRaises(ValueError):
            check.verify_delivery(self.receipt)

    def test_publisher_manifest_binding_refused(self):
        path = Path(self.receipt['file_root']) / 'manifest.json'
        value = json.loads(path.read_text())
        value['files'][0]['hash'] = 'sha256:' + '0' * 64
        path.write_text(json.dumps(value))
        self.receipt['files'][-1] = entry(path, 'manifest')
        with self.assertRaises(ValueError):
            check.verify_delivery(self.receipt)

    def test_publisher_metadata_scope_refused(self):
        path = Path(self.receipt['file_root']) / 'metadata.json'
        value = json.loads(path.read_text())
        value['query']['schema'] = 'mbo'
        path.write_text(json.dumps(value))
        index = next(i for i, e in enumerate(self.receipt['files']) if e['path'] == 'metadata.json')
        self.receipt['files'][index] = entry(path, 'metadata')
        with self.assertRaises(ValueError):
            check.verify_delivery(self.receipt)

    def test_full_synthetic_delivery_stays_unverified_and_no_state_carry(self):
        receipt_path = self.parent / 'receipt.json'
        receipt_path.write_text(json.dumps(self.receipt))
        result = check.run(receipt_path, self.parent / 'technical')
        self.assertEqual(result['total_records'], 15)
        self.assertFalse(result['mapping_verified'])
        self.assertFalse(result['training_ready'])
        self.assertFalse(result['mbo_flag64_cleared'])
        self.assertFalse(result['labels_computed'])
        self.assertEqual(result['optimizer_updates'], 0)
        for item in result['days'].values():
            self.assertEqual(item['tri_state_counts']['is_trading'], {'Y': 1, '~': 1, 'N': 1})
            rows = [json.loads(line) for line in Path(item['timeline_path']).read_text().splitlines()]
            self.assertEqual([row['is_trading'] for row in rows], ['Y', '~', 'N'])
            self.assertEqual(rows[0]['action'], 7)
            self.assertNotIn('regular', rows[0])
        with self.assertRaises(ValueError):
            check.run(receipt_path, self.parent / 'technical')


class MetadataTests(unittest.TestCase):
    def test_daily_id_and_metadata_match(self):
        ids, result = check.validate_metadata(meta(), DAY)
        self.assertEqual(ids, {INSTRUMENT})
        self.assertEqual(result['schema'], 'status')

    def test_wrong_dbn_scope_refused(self):
        for key, wrong in [('dataset', 'GLBX.MDP3'), ('schema', dbn.Schema.MBO), ('symbols', ['MSFT']),
                           ('partial', ['AAPL']), ('not_found', ['AAPL']), ('ts_out', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                check.validate_metadata(meta(**{key: wrong}), DAY)

    def test_wrong_split_interval_refused(self):
        with self.assertRaises(ValueError):
            check.validate_metadata(meta(start=check.midnight_ns(DAY) - 1), DAY)
        with self.assertRaises(ValueError):
            check.validate_metadata(meta(end=check.midnight_ns(DAY) + check.DAY_NS - 1), DAY)

    def test_finite_record_limit_refused(self):
        with self.assertRaises(ValueError):
            check.validate_metadata(meta(limit=1), DAY)

    def test_missing_or_ambiguous_instrument_mapping_refused(self):
        today = date.fromisoformat(DAY)
        first = SimpleNamespace(start_date=today, end_date=today + timedelta(days=1), symbol='38')
        second = SimpleNamespace(start_date=today, end_date=today + timedelta(days=1), symbol='39')
        with self.assertRaises(ValueError):
            check.validate_metadata(meta(mappings=[]), DAY)
        with self.assertRaises(ValueError):
            check.validate_metadata(meta(mappings=[SimpleNamespace(raw_symbol='AAPL', intervals=[first, second])]), DAY)


class RecordTests(unittest.TestCase):
    def test_valid_record_values_are_facts_only(self):
        result = check.validate_record(status(), DAY, {INSTRUMENT})
        self.assertEqual(result['is_trading'], 'Y')
        self.assertEqual(result['action'], 7)
        self.assertEqual(result['ts_index'], result['ts_recv'])
        self.assertNotIn('native_system_phase', result)

    def test_wrong_publisher_refused(self):
        with self.assertRaises(ValueError):
            check.validate_record(status(publisher_id=3), DAY, {INSTRUMENT})

    def test_wrong_instrument_refused(self):
        with self.assertRaises(ValueError):
            check.validate_record(status(instrument_id=39), DAY, {INSTRUMENT})

    def test_timestamp_out_of_day_refused(self):
        for timestamp in (check.midnight_ns(DAY) - 1, check.midnight_ns(DAY) + check.DAY_NS):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                check.validate_record(status(ts_recv=timestamp), DAY, {INSTRUMENT})

    def test_recv_regression_refused_equal_timestamp_preserved(self):
        with self.assertRaises(ValueError):
            check.validate_record(status(offset=2), DAY, {INSTRUMENT}, check.midnight_ns(DAY) + 3)
        self.assertEqual(check.validate_record(status(offset=2), DAY, {INSTRUMENT},
                                             check.midnight_ns(DAY) + 2)['ts_recv'], check.midnight_ns(DAY) + 2)

    def test_unknown_tristate_is_not_previous_true(self):
        yes = check.validate_record(status(), DAY, {INSTRUMENT})
        unknown = check.validate_record(status(offset=3, is_trading=dbn.TriState.NOT_AVAILABLE), DAY,
                                        {INSTRUMENT}, yes['ts_recv'])
        self.assertEqual(unknown['is_trading'], '~')
        self.assertIsNot(unknown['is_trading'], yes['is_trading'])

    def test_implicit_tristate_values_refused(self):
        for value in ('~', 'Y', 0, 1, object()):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                check.tri_state(value)

    def test_invalid_raw_tristate_refused(self):
        original = status(is_trading=dbn.TriState.NOT_AVAILABLE)
        corrupt = bytearray(bytes(original))
        corrupt[30] = ord('X')
        class Proxy:
            def __getattr__(self, name):
                return getattr(original, name)
            def __bytes__(self):
                return bytes(corrupt)
        with self.assertRaisesRegex(ValueError, 'Invalid raw'):
            check.validate_record(Proxy(), DAY, {INSTRUMENT})


class CompressedDecodeTests(unittest.TestCase):
    def test_complete_zstd_frame_and_concatenation(self):
        data = payload()
        self.assertTrue(check.check_zstd_frames(io.BytesIO(data))['frame_boundaries_complete'])
        self.assertEqual(check.check_zstd_frames(io.BytesIO(data + data))['frames'], 2)

    def test_truncated_frame_rejected_even_if_native_buffer_empty(self):
        data = payload()
        decoder = dbn.DBNDecoder(compression=dbn.Compression.ZSTD)
        self.assertTrue(decoder.write_and_decode(data[:-1]))
        self.assertFalse(decoder.buffer())
        with self.assertRaises(ValueError):
            check.check_zstd_frames(io.BytesIO(data[:-1]))

    def test_trailing_compressed_garbage_refused(self):
        with self.assertRaises(ValueError):
            check.check_zstd_frames(io.BytesIO(payload() + b'garbage'))

    def test_reserved_zstd_descriptor_refused(self):
        data = bytearray(payload())
        data[4] |= 8
        with self.assertRaises(ValueError):
            check.check_zstd_frames(io.BytesIO(data))

    def test_no_data_frames_refused(self):
        with self.assertRaises(ValueError):
            check.check_zstd_frames(io.BytesIO(b''))

    def test_decode_eof_and_unknown_facts(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-status-decode-') as folder:
            folder = Path(folder)
            source = folder / 'fixture.dbn.zst'
            source.write_bytes(payload())
            result = check.decode_file(source, DAY, folder / 'timeline.jsonl')
            self.assertEqual(result['records'], 3)
            self.assertFalse(result['mapping_verified'])
            self.assertFalse(result['unknown_state_carried_forward'])
            self.assertFalse(result['native_sequence_available'])

    def test_decode_wrong_metadata_refused(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-status-reject-') as folder:
            folder = Path(folder)
            source = folder / 'fixture.dbn.zst'
            source.write_bytes(payload(metadata=meta(schema=dbn.Schema.MBO)))
            with self.assertRaises(ValueError):
                check.decode_file(source, DAY, folder / 'timeline.jsonl')

    def test_decode_empty_status_day_refused(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-status-empty-') as folder:
            folder = Path(folder)
            source = folder / 'fixture.dbn.zst'
            source.write_bytes(payload(records=[]))
            with self.assertRaises(ValueError):
                check.decode_file(source, DAY, folder / 'timeline.jsonl')


if __name__ == '__main__':
    unittest.main(verbosity=2)
