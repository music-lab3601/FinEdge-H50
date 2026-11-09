"""Synthetic MBO semantics only; no market files, model, or optimizer access."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import databento_mbo_v2 as mbo
from itch_book import Book, ROW, snapshot


DAY = '2025-12-15'
INSTRUMENT = 12345
BID = 1_000_000
ASK = 1_000_100


def record(action, sequence=50, oid=1, side='B', price=BID, size=100,
           flags=mbo.LAST, **overrides):
    base = mbo.midnight_ns(DAY)
    fields = dict(action=action, publisher_id=2, instrument_id=INSTRUMENT,
                  order_id=oid, side=side, price=price * 100_000, size=size,
                  flags=flags, sequence=sequence, channel_id=0,
                  ts_event=base + sequence * 1000,
                  ts_recv=base + 10**12 + sequence * 1000)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def fixture():
    rows = []
    adapter = mbo.Adapter(DAY, [INSTRUMENT], rows.append)
    adapter.consume(record('R', sequence=0, flags=mbo.SNAPSHOT | mbo.BAD_RECV))
    reference = Book()
    for index in range(10):
        for side, oid, price in [('B', index + 1, BID - index * 100),
                                 ('A', index + 11, ASK + index * 100)]:
            is_last = index == 9 and side == 'A'
            adapter.consume(record('A', sequence=index * 2 + (side == 'A') + 1,
                                   oid=oid, side=side, price=price,
                                   flags=mbo.SNAPSHOT | (mbo.LAST if is_last else 0)))
            reference.add(oid, 'B' if side == 'B' else 'S', price, 100)
    return adapter, rows, reference


def unpack(row):
    values = ROW.unpack(row)
    return dict(sequence=values[0], timestamp_ns=values[1], segment_id=values[2],
                flags=values[3], bid_levels=values[4], ask_levels=values[5],
                features=list(values[6:]))


class MboMappingTests(unittest.TestCase):
    def test_trade_fill_cancel_decrements_exactly_once(self):
        adapter, rows, reference = fixture()
        adapter.consume(record('T', size=40, flags=0))
        adapter.consume(record('F', size=40, flags=0))
        self.assertEqual(adapter.book.orders[1][2], 100)
        adapter.consume(record('C', size=40))
        reference.apply(dict(type='E', order_id=1, shares=40))
        self.assertEqual(adapter.book.orders[1][2], 60)
        self.assertEqual(adapter.book.top(), reference.top())
        self.assertEqual(len(rows), 1)
        self.assertEqual(adapter.finish()['counts']['complete_native_events'], 1)

    def test_partial_and_full_cancel(self):
        adapter, rows, reference = fixture()
        adapter.consume(record('C', sequence=50, size=25))
        reference.apply(dict(type='X', order_id=1, shares=25))
        self.assertEqual(adapter.book.orders, reference.orders)
        adapter.consume(record('C', sequence=51, size=75))
        reference.apply(dict(type='D', order_id=1))
        self.assertNotIn(1, adapter.book.orders)
        self.assertEqual(adapter.book.orders, reference.orders)
        self.assertEqual(len(rows), 2)
        adapter.finish()

    def test_unknown_cancel_and_modify_are_rejected(self):
        for action in ['C', 'M']:
            with self.subTest(action=action):
                adapter, rows, _ = fixture()
                with self.assertRaisesRegex(ValueError, 'Unknown active order'):
                    adapter.consume(record(action, oid=999))
                self.assertEqual(rows, [])

    def test_invalid_cancel_quantity_is_rejected(self):
        for size in [0, -1, 101]:
            with self.subTest(size=size):
                adapter, rows, _ = fixture()
                with self.assertRaisesRegex(ValueError, 'Invalid reduction'):
                    adapter.consume(record('C', size=size))
                self.assertEqual(rows, [])
                self.assertEqual(adapter.book.orders[1][2], 100)

    def test_cancel_side_and_price_must_match_active_order(self):
        for changes in [dict(side='A'), dict(price=BID + 1)]:
            with self.subTest(changes=changes):
                adapter, rows, _ = fixture()
                with self.assertRaisesRegex(ValueError, 'Cancel side/price'):
                    adapter.consume(record('C', **changes))
                self.assertEqual(rows, [])

    def test_duplicate_active_id_is_rejected(self):
        adapter, rows, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate active order'):
            adapter.consume(record('A', oid=1))
        self.assertEqual(rows, [])

    def test_modify_uses_absolute_quantity_and_new_price(self):
        adapter, rows, reference = fixture()
        adapter.consume(record('M', size=40, price=BID - 1))
        reference.apply(dict(type='D', order_id=1))
        reference.add(1, 'B', BID - 1, 40)
        self.assertEqual(adapter.book.orders[1], ('B', BID - 1, 40))
        self.assertEqual(adapter.book.top(), reference.top())
        self.assertEqual(len(rows), 1)
        adapter.finish()

    def test_atomic_replace_samples_only_final_ten_level_book(self):
        adapter, rows, reference = fixture()
        adapter.consume(record('C', size=100, flags=0))
        self.assertEqual(len(adapter.book.top()['B']), 9)
        self.assertEqual(rows, [])
        adapter.consume(record('A', oid=101, price=BID - 1, size=60))
        reference.apply(dict(type='U', order_id=1, new_order_id=101,
                             price=BID - 1, shares=60))
        self.assertEqual(adapter.book.top(), reference.top())
        self.assertEqual(len(rows), 1)
        observed = unpack(rows[0])
        self.assertEqual(observed['bid_levels'], 10)
        self.assertFalse(observed['flags'] & 2)
        self.assertEqual(observed['features'], snapshot(reference, False, False)[1])
        adapter.finish()

    def test_aggregate_unchanged_replace_still_samples(self):
        adapter, rows, reference = fixture()
        before = adapter.book.top()
        adapter.consume(record('C', size=100, flags=0))
        adapter.consume(record('A', oid=101, size=100))
        reference.apply(dict(type='U', order_id=1, new_order_id=101,
                             price=BID, shares=100))
        self.assertEqual(adapter.book.top(), before)
        self.assertEqual(adapter.book.orders, reference.orders)
        self.assertEqual(len(rows), 1)
        adapter.finish()

    def test_deep_changes_trigger_rows_even_when_top_ten_unchanged(self):
        adapter, rows, _ = fixture()
        before = adapter.book.top()
        adapter.consume(record('A', sequence=50, oid=101, price=BID - 1000))
        adapter.consume(record('C', sequence=51, oid=101, price=BID - 1000, size=40))
        self.assertEqual(adapter.book.top(), before)
        self.assertEqual(len(rows), 2)
        self.assertEqual(unpack(rows[0])['features'], unpack(rows[1])['features'])
        adapter.finish()

    def test_pure_trade_and_no_action_do_not_sample(self):
        adapter, rows, reference = fixture()
        for sequence, action in enumerate(['T', 'N'], start=50):
            # Pure trades may use non-book references; they are not reductions.
            adapter.consume(record(action, sequence=sequence, oid=999, size=40))
        self.assertEqual(rows, [])
        self.assertEqual(adapter.book.orders, reference.orders)
        adapter.consume(record('C', sequence=52, size=1))
        self.assertEqual(len(rows), 1)
        self.assertEqual(adapter.finish()['counts']['complete_native_events'], 3)


class MboBoundaryTests(unittest.TestCase):
    def test_provider_flags_130_are_normal_last_event(self):
        adapter, rows, _ = fixture()
        adapter.consume(record('C', flags=130, size=1))
        self.assertEqual(len(rows), 1)
        self.assertFalse(unpack(rows[0])['flags'] & mbo.BAD_BOOK)
        self.assertEqual(adapter.finish()['provider_flag_counts']['130'], 1)

    def test_reset_with_flag_8_is_independent_and_clears_state(self):
        adapter, rows, _ = fixture()
        adapter.consume(record('C', flags=mbo.LAST | mbo.MAYBE_BAD, size=1))
        self.assertTrue(adapter.bad_book)
        adapter.consume(record('R', sequence=90, flags=8))
        self.assertEqual(adapter.book.orders, {})
        self.assertFalse(adapter.bad_book)
        self.assertIsNone(adapter.group_key)
        self.assertIsNone(adapter.previous_sequence)
        self.assertFalse(adapter.snapshot_active)
        # Resetting native sequence is permitted after verified book clearing.
        adapter.consume(record('A', sequence=1, oid=101,
                               ts_recv=mbo.midnight_ns(DAY) + 10**12 + 91_000))
        self.assertEqual(len(rows), 2)
        self.assertFalse(unpack(rows[-1])['flags'] & mbo.BAD_BOOK)
        adapter.finish()

    def test_snapshot_rows_do_not_sample(self):
        adapter, rows, reference = fixture()
        self.assertEqual(rows, [])
        self.assertEqual(adapter.book.orders, reference.orders)
        self.assertEqual(adapter.counts['snapshot_records'], 20)
        adapter.consume(record('C', size=1))
        self.assertEqual(len(rows), 1)
        adapter.finish()

    def test_uninitialized_stream_is_rejected(self):
        adapter = mbo.Adapter(DAY, [INSTRUMENT], lambda row: None)
        with self.assertRaisesRegex(ValueError, 'No verified reset'):
            adapter.consume(record('A'))

    def test_missing_last_is_rejected_at_eof(self):
        adapter, rows, _ = fixture()
        adapter.consume(record('C', size=1, flags=0))
        self.assertEqual(rows, [])
        with self.assertRaisesRegex(ValueError, 'Truncated native event'):
            adapter.finish()

    def test_same_sequence_group_key_conflicts_are_rejected(self):
        for changes in [dict(channel_id=1),
                        dict(ts_event=mbo.midnight_ns(DAY) + 50_001)]:
            with self.subTest(changes=changes):
                adapter, rows, _ = fixture()
                adapter.consume(record('C', size=1, flags=0))
                with self.assertRaisesRegex(ValueError, 'Native event changed'):
                    adapter.consume(record('F', **changes))
                self.assertEqual(rows, [])

    def test_different_sequence_channel_conflict_is_rejected(self):
        adapter, rows, _ = fixture()
        adapter.consume(record('C', size=1, flags=0))
        with self.assertRaisesRegex(ValueError, 'Instrument/channel changed'):
            adapter.consume(record('A', sequence=51, channel_id=1, oid=101))
        self.assertEqual(rows, [])

    def test_multiple_add_sequences_share_one_venue_last(self):
        adapter, rows, reference = fixture()
        adapter.consume(record('A', sequence=50, oid=101, price=BID - 1000, flags=0))
        self.assertEqual(rows, [])
        adapter.consume(record('A', sequence=51, oid=102, price=BID - 1100))
        reference.add(101, 'B', BID - 1000, 100)
        reference.add(102, 'B', BID - 1100, 100)
        self.assertEqual(adapter.book.orders, reference.orders)
        self.assertEqual([unpack(row)['sequence'] for row in rows], [50, 51])
        report = adapter.finish()
        self.assertEqual(report['counts']['complete_native_events'], 2)
        self.assertEqual(report['counts']['venue_event_last_records'], 1)
        self.assertEqual(report['counts']['native_samples_before_venue_last'], 1)

    def test_replace_and_next_add_preserve_two_native_samples(self):
        adapter, rows, reference = fixture()
        adapter.consume(record('C', sequence=50, size=100, flags=0))
        self.assertEqual(rows, [])
        adapter.consume(record('A', sequence=50, oid=101, price=BID - 1,
                               size=60, flags=0))
        self.assertEqual(rows, [])
        reference.apply(dict(type='U', order_id=1, new_order_id=101,
                             price=BID - 1, shares=60))
        first_expected = snapshot(reference, False, False)[1]
        adapter.consume(record('A', sequence=51, oid=102, price=BID - 1000))
        reference.add(102, 'B', BID - 1000, 100)
        self.assertEqual(adapter.book.orders, reference.orders)
        self.assertEqual([unpack(row)['sequence'] for row in rows], [50, 51])
        self.assertEqual(unpack(rows[0])['features'], first_expected)
        self.assertEqual(adapter.finish()['counts']['native_pattern_CA'], 1)

    def test_trade_fill_cancel_then_next_sequence_add_samples_twice(self):
        adapter, rows, reference = fixture()
        for action in ['T', 'F', 'C']:
            adapter.consume(record(action, sequence=50, size=40, flags=0))
        self.assertEqual(rows, [])
        self.assertEqual(adapter.book.orders[1][2], 60)
        reference.apply(dict(type='E', order_id=1, shares=40))
        first_expected = snapshot(reference, False, False)[1]
        adapter.consume(record('A', sequence=51, oid=101, price=BID - 1000))
        reference.add(101, 'B', BID - 1000, 100)
        self.assertEqual(adapter.book.orders, reference.orders)
        self.assertEqual([unpack(row)['sequence'] for row in rows], [50, 51])
        self.assertEqual(unpack(rows[0])['features'], first_expected)
        self.assertEqual(adapter.finish()['counts']['native_pattern_TFC'], 1)

    def test_different_sequences_with_same_receive_time_are_allowed(self):
        adapter, rows, _ = fixture()
        receive_time = mbo.midnight_ns(DAY) + 10**12 + 50_000
        adapter.consume(record('A', sequence=50, oid=101, flags=0,
                               ts_recv=receive_time))
        adapter.consume(record('A', sequence=51, oid=102, ts_recv=receive_time))
        self.assertEqual([unpack(row)['sequence'] for row in rows], [50, 51])
        adapter.finish()

    def test_fill_without_same_sequence_cancel_is_rejected(self):
        for pattern in ['F', 'TF']:
            with self.subTest(pattern=pattern):
                adapter, rows, _ = fixture()
                for action in pattern:
                    adapter.consume(record(action, sequence=50, flags=0))
                with self.assertRaisesRegex(ValueError, 'Incomplete/unsupported XNAS native action pattern'):
                    adapter.consume(record('A', sequence=51, oid=101))
                self.assertEqual(rows, [])

    def test_eof_add_without_venue_last_cannot_commit(self):
        adapter, rows, _ = fixture()
        adapter.consume(record('A', sequence=50, oid=101, flags=0))
        self.assertEqual(rows, [])
        with self.assertRaisesRegex(ValueError, 'Truncated native event'):
            adapter.finish()
        # A completed native sample may exist as diagnostic output before venue LAST.
        # The whole file still cannot receive a successful finish/commit receipt.
        adapter.consume(record('A', sequence=51, oid=102, flags=0))
        self.assertEqual([unpack(row)['sequence'] for row in rows], [50])
        with self.assertRaisesRegex(ValueError, 'Truncated native event'):
            adapter.finish()

    def test_wrong_publisher_or_instrument_is_rejected(self):
        for changes in [dict(publisher_id=1), dict(instrument_id=999)]:
            with self.subTest(changes=changes):
                adapter, _, _ = fixture()
                with self.assertRaisesRegex(ValueError, 'Wrong publisher'):
                    adapter.consume(record('C', **changes))

    def test_sequence_gaps_are_allowed_but_non_increasing_groups_rejected(self):
        adapter, rows, _ = fixture()
        adapter.consume(record('C', sequence=50, size=1))
        adapter.consume(record('C', sequence=100, size=1))
        self.assertEqual(len(rows), 2)
        # Keep receive time increasing to isolate the native sequence failure.
        with self.assertRaisesRegex(ValueError, 'Non-increasing native sequence'):
            adapter.consume(record('C', sequence=100, size=1,
                                   ts_recv=mbo.midnight_ns(DAY) + 10**12 + 101_000))

    def test_reset_or_snapshot_inside_live_group_is_rejected(self):
        for action, flags in [('R', 8), ('A', mbo.SNAPSHOT | mbo.LAST)]:
            with self.subTest(action=action):
                adapter, rows, _ = fixture()
                adapter.consume(record('C', size=1, flags=0))
                with self.assertRaisesRegex(ValueError, 'inside unfinished native event'):
                    adapter.consume(record(action, oid=101, flags=flags))
                self.assertEqual(rows, [])

    def test_unfinished_snapshot_rejects_live_record_and_eof(self):
        adapter = mbo.Adapter(DAY, [INSTRUMENT], lambda row: None)
        adapter.consume(record('R', flags=mbo.SNAPSHOT | 8))
        adapter.consume(record('A', flags=mbo.SNAPSHOT))
        with self.assertRaisesRegex(ValueError, 'Snapshot unfinished'):
            adapter.consume(record('C', sequence=51, size=1))
        with self.assertRaisesRegex(ValueError, 'Truncated native event/snapshot'):
            adapter.finish()

    def test_all_rows_remain_control_unverified_with_zero_segments(self):
        adapter, rows, _ = fixture()
        for sequence in range(50, 53):
            adapter.consume(record('C', sequence=sequence, size=1))
        for raw in rows:
            row = unpack(raw)
            self.assertEqual(len(raw), 352)
            self.assertEqual(row['segment_id'], 0)
            self.assertTrue(row['flags'] & mbo.CONTROL_UNVERIFIED)
            self.assertTrue(row['flags'] & 1)
            self.assertTrue(row['flags'] & 16)
        report = adapter.finish()
        self.assertEqual(report['valid_rows'], 0)
        self.assertEqual(report['segments'], 0)
        self.assertFalse(report['training_ready'])
        self.assertFalse(report['labels_computed'])
        self.assertEqual(report['model_forward_count'], 0)
        self.assertEqual(report['optimizer_updates'], 0)


class MboQualityTests(unittest.TestCase):
    def test_price_scaling_is_exact_and_bad_price_rejected(self):
        self.assertEqual(mbo.price4(100_000), 1)
        self.assertEqual(mbo.price4(123_456_700_000), 1_234_567)
        for raw in [0, -100_000, 1, 100_001, 2**63 - 1, 2**63]:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, 'not exactly representable'):
                    mbo.price4(raw)
        adapter, rows, _ = fixture()
        bad_record = record('A', oid=101)
        bad_record.price = 123_456_700_001
        with self.assertRaisesRegex(ValueError, 'not exactly representable'):
            adapter.consume(bad_record)
        self.assertEqual(rows, [])

    def test_event_timestamp_regression_marks_only_affected_row(self):
        adapter, rows, _ = fixture()
        adapter.consume(record('C', sequence=50, size=1))
        adapter.consume(record('C', sequence=51, size=1,
                               ts_event=mbo.midnight_ns(DAY) + 49_000))
        adapter.consume(record('C', sequence=52, size=1))
        self.assertEqual([bool(unpack(row)['flags'] & 32) for row in rows],
                         [False, True, False])
        self.assertEqual(adapter.finish()['counts']['event_timestamp_regressions'], 1)

    def test_receive_regression_and_wrong_local_date_are_rejected(self):
        adapter, _, _ = fixture()
        adapter.consume(record('C', sequence=50, size=1))
        with self.assertRaisesRegex(ValueError, 'Receive timestamp regression'):
            adapter.consume(record('C', sequence=51, size=1,
                                   ts_recv=mbo.midnight_ns(DAY) + 10**12 + 49_999))
        for ts in [mbo.midnight_ns(DAY) - 1, mbo.midnight_ns(DAY) + mbo.DAY_NS]:
            with self.subTest(ts=ts):
                adapter, rows, _ = fixture()
                with self.assertRaisesRegex(ValueError, 'outside local trading date'):
                    adapter.consume(record('C', size=1, ts_event=ts))
                self.assertEqual(rows, [])

    def test_maybe_bad_book_persists_until_reset(self):
        adapter, rows, _ = fixture()
        adapter.consume(record('C', sequence=50, size=1,
                               flags=mbo.LAST | mbo.MAYBE_BAD))
        adapter.consume(record('C', sequence=51, size=1))
        self.assertEqual([bool(unpack(row)['flags'] & mbo.BAD_BOOK) for row in rows],
                         [True, True])
        self.assertEqual(adapter.finish()['counts']['maybe_bad_book_records'], 1)

    def test_bad_receive_flag_is_group_local(self):
        adapter, rows, _ = fixture()
        adapter.consume(record('F', sequence=50, flags=mbo.BAD_RECV))
        adapter.consume(record('C', sequence=50, size=1))
        adapter.consume(record('C', sequence=51, size=1))
        self.assertEqual([bool(unpack(row)['flags'] & mbo.BAD_BOOK) for row in rows],
                         [True, False])
        adapter.finish()

    def test_top_of_book_and_mbp_flags_are_rejected(self):
        for flag in [mbo.TOB, mbo.MBP]:
            with self.subTest(flag=flag):
                adapter, rows, _ = fixture()
                with self.assertRaisesRegex(ValueError, 'TOB/MBP'):
                    adapter.consume(record('C', size=1, flags=mbo.LAST | flag))
                self.assertEqual(rows, [])

    def test_scope_and_missing_instrument_metadata_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'metadata mapping absent'):
            mbo.Adapter(DAY, [], lambda row: None)
        for day in ['2025-11-15', '2026-12-15']:
            with self.subTest(day=day):
                with self.assertRaisesRegex(ValueError, 'only approved'):
                    mbo.midnight_ns(day)


if __name__ == '__main__':
    unittest.main()
