"""Independent small-array invariants for P1 research screening.

Fixtures contain generated books only. No source file, label, statistic, model,
or accelerator is accessed. Expected windows are computed from explicit
original-row intervals, not from the implementation's segment algorithm.
"""

from dataclasses import replace
import unittest

import numpy as np

from software.data import tcn_screening_reference as reference
from software.data import tcn_research_screening_core as core


DAY = "2025-12-15"
START = (9 * 3600 + 35 * 60) * 1_000_000_000
STOP = (15 * 3600 + 55 * 60) * 1_000_000_000
DAY_NS = 86_400 * 1_000_000_000


def books(count, kind="engineering", start=START):
    rows = np.zeros(count, dtype=core.RECORD_DTYPE)
    rows["sequence"] = np.arange(1, count + 1, dtype=np.uint64)
    rows["timestamp_ns"] = np.arange(start, start + count, dtype=np.uint64)
    rows["segment_id"] = 0 if kind == "engineering" else 1
    rows["flags"] = 1 | 16 | 64 if kind == "engineering" else 0
    rows["bid_levels"] = rows["ask_levels"] = 10
    for level in range(10):
        rows["features"][:, 4 * level] = 10_100 + 10 * level
        rows["features"][:, 4 * level + 1] = 20 + level
        rows["features"][:, 4 * level + 2] = 10_000 - 10 * level
        rows["features"][:, 4 * level + 3] = 30 + level
    return rows


def native_span(start, stop, segment_id=1):
    return {
        "start_row": start, "end_row": stop - 1,
        "rows": stop - start, "segment_id": segment_id,
    }


def control(t=START - 1, action="TRADING", day=DAY, **kwargs):
    return reference.Status(day=day, ts_event=t, action=action, **kwargs)


def screen(rows, kind="engineering", *, statuses=None, spans=None, day=DAY,
           adapter=reference.ADAPTER_VERSION, chunk=100_000):
    if statuses is None:
        statuses = (control(day=day),) if kind == "engineering" else ()
    if spans is None and kind == "native" and len(rows):
        spans = [native_span(0, len(rows))]
    return core.screen_array(
        rows, day=day, source_kind=kind, native_segments=spans,
        statuses=statuses, adapter_version=adapter, chunk_size=chunk,
    )


class ScreeningAssertions(unittest.TestCase):
    def assert_intervals(self, result, intervals):
        masks = result["reason_masks"]
        self.assertEqual(masks.dtype, np.dtype(np.uint32))
        ranges = result["ranges"]
        self.assertEqual(
            [(item["start_row"], item["stop_row"]) for item in ranges],
            list(intervals),
        )
        accepted = np.zeros(len(masks), dtype=bool)
        for ordinal, (item, (start, stop)) in enumerate(zip(ranges, intervals), 1):
            accepted[start:stop] = True
            self.assertEqual(item["research_segment_id"], ordinal)
            self.assertEqual(item["record_count"], stop - start)
            self.assertEqual(item["window_count"], max(0, stop - start - 149))
        np.testing.assert_array_equal(masks == 0, accepted)
        summary = result["summary"]
        count = sum(stop - start for start, stop in intervals)
        self.assertEqual(summary["eligible_records"], count)
        self.assertEqual(summary["rejected_records"], len(masks) - count)
        self.assertEqual(summary["segment_count"], len(intervals))
        self.assertEqual(summary["window_count"], sum(max(0, b - a - 149) for a, b in intervals))
        self.assertEqual(sum(summary["primary_reason_counts"].values()), len(masks) - count)
        for name, bit in core.REASON_BITS.items():
            self.assertEqual(summary["reason_counts"].get(name, 0), int(np.count_nonzero(masks & bit)))


class ContinuityTests(ScreeningAssertions):
    def test_exact_352_byte_generated_schema(self):
        self.assertEqual(core.RECORD_DTYPE.itemsize, 352)
        self.assertEqual(core.RECORD_DTYPE["features"].shape, (40,))

    def test_149_150_151_original_rows_need_entire_footprint(self):
        for count, windows in ((149, 0), (150, 1), (151, 2)):
            with self.subTest(count=count):
                result = screen(books(count))
                self.assert_intervals(result, [(0, count)])
                self.assertEqual(result["summary"]["window_count"], windows)

    def test_bad_future_middle_cannot_be_deleted_and_compressed(self):
        rows = books(151)
        rows["flags"][125] |= 128
        self.assert_intervals(screen(rows), [(0, 125), (126, 151)])

    def test_bad_input_middle_cannot_be_deleted_and_compressed(self):
        rows = books(151)
        rows["flags"][50] |= 128
        self.assert_intervals(screen(rows), [(0, 50), (51, 151)])

    def test_reset_boundary_requires_two_new_150_row_footprints(self):
        rows = books(301)
        rows["flags"][150] |= 8
        result = screen(rows)
        self.assert_intervals(result, [(0, 150), (151, 301)])
        self.assertEqual(result["summary"]["window_count"], 2)

    def test_adjacent_native_segments_cannot_join(self):
        rows = books(150, "native")
        rows["segment_id"][75:] = 2
        spans = [native_span(0, 75, 1), native_span(75, 150, 2)]
        self.assert_intervals(screen(rows, "native", spans=spans), [(0, 75), (75, 150)])

    def test_native_unbound_gap_cannot_be_readmitted(self):
        rows = books(301, "native")
        rows["segment_id"][151:] = 2
        spans = [native_span(0, 150, 1), native_span(151, 301, 2)]
        self.assert_intervals(
            screen(rows, "native", spans=spans, statuses=[control()]),
            [(0, 150), (151, 301)],
        )

    def test_native_flagged_row_remains_invalid_despite_positive_status(self):
        rows = books(301, "native")
        rows["flags"][150] = 16
        self.assert_intervals(
            screen(rows, "native", statuses=[control()]), [(0, 150), (151, 301)]
        )

    def test_missing_native_segment_manifest_rejects_entire_day(self):
        self.assert_intervals(screen(books(150, "native"), "native", spans=[]), [])

    def test_chunk_boundaries_do_not_change_segments_masks_or_counts(self):
        rows = books(501)
        rows["flags"][149] |= 128
        statuses = [control(), control(START + 300), control(START + 450, "SSR_CHANGE")]
        expected = screen(rows, statuses=statuses, chunk=1000)
        self.assert_intervals(expected, [(0, 149), (150, 300), (301, 450), (451, 501)])
        for chunk in (1, 2, 17, 149, 150, 151, 300):
            with self.subTest(chunk=chunk):
                actual = screen(rows, statuses=statuses, chunk=chunk)
                np.testing.assert_array_equal(actual["reason_masks"], expected["reason_masks"])
                self.assertEqual(actual["ranges"], expected["ranges"])
                self.assertEqual(actual["summary"], expected["summary"])


class ObservableStateTests(ScreeningAssertions):
    def test_initial_unknown_requires_later_explicit_trading(self):
        rows = books(301)
        self.assert_intervals(screen(rows, statuses=[control(START + 150)]), [(151, 301)])

    def test_missing_status_rejects_even_quality_valid_engineering(self):
        self.assert_intervals(screen(books(150), statuses=[]), [])

    def test_all_rows_equal_to_status_timestamp_are_rejected(self):
        rows = books(153)
        rows["timestamp_ns"][:3] = START
        rows["timestamp_ns"][3:] += 1
        self.assert_intervals(screen(rows, statuses=[control(START)]), [(3, 153)])

    def test_duplicate_unchanged_status_still_cuts(self):
        rows = books(400)
        statuses = [control(), control(START + 200), control(START + 200)]
        self.assert_intervals(screen(rows, statuses=statuses), [(0, 200), (201, 400)])

    def test_status_between_book_rows_still_splits_without_removing_rows(self):
        rows = books(300)
        rows["timestamp_ns"] = START + np.arange(300, dtype=np.uint64) * 2
        statuses = [control(), control(START + 299)]
        self.assert_intervals(screen(rows, statuses=statuses, chunk=17), [(0, 150), (150, 300)])

    def test_conflicting_same_time_actions_never_last_record_win(self):
        for actions in (("HALT", "TRADING"), ("TRADING", "HALT")):
            with self.subTest(actions=actions):
                statuses = [control(), *(control(START + 150, a) for a in actions), control(START + 350)]
                self.assert_intervals(screen(books(501), statuses=statuses), [(0, 150), (351, 501)])

    def test_one_negative_record_blocks_whole_timestamp_group(self):
        statuses = [control(), control(START + 150), control(START + 150, is_quoting="N")]
        self.assert_intervals(screen(books(301), statuses=statuses), [(0, 150)])

    def test_unknown_field_latches_until_explicit_trading_not_preserving_action(self):
        for field in ("is_trading", "is_quoting"):
            for value in ("N", "~"):
                with self.subTest(field=field, value=value):
                    statuses = [
                        control(), control(START + 150, **{field: value}),
                        control(START + 180, "SSR_CHANGE"),
                        control(START + 200, "NONE"),
                        control(START + 220, "NEW_PRICE_INDICATION"),
                        control(START + 350),
                    ]
                    self.assert_intervals(screen(books(501), statuses=statuses), [(0, 150), (351, 501)])

    def test_phase_blockers_override_y_y(self):
        for action in reference.BLOCKING_ACTIONS:
            with self.subTest(action=action):
                self.assert_intervals(screen(books(150), statuses=[control(action=action)]), [])

    def test_preserving_actions_cannot_establish_initial_phase(self):
        for action in ("NONE", "SSR_CHANGE", "NEW_PRICE_INDICATION"):
            with self.subTest(action=action):
                self.assert_intervals(screen(books(150), statuses=[control(action=action)]), [])

    def test_trading_plus_preserving_group_can_establish_phase(self):
        statuses = [control(), control(action="SSR_CHANGE")]
        self.assert_intervals(screen(books(150), statuses=statuses), [(0, 150)])

    def test_ssr_unknown_does_not_make_trading_unknown(self):
        self.assert_intervals(screen(books(150), statuses=[control(ssr="~")]), [(0, 150)])

    def test_operational_and_marketwide_quarantine_cannot_generic_resume(self):
        reasons = ("OPERATIONAL", "MARKET_WIDE_HALT_LEVEL1", "MARKET_WIDE_HALT_RESUMPTION")
        for reason in reasons:
            with self.subTest(reason=reason):
                statuses = [control(), control(START + 150, reason=reason), control(START + 200)]
                self.assert_intervals(screen(books(400), statuses=statuses), [(0, 150)])

    def test_new_day_state_does_not_inherit_prior_trading_or_quarantine(self):
        screen(books(150), statuses=[control(reason="OPERATIONAL")])
        day = "2025-12-16"
        self.assert_intervals(screen(books(150), day=day, statuses=[control(day=day, action="NONE")]), [])
        self.assert_intervals(screen(books(150), day=day, statuses=[control(day=day)]), [(0, 150)])

    def test_late_unrecognized_status_rejects_earlier_rows_too(self):
        for change in ({"action": "FUTURE_ACTION"}, {"reason": "FUTURE_REASON"}, {"symbol": "MSFT"}):
            with self.subTest(change=change):
                late = replace(control(STOP + 10), **change)
                self.assert_intervals(screen(books(150), statuses=[control(), late]), [])

    def test_status_sequence_regression_cannot_be_repaired_by_sorting(self):
        self.assert_intervals(screen(books(150), statuses=[control(), control(START - 2)]), [])


class ProvenanceAndBookTests(ScreeningAssertions):
    def test_engineering_placeholder_flags_require_exact_adapter(self):
        for adapter in (None, "Databento-MBO-engineering-adapter-1.0", "Databento-MBO-engineering-adapter-2.0 "):
            with self.subTest(adapter=adapter):
                self.assert_intervals(screen(books(150), adapter=adapter), [])

    def test_engineering_flag64_must_remain_present(self):
        rows = books(150)
        rows["flags"] = 1 | 16
        self.assert_intervals(screen(rows), [])

    def test_engineering_quality_and_unexplained_bits_are_not_provider_flags(self):
        for bad_bit in (2, 4, 8, 32, 128, 256, 1 << 31):
            with self.subTest(bad_bit=bad_bit):
                rows = books(150)
                rows["flags"] = 64 | bad_bit
                self.assert_intervals(screen(rows), [])
        # Provider F_LAST=128 cannot exempt a derived-row BAD_BOOK=128.
        good = books(150)
        good["flags"] = 64
        self.assert_intervals(screen(good), [(0, 150)])

    def test_original_native_gate_flags_are_never_placeholders(self):
        for flag in (1, 16, 64, 1 | 16 | 64):
            with self.subTest(flag=flag):
                rows = books(150, "native")
                rows["flags"] = flag
                self.assert_intervals(screen(rows, "native"), [])

    def test_engineering_positive_source_segment_is_provenance_error(self):
        rows = books(150)
        rows["segment_id"] = 1
        self.assert_intervals(screen(rows), [])

    def test_native_row_segment_must_match_the_bound_manifest(self):
        rows = books(301, "native")
        rows["segment_id"][150] = 2
        self.assert_intervals(screen(rows, "native"), [(0, 150), (151, 301)])

    def test_ten_visible_levels_are_required_on_both_sides(self):
        for side in ("bid_levels", "ask_levels"):
            rows = books(301)
            rows[side][150] = 9
            with self.subTest(side=side):
                self.assert_intervals(screen(rows), [(0, 150), (151, 301)])

    def test_nonpositive_price_or_quantity_invalidates_only_original_row(self):
        for feature in (0, 1, 2, 3, 36, 37, 38, 39):
            rows = books(301)
            rows["features"][150, feature] = 0
            with self.subTest(feature=feature):
                self.assert_intervals(screen(rows), [(0, 150), (151, 301)])

    def test_level_prices_must_be_strictly_ordered(self):
        for index, value in ((4, 10_100), (6, 10_000), (36, 10_150), (38, 9990)):
            rows = books(301)
            rows["features"][150, index] = value
            with self.subTest(index=index, value=value):
                self.assert_intervals(screen(rows), [(0, 150), (151, 301)])

    def test_locked_or_crossed_top_of_book_is_invalid(self):
        for top_bid in (10_100, 10_110):
            rows = books(301)
            rows["features"][150, 2] = top_bid
            with self.subTest(top_bid=top_bid):
                self.assert_intervals(screen(rows), [(0, 150), (151, 301)])

    def test_read_only_source_bytes_flags_and_statuses_are_unchanged(self):
        rows = books(301)
        rows["flags"][150] |= 128
        before = rows.tobytes()
        statuses = (control(), control(START + 200, "SSR_CHANGE"))
        controls_before = tuple(statuses)
        rows.setflags(write=False)
        screen(rows, statuses=statuses, chunk=17)
        self.assertEqual(rows.tobytes(), before)
        self.assertEqual(statuses, controls_before)

    def test_sequence_gaps_do_not_imply_loss_in_single_symbol_rows(self):
        rows = books(150)
        rows["sequence"] *= 123
        self.assert_intervals(screen(rows), [(0, 150)])

    def test_non_increasing_sequence_quarantines_whole_source_day(self):
        for kind in ("native", "engineering"):
            for sequence in (1, 149):
                rows = books(301, kind)
                rows["sequence"][150] = sequence
                with self.subTest(kind=kind, sequence=sequence):
                    self.assert_intervals(screen(rows, kind), [])

    def test_event_regression_quarantines_engineering_day_without_reordering(self):
        rows = books(301)
        rows["timestamp_ns"][250] = START - 1
        before = rows.tobytes()
        self.assert_intervals(screen(rows, chunk=17), [])
        self.assertEqual(rows.tobytes(), before)


class CalendarAndInputTests(ScreeningAssertions):
    def test_clock_start_inclusive_stop_exclusive(self):
        rows = books(6)
        rows["timestamp_ns"] = [START - 1, START, START + 1, STOP - 1, STOP, STOP + 1]
        self.assert_intervals(screen(rows), [(1, 4)])

    def test_unregistered_and_nontrading_day_rejected(self):
        for day in ("2025-12-13", "2025-12-24", "2026-12-15"):
            with self.subTest(day=day):
                self.assert_intervals(screen(books(150), day=day), [])

    def test_out_of_day_event_ns_not_silently_wrapped(self):
        rows = books(150)
        rows["timestamp_ns"][149] = DAY_NS
        self.assert_intervals(screen(rows), [])

    def test_out_of_day_status_ns_not_silently_wrapped(self):
        self.assert_intervals(screen(books(150), statuses=[control(), control(DAY_NS)]), [])

    def test_empty_fixture_has_no_windows_and_no_pass_claim(self):
        result = screen(books(0), statuses=[control()])
        self.assert_intervals(result, [])
        self.assertFalse(result["summary"]["screening_passed"])

    def test_wrong_dtype_and_multidimensional_records_rejected(self):
        for records in (np.zeros(150, dtype=np.uint8), np.zeros((2, 75), dtype=core.RECORD_DTYPE)):
            with self.subTest(shape=records.shape):
                with self.assertRaises(ValueError):
                    screen(records)

    def test_malformed_native_segment_metadata_not_treated_as_accepted(self):
        for span in (
            native_span(0, 151),
            {**native_span(0, 150), "rows": 149},
            {**native_span(0, 150), "segment_id": 0},
            {**native_span(0, 150), "start_row": -1},
        ):
            with self.subTest(span=span):
                with self.assertRaises(ValueError):
                    screen(books(150, "native"), "native", spans=[span])

    def test_overlapping_or_boolean_native_segment_fields_rejected(self):
        examples = (
            [native_span(0, 100, 1), native_span(99, 150, 2)],
            [{**native_span(0, 150), "start_row": False}],
            [native_span(0, 75, 1), native_span(75, 150, 1)],
        )
        for spans in examples:
            with self.subTest(spans=spans):
                with self.assertRaises(ValueError):
                    screen(books(150, "native"), "native", spans=spans)


if __name__ == "__main__":
    unittest.main()
