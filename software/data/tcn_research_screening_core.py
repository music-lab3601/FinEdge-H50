"""Array-only research-B screening of immutable, already rebuilt book rows.

This module performs no file access, DBN decoding, labeling, normalization, or
model work. A research-eligible engineering row still has unverified source
control coverage. Source flags and original row positions are never rewritten.
"""

from collections.abc import Mapping

import numpy as np

if __package__:
    from software.data.tcn_screening_reference import ADAPTER_VERSION, ObservedState, SCREEN_END_NS, SCREEN_START_NS, TARGET_DATES, Status, _status_groups, apply_status_group
else:
    from software.data.tcn_screening_reference import ADAPTER_VERSION, ObservedState, SCREEN_END_NS, SCREEN_START_NS, TARGET_DATES, Status, _status_groups, apply_status_group


VERSION = "TCN-research-B-array-screening-1.0"
DAY_NS = 86_400 * 1_000_000_000
FOOTPRINT = 150
RECORD_DTYPE = np.dtype([
    ("sequence", "<u8"), ("timestamp_ns", "<u8"),
    ("segment_id", "<u4"), ("flags", "<u4"),
    ("bid_levels", "<u4"), ("ask_levels", "<u4"),
    ("features", "<u8", (40,)),
])
assert RECORD_DTYPE.itemsize == 352

# Ordered by primary rejection priority. Counts for individual reasons overlap;
# primary counts below assign each rejected source row to its first reason only.
REASON_BITS = {
    "day_quarantine": 2,
    "unbound_engineering_provenance": 16,
    "invalid_event_timestamp": 2048,
    "sequence_non_increasing": 4096,
    "event_time_regression": 8192,
    "unexpected_row_bits": 256,
    "native_control_or_quality_flag": 4,
    "native_segment": 8,
    "missing_unverified_source_flag": 32,
    "engineering_source_segment": 64,
    "engineering_quality_flag": 128,
    "insufficient_depth": 512,
    "invalid_book_geometry": 1024,
    "outside_research_clock": 1,
    "equal_status_timestamp": 16384,
    "sticky_control_quarantine": 65536,
    "observed_state_blocked": 32768,
}
ROW_QUALITY_REJECT = 2 | 4 | 8 | 32 | 128
ROW_KNOWN_BITS = 255


def _integer(value):
    return type(value) is int


def _native_segments(segments, rows):
    """Validate original inclusive ranges without sorting or merging them."""
    if segments is None:
        return (), "missing_native_segment_binding"
    if not isinstance(segments, (list, tuple)):
        raise ValueError("native_segments must be an ordered list or tuple")
    normalized, previous_stop, used_ids = [], 0, set()
    for item in segments:
        if not isinstance(item, Mapping):
            raise ValueError("native segment metadata must be a mapping")
        fields = (item.get("start_row"), item.get("end_row"),
                  item.get("rows"), item.get("segment_id"))
        if not all(_integer(value) for value in fields):
            raise ValueError("native segment fields must be integers")
        start, end, count, segment_id = fields
        if (start < previous_stop or end < start or end >= rows
                or count != end - start + 1 or not 0 < segment_id < 2**32
                or segment_id in used_ids):
            raise ValueError("invalid or overlapping native segment geometry")
        normalized.append((start, end + 1, segment_id))
        previous_stop = end + 1
        used_ids.add(segment_id)
    return tuple(normalized), None


def _add(mask, condition, reason):
    mask[condition] |= np.uint32(REASON_BITS[reason])


def _native_membership(start, stop, segments, first_segment):
    members = np.zeros(stop - start, dtype=np.uint32)
    while first_segment < len(segments) and segments[first_segment][1] <= start:
        first_segment += 1
    cursor = first_segment
    while cursor < len(segments) and segments[cursor][0] < stop:
        left, right, segment_id = segments[cursor]
        members[max(start, left) - start:min(stop, right) - start] = segment_id
        cursor += 1
    return members, first_segment


def _status_table(statuses, day):
    groups, problem = _status_groups(statuses, day)
    if problem:
        return (), (), (), problem
    if any(group[0].ts_event >= DAY_NS for group in groups):
        return (), (), (), "invalid_status_local_day_timestamp"
    state = ObservedState(day)
    timestamps, blocked, sticky = [], [], []
    for group in groups:
        state = apply_status_group(state, group)
        timestamps.append(group[0].ts_event)
        blocked.append(not state.eligible)
        sticky.append(state.quarantined)
    return (np.asarray(timestamps, dtype=np.uint64),
            np.asarray(blocked, dtype=np.bool_),
            np.asarray(sticky, dtype=np.bool_), None)


def screen_array(records, *, day, source_kind, native_segments=None,
                 statuses=(), adapter_version=None, chunk_size=100_000,
                 progress=None):
    """Return masks and ranges referencing original rows; never compress data.

    ``timestamp_ns`` and ``Status.ts_event`` are New York local nanoseconds of
    this day. ``native_segments`` is the accepted source segments.json format
    (inclusive end_row). Status order is preserved and adjacent equal timestamps
    are aggregated without last-record wins. Source integrity anomalies fail
    closed for the whole day, including previously processed rows.
    """
    if (not isinstance(records, np.ndarray) or records.ndim != 1
            or records.dtype != RECORD_DTYPE):
        raise ValueError("records must be a one-dimensional RECORD_DTYPE array")
    if source_kind not in ("native", "engineering"):
        raise ValueError("unknown source_kind")
    if not _integer(chunk_size) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable")
    try:
        statuses = tuple(statuses)
    except TypeError as exc:
        raise ValueError("statuses must be an iterable of Status") from exc
    n = len(records)
    segments, segment_problem = ((), None)
    if source_kind == "native":
        segments, segment_problem = _native_segments(native_segments, n)
    day_problems = []
    if type(day) is not str or day not in TARGET_DATES:
        day_problems.append("unknown_or_nonstandard_calendar")
    if segment_problem:
        day_problems.append(segment_problem)
    status_ts, status_blocked, status_sticky, status_problem = _status_table(statuses, day)
    if status_problem:
        day_problems.append(status_problem)
    if source_kind == "engineering" and not statuses:
        day_problems.append("missing_status")
    masks = np.zeros(n, dtype=np.uint32)
    # Keep one lightweight boundary key per row. Geometry temporaries remain
    # bounded by chunk_size even for multi-million-row memory mapped sources.
    epochs = np.zeros(n, dtype=np.int64)
    invalid_timestamp_count = sequence_anomaly_count = event_regression_count = 0
    previous_ts = previous_seq = previous_member = None
    native_cursor = 0
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        chunk = records[start:stop]
        mask = masks[start:stop]
        ts, sequence, flags = chunk["timestamp_ns"], chunk["sequence"], chunk["flags"]
        invalid_ts = ts >= DAY_NS
        invalid_timestamp_count += int(np.count_nonzero(invalid_ts))
        _add(mask, invalid_ts, "invalid_event_timestamp")
        seq_bad = np.zeros(stop - start, dtype=np.bool_)
        seq_bad[1:] = sequence[1:] <= sequence[:-1]
        if previous_seq is not None:
            seq_bad[0] = int(sequence[0]) <= previous_seq
        sequence_anomaly_count += int(np.count_nonzero(seq_bad))
        _add(mask, seq_bad, "sequence_non_increasing")
        regression = np.zeros(stop - start, dtype=np.bool_)
        regression[1:] = ts[1:] < ts[:-1]
        if previous_ts is not None:
            regression[0] = int(ts[0]) < previous_ts
        _add(mask, (flags & ~np.uint32(ROW_KNOWN_BITS)) != 0, "unexpected_row_bits")
        if source_kind == "native":
            members, native_cursor = _native_membership(start, stop, segments, native_cursor)
            _add(mask, flags != 0, "native_control_or_quality_flag")
            _add(mask, (members == 0) | (chunk["segment_id"] != members), "native_segment")
            # Regressions outside accepted segments cannot admit any new rows;
            # only time integrity inside a bound original segment is asserted.
            same_member = np.zeros(stop - start, dtype=np.bool_)
            same_member[1:] = (members[1:] != 0) & (members[1:] == members[:-1])
            if previous_member is not None:
                same_member[0] = int(members[0]) != 0 and int(members[0]) == previous_member
            regression &= same_member
            previous_member = int(members[-1])
        else:
            if adapter_version != ADAPTER_VERSION:
                mask[:] |= np.uint32(REASON_BITS["unbound_engineering_provenance"])
            _add(mask, (flags & 64) == 0, "missing_unverified_source_flag")
            _add(mask, chunk["segment_id"] != 0, "engineering_source_segment")
            _add(mask, (flags & ROW_QUALITY_REJECT) != 0, "engineering_quality_flag")
        event_regression_count += int(np.count_nonzero(regression))
        _add(mask, regression, "event_time_regression")
        _add(mask, (chunk["bid_levels"] < 10) | (chunk["ask_levels"] < 10), "insufficient_depth")
        f = chunk["features"]
        ask, bid = f[:, 0::4], f[:, 2::4]
        geometry_bad = (np.any(f == 0, axis=1)
                        | np.any(ask[:, 1:] <= ask[:, :-1], axis=1)
                        | np.any(bid[:, 1:] >= bid[:, :-1], axis=1)
                        | (ask[:, 0] <= bid[:, 0]))
        _add(mask, geometry_bad, "invalid_book_geometry")
        _add(mask, (ts < SCREEN_START_NS) | (ts >= SCREEN_END_NS), "outside_research_clock")
        if not status_problem and len(status_ts):
            epoch = np.searchsorted(status_ts, ts, side="right")
            epochs[start:stop] = epoch
            has_status = epoch > 0
            idx = np.maximum(epoch - 1, 0)
            _add(mask, has_status & (status_ts[idx] == ts), "equal_status_timestamp")
            if source_kind == "engineering":
                sticky = has_status & status_sticky[idx]
                blocked = ~has_status | status_blocked[idx]
                _add(mask, sticky, "sticky_control_quarantine")
                _add(mask, blocked & ~sticky, "observed_state_blocked")
        elif source_kind == "engineering" and not status_problem:
            mask[:] |= np.uint32(REASON_BITS["observed_state_blocked"])
        previous_ts, previous_seq = int(ts[-1]), int(sequence[-1])
        if progress is not None:
            progress({"day": day, "processed_records": stop, "total_records": n})
    if invalid_timestamp_count:
        day_problems.append("invalid_record_local_day_timestamp")
    if sequence_anomaly_count:
        day_problems.append("source_sequence_non_increasing")
    if event_regression_count:
        day_problems.append("book_event_time_regression" if source_kind == "engineering"
                            else "native_accepted_segment_event_time_regression")
    if day_problems:
        masks[:] |= np.uint32(REASON_BITS["day_quarantine"])
    ranges = []
    # Extract contiguous original-row ranges. Status epochs also split an
    # otherwise eligible range when no book row lands exactly on a status time.
    segment_id, previous_key, previous_eligible = 0, None, False
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        valid = masks[start:stop] == 0
        source_segments = records["segment_id"][start:stop]
        epoch = epochs[start:stop]
        cut = np.ones(stop - start, dtype=np.bool_)
        if len(cut) > 1:
            cut[1:] = ~valid[:-1] | (epoch[1:] != epoch[:-1])
            if source_kind == "native":
                cut[1:] |= source_segments[1:] != source_segments[:-1]
        first_key = (int(epoch[0]), int(source_segments[0]) if source_kind == "native" else 0)
        cut[0] = not previous_eligible or first_key != previous_key
        starts = np.flatnonzero(valid & cut)
        ends = np.flatnonzero(valid & np.concatenate((cut[1:] | ~valid[1:], [True]))) + 1
        if valid[0] and not cut[0]:
            starts = np.concatenate(([0], starts))
        if len(starts) != len(ends):
            raise AssertionError("internal range start/stop mismatch")
        for left, right in zip(starts.tolist(), ends.tolist()):
            absolute_left, absolute_right = start + left, start + right
            if left == 0 and not cut[0]:
                row = ranges[-1]
                row["stop_row"] = absolute_right
                row["record_count"] = absolute_right - row["start_row"]
                row["window_count"] = max(0, row["record_count"] - FOOTPRINT + 1)
            else:
                segment_id += 1
                count = absolute_right - absolute_left
                ranges.append({"start_row": absolute_left, "stop_row": absolute_right,
                               "research_segment_id": segment_id, "record_count": count,
                               "window_count": max(0, count - FOOTPRINT + 1)})
        previous_eligible = bool(valid[-1])
        previous_key = (int(epoch[-1]), int(source_segments[-1]) if source_kind == "native" else 0)
    eligible_count = int(np.count_nonzero(masks == 0))
    reason_counts, primary_counts = {}, {}
    unassigned = masks != 0
    for reason, bit in REASON_BITS.items():
        present = (masks & np.uint32(bit)) != 0
        reason_counts[reason] = int(np.count_nonzero(present))
        selected = present & unassigned
        primary_counts[reason] = int(np.count_nonzero(selected))
        unassigned[selected] = False
    if int(np.count_nonzero(unassigned)) or sum(primary_counts.values()) != n - eligible_count:
        raise AssertionError("internal reason accounting mismatch")
    window_count = sum(item["window_count"] for item in ranges)
    if sum(item["record_count"] for item in ranges) != eligible_count:
        raise AssertionError("internal range accounting mismatch")
    summary = {
        "version": VERSION, "day": day, "source_kind": source_kind,
        "total_records": n, "eligible_records": eligible_count,
        "rejected_records": n - eligible_count, "segment_count": len(ranges),
        "window_count": window_count, "footprint_length": FOOTPRINT,
        "day_problem_codes": list(dict.fromkeys(day_problems)),
        "invalid_timestamp_count": invalid_timestamp_count,
        "sequence_anomaly_count": sequence_anomaly_count,
        "event_regression_count": event_regression_count,
        "status_records": len(statuses),
        "status_timestamp_groups": len(status_ts),
        "reason_counts": reason_counts, "primary_reason_counts": primary_counts,
        "reason_counts_overlap": True,
        "source_control_verified": source_kind == "native",
        "coverage_unknown": source_kind == "engineering",
        "screening_passed": bool(n and eligible_count and not day_problems),
        "source_rows_and_flags_unchanged": True,
        "labels_computed": False, "normalization_computed": False,
        "training_ready": False, "accuracy": None,
    }
    return {"reason_masks": masks, "ranges": ranges, "summary": summary}
