"""Pure synthetic reference for screening proposal 3.0; not a data adapter.

The functions accept small in-memory fixtures only. They neither activate the
draft nor read files, decode market data, compute labels, fit statistics, or
import an ML runtime. Named enums below are locked to databento-dbn 0.71.0.
An accepted synthetic result does not prove vendor control-event coverage.
"""

from dataclasses import dataclass, replace


REFERENCE_VERSION = "TCN-screening-synthetic-reference-1.0"
PROPOSAL_VERSION = "TCN-screening-proposal-3.0-draft"
ADAPTER_VERSION = "Databento-MBO-engineering-adapter-2.0"
NS_SECOND = 1_000_000_000
SCREEN_START_NS = (9 * 3600 + 35 * 60) * NS_SECOND
SCREEN_END_NS = (15 * 3600 + 55 * 60) * NS_SECOND
TARGET_DATES = frozenset(
    f"2025-12-{day:02d}" for day in (8, 9, 10, 11, 12, 15, 16, 17, 18, 19)
)
PRESERVING_ACTIONS = frozenset(("NONE", "SSR_CHANGE", "NEW_PRICE_INDICATION"))
BLOCKING_ACTIONS = frozenset((
    "PRE_OPEN", "PRE_CROSS", "QUOTING", "CROSS", "ROTATION", "PRE_CLOSE",
    "CLOSE", "POST_CLOSE", "HALT", "PAUSE", "SUSPEND",
    "NOT_AVAILABLE_FOR_TRADING",
))
ACTIONS = PRESERVING_ACTIONS | BLOCKING_ACTIONS | {"TRADING"}
REASONS = frozenset((
    "NONE", "SCHEDULED", "SURVEILLANCE_INTERVENTION", "MARKET_EVENT",
    "INSTRUMENT_ACTIVATION", "INSTRUMENT_EXPIRATION", "RECOVERY_IN_PROCESS",
    "REGULATORY", "ADMINISTRATIVE", "NON_COMPLIANCE", "FILINGS_NOT_CURRENT",
    "SEC_TRADING_SUSPENSION", "NEW_ISSUE", "ISSUE_AVAILABLE", "ISSUES_REVIEWED",
    "FILING_REQS_SATISFIED", "NEWS_PENDING", "NEWS_RELEASED",
    "NEWS_AND_RESUMPTION_TIMES", "NEWS_NOT_FORTHCOMING", "ORDER_IMBALANCE",
    "LULD_PAUSE", "OPERATIONAL", "ADDITIONAL_INFORMATION_REQUESTED",
    "MERGER_EFFECTIVE", "ETF", "CORPORATE_ACTION", "NEW_SECURITY_OFFERING",
    "MARKET_WIDE_HALT_LEVEL1", "MARKET_WIDE_HALT_LEVEL2",
    "MARKET_WIDE_HALT_LEVEL3", "MARKET_WIDE_HALT_CARRYOVER",
    "MARKET_WIDE_HALT_RESUMPTION", "QUOTATION_NOT_AVAILABLE",
))
STICKY_REASONS = frozenset(
    reason for reason in REASONS
    if reason == "OPERATIONAL" or reason.startswith("MARKET_WIDE_HALT_")
)
TRISTATES = frozenset(("Y", "N", "~"))
ROW_CONTROL_UNVERIFIED = 64
ROW_QUALITY_REJECT = 2 | 4 | 8 | 32 | 128
ROW_KNOWN_BITS = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128
PROVIDER_LAST = 128  # Separate namespace: this is not row BAD_BOOK.


@dataclass(frozen=True)
class Status:
    day: str
    ts_event: int
    action: str
    is_trading: str = "Y"
    is_quoting: str = "Y"
    reason: str = "NONE"
    ssr: str = "~"
    symbol: str = "AAPL"


@dataclass(frozen=True)
class ObservedState:
    day: str
    phase: str = "unknown"
    blocked: bool = True
    quarantined: bool = False
    boundary: int | None = None
    status_epoch: int = 0

    @property
    def eligible(self):
        return not self.quarantined and not self.blocked and self.phase == "TRADING"


def _valid_timestamp(value):
    return type(value) is int and value >= 0 and value < (1 << 64) - 1


def _valid_status(status, day):
    return (
        isinstance(status, Status)
        and all(type(value) is str for value in (
            status.day, status.symbol, status.action, status.reason,
            status.is_trading, status.is_quoting, status.ssr,
        ))
        and status.day == day and status.symbol == "AAPL"
        and _valid_timestamp(status.ts_event) and status.action in ACTIONS
        and status.reason in REASONS and status.is_trading in TRISTATES
        and status.is_quoting in TRISTATES and status.ssr in TRISTATES
    )


def apply_status_group(state, group):
    """Apply one already aggregated timestamp group, without last-record wins.

    Every group cuts the segment. N/unknown and phase conflicts latch blocked;
    only a later unambiguous TRADING group, all Y/Y, releases an ordinary latch.
    Operational/MWCB quarantine is sticky until a fresh day's initial state.
    """
    group = tuple(group)
    next_state = replace(state, status_epoch=state.status_epoch + 1)
    if not group or any(not _valid_status(item, state.day) for item in group):
        return replace(next_state, blocked=True, quarantined=True)
    timestamp = group[0].ts_event
    if any(item.ts_event != timestamp for item in group):
        return replace(next_state, blocked=True, quarantined=True)
    if state.boundary is not None and timestamp <= state.boundary:
        return replace(next_state, blocked=True, quarantined=True)
    next_state = replace(next_state, boundary=timestamp)
    if state.quarantined or any(item.reason in STICKY_REASONS for item in group):
        return replace(next_state, blocked=True, quarantined=True)
    phase_updates = {item.action for item in group if item.action not in PRESERVING_ACTIONS}
    all_yy = all(item.is_trading == "Y" and item.is_quoting == "Y" for item in group)
    if len(phase_updates) > 1:
        return replace(next_state, phase="conflict", blocked=True)
    phase = next(iter(phase_updates)) if phase_updates else state.phase
    if not all_yy or phase_updates & BLOCKING_ACTIONS:
        return replace(next_state, phase=phase, blocked=True)
    if phase_updates == {"TRADING"}:
        return replace(next_state, phase="TRADING", blocked=False)
    # A phase-preserving Y/Y update can continue an eligible state, not resume it.
    return replace(next_state, phase=phase, blocked=state.blocked or phase != "TRADING")


def clock_contains(local_event_ns, day, calendar_is_known=True):
    """Input is an explicitly supplied New York local nanosecond-of-day."""
    return (
        calendar_is_known and day in TARGET_DATES and type(local_event_ns) is int
        and SCREEN_START_NS <= local_event_ns < SCREEN_END_NS
    )


@dataclass(frozen=True)
class Row:
    source_index: int
    day: str
    ts_event: int
    row_flags: int
    symbol: str = "AAPL"
    native_segment: int = 0
    provider_flags: int = PROVIDER_LAST
    initialized: bool = True
    complete_group: bool = True
    is_reset_or_snapshot: bool = False


@dataclass(frozen=True)
class ResearchIndexRow:
    source_index: int
    day: str
    symbol: str
    ts_event: int
    source_flags: int
    research_eligible: bool
    research_segment_id: int
    source_control_verified: bool
    coverage_unknown: bool
    reject_reasons: tuple[str, ...]


def row_reject_reasons(row, source_kind, adapter_version=None):
    """Interpret derived row flags only; provider F_LAST is never BAD_BOOK."""
    reasons = []
    if row.symbol != "AAPL" or type(row.source_index) is not int or row.source_index < 0:
        reasons.append("identity_or_index")
    if not _valid_timestamp(row.ts_event):
        reasons.append("timestamp")
    if type(row.row_flags) is not int or row.row_flags < 0 or row.row_flags & ~ROW_KNOWN_BITS:
        reasons.append("unexpected_row_bits")
    elif source_kind == "native":
        if row.row_flags != 0:
            reasons.append("native_control_or_quality_flag")
        if type(row.native_segment) is not int or row.native_segment <= 0:
            reasons.append("native_segment")
    elif source_kind == "engineering":
        if adapter_version != ADAPTER_VERSION:
            reasons.append("unbound_engineering_provenance")
        if not row.row_flags & ROW_CONTROL_UNVERIFIED:
            reasons.append("missing_unverified_source_flag")
        if row.row_flags & ROW_QUALITY_REJECT:
            reasons.append("engineering_quality_flag")
    else:
        reasons.append("unknown_source_kind")
    if not row.initialized or not row.complete_group:
        reasons.append("incomplete_book_or_group")
    if row.is_reset_or_snapshot:
        reasons.append("reset_or_snapshot")
    return tuple(reasons)


def _status_groups(statuses, day):
    """Validate fixture order, aggregate adjacent equal ts, never sort events."""
    groups = []
    previous = None
    for status in statuses:
        if not _valid_status(status, day):
            return (), "invalid_status_or_identity"
        if previous is not None and status.ts_event < previous:
            return (), "status_event_time_regression"
        if groups and groups[-1][0].ts_event == status.ts_event:
            groups[-1].append(status)
        else:
            groups.append([status])
        previous = status.ts_event
    return tuple(tuple(group) for group in groups), None


def screen_research_rows(rows, statuses, day, source_kind, adapter_version=None,
                         calendar_is_known=True):
    """Screen synthetic row fixtures; source indices and flags are immutable.

    For engineering fixtures, day-wide identity/time/enum problems reject the
    entire fixture, including earlier rows. Operational events reject the event
    and the rest of the day. Native fixtures take the intersection of already
    valid native segments and the clock; no weaker status gate readmits them.
    ts_event values here are supplied New York local nanoseconds-of-day; this
    prototype deliberately performs no UTC/calendar conversion or data reading.
    """
    rows, statuses = tuple(rows), tuple(statuses)
    groups, day_problem = _status_groups(statuses, day)
    if not calendar_is_known or day not in TARGET_DATES:
        day_problem = "unknown_or_nonstandard_calendar"
    if source_kind == "engineering" and not statuses:
        day_problem = "missing_status"
    if any(row.day != day or row.symbol != "AAPL" for row in rows):
        day_problem = "mixed_day_or_instrument_identity"
    if source_kind == "engineering":
        timestamps = [row.ts_event for row in rows]
        if any(not _valid_timestamp(timestamp) for timestamp in timestamps):
            day_problem = "invalid_event_timestamp"
        elif any(left > right for left, right in zip(timestamps, timestamps[1:])):
            day_problem = "book_event_time_regression"
    state = ObservedState(day)
    outputs, group_index, segment_id = [], 0, 0
    active_key, previous_index = None, None
    for row in rows:
        while (not day_problem and _valid_timestamp(row.ts_event)
               and group_index < len(groups)
               and groups[group_index][0].ts_event <= row.ts_event):
            state = apply_status_group(state, groups[group_index])
            group_index += 1
        reasons = list(row_reject_reasons(row, source_kind, adapter_version))
        if day_problem:
            reasons.append(day_problem)
        if not clock_contains(row.ts_event, day, calendar_is_known):
            reasons.append("outside_research_clock")
        if source_kind == "engineering":
            if state.quarantined:
                reasons.append("sticky_control_quarantine")
            elif not state.eligible:
                reasons.append("observed_state_blocked")
        if state.boundary == row.ts_event:
            reasons.append("equal_status_timestamp")
        eligible = not reasons
        key = (row.day, row.symbol, state.status_epoch,
               row.native_segment if source_kind == "native" else None)
        if eligible:
            if key != active_key or previous_index is None or row.source_index != previous_index + 1:
                segment_id += 1
            active_key = key
        else:
            active_key = None
        outputs.append(ResearchIndexRow(
            row.source_index, row.day, row.symbol, row.ts_event, row.row_flags,
            eligible, segment_id if eligible else 0, source_kind == "native",
            source_kind != "native", tuple(dict.fromkeys(reasons)),
        ))
        previous_index = row.source_index
    return tuple(outputs)


def window_endpoints(index_rows):
    """Return endpoint original indices; all 150 consecutive rows must pass.

    Neither ineligible rows nor absent original indices are compressed. The
    endpoint offset is 99 and label target offset is 149 in a footprint.
    No label or market price is computed by this helper.
    """
    index_rows = tuple(index_rows)
    endpoints = []
    for start in range(max(0, len(index_rows) - 149)):
        footprint = index_rows[start:start + 150]
        first = footprint[0]
        if not first.research_eligible or first.research_segment_id <= 0:
            continue
        identity = (first.day, first.symbol, first.research_segment_id)
        if all(
            row.research_eligible
            and (row.day, row.symbol, row.research_segment_id) == identity
            and row.source_index == first.source_index + offset
            for offset, row in enumerate(footprint)
        ):
            endpoints.append(footprint[99].source_index)
    return tuple(endpoints)
