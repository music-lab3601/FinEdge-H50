"""MBO -> legacy visible-book rows. Control state absent: fail closed.

This is an engineering adapter, not a replacement for ITCH Q/M/H/h gating.
No labels, model, optimizer, network or account access.
"""
from collections import Counter
from datetime import date, datetime, time, timezone, timedelta

from itch_book import Book, ROW, snapshot

VERSION = 'Databento-MBO-engineering-adapter-2.0'
CONTROL_UNVERIFIED = 64
BAD_BOOK = 128
LAST, TOB, SNAPSHOT, MBP, BAD_RECV, MAYBE_BAD = 128, 64, 32, 16, 8, 4
DAY_NS = 86400 * 10**9


def price4(value):
    if value <= 0 or value >= 2**63 - 1 or value % 100000:
        raise ValueError('Order price not exactly representable as positive ITCH Price(4)')
    return value // 100000


def midnight_ns(day):
    # Explicitly scoped to these December 2025 dates; all are New York EST.
    d = date.fromisoformat(day)
    if d.year != 2025 or d.month != 12:
        raise ValueError('EST conversion only approved for December 2025')
    return int(datetime.combine(d, time(), timezone(timedelta(hours=-5))).timestamp()) * 10**9


class Adapter:
    def __init__(self, day, instrument_ids, write_row):
        self.midnight = midnight_ns(day)
        self.instrument_ids = set(instrument_ids)
        if not self.instrument_ids:
            raise ValueError('AAPL metadata mapping absent')
        self.write_row = write_row
        self.book = Book()
        self.group_key = None
        self.mutated = False
        self.group_actions = []
        self.group_bad = False
        self.bad_book = False
        self.snapshot_active = False
        self.previous_event_ts = None
        self.previous_recv = None
        self.previous_sequence = None
        self.initialized = False
        self.counts = Counter()
        self.actions = Counter()
        self.flag_counts = Counter()
        self.row_flags = Counter()
        self.rows = 0
        self.first_event = None
        self.last_event = None

    def mutate(self, r):
        action, oid, size = str(r.action), int(r.order_id), int(r.size)
        side = {'B': 'B', 'A': 'S'}.get(str(r.side))
        if action == 'A':
            self.book.add(oid, side, price4(int(r.price)), size)
        elif action in ('C', 'M'):
            if oid not in self.book.orders:
                raise ValueError(f'Unknown active order in {action}')
            oldside, oldprice, remaining = self.book.orders[oid]
            if side != oldside or price4(int(r.price)) != oldprice and action == 'C':
                raise ValueError('Cancel side/price differs from resting order')
            if action == 'C':
                self.book.apply({'type': 'X', 'order_id': oid, 'shares': size})
            else:
                if size <= 0:
                    raise ValueError('Nonpositive absolute modify quantity')
                self.book.apply({'type': 'D', 'order_id': oid})
                self.book.add(oid, side, price4(int(r.price)), size)
        else:
            return False
        return True

    def consume(self, r):
        if int(r.publisher_id) != 2 or int(r.instrument_id) not in self.instrument_ids:
            raise ValueError('Wrong publisher or AAPL instrument ID')
        action, flags = str(r.action), int(r.flags)
        self.counts['decoded_mbo_records'] += 1
        self.actions[action] += 1
        self.flag_counts[str(flags)] += 1
        if flags & (TOB | MBP):
            raise ValueError('TOB/MBP flags do not represent full order-level book')
        if action not in ('A', 'C', 'M', 'R', 'T', 'F', 'N'):
            raise ValueError('Unknown MBO action')
        if action == 'R':
            if self.group_key is not None:
                raise ValueError('Reset inside unfinished native event')
            self.book = Book()
            self.initialized = True
            self.bad_book = False
            self.snapshot_active = bool(flags & SNAPSHOT)
            self.previous_event_ts = None
            self.previous_sequence = None
            self.counts['resets'] += 1
            return
        if not self.initialized:
            raise ValueError('No verified reset/initialization before order stream')
        if flags & SNAPSHOT:
            if self.group_key is not None:
                raise ValueError('Snapshot inside unfinished native event')
            self.snapshot_active = True
            self.mutate(r)
            self.counts['snapshot_records'] += 1
            if flags & LAST:
                self.snapshot_active = False
            return
        if self.snapshot_active:
            raise ValueError('Snapshot unfinished before live events')
        ts, recv, sequence = int(r.ts_event), int(r.ts_recv), int(r.sequence)
        # Input stays in original provider order, never event-time sorted.
        if self.previous_recv is not None and recv < self.previous_recv:
            raise ValueError('Receive timestamp regression')
        self.previous_recv = recv
        if not self.midnight <= ts < self.midnight + DAY_NS:
            raise ValueError('Event outside local trading date')
        key = (int(r.publisher_id), int(r.instrument_id), int(r.channel_id), sequence, ts)
        if self.group_key is not None and sequence != self.group_key[3]:
            # F_LAST is the venue-event boundary, not an ITCH message boundary.
            # Multiple different native sequences may precede one F_LAST.
            if key[:3] != self.group_key[:3]:
                raise ValueError('Instrument/channel changed inside native grouping')
            self.complete_group(False)
        if self.group_key is None:
            if self.previous_sequence is not None and sequence <= self.previous_sequence:
                raise ValueError('Non-increasing native sequence across completed groups')
            self.group_key = key
            self.mutated = False
            self.group_bad = False
            self.group_actions = []
        elif key != self.group_key:
            raise ValueError('Native event changed before F_LAST')
        if flags & MAYBE_BAD:
            self.bad_book = True
            self.counts['maybe_bad_book_records'] += 1
        if flags & BAD_RECV:
            self.group_bad = True
            self.counts['bad_receive_timestamp_records'] += 1
        self.mutated = self.mutate(r) or self.mutated
        self.group_actions.append(action)
        if not flags & LAST:
            return
        self.counts['venue_event_last_records'] += 1
        self.complete_group(True)

    def complete_group(self, venue_last):
        pattern = ''.join(self.group_actions)
        if pattern not in ('A', 'C', 'CA', 'TFC', 'FC', 'T', 'N', 'M'):
            raise ValueError('Incomplete/unsupported XNAS native action pattern: '+pattern)
        self.counts['native_pattern_'+pattern] += 1
        self.counts['complete_native_events'] += 1
        sequence, ts = self.group_key[3:]
        backwards = self.previous_event_ts is not None and ts < self.previous_event_ts
        if backwards:
            self.counts['event_timestamp_regressions'] += 1
        self.previous_event_ts = ts
        self.previous_sequence = sequence
        if self.mutated:
            # Original system/trading gate cannot be proved from MBO. All rows diagnostic.
            row_flags, features = snapshot(self.book, False, False, backwards)
            row_flags |= CONTROL_UNVERIFIED
            if self.bad_book or self.group_bad:
                row_flags |= BAD_BOOK
            self.write_row(ROW.pack(sequence, ts-self.midnight, 0, row_flags,
                                    len(self.book.prices['B']), len(self.book.prices['S']), *features))
            self.rows += 1
            if not venue_last:
                self.counts['native_samples_before_venue_last'] += 1
            self.row_flags[str(row_flags)] += 1
            self.first_event = ts if self.first_event is None else self.first_event
            self.last_event = ts
        self.group_key = None
        self.mutated = False
        self.group_actions = []

    def finish(self):
        if self.group_key is not None or self.snapshot_active:
            raise ValueError('Truncated native event/snapshot at EOF')
        if not self.initialized or not self.rows:
            raise ValueError('No complete initialized visible-book events')
        self.book.validate()
        return {
            'adapter_version': VERSION, 'rows': self.rows, 'valid_rows': 0, 'segments': 0,
            'counts': dict(self.counts), 'action_counts': dict(self.actions),
            'provider_flag_counts': dict(self.flag_counts), 'row_flag_counts': dict(self.row_flags),
            'first_sample_event_unix_ns': self.first_event, 'last_sample_event_unix_ns': self.last_event,
            'control_state_verified': False, 'regular_system_phase_verified': False,
            'stock_trading_state_verified': False, 'training_ready': False,
            'control_gate': 'fail_closed_all_rows_diagnostic',
            'labels_computed': False, 'model_forward_count': 0, 'optimizer_updates': 0,
        }
