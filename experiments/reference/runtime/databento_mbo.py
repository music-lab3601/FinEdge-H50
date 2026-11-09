"""Streaming DBN MBO reconstruction for the delivered AAPL development date.

Provider bytes are kept intact. No cached book, future labels or model inference
is read by this source. Price conversion uses integer nanodollars -> 1e-4 USD.
"""
from collections import Counter, deque
from datetime import date, datetime, timezone
from itertools import islice
from pathlib import Path
import hashlib
import struct

import numpy as np
from sortedcontainers import SortedDict

from software.runtime.delivery import read_json, sha256
from shared.model_paths import ROOT, PACKAGE
from software.runtime.stream import Record

DATASET = ROOT/'data/databento_20251215'
DAY = '2025-12-15'
MBO_SHA = '0e69b858781b8c7d82e4563b5098f70ff45b02a17c8c3a88aa8880c430ba2fd7'
STATUS_SHA = 'c1d0ee67f1bc5bdab40c4b497684ef056f205b12f2d790bd24756f58e3891fe3'
PROFILE = 'AAPL-DBN-development-20251215-v1'
EST_OFFSET_NS = -18000*10**9
DAY_START_NS = int(datetime(2025,12,15,tzinfo=timezone.utc).timestamp())*10**9
SCREEN_START = (14*3600+35*60)*10**9
SCREEN_END = (20*3600+55*60)*10**9


def iter_dbn(path, *, cancelled=lambda: False):
    """Decode bounded chunks, validate final frame and Zstd stream exhaustion."""
    import databento_dbn as dbn
    import zstandard
    decoder = dbn.DBNDecoder()
    with Path(path).open('rb') as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
        while not cancelled():
            chunk = stream.read(262144)
            if not chunk: break
            for index, message in enumerate(decoder.write_and_decode(chunk)):
                if index % 256 == 0 and cancelled(): return
                yield message
        if not cancelled() and decoder.buffer(): raise ValueError('Truncated DBN record')


def validate_metadata(meta, schema):
    if (str(meta.dataset) != 'XNAS.ITCH' or str(meta.schema) != schema
            or list(meta.symbols) != ['AAPL'] or meta.partial or meta.not_found
            or meta.start != DAY_START_NS or meta.end != DAY_START_NS+86400*10**9
            or str(meta.stype_out) != 'instrument_id' or meta.ts_out):
        raise ValueError('Unexpected DBN date/schema/symbol coverage')
    # DBN metadata mappings, rather than raw-symbol text searches, bind identity.
    mappings = meta.mappings
    if set(mappings) != {'AAPL'}:
        raise ValueError('Unexpected DBN symbol mapping')
    intervals = mappings['AAPL']
    if (len(intervals) != 1 or str(intervals[0]['symbol']) != '38'
            or str(intervals[0]['start_date']) != DAY or str(intervals[0]['end_date']) != '2025-12-16'):
        raise ValueError('DBN AAPL instrument mapping differs')


class MboBook:
    """Order-ID book and aggregated integer price levels; T/F/N do not mutate."""
    def __init__(self):
        self.orders = {}; self.bids = SortedDict(); self.asks = SortedDict()
        self.initialized = False

    def _levels(self, side): return self.bids if side == 'B' else self.asks

    def _quantity(self, side, price, delta):
        levels = self._levels(side); quantity = levels.get(price, 0)+delta
        if quantity < 0: raise ValueError('Negative aggregate order quantity')
        if quantity: levels[price] = quantity
        else: levels.pop(price, None)

    def apply(self, message):
        action = str(message.action); side = str(message.side)
        if action in ('T', 'F', 'N'): return False
        if action == 'R':
            self.orders.clear(); self.bids.clear(); self.asks.clear(); self.initialized = True
            return False
        if not self.initialized: raise ValueError('MBO stream requires clear/snapshot initialization')
        if action not in ('A', 'C', 'M') or side not in ('A', 'B'):
            raise ValueError('Unknown MBO book action/side')
        flags = int(message.flags)
        # This admitted XNAS order feed is neither MBP nor top-of-book synthetic data.
        if flags & (4|16|64): raise ValueError('Bad-book/MBP/TOB message requires another profile')
        ident, size = int(message.order_id), int(message.size)
        if ident <= 0 or size <= 0: raise ValueError('Invalid MBO order identity/size')
        old = self.orders.get(ident)
        if action == 'C':
            if old is None: raise ValueError('Cancel references unknown order')
            old_side, price, available = old
            if old_side != side or size > available: raise ValueError('Cancel side/size mismatch')
            self._quantity(side, price, -size)
            if size == available: del self.orders[ident]
            else: self.orders[ident] = (side, price, available-size)
            return True
        price_ns = int(message.price)
        if price_ns <= 0 or price_ns >= 2**63-1 or price_ns % 100000:
            raise ValueError('Price cannot be represented in delivery 1e-4 USD units')
        price = price_ns//100000
        if action == 'A' and old is not None: raise ValueError('Duplicate live order ID')
        if action == 'M':
            if old is None or old[0] != side: raise ValueError('Modify references unknown order or changes side')
            self._quantity(old[0], old[1], -old[2])
        self.orders[ident] = (side, price, size); self._quantity(side, price, size)
        return True

    def top10(self):
        asks = list(islice(self.asks.items(), 10))
        bids = list(islice(reversed(self.bids.items()), 10))
        flat = []
        for i in range(10):
            flat.extend(asks[i] if i < len(asks) else (0,0))
            flat.extend(bids[i] if i < len(bids) else (0,0))
        values = np.asarray(flat, dtype='<u8')
        return values, len(bids), len(asks)


def read_statuses(path):
    import databento_dbn as dbn
    messages = iter_dbn(path); first = next(messages)
    if not isinstance(first, dbn.Metadata): raise ValueError('STATUS metadata missing')
    validate_metadata(first, 'status')
    result = []
    for ordinal, message in enumerate(messages, 1):
        if not isinstance(message, dbn.StatusMsg) or message.instrument_id != 38 or message.publisher_id != 2:
            raise ValueError('Unexpected STATUS record identity')
        tri = lambda value: 'Y' if value is True else 'N' if value is False else '~' if value is None else 'INVALID'
        result.append(dict(ts_event=int(message.ts_event), action=message.action.name,
            reason=message.reason.name, trading_event=message.trading_event.name,
            is_trading=tri(message.is_trading), is_quoting=tri(message.is_quoting),
            ssr=tri(message.is_short_sell_restricted), ordinal=ordinal))
    if not result or any(a['ts_event'] > b['ts_event'] for a,b in zip(result,result[1:])):
        raise ValueError('Missing or regressing status records')
    protocol = read_json(PACKAGE/'protocol.json')
    reference = next(row for row in protocol['data']['per_day'] if row['date'] == DAY)['record_source_reference_only']['statuses']
    if len(reference) != len(result): raise ValueError('STATUS differs from model delivery')
    for got, expected in zip(result, reference):
        for key, ref_key in (('ts_event','source_ts_event'), ('action','action'), ('reason','reason'),
                ('trading_event','source_trading_event'), ('is_trading','is_trading'), ('is_quoting','is_quoting'), ('ssr','ssr')):
            if got[key] != expected[ref_key]: raise ValueError('STATUS facts differ from model delivery')
    return result


class StatusGate:
    """Conservative timestamp-group gate; same-timestamp records are rejected."""
    def __init__(self, statuses):
        self.groups = []
        for status in statuses:
            if not self.groups or self.groups[-1][0] != status['ts_event']:
                self.groups.append((status['ts_event'], []))
            self.groups[-1][1].append(status)
        self.index = 0; self.eligible = False; self.blocked = False; self.boundary = None

    def accept(self, timestamp):
        while self.index < len(self.groups) and self.groups[self.index][0] <= timestamp:
            boundary, items = self.groups[self.index]; self.index += 1; self.boundary = boundary
            phases = {item['action'] for item in items if item['action'] not in ('NONE','SSR_CHANGE','NEW_PRICE_INDICATION')}
            if any(item['trading_event'] != 'NONE' for item in items): self.blocked = True
            good = all(item['is_trading'] == item['is_quoting'] == 'Y' for item in items)
            if phases == {'TRADING'} and good and not self.blocked: self.eligible = True
            elif phases or not good: self.eligible = False
        return self.eligible and not self.blocked and timestamp != self.boundary


class DatabentoMboSource:
    """Source admission is separate from offline census generation."""
    def __init__(self, dataset=DATASET, *, require_admission=True):
        self.dataset = Path(dataset); raw = self.dataset/'raw'
        self.path = raw/'xnas-itch-20251215.mbo.dbn.zst'
        self.status_path = raw/'xnas-itch-20251215.status.dbn.zst'
        if sha256(self.path) != MBO_SHA or sha256(self.status_path) != STATUS_SHA:
            raise ValueError('Current independent DBN copy hash differs')
        self.statuses = read_statuses(self.status_path)
        self.admission = read_json(self.dataset/'admission.json') if require_admission else None
        if self.admission is not None and (self.admission.get('profile') != PROFILE
                or self.admission.get('source_sha256') != MBO_SHA or self.admission.get('status_sha256') != STATUS_SHA
                or self.admission.get('status') != 'passed'
                or self.admission.get('adapter_sha256') != sha256(__file__)
                or self.admission.get('original_validation_alignment') is not False):
            raise ValueError('DBN source census is not admitted')
        self.expected_counts = dict(self.admission['expected_counts']) if self.admission else None
        self.provenance = dict(mode='original_databento_mbo_dbn', source_sha256=MBO_SHA,
            status_sha256=STATUS_SHA, profile=PROFILE, date=DAY, symbol='AAPL',
            dataset_role='development_validation', source_control_verified=False, coverage_unknown=True,
            prepared_book_read=False, prepared_labels_read=False, state_initialization='original_clear_then_full_prefix')
        self.cancelled = lambda: False; self.metrics = {}; self.on_progress = lambda metrics: None

    def iter_records(self):
        import databento_dbn as dbn
        book = MboBook(); gate = StatusGate(self.statuses)
        stream = iter_dbn(self.path, cancelled=self.cancelled)
        meta = next(stream)
        if not isinstance(meta, dbn.Metadata): raise ValueError('MBO metadata missing')
        validate_metadata(meta, 'mbo')
        self.metrics = dict(frames_read=0, mutation_rows=0, eligible_records=0, completed_full_stream=False,
            segments=0, rejected_rows=0, max_orders=0)
        last_ts = None; previous_eligible = False; segment = 0; row = 0; boundary = None
        try:
            for ordinal, msg in enumerate(stream, 1):
                if self.cancelled(): return
                self.metrics['frames_read'] += 1
                if not isinstance(msg, dbn.MBOMsg) or msg.instrument_id != 38 or msg.publisher_id != 2:
                    raise ValueError('Unexpected MBO record identity')
                ts = int(msg.ts_event)
                if last_ts is not None and ts < last_ts: raise ValueError('MBO event time regressed')
                last_ts = ts
                status_ok = gate.accept(ts)
                if gate.boundary != boundary:
                    previous_eligible = False; boundary = gate.boundary
                mutation = book.apply(msg)
                if str(msg.action) == 'R': previous_eligible = False
                if not mutation: continue
                values, bid_levels, ask_levels = book.top10()
                self.metrics['mutation_rows'] += 1
                current_row = row; row += 1
                valid = (SCREEN_START <= ts-DAY_START_NS < SCREEN_END and status_ok
                    and bid_levels == ask_levels == 10 and 0 < int(values[2]) < int(values[0]))
                if not valid:
                    previous_eligible = False; self.metrics['rejected_rows'] += 1; continue
                if not previous_eligible: segment += 1
                previous_eligible = True
                self.metrics['segments'] = segment; self.metrics['eligible_records'] += 1
                self.metrics['max_orders'] = max(self.metrics['max_orders'], len(book.orders))
                metadata = dict(date=DAY, symbol='AAPL', snippet_id=PROFILE,
                    research_segment_id=segment, native_segment_id=segment,
                    source_record_row_0based=current_row, sample_row_0based=current_row,
                    source_message_position_1based=ordinal,
                    timestamp_ns_since_exchange_midnight=ts-DAY_START_NS+EST_OFFSET_NS,
                    research_reject_mask=0, bid_levels=10, ask_levels=10,
                    source_control_verified=False, coverage_unknown=True)
                yield Record(values, metadata)
            if not self.cancelled():
                if self.expected_counts is not None and (
                        self.metrics['frames_read'] != self.expected_counts['frames']
                        or self.metrics['eligible_records'] != self.expected_counts['records']):
                    raise ValueError('DBN full-stream census differs from frozen admission')
                self.metrics['completed_full_stream'] = True
        finally:
            stream.close()
