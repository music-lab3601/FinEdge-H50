"""Independent native ITCH 5 file adapter and visible-order-book reconstruction.

Only current-project files are used. Extraction retains exact selected frames
and all system/MWCB controls; it does not reconstruct or resample orders.
"""
from collections import Counter
from datetime import datetime, timezone
import gzip
from itertools import islice
from pathlib import Path
import struct
import time

import numpy as np
from sortedcontainers import SortedDict

from .delivery import read_json, sha256, write_json
from .paths import ROOT

DATASET = ROOT / 'data/nasdaq_20190730'
SOURCE_SHA = 'c65784c48c28735901ae442dc00e215834218a359bc12a139ab4eec209bc2d4a'
LENGTHS = dict(zip('SRHYL VWKJhAFECXDUPQBIN'.replace(' ', ''),
                   [12,39,25,20,26,35,12,28,35,21,36,40,31,36,23,19,35,44,40,19,50,20]))
MUTATIONS = frozenset('AFECXDU')
GLOBAL = frozenset('SVW')
BOOK_DTYPE = np.dtype([('source_sequence','<u8'), ('timestamp_ns','<u8'),
    ('segment_id','<u4'), ('flags','<u4'), ('bid_levels','<u4'), ('ask_levels','<u4'),
    ('source_offset','<u8'), ('features','<u8',(40,))])
PROFILE = 'native-itch5-visible-top10-20190730-v1'
SCREEN_START = (9*3600+35*60)*10**9
SCREEN_END = (15*3600+55*60)*10**9


def extract_aapl(dataset=DATASET):
    dataset = Path(dataset)
    source = dataset / 'raw/07302019.NASDAQ_ITCH50.gz'
    admission = read_json(source.parent / 'admission.json')
    if admission['sha256'] != SOURCE_SHA or sha256(source) != SOURCE_SHA:
        raise ValueError('Raw ITCH source identity mismatch')
    out = dataset / 'extracted'; out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old.get('status') == 'passed' and old.get('source_sha256') == SOURCE_SHA:
            if all(sha256(out/n) == ref['sha256'] for n, ref in old['files'].items()):
                return old
        raise ValueError('Existing extraction is not verified; choose a fresh output directory')
    counts, selected_counts, locates = Counter(), Counter(), set()
    total = selected = offset = 0
    start = last_progress = time.perf_counter()
    events_tmp = out / 'aapl.itch.partial'
    positions_tmp = out / 'positions.u64le.bin.partial'
    systems = []
    try:
        with source.open('rb') as compressed, gzip.GzipFile(fileobj=compressed) as stream, \
             events_tmp.open('wb') as events, positions_tmp.open('wb') as positions:
            carry = b''
            while True:
                chunk = stream.read(8*1024*1024)
                if not chunk:
                    if carry: raise ValueError('Truncated final ITCH frame')
                    break
                data = carry + chunk; p = 0; end = len(data)
                while p+2 <= end:
                    length = (data[p]<<8) | data[p+1]
                    if length < 11 or length > 2048: raise ValueError(f'Invalid frame at {offset+p}')
                    stop = p+2+length
                    if stop > end: break
                    kind = chr(data[p+2])
                    if LENGTHS.get(kind) != length:
                        raise ValueError(f'Unknown type/length {kind!r}/{length} at {offset+p}')
                    locate = (data[p+3]<<8) | data[p+4]
                    total += 1; counts[kind] += 1
                    if kind == 'R':
                        symbol = data[p+13:p+21].decode('ascii').strip()
                        if symbol == 'AAPL': locates.add(locate)
                        elif locate in locates: raise ValueError('AAPL locate was reused by another symbol')
                    if kind in GLOBAL or locate in locates:
                        frame = data[p:stop]
                        events.write(frame)
                        positions.write(struct.pack('<QQ', total, offset+p))
                        selected += 1; selected_counts[kind] += 1
                        if kind == 'S':
                            systems.append({'sequence':total, 'event':chr(data[p+13]),
                                            'timestamp_ns':int.from_bytes(data[p+7:p+13],'big')})
                    p = stop
                offset += p; carry = data[p:]
                if time.perf_counter()-last_progress >= 5:
                    print(f'EXTRACT {compressed.tell()/source.stat().st_size:.1%} messages={total} selected={selected}',flush=True)
                    last_progress = time.perf_counter()
        if len(locates) != 1 or not systems or systems[0]['event'] != 'O' or systems[-1]['event'] != 'C':
            raise ValueError('Missing unique AAPL directory or complete system start/end')
        events_tmp.replace(out/'aapl.itch'); positions_tmp.replace(out/'positions.u64le.bin')
        report = {'status':'passed','version':'native-itch5-aapl-extraction-v1', 'date':'2019-07-30',
                  'symbol':'AAPL','source_sha256':SOURCE_SHA,'source_bytes':source.stat().st_size,
                  'uncompressed_bytes':offset,'total_messages':total,'selected_messages':selected,
                  'all_type_counts':dict(counts),'selected_type_counts':dict(selected_counts),
                  'stock_locate':next(iter(locates)),'system_events':systems,
                  'gzip_full_crc_verified':True,'original_selected_frame_bytes_preserved':True,
                  'book_reconstructed':False,'elapsed_seconds':time.perf_counter()-start,
                  'files':{n:{'bytes':(out/n).stat().st_size,'sha256':sha256(out/n)}
                           for n in ('aapl.itch','positions.u64le.bin')}}
        write_json(manifest_path,report)
        return report
    except BaseException as exc:
        write_json(out/'extraction_failure.json',{'status':'failed','error':str(exc),'messages':total,'offset':offset})
        raise


def selected_frames(dataset=DATASET):
    dataset = Path(dataset); out = dataset/'extracted'
    manifest = read_json(out/'manifest.json')
    if manifest['status'] != 'passed': raise ValueError('Extraction did not pass')
    for name, ref in manifest['files'].items():
        if sha256(out/name) != ref['sha256']: raise ValueError('Extracted file was modified')
    positions = np.memmap(out/'positions.u64le.bin',mode='r',dtype='<u8').reshape(-1,2)
    if len(positions) != manifest['selected_messages']: raise ValueError('Event identity count mismatch')
    with (out/'aapl.itch').open('rb') as f:
        for sequence, offset in positions:
            header = f.read(2)
            if len(header) != 2: raise ValueError('Truncated selected frame header')
            payload = f.read(int.from_bytes(header,'big'))
            if not payload or LENGTHS.get(chr(payload[0])) != len(payload): raise ValueError('Invalid selected frame')
            yield int(sequence), int(offset), payload
        if f.read(1): raise ValueError('Extra selected frames without identities')


class VisibleBook:
    """Order IDs are distinct from price levels; execution price never moves an order."""
    def __init__(self):
        self.orders = {}
        self.seen_ids = set()
        self.bids, self.asks = SortedDict(), SortedDict()

    def _level(self, side, price, delta):
        levels = self.bids if side == 'B' else self.asks
        value = levels.get(price,0) + delta
        if value < 0: raise ValueError('Negative aggregate size')
        if value: levels[price] = value
        elif price in levels: del levels[price]

    def add(self, oid, side, quantity, price):
        if oid in self.seen_ids: raise ValueError('Duplicate day-unique order ID')
        if side not in ('B','S') or quantity <= 0 or price <= 0: raise ValueError('Invalid added order')
        self.orders[oid] = (side,price,quantity)
        self.seen_ids.add(oid)
        self._level(side,price,quantity)

    def reduce(self, oid, quantity=None):
        if oid not in self.orders: raise ValueError('Unknown order reference')
        side, price, remaining = self.orders[oid]
        quantity = remaining if quantity is None else quantity
        if quantity <= 0 or quantity > remaining: raise ValueError('Invalid/oversized reduction')
        self._level(side,price,-quantity)
        if quantity == remaining: del self.orders[oid]
        else: self.orders[oid] = (side,price,remaining-quantity)

    def apply(self, payload, profiler=None):
        mark = profiler.start() if profiler is not None else None
        kind = chr(payload[0])
        if kind not in MUTATIONS:
            if profiler is not None: profiler.end('order_decode', mark)
            return False
        oid = int.from_bytes(payload[11:19],'big')
        if kind in 'AF':
            if payload[24:32].decode('ascii').strip() != 'AAPL': raise ValueError('Wrong symbol in AAPL order')
            side, quantity, price = chr(payload[19]), int.from_bytes(payload[20:24],'big'), int.from_bytes(payload[32:36],'big')
        elif kind in 'ECX': quantity = int.from_bytes(payload[19:23],'big')
        elif kind == 'U':
            if oid not in self.orders: raise ValueError('Unknown replaced order')
            new = int.from_bytes(payload[19:27],'big')
            quantity, price = struct.unpack('>II',payload[27:35])
        if profiler is not None:
            profiler.end('order_decode', mark)
            mark = profiler.start()
        if kind in 'AF': self.add(oid,side,quantity,price)
        elif kind in 'ECX': self.reduce(oid,quantity)
        elif kind == 'D': self.reduce(oid)
        elif kind == 'U':
            if new in self.seen_ids or quantity <= 0 or price <= 0: raise ValueError('Invalid replacement')
            side = self.orders[oid][0]
            self.reduce(oid); self.add(new,side,quantity,price)
        if profiler is not None: profiler.end('book_update', mark)
        return True

    def top10(self):
        # Build Python scalars once, then perform a single uint64 conversion.
        # Avoid twenty small NumPy slice assignments in this hot path.
        values = [0] * 40
        for i,(price,size) in enumerate(islice(self.asks.items(),10)):
            values[4*i] = price
            values[4*i+1] = size
        for i,(price,size) in enumerate(islice(reversed(self.bids.items()),10)):
            values[4*i+2] = price
            values[4*i+3] = size
        return np.array(values,dtype='<u8')


class NativeBookProcessor:
    """One event in, one owned mutation row or None out; no file access."""
    def __init__(self, profiler=None):
        self.profiler = profiler
        self.book = VisibleBook()
        self.phase = self.trading = None
        self.operational = 'T'
        self.quarantine = False
        self.previous_time = None
        self.previous_sequence = self.previous_offset = None
        self.segment = self.count = self.accepted = self.events = 0
        self.was_valid = False
        self.reasons = Counter()
        self.controls, self.segments = [], []
        self.current_segment = None

    def push(self, sequence, offset, payload, emit_row=True):
        probe = self.profiler
        mark = probe.start() if probe is not None else None
        if not payload or LENGTHS.get(chr(payload[0])) != len(payload):
            raise ValueError('Invalid native ITCH type/length')
        if (self.previous_sequence is not None and
                (sequence <= self.previous_sequence or offset <= self.previous_offset)):
            raise ValueError('Raw event identity regressed or duplicated')
        self.previous_sequence, self.previous_offset = sequence, offset
        self.events += 1
        kind = chr(payload[0])
        timestamp = int.from_bytes(payload[5:11], 'big')
        if kind in 'SHhW':
            self.was_valid = False
            if kind == 'S':
                self.phase = chr(payload[11])
                if self.phase not in 'OSQMEC': self.quarantine = True
            elif kind == 'H':
                self.trading = chr(payload[19])
                if self.trading not in 'HPQT': self.quarantine = True
            elif kind == 'h':
                market, action = chr(payload[19]), chr(payload[20])
                if market == 'Q':
                    self.operational = action
                    if action != 'T': self.quarantine = True
            elif kind == 'W': self.quarantine = True
            self.controls.append({'sequence': sequence, 'type': kind,
                                  'timestamp_ns': timestamp, 'payload_hex': payload.hex()})
        if probe is not None: probe.end('event_header_control', mark)
        if not self.book.apply(payload, profiler=probe): return None
        mark = probe.start() if probe is not None else None
        features = self.book.top10() if emit_row else None
        if probe is not None:
            probe.end('top10_extract', mark)
            mark = probe.start()
        flags = 0
        if self.phase != 'Q' or not SCREEN_START <= timestamp < SCREEN_END: flags |= 1
        if len(self.book.bids) < 10 or len(self.book.asks) < 10: flags |= 2
        if self.book.bids and self.book.asks:
            bid, ask = self.book.bids.peekitem(-1)[0], self.book.asks.peekitem(0)[0]
            if bid == ask: flags |= 4
            if bid > ask: flags |= 8
        if self.trading != 'T' or self.operational != 'T' or self.quarantine: flags |= 16
        if self.previous_time is not None and timestamp < self.previous_time:
            flags |= 32
            self.quarantine = True
        self.previous_time = timestamp
        valid = flags == 0
        if valid and not self.was_valid:
            self.segment += 1
            self.current_segment = {'segment_id': self.segment, 'first_row': self.count,
                                    'count': 0, 'first_timestamp_ns': timestamp}
            self.segments.append(self.current_segment)
        if valid:
            self.accepted += 1
            self.current_segment['count'] += 1
            self.current_segment.update(last_row=self.count, last_timestamp_ns=timestamp)
        else:
            for bit in (1, 2, 4, 8, 16, 32):
                if flags & bit: self.reasons[str(bit)] += 1
        self.was_valid = valid
        if probe is not None: probe.end('screen_segment', mark)
        if not emit_row:
            # Warm-start prefix: keep every order/control and screening/segment
            # transition, but avoid materializing observations nobody consumes.
            self.count += 1
            return None
        mark = probe.start() if probe is not None else None
        row = np.zeros((), dtype=BOOK_DTYPE)
        row['source_sequence'], row['timestamp_ns'] = sequence, timestamp
        row['segment_id'], row['flags'] = self.segment if valid else 0, flags
        row['bid_levels'], row['ask_levels'] = len(self.book.bids), len(self.book.asks)
        row['source_offset'], row['features'] = offset, features
        self.count += 1
        if probe is not None: probe.end('row_materialize', mark)
        return row


def record_from_book_row(row, index):
    from .stream import Record
    segment = int(row['segment_id'])
    return Record(row['features'].copy(), {
        'date': '2019-07-30', 'symbol': 'AAPL', 'snippet_id': 'AAPL_20190730_native_full_day',
        'research_segment_id': segment, 'native_segment_id': segment,
        'sample_row_0based': int(index), 'sample_order_1based': int(index)+1,
        'source_record_row_0based': int(index),
        'source_message_position_1based': int(row['source_sequence']),
        'source_uncompressed_byte_offset': int(row['source_offset']),
        'timestamp_ns_since_exchange_midnight': int(row['timestamp_ns']),
        'source_flags': int(row['flags']), 'research_reject_mask': int(row['flags']),
        'bid_levels': int(row['bid_levels']), 'ask_levels': int(row['ask_levels']),
        'source_profile': PROFILE, 'dataset_role': 'cross_period_engineering_validation'})


def rebuild(dataset=DATASET):
    dataset = Path(dataset); extraction = read_json(dataset/'extracted/manifest.json')
    out = dataset/'prepared'; out.mkdir(parents=True,exist_ok=True)
    target = out/'book_records.bin'
    if (out/'manifest.json').exists():
        old = read_json(out/'manifest.json')
        if old.get('status') == 'passed' and old.get('profile') == PROFILE and sha256(target) == old['files']['book_records.bin']['sha256']:
            return old
        raise ValueError('Existing prepared output is not verified')
    processor = NativeBookProcessor()
    count = 0
    start = progress = time.perf_counter()
    try:
        with (out/'book_records.bin.partial').open('wb') as output:
            for sequence, offset, payload in selected_frames(dataset):
                row = processor.push(sequence, offset, payload)
                if row is None: continue
                output.write(row.tobytes())
                count = processor.count
                if time.perf_counter()-progress>=5:
                    print(f'REBUILD mutations={count} eligible={processor.accepted} live_orders={len(processor.book.orders)}',flush=True)
                    progress=time.perf_counter()
        (out/'book_records.bin.partial').replace(target)
        segments = processor.segments
        window_count = sum(max(0,s['count']-99) for s in segments)
        if window_count == 0: raise ValueError('No complete eligible window')
        records=np.memmap(target,mode='r',dtype=BOOK_DTYPE)
        labels=np.full(len(records),-1,dtype='i1')
        endpoints=np.arange(99,max(99,len(records)-50))
        ids=records['segment_id']
        if len(endpoints):
            good=(ids[endpoints]>0)&(ids[endpoints]==ids[endpoints-99])&(ids[endpoints]==ids[endpoints+50])
            e=endpoints[good]
            mids=records['features'][:,0]+records['features'][:,2]
            labels[e]=np.where(mids[e+50]>mids[e],2,np.where(mids[e+50]<mids[e],0,1))
        labels.tofile(out/'labels.i8.bin')
        report={'status':'passed','profile':PROFILE,'date':'2019-07-30','symbol':'AAPL',
                'role':'cross_period_engineering_validation','record_count':count,'eligible_count':processor.accepted,
                'window_count':window_count,'labeled_windows':int(np.sum(labels>=0)),
                'segments':segments,'reject_reason_counts':dict(processor.reasons),'control_events':processor.controls,
                'unknown_order_references':0,'duplicate_live_ids':0,'oversized_reductions':0,
                'full_extracted_stream_consumed':True,'gzip_full_crc_verified':extraction['gzip_full_crc_verified'],
                'source_sha256':SOURCE_SHA,'extraction_manifest_sha256':sha256(dataset/'extracted/manifest.json'),
                'record_dtype':BOOK_DTYPE.descr,'record_bytes':BOOK_DTYPE.itemsize,
                'feature_order_per_level':['ask_price','ask_size','bid_price','bid_size'],
                'price_divisor_for_display':10000,'quantity_unit':'shares','normalization_refitted':False,
                'screen_local_exchange_time':'09:35:00 inclusive to 15:55:00 exclusive; EDT UTC-04:00',
                'profile_note':'Native displayed Nasdaq book; not a Databento MBO adapter or same-source match to 2025 delivery.',
                'operational_halt_policy':'Any native Nasdaq h/H or MWCB W quarantines remainder; h absence is interpreted by native incremental protocol, not by provider status silence.',
                'elapsed_seconds':time.perf_counter()-start,
                'files':{n:{'bytes':(out/n).stat().st_size,'sha256':sha256(out/n)} for n in ('book_records.bin','labels.i8.bin')}}
        write_json(out/'manifest.json',report)
        return report
    except BaseException as exc:
        write_json(out/'rebuild_failure.json',{'status':'failed','error':str(exc),'mutation_rows':count})
        raise
