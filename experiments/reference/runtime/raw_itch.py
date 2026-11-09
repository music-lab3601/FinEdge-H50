"""Raw file event source: reconstruct during iteration, never open prepared data."""
import gzip
from pathlib import Path
import struct
import time

from software.runtime.delivery import read_json, sha256
from experiments.reference.runtime.itch import DATASET, GLOBAL, LENGTHS, MUTATIONS, SOURCE_SHA, PROFILE, NativeBookProcessor, record_from_book_row


class ItchFramer:
    """Length-prefixed ITCH framing independent of read/chunk boundaries."""
    def __init__(self):
        self.carry = b''
        self.offset = 0

    def feed(self, chunk):
        data = self.carry + chunk
        p = 0
        while p+2 <= len(data):
            length = int.from_bytes(data[p:p+2], 'big')
            if not 11 <= length <= 2048:
                raise ValueError(f'Invalid ITCH frame length at byte {self.offset+p}')
            end = p+2+length
            if end > len(data):
                break
            payload = data[p+2:end]
            if LENGTHS.get(chr(payload[0])) != length:
                raise ValueError(f'Unknown ITCH type/length at byte {self.offset+p}')
            yield self.offset+p, payload
            p = end
        self.offset += p
        self.carry = data[p:]

    def finish(self):
        if self.carry:
            raise ValueError(f'Truncated final ITCH frame at byte {self.offset}')


def framed_stream(stream, chunk_size=1024*1024, cancelled=lambda: False):
    framer = ItchFramer()
    while not cancelled():
        chunk = stream.read(chunk_size)
        if not chunk:
            framer.finish()
            return
        for i, frame in enumerate(framer.feed(chunk)):
            if i % 1024 == 0 and cancelled():
                return
            yield frame


class RawItchSource:
    """Re-iterable fresh book. Extracted and gzip routes share the same engine.

    File backpressure is intentional; this is not a live exchange receiver.
    No __len__: eligible count is discovered by processing events, not by
    reading a future/prepared book manifest.
    """
    def __init__(self, dataset=DATASET, full_gzip=False, chunk_size=1024*1024,
                 before_event=None, measure_stages=False, feature_start_ns=None):
        self.dataset = Path(dataset)
        self.full_gzip = full_gzip
        self.chunk_size = chunk_size
        # Optional event-time clock. Called before book mutation, including
        # controls and rejected rows; an inference slowdown must not retime arrivals.
        self.before_event = before_event
        self.measure_stages = measure_stages
        self.feature_start_ns = feature_start_ns
        self.cancelled = lambda: False
        self.on_progress = lambda metrics: None
        self.metrics = {}
        if full_gzip:
            self.path = self.dataset/'raw'
            self.manifest_path = self.path/'admission.json'
            self.manifest = read_json(self.manifest_path)
            raw = self.path/'07302019.NASDAQ_ITCH50.gz'
            if self.manifest['sha256'] != SOURCE_SHA or sha256(raw) != SOURCE_SHA:
                raise ValueError('Raw gzip identity mismatch')
            self.provenance = {'mode': 'full_market_gzip', 'source_sha256': SOURCE_SHA,
                               'raw_bytes': raw.stat().st_size}
        else:
            self.path = self.dataset/'extracted'
            self.manifest_path = self.path/'manifest.json'
            self.manifest = read_json(self.manifest_path)
            m = self.manifest
            if (m.get('status') != 'passed' or m.get('source_sha256') != SOURCE_SHA
                    or not m.get('gzip_full_crc_verified') or m.get('book_reconstructed') is not False):
                raise ValueError('Raw selected-event identity is not admitted')
            for name in ('aapl.itch', 'positions.u64le.bin'):
                ref = m['files'][name]
                p = self.path/name
                if p.stat().st_size != ref['bytes'] or sha256(p) != ref['sha256']:
                    raise ValueError('Raw selected-event integrity failure: '+name)
            if (self.path/'positions.u64le.bin').stat().st_size != m['selected_messages']*16:
                raise ValueError('Raw event identity count mismatch')
            self.provenance = {'mode': 'selected_original_frames', 'source_sha256': SOURCE_SHA,
                               'selected_messages': m['selected_messages'], 'files': m['files']}
        self.provenance.update(profile=PROFILE, prepared_book_read=False,
                               prepared_labels_read=False, state_initialization='system_O_then_full_prefix')
        self.provenance['feature_materialization_start_ns'] = feature_start_ns

    def _frames(self, cancelled):
        if self.full_gzip:
            with gzip.open(self.path/'07302019.NASDAQ_ITCH50.gz', 'rb') as stream:
                for sequence, (offset, payload) in enumerate(framed_stream(stream, self.chunk_size, cancelled), 1):
                    yield sequence, offset, payload
        else:
            with (self.path/'aapl.itch').open('rb') as stream, (self.path/'positions.u64le.bin').open('rb') as positions:
                for _, payload in framed_stream(stream, self.chunk_size, cancelled):
                    identity = positions.read(16)
                    if len(identity) != 16:
                        raise ValueError('Missing original frame identity')
                    sequence, offset = struct.unpack('<QQ', identity)
                    yield sequence, offset, payload
                if not cancelled() and positions.read(1):
                    raise ValueError('Extra frame identities without raw events')

    def iter_rows(self, cancel=lambda: False):
        engine = NativeBookProcessor()
        m = {'frames_read': 0, 'selected_events': 0, 'mutation_rows': 0,
             'eligible_records': 0, 'last_source_sequence': None,
             'last_timestamp_ns': None, 'completed_full_stream': False,
             'stream_mode': self.provenance['mode']}
        if self.measure_stages:
            m.update(book_parse_screen_ns=0, book_parse_screen_count=0, book_parse_screen_max_ns=0)
        self.metrics = m
        cancelled = lambda: self.cancelled() or cancel()
        locate = None
        first = last_system = None
        last_progress = 0.
        try:
            for sequence, offset, payload in self._frames(cancelled):
                m['frames_read'] += 1
                kind = chr(payload[0])
                stock_locate = int.from_bytes(payload[1:3], 'big')
                if first is None:
                    first = payload
                    if kind != 'S' or chr(payload[11]) != 'O':
                        raise ValueError('Full system O prefix required; cannot start with an empty intraday book')
                if kind == 'R':
                    symbol = payload[11:19].decode('ascii').strip()
                    if symbol == 'AAPL':
                        if locate is not None and locate != stock_locate:
                            raise ValueError('AAPL locate mapping changed')
                        locate = stock_locate
                    elif stock_locate == locate:
                        raise ValueError('AAPL locate reused by another symbol')
                selected = kind in GLOBAL or (locate is not None and stock_locate == locate)
                if not selected:
                    if not self.full_gzip:
                        raise ValueError('Unexpected symbol or missing directory in selected AAPL events')
                else:
                    if self.before_event is not None:
                        self.before_event(sequence, offset, payload)
                    if kind in MUTATIONS and locate is None:
                        raise ValueError('Order event before AAPL security mapping')
                    m['selected_events'] += 1
                    m['last_source_sequence'] = sequence
                    m['last_timestamp_ns'] = int.from_bytes(payload[5:11], 'big')
                    if kind == 'S': last_system = chr(payload[11])
                    began = time.perf_counter_ns() if self.measure_stages else 0
                    row = engine.push(sequence, offset, payload, emit_row=(self.feature_start_ns is None or
                                      m['last_timestamp_ns'] >= self.feature_start_ns))
                    if self.measure_stages:
                        cost = time.perf_counter_ns()-began
                        m['book_parse_screen_ns'] += cost
                        m['book_parse_screen_count'] += 1
                        m['book_parse_screen_max_ns'] = max(m['book_parse_screen_max_ns'], cost)
                    m['last_processed_source_sequence'] = sequence
                    m['last_processed_timestamp_ns'] = m['last_timestamp_ns']
                    m['mutation_rows'], m['eligible_records'] = engine.count, engine.accepted
                    if row is not None:
                        if int(row['flags']) & 32:
                            raise ValueError('Raw mutation timestamp regressed; inference stream stopped')
                        yield engine.count-1, row
                if time.perf_counter()-last_progress >= .5:
                    self.on_progress(dict(m))
                    last_progress = time.perf_counter()
            if cancelled():
                return
            if locate is None or last_system != 'C':
                raise ValueError('Incomplete raw session: missing AAPL directory or final system C')
            if not self.full_gzip and m['selected_events'] != self.manifest['selected_messages']:
                raise ValueError('Selected raw frame count differs from manifest')
            m['completed_full_stream'] = True
        except BaseException as exc:
            if not isinstance(exc, GeneratorExit):
                m['error'] = str(exc)
            raise
        finally:
            m['segments'] = [dict(s) for s in engine.segments]
            m['reject_reason_counts'] = dict(engine.reasons)
            m['live_orders'] = len(engine.book.orders)
            self.on_progress(dict(m))

    def iter_records(self, cancel=lambda: False):
        rows = self.iter_rows(cancel)
        try:
            for index, row in rows:
                if int(row['flags']) == 0:
                    yield record_from_book_row(row, index)
        finally:
            rows.close()

    def __iter__(self):
        return self.iter_records()
