"""Bounded serial PSW streaming framing; standard library, Python 3.8 only.

This module never opens a socket, imports an SDK, retries a request, reconnects,
logs a token, or retains a complete trading day. Callbacks are blocking
``recv/read(n) -> bytes`` and ``send/write(bytes) -> count``; callers set their
own transport deadlines. Fragmented reads/writes are assembled exactly. EOF,
zero progress, timeout, malformed data and sequence errors are terminal.

Request header (32 bytes, little endian): ``<4sBBHQQII`` = magic, version,
kind, reserved, sequence, window budget, payload bytes, reserved. HELLO sequence
0 supplies the entire session's positive window ceiling and a JSON token.
BATCH carries 1..32 opaque 256-byte WDM records and a 0..record-count ceiling.
FINISH carries no payload and budget 0. Responses have a uint32 little-endian
length followed by at most 256 KiB of strict UTF-8 JSON. A BATCH result window
has ``start`` and ``end`` identities [segment,row,source_id,timestamp_ns].

ClientCodec/ServerCodec are deliberately single-threaded. Exactly one request
is pending at a time. On any error discard the codec and connection; the caller
must separately retain/quarantine hardware ownership. Protocol success does
not itself prove hardware visibility, inference correctness, or full-day EOF.
"""
from dataclasses import dataclass, field
from enum import IntEnum
import hmac
import json
import math
import re
import struct

MAGIC = b'PST1'
VERSION = 1
HEADER = struct.Struct('<4sBBHQQII')
LENGTH = struct.Struct('<I')
MAX_BATCH_BYTES = 8192
RECORD_BYTES = 256
MAX_BATCH_WINDOWS = 32
MAX_RESPONSE_BYTES = 256 * 1024
MAX_HELLO_BYTES = 1024
MAX_WINDOW_BUDGET = (1 << 64) - 1
MAX_SEQUENCE = (1 << 64) - 1


class ProtocolError(ValueError):
    """Terminal framing, ordering or resource-limit failure."""


class AuthenticationError(ProtocolError):
    """HELLO authentication failed; exception never contains the token."""


class RemoteError(ProtocolError):
    def __init__(self, code):
        self.code = code
        super().__init__('Remote request failed: ' + code)


class Kind(IntEnum):
    HELLO = 1
    BATCH = 2
    FINISH = 3


@dataclass(frozen=True)
class Request:
    kind: Kind
    seq: int
    window_budget: int
    payload: bytes = field(repr=False)


def _require(condition, message):
    if not condition:
        raise ProtocolError(message)


def _uint(value, maximum=MAX_SEQUENCE):
    return type(value) is int and 0 <= value <= maximum


def read_exact(read, size):
    """Consume exactly size bytes; no retry of failed transport operations."""
    _require(_uint(size, MAX_RESPONSE_BYTES), 'Invalid bounded read size')
    chunks = []
    remaining = size
    while remaining:
        part = read(remaining)  # transport exceptions propagate; no retry
        _require(isinstance(part, bytes), 'Reader must return owned bytes')
        _require(0 < len(part) <= remaining, 'EOF, truncated frame or oversized read')
        chunks.append(part)
        remaining -= len(part)
    return b''.join(chunks)


def write_exact(write, data):
    """A short successful write is progress, never a request retransmission."""
    _require(isinstance(data, bytes) and len(data) <= MAX_RESPONSE_BYTES + 4,
             'Invalid bounded write payload')
    offset = 0
    view = memoryview(data)
    while offset < len(data):
        count = write(view[offset:])
        _require(type(count) is int and 0 < count <= len(data) - offset,
                 'Writer made invalid or zero progress')
        offset += count


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate JSON member')
        result[key] = value
    return result


def _constant(value):
    raise ProtocolError('Nonfinite JSON number')


def _float(value):
    number = float(value)
    _require(math.isfinite(number), 'Nonfinite JSON number')
    return number


def _load_json(data):
    try:
        result = json.loads(data.decode('utf-8'), object_pairs_hook=_pairs,
                            parse_constant=_constant, parse_float=_float)
    except (ValueError, UnicodeError, RecursionError):
        raise ProtocolError('Invalid strict JSON object') from None
    _require(type(result) is dict, 'JSON object required')
    return result


def _dump_json(value, limit):
    try:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':')).encode('utf-8')
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ProtocolError('Response must contain finite JSON values') from None
    _require(0 < len(data) <= limit, 'JSON frame exceeds bounded size')
    return data


def _token_bytes(token):
    if type(token) is not str:
        raise AuthenticationError('Invalid authentication token')
    try:
        encoded = token.encode('utf-8')
    except UnicodeError:
        raise AuthenticationError('Invalid authentication token') from None
    if not 16 <= len(encoded) <= 256:
        raise AuthenticationError('Invalid authentication token')
    return encoded


def _header_fields(kind, seq, budget, size):
    _require(type(kind) is Kind, 'Unknown message kind')
    _require(_uint(seq) and _uint(budget) and _uint(size, MAX_BATCH_BYTES), 'Invalid header values')
    if kind == Kind.HELLO:
        _require(seq == 0 and 0 < budget <= MAX_WINDOW_BUDGET and 0 < size <= MAX_HELLO_BYTES,
                 'Invalid HELLO header')
    elif kind == Kind.BATCH:
        _require(seq > 0 and 0 < size <= MAX_BATCH_BYTES and size % RECORD_BYTES == 0
                 and budget <= size // RECORD_BYTES, 'Invalid BATCH header')
    else:
        _require(seq > 0 and budget == 0 and size == 0, 'Invalid FINISH header')


def encode_request(request):
    _require(type(request) is Request and isinstance(request.payload, bytes), 'Owned request payload required')
    _header_fields(request.kind, request.seq, request.window_budget, len(request.payload))
    if request.kind == Kind.HELLO:
        hello = _load_json(request.payload)
        _require(set(hello) == {'token'}, 'HELLO contains unexpected fields')
        _token_bytes(hello['token'])
    return HEADER.pack(MAGIC, VERSION, int(request.kind), 0, request.seq,
                       request.window_budget, len(request.payload), 0) + request.payload


def read_request(read, expected_seq):
    _require(_uint(expected_seq), 'Invalid expected sequence')
    magic, version, kind, flags, seq, budget, size, reserved = HEADER.unpack(read_exact(read, HEADER.size))
    _require(magic == MAGIC and version == VERSION and flags == reserved == 0, 'Invalid protocol header')
    _require(seq == expected_seq, 'Unexpected request sequence')
    try:
        kind = Kind(kind)
    except ValueError:
        raise ProtocolError('Unknown message kind') from None
    _header_fields(kind, seq, budget, size)  # reject length before reading body
    return Request(kind, seq, budget, read_exact(read, size))


def _identity(value):
    return type(value) is list and len(value) == 4 and all(_uint(v) for v in value)


def validate_windows(windows, budget):
    _require(type(windows) is list and _uint(budget, MAX_BATCH_WINDOWS) and len(windows) <= budget,
             'Window count exceeds request budget')
    for window in windows:
        _require(type(window) is dict and _identity(window.get('start')) and _identity(window.get('end')),
                 'Window needs four-part segment endpoint identities')
        start, end = window['start'], window['end']
        _require(start[0] == end[0] and end[1] - start[1] == 99 and end[2] > start[2]
                 and end[3] >= start[3], 'Invalid or cross-segment 100-record window')


def encode_response(response):
    _require(type(response) is dict, 'Response object required')
    payload = _dump_json(response, MAX_RESPONSE_BYTES)
    return LENGTH.pack(len(payload)) + payload


def read_response(read):
    size, = LENGTH.unpack(read_exact(read, LENGTH.size))
    _require(0 < size <= MAX_RESPONSE_BYTES, 'Response exceeds bounded size')
    return _load_json(read_exact(read, size))


def _validate_response(response, request, completed):
    required = {'version','kind','seq','ok','total_windows','windows','data'}
    _require(set(response) in (required, required | {'error_code'}), 'Unexpected response envelope')
    _require(type(response['version']) is int and response['version'] == VERSION
             and response['kind'] == request.kind.name and _uint(response['seq'])
             and response['seq'] == request.seq and type(response['ok']) is bool,
             'Wrong response version, kind or sequence')
    _require(type(response['data']) is dict and _uint(response['total_windows']), 'Invalid response summary')
    budget = request.window_budget if request.kind == Kind.BATCH and response['ok'] else 0
    validate_windows(response['windows'], budget)
    _require(response['total_windows'] == completed + len(response['windows']), 'Wrong cumulative window count')
    if response['ok']:
        _require('error_code' not in response, 'Successful response contains error')
    else:
        code = response.get('error_code')
        _require(type(code) is str and re.fullmatch(r'[A-Z][A-Z0-9_]{0,63}', code), 'Invalid stable error code')
    return len(response['windows'])


class ClientCodec:
    def __init__(self):
        self.state = 'NEW'
        self.sequence = 0
        self.total_windows = 0
        self.window_budget = 0
        self._pending = None

    def _send(self, write, request):
        try:
            expected_state = 'NEW' if request.kind == Kind.HELLO else 'READY'
            _require(self.state == expected_state and self._pending is None, 'Only one serial request may be pending')
            if request.kind == Kind.BATCH:
                _require(request.window_budget <= self.window_budget - self.total_windows, 'Session window budget exceeded')
            frame = encode_request(request)
            self.state = 'WAITING'
            # Never retain a HELLO token in session metadata or its repr.
            self._pending = Request(request.kind,request.seq,request.window_budget,b'')
            write_exact(write,frame)
        except BaseException:
            self.state = 'FAILED'
            raise

    def send_hello(self, write, token, window_budget):
        try:
            _token_bytes(token)
            payload = _dump_json({'token':token},MAX_HELLO_BYTES)
            self._send(write,Request(Kind.HELLO,0,window_budget,payload))
        except BaseException:
            self.state = 'FAILED'
            raise

    def send_batch(self, write, payload, window_budget):
        self._send(write,Request(Kind.BATCH,self.sequence,window_budget,payload))

    def send_finish(self, write):
        self._send(write,Request(Kind.FINISH,self.sequence,0,b''))

    def receive_response(self, read):
        try:
            _require(self.state == 'WAITING' and self._pending is not None, 'No pending response')
            request = self._pending
            response = read_response(read)
            count = _validate_response(response,request,self.total_windows)
            if not response['ok']:
                raise RemoteError(response['error_code'])
            if request.kind == Kind.HELLO:
                self.window_budget = request.window_budget
            self.total_windows += count
            _require(self.total_windows <= self.window_budget, 'Session window budget exceeded')
            self.sequence = request.seq + 1
            self._pending = None
            self.state = 'FINISHED' if request.kind == Kind.FINISH else 'READY'
            return response
        except BaseException:
            self.state = 'FAILED'
            raise


class ServerCodec:
    def __init__(self, token, max_window_budget=MAX_WINDOW_BUDGET):
        self._token = _token_bytes(token)
        _require(_uint(max_window_budget) and max_window_budget > 0, 'Invalid server window limit')
        self.max_window_budget = max_window_budget
        self.state = 'NEW'
        self.sequence = 0
        self.total_windows = 0
        self.window_budget = 0
        self._pending = None

    def receive_request(self, read):
        try:
            _require(self.state in ('NEW','READY') and self._pending is None, 'Response required before next request')
            request = read_request(read,self.sequence)
            if self.state == 'NEW':
                _require(request.kind == Kind.HELLO, 'HELLO required before BATCH')
                hello = _load_json(request.payload)
                _require(set(hello) == {'token'}, 'HELLO contains unexpected fields')
                if not hmac.compare_digest(_token_bytes(hello['token']),self._token):
                    raise AuthenticationError('HELLO authentication failed')
                _require(request.window_budget <= self.max_window_budget, 'Requested session exceeds server window limit')
                self.window_budget = request.window_budget
                self._token = None
                request = Request(request.kind,request.seq,request.window_budget,b'')
            else:
                _require(request.kind != Kind.HELLO, 'Repeated HELLO is prohibited')
                if request.kind == Kind.BATCH:
                    _require(request.window_budget <= self.window_budget - self.total_windows, 'Session window budget exceeded')
            self._pending = request
            self.state = 'PENDING'
            return request
        except BaseException:
            self._token = None
            self.state = 'FAILED'
            raise

    def respond(self, write, windows=None, data=None, error_code=None):
        try:
            _require(self.state == 'PENDING' and self._pending is not None, 'No request to respond to')
            request = self._pending
            windows = [] if windows is None else windows
            _require(type(windows) is list, 'Window list required')
            response = dict(version=VERSION,kind=request.kind.name,seq=request.seq,ok=error_code is None,
                            total_windows=self.total_windows+len(windows),windows=windows,data={} if data is None else data)
            if error_code is not None:
                response['error_code'] = error_code
            count = _validate_response(response,request,self.total_windows)
            _require(self.total_windows + count <= self.window_budget, 'Session window budget exceeded')
            write_exact(write,encode_response(response))
            self.total_windows += count
            self.sequence = request.seq + 1
            self._pending = None
            self.state = 'FAILED' if error_code is not None else 'FINISHED' if request.kind == Kind.FINISH else 'READY'
            return response
        except BaseException:
            self.state = 'FAILED'
            raise
