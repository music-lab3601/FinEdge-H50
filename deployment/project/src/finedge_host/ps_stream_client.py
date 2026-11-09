"""Serial PST1 host wrapper with independent bounded tensor/identity checking.

No connection creation, reconnect, retry, SDK import or background worker.
Pause by making no submit calls; the same authenticated connection remains in
use. FINISH is terminal. Only one 100-record shadow and the current bounded
batch survive; callers stream accepted results to their own journal/UI.
"""
from collections import deque
import hashlib
import struct

from .ps_stream_protocol import ClientCodec, ProtocolError
from .ps_stream_server import DEFAULT_WINDOW_BUDGET, advance_digest, validate_prediction_row
from .ps_window_runtime import selected_endpoints


def _require(condition, message):
    if not condition:
        raise ProtocolError(message)


class StreamClient:
    def __init__(self, read, write, *, token, window_budget=DEFAULT_WINDOW_BUDGET,
                 expected_boot_sha256=None, expected_boot_id=None, expected_identity=None):
        self.read, self.write = read,write
        self._token = token
        self.codec = ClientCodec()
        self.window_budget = window_budget
        self.expected_boot_sha256 = expected_boot_sha256
        self.expected_boot_id = expected_boot_id
        self.expected_identity = {} if expected_identity is None else dict(expected_identity)
        self.history = deque(maxlen=100)
        self.records = self.windows = self.batches = 0
        self.digest = hashlib.sha256()
        self.state = 'NEW'
        self.failure = None
        self.retained = {}
        self.hello_data = None

    def _fail(self, exc):
        self.state = 'FAILED'
        self.codec.state = 'FAILED'
        self._token = None
        if self.failure is None:
            self.failure = exc

    def hello(self):
        try:
            _require(self.state == 'NEW', 'HELLO is single-use')
            self.codec.send_hello(self.write,self._token,self.window_budget)
            self._token = None
            reply = self.codec.receive_response(self.read)
            self.retained['response'] = reply
            data = reply['data']
            _require(type(data.get('boot_id')) is str and data['boot_id']
                     and type(data.get('firmware_sha256')) is str and len(data['firmware_sha256']) == 64
                     and all(c in '0123456789abcdef' for c in data['firmware_sha256']),
                     'Missing board boot identity')
            if self.expected_boot_sha256 is not None:
                _require(data['firmware_sha256'] == self.expected_boot_sha256, 'Unexpected board firmware')
            if self.expected_boot_id is not None:
                _require(data['boot_id'] == self.expected_boot_id, 'Unexpected board boot session')
            _require(type(data.get('identity')) is dict and all(data['identity'].get(k) == v for k,v in self.expected_identity.items()),
                     'Unexpected frozen service identity')
            _require(data.get('window_budget') == self.window_budget
                     and type(data.get('initial_npu_calls')) is int and data['initial_npu_calls'] >= 0
                     and type(data.get('initial_completed_invocations')) is int and data['initial_completed_invocations'] >= 0,
                     'Unexpected session budget or initial NPU count')
            self.hello_data = data
            self._runtime(data,0)
            self.state = 'READY'
            self.retained = {}
            return reply
        except BaseException as exc:
            self._fail(exc)
            raise

    def _runtime(self, data, windows):
        snapshot = data.get('runtime_snapshot')
        _require(type(snapshot) is dict and snapshot.get('initialized') is True
                 and snapshot.get('blocked') is False and type(snapshot.get('active_refs')) is int
                 and snapshot['active_refs'] == 0 and snapshot.get('per_window_history') == 0
                 and type(snapshot.get('NPU_calls')) is int
                 and snapshot['NPU_calls'] == self.hello_data['initial_npu_calls']+windows
                 and type(snapshot.get('completed_invocations')) is int
                 and snapshot['completed_invocations'] == self.hello_data['initial_completed_invocations']+windows,
                 'Runtime owner snapshot contradicts completed windows')
        if 'path_profile' in self.expected_identity:
            measurement=snapshot.get('path_measurement')
            _require(type(measurement) is dict
                     and measurement.get('profile')==self.expected_identity['path_profile'],
                     'Runtime path profile differs from frozen service identity')

    def _expected(self, payload):
        endpoints, _ = selected_endpoints(payload,[identity for identity,_ in self.history])
        history = deque(self.history,maxlen=100)
        expected = []
        for offset in range(0,len(payload),256):
            identity = struct.unpack_from('<4Q',payload,offset)
            emit, = struct.unpack_from('<I',payload,offset+32)
            if history and history[-1][0][0] != identity[0]:
                history.clear()
            history.append((identity,payload[offset+64:offset+224]))
            if len(history) == 100 and emit:
                digest = hashlib.sha256()
                # Reorder opaque 32-bit words, never convert floating values.
                # Forty C-level strided copies avoid 4000 Python hash calls.
                words = memoryview(b''.join(values for _,values in history)).cast('I')
                for feature in range(40):
                    digest.update(words[feature::40].tobytes())
                expected.append(dict(window_id=self.windows+len(expected)+1,start=list(history[0][0]),
                                     end=list(identity),tensor_sha256=digest.hexdigest()))
        _require([(tuple(x['start']),tuple(x['end'])) for x in expected] == endpoints,'Host shadow identity mismatch')
        return expected,history

    def submit(self, payload, expected_windows=None):
        try:
            _require(self.state == 'READY', 'Stream session is not ready')
            expected,history = self._expected(payload)
            self.retained = dict(payload=payload,expected=expected)
            if expected_windows is not None:
                _require(type(expected_windows) is list and len(expected_windows) == len(expected), 'Reference window count mismatch')
                for actual,reference in zip(expected,expected_windows):
                    _require(type(reference) is dict and all(reference.get(k) == v for k,v in actual.items()), 'Reference window identity/hash mismatch')
            self.codec.send_batch(self.write,payload,len(expected))
            reply = self.codec.receive_response(self.read)
            self.retained['response'] = reply
            rows = reply['windows']
            _require(len(rows) == len(expected), 'Missing or extra window results')
            for row,reference in zip(rows,expected):
                validate_prediction_row(row)
                _require(all(row[k] == v for k,v in reference.items()), 'Wrong window order, identity or tensor hash')
            data = reply['data']
            records = len(payload)//256
            batch_id = self.batches+1
            _require(all(type(data.get(k)) is int for k in ('batch_id','records','generation','total_records','total_batches','live_npu_calls','live_completed_invocations'))
                     and data['batch_id'] == batch_id
                     and type(data.get('generation')) is int and data['generation'] == batch_id
                     and data.get('records') == records and data.get('total_records') == self.records+records
                     and data.get('total_batches') == batch_id, 'Batch identity/count/generation mismatch')
            _require(type(data.get('board_batch_ns')) is int and data['board_batch_ns'] >= 0
                     and data.get('live_npu_calls') == self.windows+len(expected)
                     and data.get('live_completed_invocations') == self.windows+len(expected), 'Inference accounting mismatch')
            self._runtime(data,self.windows+len(expected))
            digest = self.digest.copy()
            advance_digest(digest,rows)
            _require(data.get('rolling_output_sha256') == digest.hexdigest(), 'Rolling output digest mismatch')
            self.history = history
            self.records += records
            self.windows += len(expected)
            self.batches = batch_id
            self.digest = digest
            self.retained = {}
            return reply
        except BaseException as exc:
            self._fail(exc)
            raise

    def finish(self):
        try:
            _require(self.state == 'READY', 'Stream session is not ready')
            self.codec.send_finish(self.write)
            reply = self.codec.receive_response(self.read)
            self.retained['response'] = reply
            data = reply['data']
            _require(all(type(data.get(k)) is int for k in ('records','windows','batches','live_npu_calls','live_completed_invocations'))
                     and data.get('records') == self.records and data.get('windows') == self.windows
                     and data.get('batches') == self.batches and data.get('live_npu_calls') == self.windows
                     and data.get('live_completed_invocations') == self.windows
                     and data.get('rolling_output_sha256') == self.digest.hexdigest()
                     and data.get('owner_retained') is True and data.get('controller_state') == 'FREE',
                     'FINISH totals/ownership mismatch')
            self._runtime(data,self.windows)
            self.state = 'FINISHED'
            self.retained = {}
            return reply
        except BaseException as exc:
            self._fail(exc)
            raise
