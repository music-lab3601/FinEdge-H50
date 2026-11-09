"""One serial connection over initialized PSW/SDK owners; no opening or freeing.

The service entry retains this session, owner and controller after serve ends
and parks the process. All controller B-response, full-arena readbacks, guards,
tensor validation and post-consumer REARM remain in PsWindowRuntime.submit.
Only successful diagnostics are aggregated; the first failure keeps traceback
and current batch references. No per-API file is opened here.
"""
import hashlib
import json
import math
import time

from .ps_stream_protocol import Kind, ProtocolError, ServerCodec, validate_windows
from .ps_window_runtime import PsWindowRuntime, selected_endpoints

DEFAULT_WINDOW_BUDGET = 1126606


def _require(condition, message):
    if not condition:
        raise ProtocolError(message)


def _integer(value):
    return type(value) is int and value >= 0


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_prediction_row(row):
    required = {'window_id','start','end','tensor_sha256','logits','class_index','backend','stages_ns','whole_predict_ns'}
    _require(type(row) is dict and set(row) == required, 'Unexpected prediction fields')
    validate_windows([row],1)
    _require(_integer(row['window_id']) and row['window_id'] > 0 and _sha(row['tensor_sha256']), 'Invalid window identity/hash')
    logits = row['logits']
    _require(type(logits) is list and len(logits) == 3
             and all(type(v) in (int,float) and math.isfinite(v) for v in logits), 'Three finite logits required')
    _require(type(row['class_index']) is int and row['class_index'] == max(range(3),key=logits.__getitem__), 'Class differs from raw argmax')
    _require(row['backend'] == 'board_socket_npu', 'Actual board NPU backend required')
    stages = row['stages_ns']
    _require(type(stages) is dict and 0 < len(stages) <= 32
             and all(type(k) is str and 0 < len(k) <= 96 and _integer(v) for k,v in stages.items())
             and _integer(row['whole_predict_ns']), 'Invalid bounded inference timings')


def advance_digest(digest, windows):
    """Canonical JSON-lines digest; state is fixed size, no result history."""
    for row in windows:
        digest.update(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')+b'\n')


class BoundedDiagnostics:
    def __init__(self, max_stages=128):
        _require(type(max_stages) is int and 1 <= max_stages <= 256, 'Invalid diagnostic stage bound')
        self.max_stages = max_stages
        self.stages = {}
        self.api_events = self.predictions = self.other_events = 0
        self.other_api = dict(count=0,total_ns=0,max_ns=0,min_ns=None)

    def __call__(self, event):
        if type(event) is not dict:
            self.other_events += 1
            return
        if event.get('kind') == 'prediction_complete':
            self.predictions += 1
            return
        if event.get('kind') != 'api' or not _integer(event.get('ns')):
            self.other_events += 1
            return
        self.api_events += 1
        name = event.get('stage')
        if type(name) is str and 0 < len(name) <= 128 and (name in self.stages or len(self.stages) < self.max_stages):
            stat = self.stages.setdefault(name,dict(count=0,total_ns=0,max_ns=0,min_ns=None))
        else:
            stat = self.other_api
        ns = event['ns']
        stat['count'] += 1
        stat['total_ns'] += ns
        stat['max_ns'] = max(stat['max_ns'],ns)
        stat['min_ns'] = ns if stat['min_ns'] is None else min(stat['min_ns'],ns)

    def snapshot(self):
        return dict(api_events=self.api_events,predictions=self.predictions,other_events=self.other_events,
                    stages={key:dict(value) for key,value in self.stages.items()},other_api=dict(self.other_api),
                    retained_events=0,stage_bound=self.max_stages)


class ServerSession:
    def __init__(self, owner, controller, *, token, max_windows=DEFAULT_WINDOW_BUDGET,
                 observe_memory=None, diagnostics=None, clock=time.perf_counter_ns,
                 cancelled=None, identity=None):
        _require(isinstance(controller,PsWindowRuntime) and controller.state == 'FREE'
                 and controller.request_id == controller.generation == 0 and not controller.identities,
                 'A freshly initialized protected PSW controller is required')
        _require(owner.initialized and not owner.blocked and owner.ps_arena is not None
                 and owner.active is None, 'Initialized idle NPU owner and PS arena required')
        self.owner, self.controller = owner,controller  # strong refs also after FINISH/EOF
        self.codec = ServerCodec(token,max_windows)
        self.diagnostics = diagnostics if diagnostics is not None else BoundedDiagnostics()
        _require(isinstance(self.diagnostics,BoundedDiagnostics), 'Bounded diagnostic aggregator required')
        owner.diagnostic = self.diagnostics
        self.clock = clock
        self.observe_memory = observe_memory or (lambda:dict(available=False))
        self.cancelled = cancelled
        self.identity = {} if identity is None else dict(identity)
        _require(len(json.dumps(self.identity,allow_nan=False).encode('utf-8')) <= 8192, 'Identity metadata too large')
        self.initial_npu_calls = owner.npu_calls
        self.initial_completed_invocations = owner.completed_invocations
        self.records = self.windows = self.batches = 0
        self.digest = hashlib.sha256()
        self.state = 'NEW'
        self.current_batch = self.failure = self.error_response_failure = None
        self._cancelled = False
        self.memory_peak_kib = None

    def cancel(self):
        self._cancelled = True
        self.controller.cancel()
        self.owner.cancel()

    def _check_cancel(self):
        if self._cancelled or (self.cancelled is not None and self.cancelled()):
            raise RuntimeError('Stream session cancelled; ownership retained')
        self.owner.stop.checkpoint('ps_stream')

    def _isolate(self, exc):
        self.state = 'FAILED'
        if self.failure is None:
            self.failure = exc
            self.owner.retain_failure(exc)
            self.controller._fail(exc)  # records the original exception; performs no SDK operation

    def _memory(self):
        value = self.observe_memory()
        _require(type(value) is dict and len(json.dumps(value,allow_nan=False).encode('utf-8')) <= 4096,
                 'Bounded process-memory observation required')
        rss = value.get('VmRSS_kib')
        if rss is not None:
            _require(_integer(rss), 'Invalid RSS observation')
            self.memory_peak_kib = rss if self.memory_peak_kib is None else max(self.memory_peak_kib,rss)
        return value

    def summary(self):
        return dict(records=self.records,windows=self.windows,batches=self.batches,
                    initial_npu_calls=self.initial_npu_calls,
                    live_npu_calls=self.owner.npu_calls-self.initial_npu_calls,
                    live_completed_invocations=self.owner.completed_invocations-self.initial_completed_invocations,
                    rolling_output_sha256=self.digest.hexdigest(),runtime_snapshot=self.owner.snapshot(),
                    controller_state=self.controller.state,peak_RSS_kib=self.memory_peak_kib,
                    diagnostics=self.diagnostics.snapshot(),state=self.state,
                    owner_retained=True,strict_visibility_proven=False)

    def _hello(self, write):
        self._check_cancel()
        contract = self.controller.access_contract
        data = dict(boot_id=contract.boot_id,firmware_sha256=contract.firmware_sha256,
                    boot_sha256=contract.firmware_sha256,run_id=self.controller.run_id,
                    versions=dict(self.owner.versions),bindings=[list(x) for x in self.owner.bindings],
                    arena_metadata=dict(self.owner.arena_metadata),initial_npu_calls=self.initial_npu_calls,
                    initial_completed_invocations=self.initial_completed_invocations,
                    identity=self.identity,window_budget=self.codec.window_budget,
                    process_memory=self._memory(),runtime_snapshot=self.owner.snapshot())
        result = self.codec.respond(write,data=data)
        self.state = 'READY'
        return result

    def _batch(self, request, write):
        self._check_cancel()
        expected, _ = selected_endpoints(request.payload,self.controller.identities)
        _require(len(expected) <= request.window_budget, 'Batch window ceiling below required outputs')
        _require(request.seq == self.batches+1, 'Unexpected batch identity')
        started = self.clock()
        rows = []
        self.current_batch = dict(batch_id=request.seq,payload=request.payload,expected=expected,predictions=rows)

        def consume(tensor, metadata):
            self._check_cancel()
            index = len(rows)
            _require(index < len(expected) and metadata['window_index'] == index
                     and (tuple(metadata['first']),tuple(metadata['last'])) == expected[index]
                     and metadata['request_id'] == request.seq and metadata['generation'] == request.seq,
                     'Controller window identity mismatch')
            tensor_hash = hashlib.sha256(tensor).hexdigest()
            _require(metadata['tensor_sha256'] == tensor_hash, 'Controller tensor hash mismatch')
            prediction = self.owner.predict(tensor)
            _require(prediction['tensor_sha256'] == tensor_hash, 'NPU input tensor hash mismatch')
            row = dict(window_id=self.windows+index+1,start=list(expected[index][0]),end=list(expected[index][1]),
                       tensor_sha256=tensor_hash,logits=prediction['logits'],class_index=prediction['class_index'],
                       backend=prediction['backend'],stages_ns=prediction['stages_ns'],whole_predict_ns=prediction['whole_predict_ns'])
            validate_prediction_row(row)  # failure here prevents the controller's REARM
            rows.append(row)
            self._check_cancel()
            return row

        with self.owner.operation('ps_stream.batch'):
            result = self.controller.submit(request.payload,consume)
            _require(result['records'] == len(request.payload)//256 and result['windows'] == len(expected)
                     and len(rows) == len(expected) and result['request_id'] == request.seq
                     and result['generation'] == request.seq and list(result['endpoints']) == expected,
                     'Controller batch accounting mismatch')
        self._check_cancel()
        snapshot = self.owner.snapshot()
        _require(self.controller.state == 'FREE' and not self.controller.retained and not snapshot['active_refs'],
                 'Successful batch still holds an active operation or slot')
        _require(self.owner.npu_calls-self.initial_npu_calls == self.windows+len(rows)
                 and self.owner.completed_invocations-self.initial_completed_invocations == self.windows+len(rows),
                 'NPU call/completion count differs from live windows')
        board_ns = self.clock()-started
        _require(_integer(board_ns), 'Invalid batch clock')
        self.records += result['records']
        self.windows += len(rows)
        self.batches += 1
        advance_digest(self.digest,rows)
        memory = self._memory()
        data = dict(batch_id=self.batches,records=result['records'],generation=result['generation'],
                    board_batch_ns=board_ns,runtime_snapshot=snapshot,process_memory=memory,RSS_kib=memory.get('VmRSS_kib'),
                    total_records=self.records,total_batches=self.batches,rolling_output_sha256=self.digest.hexdigest(),
                    live_npu_calls=self.owner.npu_calls-self.initial_npu_calls,
                    live_completed_invocations=self.owner.completed_invocations-self.initial_completed_invocations)
        reply = self.codec.respond(write,windows=rows,data=data)
        self.current_batch = None
        self.state = 'READY'
        return reply

    def handle_once(self, read, write):
        try:
            _require(self.state in ('NEW','READY'), 'Stream connection is terminal')
            self._check_cancel()
            request = self.codec.receive_request(read)
            if request.kind == Kind.HELLO:
                return self._hello(write)
            if request.kind == Kind.BATCH:
                return self._batch(request,write)
            self._check_cancel()
            _require(self.controller.state == 'FREE' and self.owner.active is None, 'FINISH requires idle owned hardware')
            self.state = 'FINISHED'
            summary = self.summary()
            summary['process_memory'] = self._memory()
            return self.codec.respond(write,data=summary)
        except BaseException as exc:
            self._isolate(exc)
            if self.codec.state == 'PENDING':
                try:
                    self.codec.respond(write,error_code='SESSION_FAILED')
                except BaseException as response_error:
                    self.error_response_failure = response_error
            raise

    def serve(self, read, write):
        while self.state != 'FINISHED':
            self.handle_once(read,write)
        return self.summary()
