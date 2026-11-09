"""Serial socket-owned NPU/PS memory runtime, prepared locally for PSW integration.

No service or deployment entry point is provided here. The caller must verify the
frozen model, SDK and boot identity and establish producer visibility separately.
Use one ``operation('batch')`` scope for one controller transaction. PS transport
returns survive that scope; each completed prediction releases its own returns.
An error retains the current and parent scopes and stops this owner.
There is deliberately no free, reset, reconnect or address-import operation.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
import hashlib
import io
from pathlib import Path
import sys
import threading
import time

import numpy as np

from experiments.reference.runtime.udw_probe_stop import ProbeStop

DEVICE_URL = 'socket://zg330aiu@127.0.0.1:9981'
VERSIONS = {'device': '25122301', 'icore': 'FMSHZGV3TECH-AID - 24160628'}
TENSOR_BYTES = 16000
EXPECTED_BINDINGS = (
    (0, 'icraft::xir::InputNode', 'icraft::xrt::HostBackendNode'),
    (2229, 'icraft::xir::HardOpNode', 'icraft::xrt::zg330::ZG330BackendNode'),
    (255, 'icraft::xir::OutputNode', 'icraft::xrt::HostBackendNode'),
)


class BoardNpuRuntime:
    """One Device, one Session, one Host input and at most one PS arena.

    ``startup_guard(runtime)`` must raise on a dirty or incompatible hardware
    state. It runs after Device/version reads and before Session creation/apply
    or memory allocation. SDK imports, file hashes and power-cycle admission are
    the caller's responsibility. ``diagnostic(event)`` must stream its input;
    this runtime stores no per-window trace or results history.
    """

    def __init__(self, *, diagnostic=None, stop=None, clock=time.perf_counter_ns):
        self.stop = stop or ProbeStop()
        self.stop.bind_owner(self)
        self.diagnostic = diagnostic
        self.clock = clock
        self.blocked = self.initialized = self.initialize_attempted = False
        self.reason = self.failure = self.active = None
        self.orphan_callback_refs = []  # failure-only; no successful history
        self.device = self.session = self.input = None
        self.ps_arena = self.ps_region = self.ps_manager = None
        self.ps_inventory_objects = None
        self.arena_metadata = self.manager_inventory = None
        self.arena_attempted = self.axi_attempted = False
        self.axi_device = self.axi_registers = None
        self.bindings = ()
        self.hardop_ids = ()
        self.npu_calls = self.completed_invocations = self.peak_operation_refs = 0
        self.device_allocation_attempts = self.device_allocation_returns = 0
        self.host_allocation_attempts = self.host_allocation_returns = 0
        self._lock = threading.Lock()

    def quarantine(self, reason):
        if not self.blocked:
            self.reason = str(reason)
        self.blocked = True

    def retain_failure(self, exc):
        # The first traceback keeps all uncertain SDK objects alive. Further
        # refused calls never add an unbounded chain of exceptions/tracebacks.
        if self.failure is None:
            self.failure = (exc, sys.exc_info()[2])
        self.quarantine(str(exc))

    def cancel(self, signum=None):
        if not self.stop.cancelled:
            self.stop.request(signum)

    def _emit(self, event):
        if self.diagnostic is not None:
            self.diagnostic(event)

    def hold(self, value):
        if self.active is None or self.active['thread'] != threading.get_ident():
            raise RuntimeError('An owning serial operation is required')
        self.active['refs'].append(value)
        self.peak_operation_refs = max(self.peak_operation_refs, self._active_ref_count())
        return value

    def _active_ref_count(self):
        count, scope = 0, self.active
        while scope is not None:
            count += len(scope['refs'])
            scope = scope['parent']
        return count

    @contextmanager
    def operation(self, stage, *, isolated=False):
        """Retain one transaction's temporary objects; nested calls share it.

        The outermost scope must cover one finite controller batch, never a
        replay loop. ``predict`` uses an isolated child scope so completed NPU
        objects do not accumulate while a batch still owns transport objects.
        Concurrent threads are rejected before issuing any SDK operation.
        """
        tid = threading.get_ident()
        if self.active is not None and self.active['thread'] != tid:
            raise RuntimeError('BoardNpuRuntime supports one serial caller')
        self.stop.checkpoint(stage + ':before')
        parent = self.active
        owns_lock = parent is None
        new_scope = owns_lock or isolated
        if owns_lock:
            if not self._lock.acquire(blocking=False):
                raise RuntimeError('BoardNpuRuntime supports one serial caller')
        if new_scope:
            self.active = dict(stage=stage, thread=tid, refs=[], hardops=[],
                               prediction_started=False, npu_may_be_reading=False,
                               output_copied=False, parent=parent)
        try:
            yield self.active
            self.stop.checkpoint(stage + ':before_release')
        except BaseException as exc:
            self.retain_failure(exc)
            raise
        else:
            if new_scope:
                self.active = parent
        finally:
            if owns_lock:
                self._lock.release()

    def safe_call(self, stage, operation):
        """Return (exact SDK return, elapsed_ns), retaining before any callback.

        Within an outer operation the return survives until the full request
        completes. A standalone call has its own short scope; callers must keep
        returned persistent objects themselves. Failures never clear the scope.
        """
        with self.operation(stage):
            self.stop.checkpoint(stage + ':sdk_before')
            begin = self.clock()
            value = operation()
            elapsed = self.clock() - begin
            self.hold(value)
            if stage == 'ps_arena.malloc':
                self.device_allocation_returns += 1
            elif stage == 'input.malloc':
                self.host_allocation_returns += 1
            self._emit(dict(kind='api', stage=stage, ns=elapsed))
            self.stop.checkpoint(stage + ':sdk_after')
            return value, elapsed

    def initialize(self, modules, model, startup_guard):
        if self.initialize_attempted:
            raise RuntimeError('Initialize a new runtime once; no reconnect')
        if not callable(startup_guard):
            raise ValueError('Explicit startup guard required')
        self.initialize_attempted = True
        with self.operation('initialize'):
            self.xir, self.xrt = modules['xir'], modules['xrt']
            self.host, self.zg = modules['host_backend'], modules['zg330backend']
            self.device, _ = self.safe_call('device.open', lambda: self.xrt.Device.Open(DEVICE_URL))
            versions, _ = self.safe_call('device.version', self.device.version)
            self.versions = dict(versions)
            if self.versions != VERSIONS:
                raise ValueError('Device/AI version differs')
            startup_guard(self)
            self.stop.checkpoint('startup_guard:complete')
            model = Path(model)
            self.network, _ = self.safe_call('network', lambda: self.xir.Network.CreateFromJsonFile(str(model)))
            self.safe_call('parameters', lambda: self.network.loadParamsFromFile(str(model.with_suffix('.raw'))))
            self.view, _ = self.safe_call('network.view', lambda: self.network.view(0))
            self.input_type = self.view.inputs()[0].tensorType()
            if list(self.input_type.shape) != [1, 40, 100]:
                raise ValueError('Expected compiled input [1,40,100]')
            self.host_device, _ = self.safe_call('host.device', self.xrt.HostDevice.Default)
            self.session, _ = self.safe_call('session.create', lambda: self.xrt.Session.Create(
                [self.zg.ZG330Backend, self.host.HostBackend], self.view,
                [self.device, self.host_device]))
            self.safe_call('session.apply', self.session.apply)
            self.forwards, _ = self.safe_call('session.forwards', self.session.getForwards)
            self.bindings = tuple((int(op.op_id), op.typeKey(), backend.typeKey())
                                  for op, backend, *_ in self.forwards)
            if self.bindings != EXPECTED_BINDINGS:
                raise ValueError('Execution bindings differ from the frozen validated model')
            self.hardop_ids = tuple(row[0] for row in self.bindings if 'ZG330' in row[2])
            self.converter, _ = self.safe_call('converter', lambda: self.host.HostBackend(
                next(backend for op, backend, *_ in self.forwards if 'HostBackend' in backend.typeKey())))
            self.callback = self._await_hardop
            self.safe_call('session.callback', lambda: self.session.setPostCallBack(self.callback))
            self.software_type, _ = self.safe_call('software.type', lambda: self.xir.TensorType(
                self.xir.FloatType.FP32(), [1, 40, 100], self.input_type.layout))
            self.input, _ = self.safe_call('input.tensor', lambda: self.xrt.Tensor(self.software_type))
            self.input_region, _ = self.safe_call('input.region', self.xrt.HostDevice.MemRegion)
            self.host_allocation_attempts += 1
            self.safe_call('input.malloc', lambda: self.input.mallocOn(self.input_region))
            self.initialized = True

    def open_axi_register_region(self):
        """Optional register-only AXI Device; never obtains its memory regions."""
        if self.device is None or getattr(self, 'versions', None) != VERSIONS:
            raise RuntimeError('Verified socket owner required first')
        if self.axi_attempted:
            raise RuntimeError('AXI register device may be opened only once')
        with self.operation('axi_registers'):
            self.axi_attempted = True
            self.axi_device, _ = self.safe_call('axi.open', lambda: self.xrt.Device.Open('axi://zg330aiu'))
            self.axi_registers, _ = self.safe_call('axi.registers', self.axi_device.defaultRegRegion)
            return self.axi_registers

    def allocate_ps_arena(self, size, verified_range):
        """Allocate once from this Session's socket ``udma`` manager.

        verified_range is the caller's freshly checked sysfs physical base/size.
        Inventory proves only same-manager nonoverlap, not system-wide exclusion.
        """
        if not self.initialized:
            raise RuntimeError('Initialized runtime required')
        if self.arena_attempted:
            raise RuntimeError('One arena allocation only; no replacement/retry')
        if (type(size) is not int or size <= 0 or size % 64
                or not isinstance(verified_range, dict) or set(verified_range) != {'base', 'bytes'}
                or any(type(verified_range[k]) is not int for k in verified_range)
                or verified_range['base'] < 0 or verified_range['bytes'] < size
                or verified_range['base'] + verified_range['bytes'] > 2**32):
            raise ValueError('Aligned arena and verified 32-bit PS extent required')
        with self.operation('allocate_ps_arena'):
            self.arena_attempted = True
            self.ps_region, _ = self.safe_call('ps_arena.region', lambda: self.device.getMemRegion('udma'))
            self.ps_manager, _ = self.safe_call('ps_arena.manager', self.ps_region.memManager)
            self.ps_inventory_objects, _ = self.safe_call('ps_arena.inventory', self.ps_manager.getAllMemChunk)
            existing = []
            for chunk in self.ps_inventory_objects:
                if chunk.begin.ptype() not in (self.xrt.PtrType.ADDR, self.xrt.PtrType.BOTH):
                    raise ValueError('Existing PS chunk has no device address')
                address, length = int(chunk.begin.addr()), int(chunk.byte_size)
                if address < 0 or length <= 0 or not chunk.isOn(self.ps_region):
                    raise ValueError('Invalid same-manager allocation metadata')
                existing.append(dict(address=address, bytes=length))
            self.manager_inventory = tuple(existing)
            self.device_allocation_attempts += 1
            self.ps_arena, _ = self.safe_call('ps_arena.malloc', lambda: self.ps_region.malloc(size, False, 64))
            kind = self.ps_arena.begin.ptype()
            if kind not in (self.xrt.PtrType.ADDR, self.xrt.PtrType.BOTH):
                raise ValueError('PS arena requires ADDR or BOTH pointer')
            address = int(self.ps_arena.begin.addr())
            if (int(self.ps_arena.byte_size) != size or address % 64 or not self.ps_arena.isOn(self.ps_region)
                    or address < verified_range['base']
                    or address + size > verified_range['base'] + verified_range['bytes']):
                raise ValueError('Arena outside verified PS extent or wrong owner/alignment')
            for prior in existing:
                if max(address, prior['address']) < min(address + size, prior['address'] + prior['bytes']):
                    raise ValueError('Arena overlaps a same-manager allocation')
            self.arena_metadata = dict(address=address, bytes=size, region='udma',
                pointer_type='ADDR' if kind == self.xrt.PtrType.ADDR else 'BOTH',
                auto_free=False, owner_device=DEVICE_URL, verified_range=dict(verified_range))
            return self.ps_arena

    def _check_chunk(self, chunk, offset, size):
        if (chunk is not self.ps_arena or self.arena_metadata is None
                or type(offset) is not int or type(size) is not int
                or offset < 0 or size <= 0 or offset + size > self.arena_metadata['bytes']):
            raise ValueError('Access must stay within this owner\'s sole PS arena')

    def write_chunk(self, chunk, offset, payload):
        if not isinstance(payload, bytes):
            raise ValueError('Owned immutable bytes required')
        self._check_chunk(chunk, offset, len(payload))
        with self.operation('ps_write'):
            array = self.hold(np.frombuffer(payload, dtype=np.uint8))
            _, elapsed = self.safe_call('ps_write', lambda: chunk.write(offset, array, len(payload)))
            return elapsed

    def read_chunk(self, chunk, offset, size):
        self._check_chunk(chunk, offset, size)
        with self.operation('ps_read'):
            array = self.hold(np.empty(size, dtype=np.uint8))
            self.safe_call('ps_read', lambda: chunk.read(array, offset, size))
            return array.tobytes()  # independent owned result; no raw CPTR access

    def _await_hardop(self, session, op, backend, tensors):
        if self.active is None or self.active['thread'] != threading.get_ident():
            self.orphan_callback_refs.append(tensors)
            try:
                raise RuntimeError('Callback outside the owning serial operation')
            except RuntimeError as exc:
                self.retain_failure(exc)
                raise
        if 'ZG330' not in backend.typeKey() or 'HardOp' not in op.typeKey():
            return
        self.hold(tensors)
        if not self.active['prediction_started']:
            raise ValueError('HardOp callback outside prediction')
        position = len(self.active['hardops'])
        if (position >= len(self.hardop_ids) or int(op.op_id) != self.hardop_ids[position]
                or not tensors):
            raise ValueError('Unexpected HardOp callback order/count')
        for tensor in tensors:
            self.safe_call('hardop.not_ready', lambda: tensor.setReady(False))
            ready, _ = self.safe_call('hardop.wait', lambda: tensor.waitForReady(
                timedelta(seconds=10), timedelta(milliseconds=1)))
            if not ready:
                raise TimeoutError('HardOp output timeout')
        self.active['hardops'].append(int(op.op_id))

    def predict(self, tensorbytes):
        if not self.initialized:
            raise RuntimeError('Initialized runtime required')
        if (not isinstance(tensorbytes, bytes) or len(tensorbytes) != TENSOR_BYTES
                or not np.isfinite(np.frombuffer(tensorbytes, dtype='<f4')).all()):
            raise ValueError('Finite little-endian float32[1,40,100] bytes required')
        with self.operation('predict', isolated=True):
            parent = self.active['parent']
            while parent is not None:
                if parent['prediction_started']:
                    raise RuntimeError('Reentrant prediction is prohibited')
                parent = parent['parent']
            self.active['prediction_started'] = True
            begin, stages = self.clock(), {}
            tensor_hash = hashlib.sha256(tensorbytes).hexdigest()
            array = self.hold(np.frombuffer(tensorbytes, dtype=np.uint8))
            _, stages['input_write'] = self.safe_call('input.write', lambda: self.input.write(0, array, len(tensorbytes)))
            adapted, stages['input_cast'] = self.safe_call('input.cast', lambda: self.host.Cast(
                self.input, self.software_type, self.input_type, self.converter))
            self.active['npu_may_be_reading'] = True
            self.npu_calls += 1
            outputs, stages['forward_including_hardop_wait'] = self.safe_call('npu.forward', lambda: self.session.forward([adapted]))
            if tuple(self.active['hardops']) != self.hardop_ids or len(outputs) != 1:
                raise ValueError('Missing HardOp completion or wrong output count')
            self.safe_call('output.not_ready', lambda: outputs[0].setReady(False))
            ready, stages['output_wait'] = self.safe_call('output.wait', lambda: outputs[0].waitForReady(
                timedelta(seconds=10), timedelta(milliseconds=1)))
            if not ready:
                raise TimeoutError('NPU final output timeout')
            stream = self.hold(io.BytesIO())
            _, stages['output_copy'] = self.safe_call('output.dump', lambda: outputs[0].dump(stream, 'SFB'))
            raw = stream.getvalue()
            if len(raw) != 12:
                raise ValueError('Expected exactly three float32 logits')
            logits = np.frombuffer(raw, dtype='<f4').copy()
            if not np.isfinite(logits).all():
                raise ValueError('Non-finite NPU logits')
            self.active['output_copied'] = True
            result = dict(logits=logits.tolist(), class_index=int(logits.argmax()),
                tensor_sha256=tensor_hash, stages_ns=stages,
                whole_predict_ns=self.clock() - begin, backend='board_socket_npu',
                hardops=list(self.active['hardops']))
            self._emit(dict(kind='prediction_complete', result=result))
            self.stop.checkpoint('predict:completed')
            self.completed_invocations += 1
            return result

    def snapshot(self):
        return dict(initialized=self.initialized, blocked=self.blocked, reason=self.reason,
            NPU_calls=self.npu_calls, completed_invocations=self.completed_invocations,
            active_refs=self._active_ref_count(),
            peak_operation_refs=self.peak_operation_refs,
            orphan_callbacks_retained=len(self.orphan_callback_refs),
            failure_retained=self.failure is not None,
            device_allocation_attempts=self.device_allocation_attempts,
            device_allocation_returns=self.device_allocation_returns,
            host_allocation_attempts=self.host_allocation_attempts,
            host_allocation_returns=self.host_allocation_returns,
            arena_metadata=self.arena_metadata, per_window_history=0,
            SDK_internal_allocations=None, long_running_memory_validated=False)
