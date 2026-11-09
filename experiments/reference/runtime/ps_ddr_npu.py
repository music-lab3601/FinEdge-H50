"""Finite socket-owned PS/PL transport plus original NPU adaptation.

Four disjoint transfer ranges only. No RTL command, reset, free, reconnect,
implicit retry or claim of producer completion/zero-copy. Successful invocation
references are released only after every HardOp wait and owned output check.
"""
from __future__ import annotations

from datetime import timedelta
import hashlib
import io
import sys
import time

import numpy as np

from experiments.reference.runtime.udw_probe_stop import ProbeStop

SCHEMA = 'ps-ddr-npu-probe-v1'
DEVICE_URL = 'socket://zg330aiu@127.0.0.1:9981'
VERSIONS = {'device': '25122301', 'icore': 'FMSHZGV3TECH-AID - 24160628'}
ARENA = 65536
SLOT = 16384
TENSOR_BYTES = 16000
COUNT = 4
FILL = {'ps_source': 0xA3, 'pl': 0x5C, 'ps_return': 0xE7}
OFFSETS = {'ps_source': 64, 'pl': 128, 'ps_return': 192}
STARTUP = (0xC0, 0x8C, 4, 8, 12, 0x70, 0x60, 0x64, 0x68, 0x6C, 0x54)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_vectors(inputs, references):
    values = np.asarray(inputs)
    refs = np.asarray(references)
    if (values.dtype != np.dtype('<f4') or values.shape != (COUNT, 1, 40, 100)
            or not np.isfinite(values).all() or refs.shape != (COUNT, 3)
            or not np.isfinite(refs).all()):
        raise ValueError('Exactly four finite golden float32 windows and Host references required')
    return values, refs.astype(np.float32)


class SocketNpuProbe:
    def __init__(self, *, record=None, stop=None, clock=time.perf_counter_ns):
        self.stop = stop or ProbeStop()
        self.stop.bind_owner(self)
        self.record = record or (lambda event: None)
        self.clock = clock
        self.blocked = self.started = self.initialized = False
        self.reason = None
        self.device = self.session = self.input = None
        self.chunks, self.regions, self.metadata, self.snapshots = {}, {}, {}, {}
        self.startup_refs, self.failures, self.results = [], [], []
        self.active = None
        self.peak_invocation_refs = self.completed_invocations = self.npu_calls = 0
        self.allocation_attempts = dict(device=0, host_input=0)
        self.allocation_returns = dict(device=0, host_input=0)
        self.expected = {name: bytearray([fill]) * ARENA for name, fill in FILL.items()}
        self.hardop_ids, self.bindings = [], []
        self.manager_inventory = {}

    def quarantine(self, reason):
        self.blocked, self.reason = True, str(reason)

    def hold(self, value):
        refs = self.active['refs'] if self.active is not None else self.startup_refs
        refs.append(value)
        if self.active is not None:
            self.peak_invocation_refs = max(self.peak_invocation_refs, len(refs))
        return value

    def retain_failure(self, exc):
        self.failures.append((exc, sys.exc_info()[2]))
        self.quarantine(str(exc))

    def call(self, stage, operation):
        self.stop.checkpoint(stage + ':before')
        try:
            allocation_kind = ('host_input' if stage == 'input.malloc' else 'device') if stage.endswith('.malloc') else None
            if allocation_kind is not None:
                self.allocation_attempts[allocation_kind] += 1
            if stage == 'npu.forward':
                self.npu_calls += 1
            begin = self.clock()
            value = operation()
            elapsed = self.clock() - begin
            self.hold(value)  # before cancellation, logging or metadata getters
            if allocation_kind is not None:
                self.allocation_returns[allocation_kind] += 1
            self.record(dict(kind='api', stage=stage, ns=elapsed))
            self.stop.checkpoint(stage + ':after')
            return value, elapsed
        except BaseException as exc:
            self.retain_failure(exc)
            raise

    def inventory(self, region, name):
        """Optional same-region inventory; retains the startup snapshot only."""
        method = getattr(region, 'memManager', None)
        if not callable(method):
            return dict(available=False, reason='memManager binding unavailable', chunks=[])
        manager, _ = self.call(name + '.manager', method)
        getter = getattr(manager, 'getAllMemChunk', None)
        if not callable(getter):
            return dict(available=False, reason='getAllMemChunk binding unavailable', chunks=[])
        chunks, _ = self.call(name + '.inventory', getter)
        rows = []
        for chunk in chunks:
            kind = chunk.begin.ptype()
            if kind not in (self.xrt.PtrType.ADDR, self.xrt.PtrType.BOTH):
                raise ValueError('Device manager returned non-address chunk')
            address, size = int(chunk.begin.addr()), int(chunk.byte_size)
            if address < 0 or size <= 0 or not chunk.isOn(region):
                raise ValueError('Invalid manager chunk metadata')
            rows.append(dict(address=address, bytes=size))
        return dict(available=True, chunks=rows,
                    scope='Only this MemRegion manager; not cross-manager/system exclusion')

    def initialize(self, modules, model, *, ps_range):
        if self.device is not None or self.initialized:
            raise ValueError('Initialize a new owner once')
        try:
            if (set(ps_range) != {'base', 'bytes'} or
                    any(type(ps_range[k]) is not int or ps_range[k] <= 0 for k in ps_range)):
                raise ValueError('Verified PS udmabuf physical extent required')
            self.ps_range = dict(ps_range)
            self.xir, self.xrt = modules['xir'], modules['xrt']
            self.host, self.zg = modules['host_backend'], modules['zg330backend']
            xrt = self.xrt
            self.device, _ = self.call('device.open', lambda: xrt.Device.Open(DEVICE_URL))
            versions, _ = self.call('device.version', self.device.version)
            self.versions = dict(versions)
            if self.versions != VERSIONS:
                raise ValueError('Device/AI version differs')
            self.reg, _ = self.call('device.reg', self.device.defaultRegRegion)
            self.startup = {}
            for offset in STARTUP:
                value, _ = self.call('startup.' + hex(offset), lambda: self.reg.read(0x400C0000 + offset, False))
                self.startup[offset] = int(value)
                expected = 0x55445731 if offset == 0xC0 else 2 if offset == 0x8C else 0
                if int(value) != expected:
                    raise ValueError('Clean UDW1/ABI2 required before Session/allocation')
            self.network, _ = self.call('network', lambda: self.xir.Network.CreateFromJsonFile(str(model)))
            self.call('parameters', lambda: self.network.loadParamsFromFile(str(model.with_suffix('.raw'))))
            self.view, _ = self.call('network.view', lambda: self.network.view(0))
            self.input_type = self.view.inputs()[0].tensorType()
            if list(self.input_type.shape) != [1, 40, 100]:
                raise ValueError('Unexpected compiled model input shape')
            self.session, _ = self.call('session.create', lambda: xrt.Session.Create(
                [self.zg.ZG330Backend, self.host.HostBackend], self.view,
                [self.device, xrt.HostDevice.Default()]))
            self.call('session.apply', self.session.apply)
            forwards, _ = self.call('session.forwards', self.session.getForwards)
            self.bindings = [dict(op_id=int(op.op_id), op=op.typeKey(), backend=backend.typeKey())
                             for op, backend, *_ in forwards]
            self.hardop_ids = [b['op_id'] for b in self.bindings
                              if 'ZG330' in b['backend'] and 'HardOp' in b['op']]
            if not self.hardop_ids or len(set(self.hardop_ids)) != len(self.hardop_ids):
                raise ValueError('Expected distinct ZG330 HardOp bindings')
            self.converter, _ = self.call('converter', lambda: self.host.HostBackend(
                next(b for op, b, *_ in forwards if 'HostBackend' in b.typeKey())))
            self.callback = self.await_hardop
            self.call('session.callback', lambda: self.session.setPostCallBack(self.callback))
            self.software_type, _ = self.call('software.type', lambda: self.xir.TensorType(
                self.xir.FloatType.FP32(), [1, 40, 100], self.input_type.layout))
            self.input, _ = self.call('input.tensor', lambda: xrt.Tensor(self.software_type))
            self.call('input.malloc', lambda: self.input.mallocOn(xrt.HostDevice.MemRegion()))
            for name in FILL:
                region_name = 'plddr' if name == 'pl' else 'udma'
                region, _ = self.call(name + '.region', lambda: self.device.getMemRegion(region_name))
                self.regions[name] = region
                if region_name not in self.manager_inventory:
                    self.manager_inventory[region_name] = self.inventory(region, region_name)
                chunk, _ = self.call(name + '.malloc', lambda: region.malloc(ARENA, False, 64))
                self.chunks[name] = chunk
                kind = chunk.begin.ptype()
                if kind not in (xrt.PtrType.ADDR, xrt.PtrType.BOTH):
                    raise ValueError('Device address required; no CPTR coercion')
                address = int(chunk.begin.addr())
                if int(chunk.byte_size) != ARENA or address < 0 or address % 64 or not chunk.isOn(region):
                    raise ValueError('Unexpected chunk bounds, alignment or owner region')
                if region_name == 'udma' and not (ps_range['base'] <= address and
                        address + ARENA <= ps_range['base'] + ps_range['bytes']):
                    raise ValueError('PS allocation outside verified udmabuf physical range')
                for previous in self.manager_inventory[region_name]['chunks']:
                    if max(address, previous['address']) < min(address + ARENA, previous['address'] + previous['bytes']):
                        raise ValueError('User allocation overlaps existing same-manager allocation')
                self.metadata[name] = dict(address=address, bytes=ARENA, region=region_name,
                    pointer_type='ADDR' if kind == xrt.PtrType.ADDR else 'BOTH', auto_free=False,
                    owner_device=DEVICE_URL)
            a, b = [self.metadata[n]['address'] for n in ('ps_source', 'ps_return')]
            if max(a, b) < min(a + ARENA, b + ARENA):
                raise ValueError('PS chunks overlap')
            self.initialized = True
        except BaseException as exc:
            self.retain_failure(exc)
            raise

    def await_hardop(self, session, op, backend, tensors):
        if 'ZG330' not in backend.typeKey() or 'HardOp' not in op.typeKey():
            return
        self.hold(tensors)  # even a callback failure must retain returned objects
        if self.active is None:
            self.quarantine('HardOp callback outside invocation')
            raise ValueError(self.reason)
        op_id = int(op.op_id)
        position = len(self.active['hardops'])
        if position >= len(self.hardop_ids) or op_id != self.hardop_ids[position] or not tensors:
            self.quarantine('Unexpected HardOp callback order/count')
            raise ValueError(self.reason)
        for tensor in tensors:
            self.call('hardop.not_ready', lambda: tensor.setReady(False))
            ready, _ = self.call('hardop.wait', lambda: tensor.waitForReady(
                timedelta(seconds=10), timedelta(milliseconds=1)))
            if not ready:
                self.quarantine('HardOp output timeout')
                raise TimeoutError(self.reason)
        self.active['hardops'].append(op_id)

    @staticmethod
    def extent(offset, size):
        if type(offset) is not int or type(size) is not int or offset < 0 or size <= 0 or offset + size > ARENA:
            raise ValueError('Transfer outside fixed allocation')

    def write(self, name, offset, data, tag):
        self.extent(offset, len(data))
        array = self.hold(np.frombuffer(data, dtype=np.uint8))
        _, elapsed = self.call(tag, lambda: self.chunks[name].write(offset, array, len(data)))
        return elapsed

    def read(self, name, offset, size, tag):
        self.extent(offset, size)
        array = self.hold(np.full(size, 0x39, dtype=np.uint8))
        _, elapsed = self.call(tag, lambda: self.chunks[name].read(array, offset, size))
        actual = array.tobytes()
        self.snapshots[tag] = actual
        if actual != bytes(self.expected[name][offset:offset + size]):
            self.quarantine('Data/guard mismatch: ' + tag)
            raise ValueError(self.reason)
        return actual, elapsed

    def copy(self, dst, src, dst_offset, src_offset, size, tag):
        if (dst, src) not in (('pl', 'ps_source'), ('ps_return', 'pl')):
            raise ValueError('Only planned PS→PL→PS copies allowed')
        self.extent(dst_offset, size)
        self.extent(src_offset, size)
        _, elapsed = self.call(tag, lambda: self.chunks[dst].copyFrom(
            dst_offset, self.chunks[src], src_offset, size))
        return elapsed

    def predict(self, tensor, reference, stages):
        self.stop.checkpoint('predict:before')
        if self.active is None or self.npu_calls >= COUNT or len(tensor) != TENSOR_BYTES:
            raise ValueError('Finite invocation required')
        array = self.hold(np.frombuffer(tensor, dtype=np.uint8))
        _, stages['input_write'] = self.call('input.write', lambda: self.input.write(0, array, len(tensor)))
        adapted, stages['input_cast'] = self.call('input.cast', lambda: self.host.Cast(
            self.input, self.software_type, self.input_type, self.converter))
        self.active['npu_may_be_reading'] = True
        outputs, stages['forward_including_hardop_wait'] = self.call('npu.forward', lambda: self.session.forward([adapted]))
        if self.active['hardops'] != self.hardop_ids or len(outputs) != 1:
            raise ValueError('Missing HardOp completion or wrong output count')
        self.call('output.not_ready', lambda: outputs[0].setReady(False))
        ready, stages['output_wait'] = self.call('output.wait', lambda: outputs[0].waitForReady(
            timedelta(seconds=10), timedelta(milliseconds=1)))
        if not ready:
            raise TimeoutError('NPU final output timeout')
        buffer = self.hold(io.BytesIO())
        _, stages['output_copy'] = self.call('output.dump', lambda: outputs[0].dump(buffer, 'SFB'))
        logits = np.frombuffer(buffer.getvalue(), dtype='<f4').copy()
        if logits.shape != (3,) or not np.isfinite(logits).all():
            raise ValueError('Invalid NPU logits')
        error = float(np.max(np.abs(logits.astype(np.float64) - reference)))
        result = dict(logits=logits.tolist(), class_index=int(logits.argmax()),
            reference_logits=reference.tolist(), max_abs_error=error,
            within_tolerance=bool(np.allclose(logits, reference, rtol=1e-3, atol=1e-4)),
            class_match=bool(logits.argmax() == reference.argmax()))
        self.active['observed_result'] = result
        if not result['within_tolerance'] or not result['class_match']:
            raise ValueError('NPU output differs from frozen compiled Host reference')
        self.active['output_copied'] = True
        return result

    def run(self, inputs, references):
        self.stop.checkpoint('run:before')
        if not self.initialized or self.started:
            raise ValueError('Initialized unused finite probe required')
        values, refs = validate_vectors(inputs, references)
        self.started = True
        try:
            for name in FILL:
                self.write(name, 0, bytes(self.expected[name]), name + '.initial_write')
                self.read(name, 0, ARENA, name + '.initial_read')
            for index in range(COUNT):
                self.active = dict(index=index, refs=[], hardops=[], npu_may_be_reading=False,
                                   output_copied=False)
                begin = self.clock()
                payload = values[index].tobytes(order='C')
                base = index * SLOT
                offsets = {name: base + off for name, off in OFFSETS.items()}
                tag, stages = 'case%02d' % index, {}
                stages['ps_write'] = self.write('ps_source', offsets['ps_source'], payload, tag + '.ps_write')
                self.expected['ps_source'][offsets['ps_source']:offsets['ps_source'] + TENSOR_BYTES] = payload
                _, stages['ps_read'] = self.read('ps_source', base, SLOT, tag + '.ps_read')
                stages['ps_to_pl'] = self.copy('pl', 'ps_source', offsets['pl'], offsets['ps_source'], TENSOR_BYTES, tag + '.ps_to_pl')
                self.expected['pl'][offsets['pl']:offsets['pl'] + TENSOR_BYTES] = payload
                _, stages['pl_read'] = self.read('pl', base, SLOT, tag + '.pl_read')
                stages['pl_to_ps'] = self.copy('ps_return', 'pl', offsets['ps_return'], offsets['pl'], TENSOR_BYTES, tag + '.pl_to_ps')
                self.expected['ps_return'][offsets['ps_return']:offsets['ps_return'] + TENSOR_BYTES] = payload
                returned, stages['ps_return_read'] = self.read('ps_return', base, SLOT, tag + '.ps_return_read')
                npu_input = returned[OFFSETS['ps_return']:OFFSETS['ps_return'] + TENSOR_BYTES]
                result = self.predict(npu_input, refs[index], stages)
                row = dict(result, input_index=index, tensor_sha256=digest(npu_input),
                           offsets=offsets, stages_ns=stages, whole_case_ns=self.clock() - begin,
                           hardops=list(self.active['hardops']), retained_refs_before_release=len(self.active['refs']))
                self.stop.checkpoint(tag + ':before_release')
                self.record(dict(kind='case_complete', result=row))
                self.results.append(row)
                self.completed_invocations += 1
                self.active = None  # all successful waits, output copy and checks precede release
            for name in FILL:
                self.read(name, 0, ARENA, name + '.final_read')
            return self.report('passed')
        except BaseException as exc:
            self.retain_failure(exc)
            raise

    def report(self, status):
        return dict(schema=SCHEMA, status=status, cases=self.results, metadata=self.metadata,
            bindings=self.bindings, hardop_ids=self.hardop_ids, versions=getattr(self, 'versions', {}),
            startup=getattr(self, 'startup', {}), device_url=DEVICE_URL,
            ps_range=getattr(self, 'ps_range', {}), manager_inventory=self.manager_inventory,
            user_device_allocations=self.allocation_returns['device'],
            user_device_bytes=self.allocation_returns['device'] * ARENA,
            user_host_input_allocations=self.allocation_returns['host_input'],
            allocation_attempts=dict(self.allocation_attempts), allocation_returns=dict(self.allocation_returns),
            internal_SDK_allocation_count=None,
            persistent_refs=len(self.startup_refs), peak_invocation_refs=self.peak_invocation_refs,
            active_invocation_refs=len(self.active['refs']) if self.active else 0,
            failed_invocation={k:v for k,v in self.active.items() if k != 'refs'} if self.active else None,
            completed_invocations=self.completed_invocations, NPU_calls=self.npu_calls, warmups=0,
            RTL_commands=0, reused_device_ranges=0, owner_retained=True, blocked=self.blocked,
            reason=self.reason, stop=self.stop.snapshot(), strict_visibility_proven=False,
            zero_copy=False, full_day_pool_authorized=False,
            timing_scope='Four un-warmed functional samples; API wall time, not throughput benchmark')
