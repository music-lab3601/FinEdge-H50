"""Finite board-resident FPGA -> local adaptation -> NPU experiment.

Python 3.8 compatible. No book reconstruction, tensor ring, reset, free or
buffer reuse. Transport carries WDM records in and logits out. PL visibility
is an experimental observation, not a proved memory fence.
"""
from collections import deque
from datetime import timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np

from experiments.reference.runtime.ddr_transport_contract import Descriptor, OUTPUT_STRIDE, unpack_record, unpack_output
from experiments.reference.runtime.udw_lab import UdwLab

REG = 0x400c0000
ARENA = 2 * 1024 * 1024
QUOTA = 1024 * 1024  # bounded inside unchanged 2 MiB SDK arena; never reused
STARTUP = (0xc0, 0x8c, 4, 8, 12, 0x70, 0x60, 0x64, 0x68, 0x6c, 0x54)
STATUS = tuple(range(4, 0x90, 4)) + (0xc0,)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require_clean(regs):
    for offset in STARTUP:
        expected = 0x55445731 if offset == 0xc0 else 2 if offset == 0x8c else 0
        if type(regs.get(offset)) is not int or regs[offset] != expected:
            raise ValueError('Fresh physical cold-session gate failed at ' + hex(offset))


def selected_endpoints(payload, history=()):
    if not payload or len(payload) % 256 or len(payload) > 256 * 256:
        raise ValueError('Expected 1..256 complete WDM records')
    ids = deque(history, maxlen=100)
    selected = []
    for start in range(0, len(payload), 256):
        words, ident, emit = unpack_record(payload[start:start + 256])
        if not np.isfinite(np.asarray(words, dtype='<u4').view('<f4')).all():
            raise ValueError('Nonfinite WDM input')
        if ids:
            old = ids[-1]
            if ident[0] != old[0]:
                ids.clear()
            elif not (ident[1] == old[1] + 1 and ident[2] > old[2] and ident[3] >= old[3]):
                raise ValueError('Discontinuous WDM identity')
        ids.append(ident)
        if len(ids) == 100 and emit:
            selected.append((ids[0], ident))
    return selected, list(ids)


class LocalTransport:
    def __init__(self, owner):
        self.owner = owner

    def status(self):
        o = self.owner
        if o.blocked:
            raise RuntimeError('Owner quarantined')
        values = {k: int(o.reg.read(REG + k, False)) for k in STATUS}
        # Sticky/error sampled last; still not proof against a future fault.
        for k in (0x70, 12, 8):
            values[k] = int(o.reg.read(REG + k, False))
        return values

    def command(self, value):
        if value not in (1, 2, 4):
            raise ValueError('No reset command')
        if value in (1, 4):
            self.owner.risk = True
        self.owner.reg.write(REG, value, False)


class BoardBridge(UdwLab):
    def __init__(self):
        # Constructed BEFORE initialize; caller retains even a partial instance.
        self.risk = self.blocked = False
        self.session = self.device = self.chunk = None
        self.used = 0
        self.identities = []
        self.retained = []
        self.events = []
        self.call_count = 0
        self.finished = False

    def initialize(self, model, expected_boot, run_id, control='loopback'):
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != expected_boot:
            raise ValueError('Boot identity changed')
        sys.path.insert(0, '/usr/lib/aarch64-linux-gnu')
        import xir, xrt, host_backend, zg330backend
        for module in (xir, xrt, host_backend, zg330backend):
            if Path(module.__file__).parent != Path('/usr/lib/aarch64-linux-gnu'):
                raise ValueError('Wrong SDK module: ' + str(module.__file__))
        self.xrt, self.host = xrt, host_backend
        self.device = xrt.Device.Open('socket://zg330aiu@127.0.0.1:9981')
        self.reg = self.device.defaultRegRegion()
        startup = {k: int(self.reg.read(REG + k, False)) for k in STARTUP}
        require_clean(startup)  # before Session creation/apply or any allocation
        if control == 'axi_registers':
            # Only the documented register region uses local AXI. NPU Session
            # and DDR allocation keep the numerically validated loopback device.
            self.control_device = xrt.Device.Open('axi://zg330aiu')
            local_reg = self.control_device.defaultRegRegion()
            direct_startup = {k: int(local_reg.read(REG + k, False)) for k in STARTUP}
            require_clean(direct_startup)
            if direct_startup != startup:
                raise ValueError('Local AXI register view differs from loopback')
            self.reg = local_reg
        elif control != 'loopback':
            raise ValueError('Unknown control transport')
        self.network = xir.Network.CreateFromJsonFile(str(model))
        self.network.loadParamsFromFile(str(model.with_suffix('.raw')))
        view = self.network.view(0)
        self.input_type = view.inputs()[0].tensorType()
        if list(self.input_type.shape) != [1, 40, 100]:
            raise ValueError('Wrong model input shape')
        self.session = xrt.Session.Create([zg330backend.ZG330Backend, host_backend.HostBackend],
                                         view, [self.device, xrt.HostDevice.Default()])
        self.session.apply()
        def await_hardop(session, op, backend, tensors):
            if 'ZG330' in backend.typeKey() and 'HardOp' in op.typeKey():
                for tensor in tensors:
                    tensor.setReady(False)
                    if not tensor.waitForReady(timedelta(seconds=10), timedelta(milliseconds=1)):
                        raise TimeoutError('NPU HardOp timeout before Host output')
        self.callback = await_hardop
        self.session.setPostCallBack(self.callback)
        forwards = self.session.getForwards()
        bindings = [{'op_id': op.op_id, 'op': op.typeKey(), 'backend': backend.typeKey()}
                    for op, backend, *_ in forwards]
        if not any('ZG330' in b['backend'] and 'HardOp' in b['op'] for b in bindings):
            raise ValueError('NPU binding absent')
        self.converter = host_backend.HostBackend(next(b for op, b, *_ in forwards
                                                       if 'HostBackend' in b.typeKey()))
        self.software_type = xir.TensorType(xir.FloatType.FP32(), [1, 40, 100], self.input_type.layout)
        self.input = xrt.Tensor(self.software_type)
        self.input.mallocOn(xrt.HostDevice.MemRegion())
        self.chunk = self.device.defaultMemRegion().malloc(ARENA, False, 64)
        base = int(self.chunk.begin.addr())
        if base % 64 or base + ARENA > 2**32:
            raise ValueError('Arena address is not valid uint32 User DDR')
        UdwLab.__init__(self, LocalTransport(self), base, ARENA, run_id=run_id,
                        generation=1, journal=self.events.append, timeout=5)
        self.info = dict(startup=startup, boot_id=expected_boot, base=base, arena_bytes=ARENA,
                         quota_bytes=QUOTA, bindings=bindings, versions=dict(self.device.version()),
                         input_type=str(self.input_type), input_logical_bytes=self.input_type.bytes(),
                         mode='board_loopback_local_copy', strict_visibility_proven=False,
                         control_transport=control, no_PC_window_return=True, zero_copy=False, pid=os.getpid())
        return self.info

    def read_local(self, offset, size):
        if offset < 0 or size < 0 or offset + size > self.used:
            raise ValueError('Read outside reserved arena')
        dest = np.empty(size, dtype=np.uint8)
        self.chunk.read(dest, offset, size)
        return dest.tobytes()

    def predict_local(self, tensor):
        if len(tensor) != 16000 or not np.isfinite(np.frombuffer(tensor, dtype='<f4')).all():
            raise ValueError('Invalid logical NPU tensor')
        begin = time.perf_counter_ns()
        self.input.write(0, np.frombuffer(tensor, dtype=np.uint8), len(tensor))
        adapted = self.host.Cast(self.input, self.software_type, self.input_type, self.converter)
        cast_end = time.perf_counter_ns()
        if not hasattr(self, 'adapted_info'):
            self.adapted_info = dict(chunk_bytes=int(adapted.chunk().byte_size),
                                     offset=int(adapted.offset()), dtype=str(adapted.dtype()))
        outputs = self.session.forward([adapted])
        forward_end = time.perf_counter_ns()
        if len(outputs) != 1:
            raise ValueError('Wrong NPU output count')
        outputs[0].setReady(False)
        if not outputs[0].waitForReady(timedelta(seconds=10), timedelta(milliseconds=1)):
            raise TimeoutError('NPU output timeout')
        ready_end = time.perf_counter_ns()
        buffer = io.BytesIO()
        outputs[0].dump(buffer, 'SFB')
        logits = np.frombuffer(buffer.getvalue(), dtype='<f4')
        end = time.perf_counter_ns()
        if logits.shape != (3,) or not np.isfinite(logits).all():
            raise ValueError('Invalid NPU logits')
        self.call_count += 1
        return dict(logits=logits.tolist(), class_index=int(logits.argmax()),
                    npu_call=self.call_count, tensor_sha256=digest(tensor),
                    stages_ns=dict(host_input_cast=cast_end-begin, forward=forward_end-cast_end,
                                   output_wait=ready_end-forward_end, output_read=end-ready_end,
                                   npu_total=end-begin))

    def software_predict(self, tensor):
        begin = time.perf_counter_ns()
        self.guard('software_before_npu')
        result = self.predict_local(tensor)
        self.guard('software_after_npu')
        result['board_total_ns'] = time.perf_counter_ns() - begin
        return result

    def submit_local(self, payload, continuation):
        begin = time.perf_counter_ns()
        selected, identities = selected_endpoints(payload, self.identities if continuation else ())
        records = len(payload) // 256
        capacity = max(64, len(selected) * OUTPUT_STRIDE)
        size = len(payload) + capacity + 256
        # Capacity is checked before REARM, allocation, descriptor or command.
        if self.used + size + 4096 > QUOTA:
            raise ValueError('Finite arena exhausted; no reuse or automatic restart')
        before = self.guard('before_batch')
        if before[8] == 2:
            self.issue(2, 0)
        elif before[8] != 0:
            raise ValueError('Invalid PL state')
        offset = self.used
        self.used += size  # burn allocation before any write, including failure
        input_offset = offset + 64
        output_offset = offset + len(payload) + 192
        image = self.GUARD + payload + self.GUARD * 2 + bytes([self.POISON]) * capacity + self.GUARD
        self.chunk.write(offset, np.frombuffer(image, dtype=np.uint8), len(image))
        if self.read_local(offset, size) != image:
            raise ValueError('Preparation readback differs')
        prepared = time.perf_counter_ns()
        request = self.request_id + 1
        descriptor = Descriptor(self.base + input_offset, len(payload), self.base + output_offset,
                                capacity, records, len(selected), self.run_id, request, self.generation)
        shadow = descriptor.shadow_registers(address_limit_exclusive=self.base + self.size)
        self.guard('before_descriptor')
        for key, value in shadow.items():
            self.reg.write(REG + key, value, False)
        actual_shadow = self.guard('after_descriptor')
        if any(actual_shadow[k] != value for k, value in shadow.items()):
            raise ValueError('Shadow register mismatch')
        terminal = self.issue(4 if continuation else 1, 2)
        checks = {0x40: records, 0x44: len(selected), 0x48: len(selected) * OUTPUT_STRIDE,
                  0x4c: records * 4, 0x50: len(selected) * 252, 0x58: 1}
        if any(terminal.get(k) != value for k, value in checks.items()):
            raise ValueError('PL terminal count/capability mismatch')
        pl_end = time.perf_counter_ns()
        self.guard('before_local_read')
        actual = self.read_local(offset, size)
        self.guard('after_local_read')
        relative = output_offset - offset
        valid = len(selected) * OUTPUT_STRIDE
        if actual[:relative] != image[:relative] or actual[relative+valid:] != image[relative+valid:]:
            raise ValueError('Input/guard/unused-output corruption')
        tensors = []
        for i, (first, last) in enumerate(selected):
            tensor, got_first, got_last = unpack_output(actual[relative+i*OUTPUT_STRIDE:relative+(i+1)*OUTPUT_STRIDE],
                run_id=self.run_id, request_id=request, window_index=i, generation=self.generation)
            if got_first != first or got_last != last:
                raise ValueError('WDO endpoint identity mismatch')
            tensors.append(tensor)
        read_end = time.perf_counter_ns()
        predictions = []
        for tensor, (first, last) in zip(tensors, selected):
            self.guard('before_local_npu')
            result = self.predict_local(tensor)
            self.guard('after_local_npu')
            result.update(first=list(first), last=list(last))
            predictions.append(result)
        npu_end = time.perf_counter_ns()
        # Current-slot check after NPU; full history audit only at experiment end.
        if self.read_local(offset, size) != actual:
            raise ValueError('Current PL slot changed during NPU')
        self.guard('after_batch')
        self.retained.append((offset, actual))
        self.identities, self.request_id = identities, request
        end = time.perf_counter_ns()
        return dict(request_id=request, records=records, predictions=predictions, used=self.used,
                    descriptor=descriptor.__dict__, terminal=terminal, input_sha256=digest(payload),
                    stages_ns=dict(validate_prepare=prepared-begin, descriptor_PL=pl_end-prepared,
                                   local_read_verify=read_end-pl_end, npu_with_health=npu_end-read_end,
                                   final_check=end-npu_end, board_total=end-begin),
                    strict_visibility_proven=False, zero_copy=False)

    def finish(self, directory):
        self.guard('before_final_audit')
        for i, (offset, data) in enumerate(self.retained):
            if self.read_local(offset, len(data)) != data:
                raise ValueError('Retained buffer changed')
            (directory / ('slot_%03d.bin' % i)).write_bytes(data)
        self.guard('after_final_audit')
        self.finished = True
        (directory / 'events.json').write_text(json.dumps(self.events))
        return dict(used=self.used, retained_slots=len(self.retained), calls=self.call_count,
                    adapted_info=self.adapted_info, strict_visibility_proven=False,
                    owner_retained=True, audit_passed=True)
