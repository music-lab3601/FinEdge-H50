"""Hardware-verified PL-DDR single slot, with bounded ARM sample reads.

Opt-in ZCW1 only. Never accepts the current PSW1 BOOT. Data stays in the
producer-owned NPU input chunk. One output per command is intentional: the
SDK input binding is fixed at apply time, so ownership is serial.
"""
from __future__ import annotations

from collections import deque
import hashlib
import struct
import threading
import time

from experiments.reference.runtime.ps_window_runtime import selected_endpoints
from experiments.reference.runtime.shared_input_contract import MAGIC, ABI, CAPABILITIES, TENSOR_BYTES, METADATA_BYTES, INPUT_BYTES, SLOT_BYTES, Lease, Ownership, crc32, validate_extent, unpack_metadata


class SharedInputController:
    def __init__(self, *, read_reg, write_reg, write_records, read_shared,
                 input_base, input_size, shared_base, shared_size,
                 boot_id, allocation_id, run_id, predict_shared,
                 sample_every=64, timeout=10.0, cancelled=lambda: False,
                 quarantine=lambda error: None, clock=time.monotonic, sleep=time.sleep):
        validate_extent(input_base, input_size, INPUT_BYTES)
        validate_extent(shared_base, shared_size, SLOT_BYTES)
        if max(input_base, shared_base) < min(input_base+input_size, shared_base+shared_size):
            raise ValueError('Record and NPU input allocations overlap')
        if type(sample_every) is not int or not 1 <= sample_every <= 4096 or not 0 < timeout <= 60:
            raise ValueError('Bounded sampling and timeout required')
        # Validate the identity before any device operation.
        Lease(boot_id, allocation_id, run_id, 1, 1, (0,0,0,0), (0,0,0,0))
        self.read_reg, self.write_reg = read_reg, write_reg
        self.write_records, self.read_shared, self.predict_shared = write_records, read_shared, predict_shared
        self.input_base, self.shared_base = input_base, shared_base
        self.boot_id, self.allocation_id, self.run_id = boot_id, allocation_id, run_id
        self.sample_every, self.timeout = sample_every, timeout
        self.cancelled, self.on_quarantine, self.clock, self.sleep = cancelled, quarantine, clock, sleep
        self.owner = Ownership()
        self.sequence = self.request = self.windows = 0
        self.history = deque(maxlen=100)
        self.initialized = False
        self.read_bytes = self.sample_reads = self.metadata_reads = 0
        self.retained = {}
        self.lock = threading.Lock()

    def _check(self):
        if self.owner.state == 'QUARANTINED': raise RuntimeError('Shared owner quarantined')
        if self.cancelled(): raise RuntimeError('Shared input cancelled; retain allocations')

    def _status(self):
        self._check()
        values = {k: self.read_reg(k) for k in (4, 8, 12, 0x54, 0x60, 0x64, 0x68, 0x6C, 0x70, 0x74, 0x78)}
        self.retained['status'] = values
        if any(type(v) is not int or not 0 <= v < 2**32 for v in values.values()):
            raise ValueError('Invalid register response')
        if values[12] or values[0x54] or values[0x70] or values[0x78] or values[8] == 3:
            raise RuntimeError('Hardware sticky fault/error/rejection')
        return values

    def _idle(self, expected):
        s = self._status()
        if (s[4] or s[8] != expected or (s[0x60] | s[0x64]<<32) != self.sequence
                or (s[0x68] | s[0x6C]<<32) != self.sequence):
            raise RuntimeError('Unexpected FPGA ownership/completion sequence')
        return s

    def _command(self, command, state):
        self._check()
        if self.sequence >= 2**64-1: raise OverflowError('Command sequence exhausted')
        previous = self.sequence
        self.sequence += 1  # consume before uncertain write
        self.write_reg(0, command)
        deadline = self.clock()+self.timeout
        while True:
            s = self._status()
            issued, done = s[0x68] | s[0x6C]<<32, s[0x60] | s[0x64]<<32
            if issued not in (previous,self.sequence) or done not in (previous,self.sequence):
                raise RuntimeError('Foreign/stale command completion')
            if not s[4] and issued == done == self.sequence:
                if s[8] != state or s[0x74] != state: raise RuntimeError('Wrong FPGA terminal state')
                return
            if self.clock() >= deadline: raise TimeoutError('FPGA completion timeout; no reuse')
            self.sleep(.0005)

    def initialize(self):
        if self.initialized: raise RuntimeError('Single-use initialize')
        try:
            if any(self.read_reg(k) != v for k,v in ((0xC0,MAGIC),(0x8C,ABI),(0x90,CAPABILITIES))):
                raise ValueError('ZCW1 hardware required; PSW1/UDW1 cannot share NPU input')
            self._idle(0)
            self.initialized = True
        except BaseException as exc:
            self._fail(exc); raise

    def _read(self, offset, size):
        self._check()
        data = self.read_shared(offset,size)
        self.retained['last_read'] = data
        if type(data) is not bytes or len(data) != size: raise ValueError('Incomplete owned memory response')
        self.read_bytes += size
        return data

    def _fail(self, error):
        self.owner.quarantine(error)
        try: self.on_quarantine(error)
        except BaseException as secondary: self.retained['secondary_error'] = secondary

    def submit(self, payload):
        if not self.lock.acquire(False): raise RuntimeError('Concurrent shared input submission refused')
        try:
            self._check()
            if not self.initialized or self.owner.state != 'FREE': raise RuntimeError('Shared slot unavailable')
            expected, history = selected_endpoints(payload,self.history)
            if len(expected) > 1: raise ValueError('Fixed binding supports at most one window per command')
            self._idle(0)
            if self.request >= 2**64-1: raise OverflowError('Request generation exhausted')
            self.request += 1
            endpoints = expected[0] if expected else (tuple(history[0]),tuple(history[-1]))
            lease = Lease(self.boot_id,self.allocation_id,self.run_id,self.request,self.request,*endpoints)
            self.owner.begin(lease)
            self.retained.update(lease=lease,payload=payload)
            self.write_records(payload)  # write only new records, never the output slot
            records = len(payload)//256
            descriptor = struct.pack('<6I3Q',self.input_base,len(payload),self.shared_base,
                SLOT_BYTES,records,len(expected),self.run_id,self.request,self.request)
            for i,v in enumerate(struct.unpack('<12I',descriptor)): self.write_reg(0x10+4*i,v)
            for i,v in enumerate(struct.unpack('<12I',descriptor)):
                if self.read_reg(0x10+4*i) != v: raise ValueError('Shared descriptor changed')
            self._command(1 if self.request == 1 else 4,2)
            counts = {0x40:records,0x44:len(expected),0x48:len(expected)*16128,
                      0x4C:records*4,0x50:len(expected)*252,0x58:1}
            if any(self.read_reg(k) != v for k,v in counts.items()): raise ValueError('FPGA output counts differ')
            self._idle(2)
            self.owner.produced(lease)
            result = None
            if expected:
                raw = self._read(TENSOR_BYTES,METADATA_BYTES); self.metadata_reads += 1
                metadata = unpack_metadata(raw,lease)
                sample = self.windows == 0 or (self.windows+1)%self.sample_every == 0
                if sample:
                    tensor = self._read(0,TENSOR_BYTES); self.sample_reads += 1
                    if crc32(tensor) != metadata['crc32']: raise ValueError('ARM sample CRC mismatch')
                    metadata['sample_sha256'] = hashlib.sha256(tensor).hexdigest()
                self._idle(2)
                self.owner.consume(lease)  # burned before possibly starting NPU
                result = self.predict_shared(lease,metadata)
                self.retained['prediction'] = result
                if not isinstance(result,dict) or result.get('input_consumed') is not True:
                    raise ValueError('Explicit NPU completion acknowledgement required')
                self._idle(2)
                if self._read(TENSOR_BYTES,METADATA_BYTES) != raw:
                    raise ValueError('Shared slot identity changed during NPU consumption')
                self.metadata_reads += 1
                self.owner.completed(lease)
                self.windows += 1
                result = dict(result,window_id=self.windows,metadata=metadata,sampled=sample)
            else:
                # No NPU input was produced in a warmup-only command.
                self.owner.consume(lease); self.owner.completed(lease)
            self._command(2,0)  # only after NPU HardOp + final output completion
            self.owner.release(lease)
            self.history = history
            self.retained = {}
            return dict(request_id=self.request,records=records,windows=[] if result is None else [result])
        except BaseException as exc:
            self._fail(exc); raise
        finally:
            self.lock.release()

    def summary(self):
        return dict(state=self.owner.state,request_id=self.request,windows=self.windows,
            explicit_shared_read_bytes=self.read_bytes,metadata_reads=self.metadata_reads,
            full_tensor_sample_reads=self.sample_reads,sample_every=self.sample_every,
            explicit_full_tensor_input_copy_bytes=0,hardware_read_compare=True,
            strict_visibility_proven=False,hardware_acceptance=False)
