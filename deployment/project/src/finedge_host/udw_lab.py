"""Protected UDW1 ABI2 experiment. Never establishes DDR completion/visibility.

Transport owns a bounded SDK allocation for its whole lifetime, including after
client disconnect. No release/reset/recovery operation is exposed here.
"""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import socket
import struct
import time

from .ddr_transport_contract import (
    Descriptor, INPUT_STRIDE, OUTPUT_STRIDE, pack_output, unpack_record, unpack_output,
)


class LabFault(RuntimeError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def expected_outputs(records, *, run_id, request_id, generation, history=()):
    """Independent batch layout reference; history is committed only on success."""
    ring = deque(history, maxlen=100)
    outputs = []
    for record in records:
        words, identity, emit = unpack_record(record)
        if ring:
            previous = ring[-1][1]
            if identity[0] != previous[0]:
                ring.clear()
            elif not (identity[1] == previous[1] + 1 and identity[2] > previous[2]
                      and identity[3] >= previous[3]):
                raise ValueError('Discontinuous identity')
        ring.append((words, identity))
        if len(ring) == 100 and emit:
            tensor = struct.pack('<4000I', *(row[0][c] for c in range(40) for row in ring))
            outputs.append(pack_output(tensor, ring[0][1], ring[-1][1], run_id=run_id,
                                       request_id=request_id, generation=generation,
                                       window_index=len(outputs)))
    return outputs, list(ring)


@dataclass(frozen=True)
class Slot:
    offset: int
    length: int
    input_offset: int
    output_offset: int
    input_bytes: int
    output_capacity: int


class RpcTransport:
    """Short-lived RPC connections; closing these NEVER closes the SDK owner."""
    def __init__(self, owner_file, timeout=45):
        self.owner_file = owner_file
        self.owner = json.loads(owner_file.read_text(encoding='utf-8'))
        self.timeout = timeout

    def call(self, op, **values):
        with socket.create_connection(('127.0.0.1', self.owner['port']), timeout=self.timeout) as s:
            s.settimeout(self.timeout)
            s.sendall((json.dumps(dict(token=self.owner['token'], op=op, **values))+'\n').encode())
            with s.makefile('rb') as f:
                line = f.readline(16 * 1024 * 1024)
        result = json.loads(line)
        if not result.get('ok'):
            raise LabFault(str(result))
        return result

    def status(self):
        return {int(k): v for k, v in self.call('status')['regs'].items()}

    def reserve(self, size):
        return self.call('reserve', size=size)['offset']

    def prepare(self, offset, data):
        self.call('prepare', offset=offset, hex=data.hex())

    def read(self, offset, size):
        return bytes.fromhex(self.call('read', offset=offset, size=size)['hex'])

    def registers(self, values):
        self.call('registers', values={str(k): v for k, v in values.items()})

    def command(self, value):
        self.call('command', value=value)


class UdwLab:
    GUARD = bytes([0xA7]) * 64
    POISON = 0xCD

    def __init__(self, transport, base, size, *, run_id, generation=1,
                 journal=lambda row: None, timeout=5):
        if (type(base) is not int or type(size) is not int or base < 0 or size <= 0
                or base % 64 or size % 64 or base + size > 2**32):
            raise ValueError('SDK allocation must fit aligned uint32 address space')
        self.t, self.base, self.size = transport, base, size
        self.run_id, self.generation = run_id, generation
        self.journal, self.timeout = journal, timeout
        self.failed = False
        self.started = False
        self.history = []
        self.request_id = 0
        self.token_next_derived = 1
        self.slots = []
        self.expected_regions = []
        self.guard('attach')
        initial = self.t.status()
        if any(initial.get(k, 0) for k in (4, 8, 12, 0x60, 0x64, 0x68, 0x6C, 0x54)):
            self.fail('Requires a fresh cold-session endpoint', initial)

    def fail(self, reason, status=None):
        self.failed = True
        self.journal(dict(event='quarantined', reason=reason, status=status,
                          owner_must_remain_alive=self.started, strict_visibility_proven=False))
        raise LabFault(reason)

    def guard(self, point):
        if self.failed:
            raise LabFault('Experiment already quarantined; recovery refused')
        try:
            s = self.t.status()
        except Exception as e:
            self.fail(f'{point}: transport status failed: {e}')
        self.journal(dict(event='status', point=point, regs=s))
        if s.get(0xC0) != 0x55445731 or s.get(0x8C) != 2:
            self.fail(f'{point}: UDW1/ABI2 mismatch', s)
        if s.get(0x70) or s.get(8) == 3 or s.get(12):
            self.fail(f'{point}: effective error or sticky fault', s)
        return s

    @staticmethod
    def u64(s, offset):
        return s[offset] | (s[offset+4] << 32)

    def issue(self, command, target_state):
        s = self.guard('before_command')
        if s[4] or s[8] != (2 if command == 2 else 0):
            self.fail('Unexpected command prestate', s)
        sequence = self.u64(s, 0x68) + 1
        if sequence >= 2**64:
            self.fail('Command sequence exhausted', s)
        rejects = s[0x54]
        # A failed RPC may have delivered START: retain even on transport error.
        if command in (1, 4):
            self.started = True
        self.t.command(command)
        deadline = time.monotonic() + self.timeout
        while True:
            s = self.guard('command_poll')
            issued, returned = self.u64(s, 0x68), self.u64(s, 0x60)
            if s[0x54] != rejects or issued != sequence or returned > sequence:
                self.fail('Rejected command or command sequence mismatch', s)
            if not s[4]:
                if returned != sequence or s[8] != target_state:
                    self.fail('Stale terminal sequence or wrong terminal state', s)
                return s
            if time.monotonic() >= deadline:
                self.fail('Command timeout: no recovery or buffer reuse permitted', s)

    def verify_retained(self):
        self.guard('before_retained_read')
        for offset, image in self.expected_regions:
            actual = self.t.read(offset, len(image))
            if actual != image:
                self.fail(f'Retained input/output/guard changed at offset {offset}')
        self.guard('after_retained_read')

    def submit(self, records, *, continuation=False, capacity_extra=0):
        """One fresh guarded slot per request. Never reuses even a failed slot."""
        try:
            return self._submit(list(records), continuation, capacity_extra)
        except Exception as e:
            if not self.failed:
                self.fail(f'Submit stopped: {type(e).__name__}: {e}')
            raise

    def _submit(self, records, continuation, extra):
        if extra < 0 or extra % 64:
            raise ValueError('Output spare capacity must be aligned')
        before = self.guard('before_next_request')
        self.verify_retained()
        if before[8] == 2:
            self.issue(2, 0)  # same previous shadow request ID; never a drain proof
        elif before[8] != 0:
            self.fail('Not ready for a request', before)
        request = self.request_id + 1
        expected, history = expected_outputs(records, run_id=self.run_id, request_id=request,
            generation=self.generation, history=self.history if continuation else ())
        data = b''.join(records)
        capacity = max(64, len(expected)*OUTPUT_STRIDE + extra)
        size = len(data) + capacity + 4*64
        offset = self.t.reserve(size)
        slot = Slot(offset, size, offset+64, offset+len(data)+192, len(data), capacity)
        if offset % 64 or offset < 0 or offset + size > self.size:
            raise ValueError('Reservation outside SDK chunk')
        if any(max(offset, s.offset) < min(offset+size, s.offset+s.length) for s in self.slots):
            raise ValueError('Reservation overlaps retained history')
        self.slots.append(slot)
        # input guard | input | guard | guard | output | guard
        image = self.GUARD + data + self.GUARD*2 + bytes([self.POISON])*capacity + self.GUARD
        d = Descriptor(self.base+slot.input_offset, len(data), self.base+slot.output_offset,
            capacity, len(records), len(expected), self.run_id, request, self.generation)
        registers = d.shadow_registers(address_limit_exclusive=self.base+self.size)
        self.journal(dict(event='prepared', request_id=request, continuation=continuation,
            descriptor=d.__dict__, slot=slot.__dict__, input_sha256=digest(data),
            expected_sha256=[digest(x) for x in expected]))
        self.t.prepare(offset, image)
        if self.t.read(offset, size) != image:
            self.fail('SDK prepare self-check failed')
        self.guard('after_prepare')
        self.t.registers(registers)
        shadow = self.guard('after_descriptor')
        if any(shadow.get(k) != v for k, v in registers.items()):
            self.fail('Shadow descriptor mismatch', shadow)
        terminal = self.issue(4 if continuation else 1, 2)
        checks = {0x40:len(records), 0x44:len(expected), 0x48:len(expected)*OUTPUT_STRIDE,
                  0x4C:len(records)*4, 0x50:len(expected)*252, 0x58:1}
        if any(terminal.get(k) != v for k, v in checks.items()):
            self.fail('Terminal counters/capability mismatch', terminal)
        self.guard('before_output_read')
        actual_image = self.t.read(offset, size)
        self.guard('after_output_read')
        wanted = bytearray(image)
        relative = slot.output_offset-offset
        for i, output in enumerate(expected):
            actual = actual_image[relative+i*OUTPUT_STRIDE:relative+(i+1)*OUTPUT_STRIDE]
            unpack_output(actual, run_id=self.run_id, request_id=request,
                          window_index=i, generation=self.generation)
            wanted[relative+i*OUTPUT_STRIDE:relative+(i+1)*OUTPUT_STRIDE] = output
        if actual_image != wanted:
            differences = [i for i,(a,b) in enumerate(zip(actual_image,wanted)) if a!=b]
            self.journal(dict(event='byte_mismatch', count=len(differences), first_offsets=differences[:32],
                              actual_sha256=digest(actual_image), expected_sha256=digest(wanted)))
            self.fail('Byte comparison failed (header/tensor/input/poison/guards)')
        self.guard('after_output_verify')
        beats = checks[0x4C]+checks[0x50]
        result = dict(request_id=request, windows=len(expected), records=len(records),
            status='host_readback_verified', strict_visibility_proven=False,
            descriptor=d.__dict__, slot=slot.__dict__, terminal=terminal,
            actual_sha256=digest(actual_image), output_sha256=[digest(x) for x in expected],
            physical_token_check='RTL checks each response token/op; no success-token register in ABI2',
            derived_not_observed_token_first=self.token_next_derived,
            derived_not_observed_token_next=self.token_next_derived+beats)
        self.token_next_derived += beats
        self.history, self.request_id = history, request
        self.expected_regions.append((offset, bytes(wanted)))
        self.verify_retained()
        self.journal(dict(event='request_observation', **result))
        return result, expected

    def predict(self, tensor, window_id):
        try:
            self.guard('before_npu')
            result = self.t.call('predict', hex=tensor.hex(), window_id=window_id)
            self.guard('after_npu')
            if result['window_id'] != window_id:
                self.fail('NPU returned wrong window identity')
            self.verify_retained()
            return result
        except Exception as e:
            if not self.failed:
                self.fail(f'NPU stopped: {e}')
            raise
