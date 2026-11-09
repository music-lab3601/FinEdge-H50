"""Serial PSW1 fixed-slot protocol candidate; no SDK imports or device opening.

The caller owns one SDK ``udma`` allocation and supplies synchronized memory
access adapters. AccessContract records concrete identities, not a proof of
cache coherence. Deployment must independently validate those identities and
the read-after-B/cache contract. This class never claims strict visibility.

Python 3.8 compatible. Pure WDM/WDO parsing below follows the current
ddr_transport_contract.py and selected_endpoints in udw_board_bridge.py without
importing their NumPy/SDK dependency chain or runtime builtin generic hints.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
import struct
import threading
import time

ROWS = 32
INPUT_STRIDE = 256
OUTPUT_STRIDE = 16128
TENSOR_BYTES = 16000
INPUT_OFFSET = 64
INPUT_CAPACITY = ROWS * INPUT_STRIDE
OUTPUT_OFFSET = INPUT_OFFSET + INPUT_CAPACITY + 64
OUTPUT_CAPACITY = ROWS * OUTPUT_STRIDE
ARENA_BYTES = OUTPUT_OFFSET + OUTPUT_CAPACITY + 64
CMA_BASE = 0x30100000
CMA_LIMIT = 0x38100000
U64 = 1 << 64
GUARD = bytes(range(64))
POISON = 0xA5


@dataclass(frozen=True)
class AccessContract:
    boot_id: str
    firmware_sha256: str
    library_sha256: str
    cache_contract_evidence: str
    region: str = "udma"
    cma_base: int = CMA_BASE
    cma_limit: int = CMA_LIMIT

    def validate(self):
        if not isinstance(self.boot_id, str) or not self.boot_id.strip():
            raise ValueError("Concrete boot identity required")
        for value in (self.firmware_sha256, self.library_sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Concrete lowercase SHA256 identity required")
        if not isinstance(self.cache_contract_evidence, str) or not self.cache_contract_evidence.strip():
            raise ValueError("Read-after-B/cache contract evidence identity required")
        if self.region != "udma" or type(self.cma_base) is not int or type(self.cma_limit) is not int or (self.cma_base, self.cma_limit) != (CMA_BASE, CMA_LIMIT):
            raise ValueError("Expected exact PSW1 udma CMA capability")


def selected_endpoints(payload, history=()):
    if not isinstance(payload, bytes) or not payload or len(payload) % INPUT_STRIDE or len(payload) > ROWS * INPUT_STRIDE:
        raise ValueError("Expected owned bytes containing 1..32 WDM records")
    ids = deque(history, maxlen=100)
    selected = []
    for offset in range(0, len(payload), INPUT_STRIDE):
        record = payload[offset:offset+INPUT_STRIDE]
        values = struct.unpack_from("<4QI", record)
        identity, emit = tuple(values[:4]), values[4]
        if emit not in (0, 1) or any(record[36:64]) or any(record[224:256]):
            raise ValueError("Invalid WDM reserved fields or emit")
        if not all(math.isfinite(value) for value in struct.unpack_from("<40f", record, 64)):
            raise ValueError("Nonfinite WDM input")
        if ids:
            old = ids[-1]
            if identity[0] != old[0]:
                ids.clear()
            elif not (identity[1] == old[1]+1 and identity[2] > old[2] and identity[3] >= old[3]):
                raise ValueError("Discontinuous WDM identity")
        ids.append(identity)
        if len(ids) == 100 and emit:
            selected.append((ids[0], identity))
    return selected, ids


def _output(data, run_id, request_id, index, generation, endpoints):
    if len(data) != OUTPUT_STRIDE:
        raise ValueError("Incomplete WDO image")
    fields = struct.unpack_from("<IIQQQ", data)
    identities = struct.unpack_from("<4Q4QIIQ", data, 32)
    if fields != (0x314F4457, 1, run_id, request_id, index) or identities[10] != generation:
        raise ValueError("Wrong WDO generation or request identity")
    if identities[8:10] != (TENSOR_BYTES, 0) or any(data[112:128]):
        raise ValueError("Invalid WDO reserved fields or length")
    if (tuple(identities[:4]), tuple(identities[4:8])) != endpoints:
        raise ValueError("Wrong WDO endpoint identity")
    return data[128:]


class PsWindowRuntime:
    """All injected calls are synchronous; all offsets are region-relative.

    read_reg(offset)/write_reg(offset,value) wrap the 0x400c0000 register bank.
    ps_write(offset,bytes) and ps_read(offset,size)->owned bytes wrap the one
    caller-owned PS arena. Caller must serialize them within its SDK operation
    context. quarantine(context) is notified once on any controller failure.
    There is deliberately no reset, retry, allocation or free API.
    """
    def __init__(self, *, read_reg, write_reg, ps_read, ps_write, arena_base,
                 arena_size, run_id, access_contract, quarantine=None,
                 timeout_seconds=5.0, monotonic=time.monotonic, sleep=time.sleep):
        if not isinstance(access_contract, AccessContract):
            raise TypeError("Explicit AccessContract required")
        access_contract.validate()
        if type(arena_base) is not int or type(arena_size) is not int or arena_base % 64 or arena_size < ARENA_BYTES or not CMA_BASE <= arena_base < arena_base + arena_size <= CMA_LIMIT:
            raise ValueError("Exclusive aligned PS arena outside verified CMA range")
        if type(run_id) is not int or not 0 < run_id < U64:
            raise ValueError("Positive uint64 run identity required")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Finite positive timeout required")
        self.read_reg, self.write_reg = read_reg, write_reg
        self.ps_read, self.ps_write = ps_read, ps_write
        self.arena_base, self.arena_size, self.run_id = arena_base, arena_size, run_id
        self.access_contract, self.quarantine_callback = access_contract, quarantine
        self.timeout_seconds, self.monotonic, self.sleep = timeout_seconds, monotonic, sleep
        self.state = "NEW"
        self.request_id = self.generation = self.sequence = 0
        self.identities = deque(maxlen=100)
        self.retained = {}
        self.failure = None
        self._cancelled = False
        self._lock = threading.Lock()

    def cancel(self):
        self._cancelled = True

    def _check_cancel(self):
        if self._cancelled:
            raise RuntimeError("PSW operation cancelled; no reuse authorized")

    def _fail(self, exc):
        self.state = "QUARANTINED"
        if self.failure is None:
            self.failure = exc  # traceback retains failed SDK/callback call frames
            self.retained["failure"] = exc
            if self.quarantine_callback is not None:
                try:
                    self.quarantine_callback(dict(error=exc, request_id=self.request_id,
                                                  generation=self.generation, sequence=self.sequence,
                                                  last_status=self.retained.get("last_status")))
                except BaseException as callback_error:
                    self.retained["quarantine_callback_failure"] = callback_error

    def _read(self, offset):
        value = self.read_reg(offset)
        if type(value) is not int or not 0 <= value < (1 << 32):
            raise ValueError("Invalid register value at "+hex(offset))
        return value

    def _status(self):
        self._check_cancel()
        offsets = (0xc0, 0x8c, 0x90, 0xc4, 0xc8, 4, 8, 12, 0x70, 0x54,
                   0x60, 0x64, 0x68, 0x6c, 0x40, 0x44, 0x48, 0x4c, 0x50, 0x58, 0x74, 0x78)
        values = {offset: self._read(offset) for offset in offsets}
        for offset in (0x70, 12, 8):  # sticky fault sampled after status/counts
            values[offset] = self._read(offset)
        self.retained["last_status"] = values  # one bounded diagnostic snapshot
        if any(values[k] != v for k, v in {0xc0: 0x50535731, 0x8c: 1, 0x90: 3, 0xc4: CMA_BASE, 0xc8: CMA_LIMIT}.items()):
            raise RuntimeError("PSW1 identity/capability/CMA mismatch")
        if values[4] not in (0, 1) or values[0x70] or values[12] or values[8] == 3 or values[0x54] or values[0x78]:
            raise RuntimeError("PSW1 sticky fault/error/rejection")
        return values

    def initialize(self):
        if not self._lock.acquire(False):
            raise RuntimeError("Concurrent PSW operation refused")
        try:
            if self.state != "NEW":
                raise RuntimeError("Initialization is single-use")
            status = self._status()
            clean = (4, 8, 12, 0x70, 0x54, 0x60, 0x64, 0x68, 0x6c, 0x74, 0x78,
                     0x40, 0x44, 0x48, 0x4c, 0x50, 0x58)
            if any(status[k] for k in clean):
                raise RuntimeError("PSW1 fresh startup required")
            self.state = "FREE"
        except BaseException as exc:
            self._fail(exc)
            raise
        finally:
            self._lock.release()

    @staticmethod
    def _seq(status, low):
        return status[low] | (status[low+4] << 32)

    def _idle(self, expected_state):
        status = self._status()
        if status[4] or status[8] != expected_state or self._seq(status, 0x68) != self.sequence or self._seq(status, 0x60) != self.sequence:
            raise RuntimeError("PSW1 unexpected busy/state/command ownership")
        return status

    def _issue(self, command, terminal_state):
        self._check_cancel()
        if self.sequence == U64-1:
            raise OverflowError("PSW command sequence exhausted")
        previous = self.sequence
        self.sequence += 1  # burned before ambiguous write failures
        self.write_reg(0, command)
        deadline = self.monotonic() + self.timeout_seconds
        while True:
            status = self._status()
            actual = self._seq(status, 0x68)
            terminal = self._seq(status, 0x60)
            if actual not in (previous, self.sequence) or terminal not in (previous, self.sequence):
                raise RuntimeError("PSW command/terminal sequence mismatch")
            if not status[4] and actual == terminal == self.sequence:
                if status[8] != terminal_state or status[0x74] != terminal_state:
                    raise RuntimeError("Wrong PSW terminal state")
                return status
            if self.monotonic() >= deadline:
                raise TimeoutError("PSW completion deadline expired; buffer retained")
            self.sleep(0.0005)

    def _bytes(self, offset, size):
        result = self.ps_read(offset, size)
        self.retained["last_memory_result"] = result
        if not isinstance(result, bytes) or len(result) != size:
            raise ValueError("PS read adapter must return exact independent owned bytes")
        return result

    def submit(self, payload, consume):
        if not self._lock.acquire(False):
            raise RuntimeError("Concurrent/reentrant PSW submission refused")
        try:
            if self.state != "FREE":
                raise RuntimeError("Fixed PS slot unavailable")
            self._check_cancel()
            selected, identities = selected_endpoints(payload, self.identities)
            if self.request_id == U64-1 or self.generation == U64-1:
                raise OverflowError("Request/generation exhausted")
            self._idle(0)
            self.state = "CPU_PREPARING"
            self.request_id += 1
            self.generation += 1
            records = len(payload)//INPUT_STRIDE
            image = GUARD + payload + bytes([POISON])*(INPUT_CAPACITY-len(payload)) + GUARD + bytes([POISON])*OUTPUT_CAPACITY + GUARD
            self.retained = dict(input=payload, prepared=image, predictions=[])
            self.ps_write(0, image)
            if self._bytes(0, ARENA_BYTES) != image:
                raise ValueError("PS preparation readback mismatch")
            shadow = dict(zip(range(0x10, 0x28, 4), (self.arena_base+INPUT_OFFSET, len(payload),
                          self.arena_base+OUTPUT_OFFSET, OUTPUT_CAPACITY, records, len(selected))))
            for offset, value in ((0x28, self.run_id), (0x30, self.request_id), (0x38, self.generation)):
                shadow[offset], shadow[offset+4] = value & 0xffffffff, value >> 32
            self._idle(0)
            for offset, value in shadow.items():
                self._check_cancel()
                self.write_reg(offset, value)
            if any(self._read(offset) != value for offset, value in shadow.items()):
                raise ValueError("PSW descriptor readback mismatch")
            self.state = "PL_INFLIGHT"
            terminal = self._issue(1 if self.request_id == 1 else 4, 2)
            counts = {0x40: records, 0x44: len(selected), 0x48: len(selected)*OUTPUT_STRIDE,
                      0x4c: records*4, 0x50: len(selected)*252, 0x58: 1}
            if any(terminal[k] != value for k, value in counts.items()):
                raise ValueError("PSW completion counts mismatch")
            self.state = "ARM_COPYING"
            actual = self._bytes(0, ARENA_BYTES)
            self.retained["completed"] = actual
            self._idle(2)
            end = OUTPUT_OFFSET + len(selected)*OUTPUT_STRIDE
            if actual[:OUTPUT_OFFSET] != image[:OUTPUT_OFFSET] or actual[end:] != image[end:]:
                raise ValueError("PS input/guard/unused-output corrupted")
            tensors = []
            for index, endpoints in enumerate(selected):
                start = OUTPUT_OFFSET+index*OUTPUT_STRIDE
                tensors.append(_output(actual[start:start+OUTPUT_STRIDE], self.run_id, self.request_id,
                                       index, self.generation, endpoints))
            self.retained["tensors"] = tensors
            self.state = "CONSUMING"
            predictions = self.retained["predictions"]
            for index, (tensor, endpoints) in enumerate(zip(tensors, selected)):
                self._idle(2)
                metadata = dict(run_id=self.run_id, request_id=self.request_id, generation=self.generation,
                                window_index=index, first=endpoints[0], last=endpoints[1],
                                tensor_sha256=hashlib.sha256(tensor).hexdigest())
                prediction = consume(tensor, metadata)
                predictions.append(prediction)
                self._idle(2)
            if self._bytes(0, ARENA_BYTES) != actual:
                raise ValueError("PS slot changed while callbacks consumed tensors")
            self._idle(2)
            self.state = "REARMING"
            rearmed = self._issue(2, 0)
            if any(rearmed[k] != value for k, value in counts.items()):
                raise ValueError("PSW counts changed during REARM")
            result = dict(run_id=self.run_id, request_id=self.request_id, generation=self.generation,
                          records=records, windows=len(selected), predictions=tuple(predictions),
                          endpoints=tuple(selected), terminal=terminal, input_sha256=hashlib.sha256(payload).hexdigest(),
                          scope="candidate protocol; actual hardware validation pending",
                          strict_visibility_proven=False, axi_b_complete=True,
                          sdk_owned_copy_checked=True, cache_contract_evidence=self.access_contract.cache_contract_evidence)
            self.identities = identities
            self.retained = {}
            self.state = "FREE"
            return result
        except BaseException as exc:
            self._fail(exc)
            raise
        finally:
            self._lock.release()
