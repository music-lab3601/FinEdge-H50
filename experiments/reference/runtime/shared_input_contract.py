"""PL-DDR shared-input ABI. No SDK imports, address casts or device access.

The existing PS-DDR BOOT cannot implement this ABI. A producer completion
means each written 64-byte block was subsequently read and compared in RTL;
it is NOT a vendor write-drain/coherence guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib

MAGIC = 0x3157435A  # ZCW1, distinct from PSW1/UDW1
ABI = 1
CAPABILITIES = 7  # payload-first, per-write read/compare, payload CRC32
TENSOR_BYTES = 16000
METADATA_BYTES = 128
SLOT_BYTES = 16384
INPUT_BYTES = 8192
REG_BASE = 0x400C0000
VALIDATION_FLAG = 1
LAYOUT_ID = 'fp32-1x40x100-channel-major-le-v1'


def crc32(data):
    return zlib.crc32(data) & 0xFFFFFFFF


@dataclass(frozen=True)
class Lease:
    boot_id: str
    allocation_id: str
    run_id: int
    request_id: int
    generation: int
    first: tuple
    last: tuple

    def __post_init__(self):
        if not self.boot_id or not self.allocation_id:
            raise ValueError('Concrete boot/allocation ownership required')
        for n in (self.run_id, self.request_id, self.generation):
            if type(n) is not int or not 0 < n < 2**64:
                raise ValueError('Positive uint64 lease identity required')
        for endpoint in (self.first, self.last):
            if len(endpoint) != 4 or any(type(n) is not int or not 0 <= n < 2**64 for n in endpoint):
                raise ValueError('Four uint64 endpoint fields required')


def validate_extent(base, size, minimum):
    if type(base) is not int or type(size) is not int or base % 4096 or size < minimum or not 0 <= base < base + size <= 2**32:
        raise ValueError('Owned 4096-aligned 32-bit PL-DDR extent required')


def unpack_metadata(raw, lease):
    if type(raw) is not bytes or len(raw) != METADATA_BYTES:
        raise ValueError('Exactly 128 independent metadata bytes required')
    fields = struct.unpack('<IIQQQ4Q4QIIQIIII', raw)
    expected = (MAGIC, ABI, lease.run_id, lease.request_id, 0,
                *lease.first, *lease.last, TENSOR_BYTES, 0, lease.generation)
    if fields[:16] != expected:
        raise ValueError('Stale/mismatched shared-window metadata')
    checksum, flags, verified_beats, reserved = fields[16:]
    if flags != VALIDATION_FLAG or verified_beats != 250 or reserved:
        raise ValueError('Hardware payload verification absent')
    return dict(crc32=checksum, hardware_read_compare=True,
                run_id=lease.run_id, request_id=lease.request_id,
                generation=lease.generation, first=list(lease.first), last=list(lease.last))


class Ownership:
    """Single-slot fail-closed producer/consumer ownership. No recovery API.

    The caller must retain SDK chunks for this object's entire lifetime,
    including on errors. A timeout never becomes permission to reuse.
    """
    def __init__(self):
        self.state = 'FREE'
        self.lease = None
        self.error = None

    def transition(self, before, after, lease):
        if self.state != before or (before != 'FREE' and self.lease != lease):
            raise RuntimeError('Shared-slot state or lease mismatch')
        if before == 'FREE':
            if not isinstance(lease, Lease):
                raise ValueError('Explicit immutable lease required')
            self.lease = lease
        self.state = after

    def begin(self, lease): self.transition('FREE', 'FPGA_WRITING', lease)
    def produced(self, lease): self.transition('FPGA_WRITING', 'FPGA_VERIFIED', lease)
    def consume(self, lease): self.transition('FPGA_VERIFIED', 'NPU_READING', lease)
    def completed(self, lease): self.transition('NPU_READING', 'NPU_COMPLETE', lease)
    def release(self, lease):
        self.transition('NPU_COMPLETE', 'FREE', lease)
        self.lease = None

    def quarantine(self, error):
        if self.error is None: self.error = str(error)
        self.state = 'QUARANTINED'
