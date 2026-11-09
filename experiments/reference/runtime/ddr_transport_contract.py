"""Draft WDM1 host codec/descriptor checks, not a driver for the current WIN1 BOOT.

All addresses below are byte addresses in an SDK-managed region, not CPU pointers.
The caller must supply an experimentally/vendor-verified exclusive address limit.
"""
from dataclasses import dataclass
import struct

INPUT_STRIDE = 256
TENSOR_BYTES = 16000
OUTPUT_HEADER_BYTES = 128
OUTPUT_STRIDE = OUTPUT_HEADER_BYTES + TENSOR_BYTES
VERSION = 0x57444D31
OUTPUT_MAGIC = 0x314F4457  # little-endian bytes WDO1
U32 = 1 << 32
U64 = 1 << 64


def _uint(value: int, bits: int, name: str) -> None:
    if type(value) is not int or not 0 <= value < 1 << bits:
        raise ValueError(f'{name} must be uint{bits}')


def pack_record(words: tuple[int, ...], identity: tuple[int, int, int, int], emit: bool) -> bytes:
    if len(words) != 40 or len(identity) != 4 or type(emit) is not bool:
        raise ValueError('Expected 40 words, four identity fields and bool emit')
    for value in words:
        _uint(value, 32, 'word')
    for value in identity:
        _uint(value, 64, 'identity')
    result = bytearray(INPUT_STRIDE)
    struct.pack_into('<4QI', result, 0, *identity, int(emit))
    struct.pack_into('<40I', result, 64, *words)
    return bytes(result)


def unpack_record(data: bytes) -> tuple[tuple[int, ...], tuple[int, ...], bool]:
    if len(data) != INPUT_STRIDE:
        raise ValueError('Record must have exactly 256 bytes')
    *identity, emit = struct.unpack_from('<4QI', data)
    if emit not in (0, 1) or any(data[36:64]) or any(data[224:256]):
        raise ValueError('Nonzero reserved fields or invalid emit')
    return struct.unpack_from('<40I', data, 64), tuple(identity), bool(emit)


def pack_output(tensor: bytes, first: tuple[int, ...], last: tuple[int, ...], *,
                run_id: int, request_id: int, window_index: int, generation: int) -> bytes:
    """Reference image for future RTL comparison, no completion/visibility assertion."""
    if len(tensor) != TENSOR_BYTES or len(first) != 4 or len(last) != 4:
        raise ValueError('Expected 16000 tensor bytes and two four-field identities')
    for value in (*first, *last, run_id, request_id, window_index, generation):
        _uint(value, 64, 'output field')
    if first[0] != last[0] or last[1] - first[1] != 99 or first[2] >= last[2] or first[3] > last[3]:
        raise ValueError('Output endpoints must describe one contiguous 100-row window')
    raw = bytearray(OUTPUT_STRIDE)
    struct.pack_into('<IIQQQ', raw, 0, OUTPUT_MAGIC, 1, run_id, request_id, window_index)
    struct.pack_into('<4Q4QIIQ', raw, 32, *first, *last, TENSOR_BYTES, 0, generation)
    raw[OUTPUT_HEADER_BYTES:] = tensor
    return bytes(raw)


def unpack_output(data: bytes, *, run_id: int, request_id: int, window_index: int,
                  generation: int) -> tuple[bytes, tuple[int, ...], tuple[int, ...]]:
    """Checks an already-read image. The caller must establish its completion separately."""
    if len(data) != OUTPUT_STRIDE:
        raise ValueError('Output must have exactly 16128 bytes')
    magic, version, actual_run, actual_request, actual_index = struct.unpack_from('<IIQQQ', data)
    *ids, length, flags, actual_generation = struct.unpack_from('<4Q4QIIQ', data, 32)
    first, last = tuple(ids[:4]), tuple(ids[4:])
    if (magic, version, actual_run, actual_request, actual_index, actual_generation) != (
            OUTPUT_MAGIC, 1, run_id, request_id, window_index, generation):
        raise ValueError('Output version or request identity mismatch')
    if length != TENSOR_BYTES or flags or any(data[112:128]):
        raise ValueError('Output length, flags or reserved bytes invalid')
    tensor = data[OUTPUT_HEADER_BYTES:]
    pack_output(tensor, first, last, run_id=run_id, request_id=request_id,
                window_index=window_index, generation=generation)
    return tensor, first, last


@dataclass(frozen=True)
class Descriptor:
    input_base: int
    input_bytes: int
    output_base: int
    output_capacity: int
    record_count: int
    max_windows: int
    run_id: int
    request_id: int
    generation: int

    def validate(self, *, address_limit_exclusive: int) -> None:
        for name in ('input_base', 'input_bytes', 'output_base', 'output_capacity',
                     'record_count', 'max_windows'):
            _uint(getattr(self, name), 32, name)
        for name in ('run_id', 'request_id', 'generation'):
            _uint(getattr(self, name), 64, name)
        if type(address_limit_exclusive) is not int or not 0 < address_limit_exclusive <= U32:
            raise ValueError('An explicit verified address limit is required')
        if self.input_base % 64 or self.output_base % 64 or self.output_capacity % 64:
            raise ValueError('Bases and output capacity must be 64-byte aligned')
        if self.input_bytes != self.record_count * INPUT_STRIDE:
            raise ValueError('Input length must equal record_count * 256 without truncation')
        if self.max_windows > self.record_count:
            raise ValueError('At most one snapshot per incoming record')
        if self.output_capacity < self.max_windows * OUTPUT_STRIDE:
            raise ValueError('Insufficient output space for declared window limit')
        if not self.record_count and (self.max_windows or self.input_bytes):
            raise ValueError('Zero-record request cannot produce windows')
        ranges = ((self.input_base, self.input_bytes), (self.output_base, self.output_capacity))
        for begin, count in ranges:
            if begin >= address_limit_exclusive or begin + count > address_limit_exclusive:
                raise ValueError('Buffer exceeds the verified address range')
        if self.input_bytes and self.output_capacity:
            if max(self.input_base, self.output_base) < min(
                    self.input_base + self.input_bytes, self.output_base + self.output_capacity):
                raise ValueError('Input/output buffers overlap')

    def shadow_registers(self, *, address_limit_exclusive: int) -> dict[int, int]:
        """Validated shadow values only; never issues START or touches any device."""
        self.validate(address_limit_exclusive=address_limit_exclusive)
        regs = dict(zip(range(0x10, 0x28, 4), (
            self.input_base, self.input_bytes, self.output_base, self.output_capacity,
            self.record_count, self.max_windows)))
        for offset, value in ((0x28, self.run_id), (0x30, self.request_id), (0x38, self.generation)):
            regs[offset], regs[offset + 4] = value & (U32 - 1), value >> 32
        return regs
