"""B0 executable specification, NOT RTL, a DDR bridge, or a board driver.

Memory acceptance, deferred commit and a test-environment visibility receipt are
distinct. The current User_Ddr interface supplies NO such receipt. Queue/domain
scheduling models bounded dataflow; it does not verify electrical CDC behavior.
"""
from collections import deque
from dataclasses import asdict, dataclass
from enum import IntEnum
import struct

from .ddr_transport_contract import Descriptor, INPUT_STRIDE, OUTPUT_STRIDE, pack_output, unpack_record
from .window_core_model import Beat, RowIdentity, WindowCoreModel

U32_MAX = (1 << 32) - 1
U64_MAX = (1 << 64) - 1


def sat32(value):
    return min(U32_MAX, value + 1)


class Command(IntEnum):
    START = 1
    ACK = 2
    CLEAR_WINDOW = 4
    SNAPSHOT = 8


class Error(IntEnum):
    NONE = 0
    BAD_COMMAND = 1
    BUSY = 2
    BAD_DESCRIPTOR = 3
    BAD_STATE = 4
    REUSED_REQUEST = 5
    INPUT_PROTOCOL = 6
    OUTPUT_CAPACITY = 7
    DDR_TIMEOUT = 8
    VISIBILITY_UNPROVEN = 9
    RECOVERY_REQUIRED = 10
    SESSION_MISMATCH = 11


class State(IntEnum):
    IDLE = 0
    BUSY = 1
    OUTPUT_ISSUED = 2
    DONE_VISIBLE = 3
    ERROR_QUARANTINED = 4
    CLEARING = 5


@dataclass(frozen=True)
class Tag:
    run_id: int
    request_id: int
    generation: int


@dataclass(frozen=True)
class Receipt:
    tag: Tag
    writes: int
    nonce: int


@dataclass(frozen=True)
class ResetCertificate:
    nonce: int
    activity_sequence: int = 0
    reset_sequence: int = 0
    covered_lease_sequence: int = 0


class AbstractMemory:
    """Test-only memory provider. No implicit fence/reset capability by default.

    A read is one 64-byte request in flight. Accepted writes occupy a bounded
    queue and may commit much later. A receipt is issued only after a CLOSED
    batch has no pending writes. This is an explicit hypothetical provider
    contract, not an inference about AI_Mate or its internal pl_bvalid.
    """
    def __init__(self, size=1 << 20, *, visibility_capability=False, read_delay=2,
                 write_delay=20, max_pending_writes=8):
        if size <= 0 or read_delay < 1 or write_delay < 1 or max_pending_writes < 1:
            raise ValueError('Invalid abstract memory configuration')
        self.data = bytearray(size)
        self.visibility_capability = visibility_capability
        self.read_delay, self.write_delay = read_delay, write_delay
        self.max_pending_writes = max_pending_writes
        self.cycle = 0
        self.read_pending = self.read_response = None
        self.writes = deque()
        self.accepted = {}; self.committed = {}; self.closed = {}
        self.receipts = {}; self.reset_certificates = {}; self.nonce = 0
        self.owned = None
        self.activity_sequence = self.reset_sequence = self.lease_sequence = 0
        self.owned_lease_sequence = None
        self.peak_pending_writes = 0

    def _note_activity(self):
        """Any new provider activity revokes all unused reset certificates."""
        self.activity_sequence += 1
        self.reset_certificates.clear()

    def quiescent(self):
        return (self.read_pending is None and self.read_response is None and not self.writes
                and self.owned is None and self.owned_lease_sequence is None)

    def _range(self, address, count):
        if address < 0 or count < 0 or address + count > len(self.data):
            raise ValueError('Abstract memory range exceeded')

    def load(self, address, data):
        self._range(address, len(data))
        if self.owned is not None:
            raise RuntimeError('Host cannot mutate memory while a request owns buffers')
        self._note_activity()
        self.data[address:address+len(data)] = data

    def lease(self, tag):
        if not self.quiescent():
            raise RuntimeError('Memory provider already leased or has outstanding traffic')
        self._note_activity()
        self.lease_sequence += 1
        self.owned_lease_sequence = self.lease_sequence
        self.owned = tag

    def submit_read(self, tag, address):
        self._range(address, 64)
        if tag != self.owned or self.read_pending is not None or self.read_response is not None:
            return False
        self._note_activity()
        self.read_pending = (self.cycle + self.read_delay, tag, address)
        return True

    def submit_write(self, tag, address, data):
        self._range(address, 64)
        if len(data) != 64: raise ValueError('B0 accepts whole 64-byte beats only')
        if tag != self.owned or tag in self.closed or len(self.writes) >= self.max_pending_writes:
            return False
        self._note_activity()
        self.writes.append((self.cycle+self.write_delay, tag, address, bytes(data)))
        self.accepted[tag] = self.accepted.get(tag, 0) + 1
        self.peak_pending_writes = max(self.peak_pending_writes, len(self.writes))
        return True

    def tick(self, *, reads_enabled=True, commits_enabled=True):
        self.cycle += 1
        if self.read_pending is not None and reads_enabled:
            due, tag, address = self.read_pending
            if due <= self.cycle:
                self._note_activity()
                self.read_response = (tag, bytes(self.data[address:address+64]))
                self.read_pending = None
        if self.writes and commits_enabled and self.writes[0][0] <= self.cycle:
            self._note_activity()
            _, tag, address, data = self.writes.popleft()
            self.data[address:address+64] = data
            self.committed[tag] = self.committed.get(tag, 0) + 1

    def close_writes(self, tag, count):
        if tag != self.owned or count != self.accepted.get(tag, 0):
            raise RuntimeError('Closed watermark does not match accepted writes')
        self._note_activity()
        self.closed[tag] = count

    def poll_receipt(self, tag):
        count = self.closed.get(tag)
        if not self.visibility_capability or count is None or self.committed.get(tag, 0) != count:
            return None
        if any(w[1] == tag for w in self.writes): return None
        if tag not in self.receipts:
            self.nonce += 1
            self.receipts[tag] = Receipt(tag, count, self.nonce)
        return self.receipts[tag]

    def authentic(self, receipt):
        return self.receipts.get(receipt.tag) is receipt

    def release(self, tag):
        if tag != self.owned or self.read_pending is not None or self.read_response is not None:
            raise RuntimeError('Lease not drained')
        if self.accepted.get(tag, 0) and self.poll_receipt(tag) is None:
            raise RuntimeError('Cannot release without provider visibility receipt')
        self._note_activity()
        self.owned = None
        self.owned_lease_sequence = None

    def full_reset_for_test(self):
        """Hypothetical complete provider reset. NEVER maps to CLEAR_WINDOW."""
        self._note_activity()
        self.read_pending = self.read_response = None
        self.writes.clear(); self.owned = None
        self.owned_lease_sequence = None
        self.accepted.clear(); self.committed.clear(); self.closed.clear(); self.receipts.clear()
        self.reset_sequence += 1
        self.nonce += 1
        certificate = ResetCertificate(self.nonce,self.activity_sequence,self.reset_sequence,self.lease_sequence)
        self.reset_certificates[certificate.nonce] = certificate
        return certificate

    def validate_reset_certificate(self, certificate, minimum_lease_sequence):
        """Pure validation: reject stale proofs or non-quiescent state without mutation.

        A full provider reset covers ALL leases up to its monotonically increasing
        lease watermark. The watermark never rewinds, including across resets.
        Thus a newer reset while already idle can still cover a canceled lease.
        """
        if (not isinstance(certificate,ResetCertificate) or
                self.reset_certificates.get(certificate.nonce) is not certificate):
            raise RuntimeError('Missing, revoked or consumed model-provider reset certificate')
        if (certificate.activity_sequence != self.activity_sequence or
                certificate.reset_sequence != self.reset_sequence):
            raise RuntimeError('Reset certificate predates current provider activity/reset')
        if (type(minimum_lease_sequence) is not int or minimum_lease_sequence < 1 or
                not minimum_lease_sequence <= certificate.covered_lease_sequence <= self.lease_sequence):
            raise RuntimeError('Reset certificate does not cover the quarantined lease')
        if not self.quiescent():
            raise RuntimeError('Provider has outstanding reads, responses, writes or an active lease')

    def consume_reset_certificate(self, certificate, minimum_lease_sequence):
        self.validate_reset_certificate(certificate,minimum_lease_sequence)
        del self.reset_certificates[certificate.nonce]


class TransportB0:
    """Two virtual domains: memory every tick, gp/core every gp_period ticks.

    FIFO capacity is in full records/images, plus one partial read row, one
    active row, and one partial output frame. Python deques model atomic async
    FIFO transfers. They are not an implementation or simulation of CDC cells.
    """
    def __init__(self, memory, *, run_id=1, generation=1, fifo_depth=2,
                 gp_period=2, timeout_cycles=10000):
        if (type(run_id) is not int or type(generation) is not int or
                not 0 <= run_id <= U64_MAX or not 0 <= generation <= U64_MAX):
            raise ValueError('Invalid session identity')
        if fifo_depth < 1 or gp_period < 1 or timeout_cycles < 1: raise ValueError('Invalid limits')
        self.memory = memory; self.run_id = run_id; self.generation = generation
        self.fifo_depth = fifo_depth; self.gp_period = gp_period; self.timeout_cycles = timeout_cycles
        self.core = WindowCoreModel()
        self.state = State.IDLE; self.error = Error.NONE; self.command_error = Error.NONE
        self.command_rejections = 0; self.last_request_id = -1; self.descriptor = None; self.tag = None
        self.request_lease_sequence = None
        self.snapshot_seq = 0; self.snapshot_pending = False; self.published = {}
        self.cycle = 0; self.clear_remaining = self.clear_memory_remaining = 0; self.ignored_receipts = 0
        self._clear_buffers(); self._reset_counters(); self._publish()

    def _clear_buffers(self):
        self.input_fifo = deque(); self.output_fifo = deque()
        self.read_partial = bytearray(); self.active_row = None; self.input_word = 0
        self.output_words = []; self.output_first = None; self.output_beat = 0
        self.requested_bytes = 0; self.idle_cycles = 0
        self.peak_input_fifo = self.peak_output_fifo = 0

    def _reset_counters(self):
        self.records_done = self.windows_done = self.read_beats = self.write_beats = 0

    def _begin_clear(self):
        # Separate abstract acknowledgments from both virtual domains. Actual
        # synchronizers/toggle mailboxes must be independently implemented/tested.
        self.state = State.CLEARING
        self.clear_remaining = self.clear_memory_remaining = 2

    @property
    def buffers_reusable(self):
        return self.memory.owned is None and self.state in (State.IDLE, State.CLEARING)

    def _reject(self, error):
        self.command_error = error; self.command_rejections = sat32(self.command_rejections)
        return False

    def command(self, value, descriptor=None):
        # Simultaneous command bits are invalid, including START|ACK.
        if type(value) not in (int, Command) or value not in tuple(int(c) for c in Command):
            return self._reject(Error.BAD_COMMAND)
        if value == Command.SNAPSHOT:
            if self.snapshot_seq == U64_MAX: return self._reject(Error.BAD_STATE)
            if self.snapshot_pending: return self._reject(Error.BUSY)
            self.snapshot_pending = True; return True
        if value == Command.START:
            if self.state != State.IDLE: return self._reject(Error.BUSY)
            if not isinstance(descriptor, Descriptor): return self._reject(Error.BAD_DESCRIPTOR)
            try: descriptor.validate(address_limit_exclusive=len(self.memory.data))
            except ValueError: return self._reject(Error.BAD_DESCRIPTOR)
            if descriptor.run_id != self.run_id or descriptor.generation != self.generation:
                return self._reject(Error.SESSION_MISMATCH)
            if descriptor.request_id <= self.last_request_id: return self._reject(Error.REUSED_REQUEST)
            # Frozen dataclass represents atomic shadow-register capture at START.
            tag = Tag(descriptor.run_id, descriptor.request_id, descriptor.generation)
            if self.memory.owned is not None: return self._reject(Error.BUSY)
            self.memory.lease(tag)
            self.request_lease_sequence = self.memory.owned_lease_sequence
            self.descriptor = descriptor; self.tag = tag; self.last_request_id = descriptor.request_id
            self._clear_buffers(); self._reset_counters()
            self.error = self.command_error = Error.NONE
            self.state = State.BUSY
            return True
        if value == Command.ACK:
            if self.state != State.DONE_VISIBLE: return self._reject(Error.BAD_STATE)
            self.memory.release(self.tag)
            self.state = State.IDLE; self.command_error = Error.NONE
            return True
        if self.state != State.IDLE: return self._reject(Error.BAD_STATE)
        self._begin_clear()
        self.command_error = Error.NONE
        return True

    def _fail(self, error):
        self.state = State.ERROR_QUARANTINED; self.error = error
        peaks = self.peak_input_fifo, self.peak_output_fifo
        self.core.clear(); self._clear_buffers()
        self.peak_input_fifo, self.peak_output_fifo = peaks
        # Deliberately do not release memory or cancel already accepted writes.

    def external_reset(self):
        if self.state in (State.BUSY, State.OUTPUT_ISSUED, State.DONE_VISIBLE, State.ERROR_QUARANTINED):
            self._fail(Error.RECOVERY_REQUIRED)
        else:
            self.core.clear(); self._clear_buffers(); self._begin_clear()

    def recover_for_test(self, certificate, new_generation):
        if self.state != State.ERROR_QUARANTINED: raise RuntimeError('No quarantined request')
        if type(new_generation) is not int or not self.generation < new_generation <= U64_MAX:
            raise ValueError('Recovery generation must increase without wrap')
        # All preconditions precede certificate consumption or controller changes.
        # Consumption revalidates atomically in this single-threaded model.
        self.memory.consume_reset_certificate(certificate,self.request_lease_sequence)
        self.generation = new_generation; self.last_request_id = -1
        self.core.clear(); self._clear_buffers(); self._reset_counters()
        self.tag = self.descriptor = None
        self.request_lease_sequence = None
        self.error = self.command_error = Error.NONE
        self._begin_clear()

    def accept_receipt(self, receipt):
        if (not isinstance(receipt, Receipt) or self.state != State.OUTPUT_ISSUED or receipt.tag != self.tag or
                receipt.writes != self.write_beats or not self.memory.authentic(receipt)):
            self.ignored_receipts = sat32(self.ignored_receipts); return False
        self.state = State.DONE_VISIBLE
        return True

    def result_bytes(self):
        if self.state != State.DONE_VISIBLE: raise RuntimeError('Output not visible; lease cannot be consumed/reused')
        d = self.descriptor
        return bytes(self.memory.data[d.output_base:d.output_base+self.windows_done*OUTPUT_STRIDE])

    def _publish(self):
        self.published = dict(state=int(self.state), error=int(self.error), command_error=int(self.command_error),
            records_done=self.records_done, windows_done=self.windows_done, read_beats=self.read_beats,
            write_beats=self.write_beats, generation=self.generation, request_id=self.tag.request_id if self.tag else None,
            run_id=self.run_id, buffers_reusable=self.buffers_reusable, snapshot_seq=self.snapshot_seq,
            clear_core_ack=self.clear_remaining==0,clear_memory_ack=self.clear_memory_remaining==0,
            command_rejections=self.command_rejections, ignored_receipts=self.ignored_receipts)

    def snapshot(self):
        return dict(self.published)

    def tick(self, *, read_ready=True, write_ready=True, commits_enabled=True, core_ready=True):
        self.cycle += 1
        self.memory.tick(reads_enabled=read_ready, commits_enabled=commits_enabled)
        gp_edge = self.cycle % self.gp_period == 0
        progress = False
        if self.state == State.CLEARING:
            self.clear_memory_remaining = max(0,self.clear_memory_remaining-1)
            if gp_edge:
                self.core.clear(); self.clear_remaining = max(0,self.clear_remaining-1)
            if self.clear_remaining == self.clear_memory_remaining == 0:
                self._clear_buffers(); self._reset_counters(); self.tag = self.descriptor = None
                self.state = State.IDLE
        if self.state == State.BUSY:
            d = self.descriptor
            response = self.memory.read_response
            if response is not None and len(self.input_fifo) < self.fifo_depth:
                tag, data = response
                if tag != self.tag:
                    self._fail(Error.INPUT_PROTOCOL)
                else:
                    self.memory.read_response = None; self.read_partial.extend(data)
                    self.read_beats = sat32(self.read_beats); progress = True
                    if len(self.read_partial) == INPUT_STRIDE:
                        self.input_fifo.append(bytes(self.read_partial)); self.read_partial.clear()
                        self.peak_input_fifo = max(self.peak_input_fifo, len(self.input_fifo))
            if self.state == State.BUSY and read_ready and self.requested_bytes < d.input_bytes and len(self.input_fifo) < self.fifo_depth:
                if self.memory.submit_read(self.tag, d.input_base+self.requested_bytes):
                    self.requested_bytes += 64; progress = True
            if self.state == State.BUSY and write_ready and self.output_fifo:
                image = self.output_fifo[0]
                address = d.output_base + (self.windows_done-len(self.output_fifo))*OUTPUT_STRIDE + self.output_beat*64
                beat = image[self.output_beat*64:(self.output_beat+1)*64]
                if self.memory.submit_write(self.tag, address, beat):
                    self.write_beats = sat32(self.write_beats); self.output_beat += 1; progress = True
                    if self.output_beat == OUTPUT_STRIDE//64:
                        self.output_fifo.popleft(); self.output_beat = 0
            if self.state == State.BUSY and gp_edge and core_ready:
                if self.active_row is None and self.input_fifo and self.core.phase == 'input' and not self.core.stage:
                    try:
                        words, ids, emit = unpack_record(self.input_fifo.popleft())
                        identity = RowIdentity(*ids)
                    except ValueError:
                        self._fail(Error.INPUT_PROTOCOL)
                    else:
                        prev = self.core.previous
                        continuous = prev is not None and prev.epoch == identity.epoch
                        if emit and continuous and self.core.count >= 99 and (
                                self.windows_done >= d.max_windows or (self.windows_done+1)*OUTPUT_STRIDE > d.output_capacity):
                            self._fail(Error.OUTPUT_CAPACITY)
                        else: self.active_row = (words, identity, emit); self.input_word = 0
                if self.state == State.BUSY:
                    source = None
                    if self.active_row is not None:
                        words, identity, emit = self.active_row; i = self.input_word
                        source = Beat(words[i], i==0, i==39, identity, emit)
                    ready = len(self.output_fifo) < self.fifo_depth
                    before_phase, before_cursor = self.core.phase, self.core.cursor
                    signal = self.core.step(source, ready)
                    if source is not None and signal['s_ready']:
                        self.input_word += 1; progress = True
                        if self.input_word == 40: self.active_row = None; self.input_word = 0
                    if before_phase == 'commit': progress = True
                    if before_phase == 'commit' and before_cursor == 39:
                        self.records_done = sat32(self.records_done)
                    if self.core.errors:
                        self._fail(Error.INPUT_PROTOCOL)
                    elif signal['m_valid'] and ready:
                        progress = True
                        if signal['m_sop']: self.output_first = signal['m_first']
                        self.output_words.append(signal['m_data'])
                        if signal['m_last']:
                            first = tuple(asdict(self.output_first).values())
                            last = tuple(asdict(signal['m_last_identity']).values())
                            image = pack_output(struct.pack('<4000I', *self.output_words), first, last,
                                run_id=d.run_id,request_id=d.request_id,window_index=self.windows_done,generation=d.generation)
                            self.output_fifo.append(image); self.output_words = []
                            self.windows_done = sat32(self.windows_done)
                            self.peak_output_fifo = max(self.peak_output_fifo,len(self.output_fifo))
            if (self.state == State.BUSY and self.records_done == d.record_count and
                    self.core.phase == 'input' and not self.core.stage and not self.input_fifo and
                    self.active_row is None and not self.output_fifo and not self.output_words and
                    self.memory.read_pending is None and self.memory.read_response is None):
                self.memory.close_writes(self.tag,self.write_beats)
                # A zero-output request has no outstanding output writes to fence.
                self.state = State.DONE_VISIBLE if self.write_beats == 0 else State.OUTPUT_ISSUED
                progress = True
        if self.state == State.OUTPUT_ISSUED:
            receipt = self.memory.poll_receipt(self.tag)
            if receipt is not None and self.accept_receipt(receipt): progress = True
        if self.state in (State.BUSY, State.OUTPUT_ISSUED):
            self.idle_cycles = 0 if progress else self.idle_cycles + 1
            if self.idle_cycles >= self.timeout_cycles:
                self._fail(Error.VISIBILITY_UNPROVEN if self.state==State.OUTPUT_ISSUED and
                    not self.memory.visibility_capability else Error.DDR_TIMEOUT)
        if gp_edge and self.snapshot_pending:
            self.snapshot_seq += 1
            self.snapshot_pending = False; self._publish()
