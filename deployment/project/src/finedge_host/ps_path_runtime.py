"""Opt-in P0/P1a candidates. Original strict runtime classes stay unchanged.

P1a changes Python dispatch/diagnostic storage and retains exactly the same
ordered register operations, three synchronized full reads and ownership gates.
"""
from __future__ import annotations

from datetime import timedelta
import hashlib
import io
import threading

import numpy as np

from .ps_npu_runtime import BoardNpuRuntime, TENSOR_BYTES
from .ps_window_runtime import (PsWindowRuntime, selected_endpoints, _output, U64,
    GUARD, POISON, ARENA_BYTES, INPUT_STRIDE, INPUT_OFFSET, INPUT_CAPACITY,
    OUTPUT_OFFSET, OUTPUT_CAPACITY, OUTPUT_STRIDE, CMA_BASE, CMA_LIMIT)
from .ps_path_profile import Measurements, Intervals, FixedDiagnostics, BoundedDiagnostics

STATUS_OFFSETS = (0xc0, 0x8c, 0x90, 0xc4, 0xc8, 4, 8, 12, 0x70, 0x54,
    0x60, 0x64, 0x68, 0x6c, 0x40, 0x44, 0x48, 0x4c, 0x50, 0x58, 0x74, 0x78)
STICKY_OFFSETS = (0x70, 12, 8)
CAPABILITIES = ((0xc0, 0x50535731), (0x8c, 1), (0x90, 3), (0xc4, CMA_BASE), (0xc8, CMA_LIMIT))


class ProfiledNpuRuntime(BoardNpuRuntime):
    def __init__(self, *, config, diagnostic=None, stop=None, clock=None):
        import time
        clock = time.perf_counter_ns if clock is None else clock
        self.measurements = Measurements(config, clock)
        self.fast_control = config['name'] == 'control-p1a-v1'
        if diagnostic is None: diagnostic = FixedDiagnostics() if self.fast_control else BoundedDiagnostics()
        super().__init__(diagnostic=diagnostic, stop=stop, clock=clock)
        self._sites = {}
        self.original_dispatch = not self.fast_control and not self.measurements.wrapper_enabled
        if self.original_dispatch:
            # Phase-only P0 uses the exact original per-SDK dispatch. Sampling
            # wrappers themselves exceeded the local intrusion budget.
            self.safe_call = BoardNpuRuntime.safe_call.__get__(self)
            self._emit = BoardNpuRuntime._emit.__get__(self)

    def _diagnostic_total(self, stage):
        if isinstance(self.diagnostic, FixedDiagnostics):
            index=self.diagnostic.indices.get(stage)
            return 0 if index is None else self.diagnostic.values[index][1]
        return self.diagnostic.stages.get(stage,{}).get('total_ns',0)

    def _emit(self, event):
        if event.get('kind') == 'api' and (self.measurements.in_forward or event['stage']=='npu.forward'):
            self.measurements.api(event['stage'], event['ns'])
        super()._emit(event)

    def _site(self, stage):
        value = self._sites.get(stage)
        if value is None:
            if len(self._sites) >= 128: raise RuntimeError('Bounded call-site table exhausted')
            value = (stage + ':before', stage + ':sdk_before', stage + ':sdk_after', stage + ':before_release')
            self._sites[stage] = value
        return value

    def _fast_call(self, stage, operation, args=(), *, plain_register=False):
        active = self.active
        if active is None or active['thread'] != threading.get_ident():
            # Standalone/concurrent admission stays in the original operation.
            return super().safe_call(stage, lambda: operation(*args))
        before, sdk_before, sdk_after, release = self._site(stage)
        self.stop.checkpoint(before)
        self.stop.checkpoint(sdk_before)
        begin = self.clock()
        value = operation(*args)  # local strong reference exists immediately
        elapsed = self.clock() - begin
        if plain_register and type(value) is int:
            self.measurements.scalar_returns_not_retained += 1
        else:
            self.hold(value)  # unknown/SDK objects held BEFORE diagnostics/cancel
        if stage == 'ps_arena.malloc': self.device_allocation_returns += 1
        elif stage == 'input.malloc': self.host_allocation_returns += 1
        if self.measurements.in_forward or stage=='npu.forward':self.measurements.api(stage, elapsed)
        diagnostic = self.diagnostic
        if isinstance(diagnostic, FixedDiagnostics):
            diagnostic.record_api(diagnostic.site(stage), elapsed)
        elif diagnostic is not None:
            diagnostic(dict(kind='api', stage=stage, ns=elapsed))
        self.stop.checkpoint(sdk_after)
        self.stop.checkpoint(release)
        return value, elapsed

    def safe_call(self, stage, operation):
        if self.active is not None and self.active['thread'] != threading.get_ident():
            return super().safe_call(stage, operation)
        if not self.fast_control and not self.measurements.wrapper_enabled:
            try:return super().safe_call(stage,operation)
            except BaseException as exc:
                self.measurements.fail(stage,exc);self.retain_failure(exc);raise
        token = self.measurements.sample_begin(stage) if self.measurements.wrapper_enabled else None
        successful = False
        sdk_ns = None
        try:
            if self.fast_control: result = self._fast_call(stage, operation)
            else: result = super().safe_call(stage, operation)
            sdk_ns = result[1]
            successful = True
        except BaseException as exc:
            self.measurements.fail(stage, exc)
            self.retain_failure(exc)
            raise
        finally:
            self.measurements.sample_end(token,successful,sdk_ns)
        return result

    def register_read(self, stage, operation, args):
        if self.active is not None and self.active['thread'] != threading.get_ident():
            return super().safe_call(stage, lambda: operation(*args))[0]
        if not self.fast_control:
            value, _ = self.safe_call(stage, lambda: operation(*args))
            return int(value)
        token = self.measurements.sample_begin(stage) if self.measurements.wrapper_enabled else None
        successful = False
        sdk_ns = None
        try:
            value, sdk_ns = self._fast_call(stage, operation, args, plain_register=True)
            result = int(value)
            successful = True
            return result
        except BaseException as exc:
            self.measurements.fail(stage, exc)
            self.retain_failure(exc)
            raise
        finally:
            self.measurements.sample_end(token,successful,sdk_ns)

    def predict(self, tensorbytes):
        if self.active is not None and self.active['thread'] != threading.get_ident():
            return super().predict(tensorbytes)
        timing = Intervals(self.measurements, 'input_check')
        try:
            if not self.initialized: raise RuntimeError('Initialized runtime required')
            if (not isinstance(tensorbytes, bytes) or len(tensorbytes) != TENSOR_BYTES
                    or not np.isfinite(np.frombuffer(tensorbytes, dtype='<f4')).all()):
                raise ValueError('Finite little-endian float32[1,40,100] bytes required')
            timing.mark('scope_hash_view')
            with self.operation('predict', isolated=True):
                parent = self.active['parent']
                while parent is not None:
                    if parent['prediction_started']: raise RuntimeError('Reentrant prediction is prohibited')
                    parent = parent['parent']
                self.active['prediction_started'] = True
                begin, stages = self.clock(), {}
                tensor_hash = hashlib.sha256(tensorbytes).hexdigest()
                array = self.hold(np.frombuffer(tensorbytes, dtype=np.uint8))
                timing.mark('write')
                _, stages['input_write'] = self.safe_call('input.write', lambda: self.input.write(0, array, len(tensorbytes)))
                timing.mark('cast')
                adapted, stages['input_cast'] = self.safe_call('input.cast', lambda: self.host.Cast(
                    self.input, self.software_type, self.input_type, self.converter))
                self.active['npu_may_be_reading'] = True
                self.npu_calls += 1
                timing.mark('forward_including_callback')
                self.measurements.in_forward = True
                self.measurements.forward_child_ns = 0
                child_before = (self._diagnostic_total('hardop.not_ready')+self._diagnostic_total('hardop.wait')) if self.original_dispatch else 0
                try:
                    outputs, stages['forward_including_hardop_wait'] = self.safe_call('npu.forward', lambda: self.session.forward([adapted]))
                finally:
                    self.measurements.in_forward = False
                if self.original_dispatch:
                    self.measurements.forward_child_ns = self._diagnostic_total('hardop.not_ready')+self._diagnostic_total('hardop.wait')-child_before
                    self.measurements.api('npu.forward',stages['forward_including_hardop_wait'])
                timing.mark('completion_and_wait')
                if tuple(self.active['hardops']) != self.hardop_ids or len(outputs) != 1:
                    raise ValueError('Missing HardOp completion or wrong output count')
                self.safe_call('output.not_ready', lambda: outputs[0].setReady(False))
                ready, stages['output_wait'] = self.safe_call('output.wait', lambda: outputs[0].waitForReady(
                    timedelta(seconds=10), timedelta(milliseconds=1)))
                if not ready: raise TimeoutError('NPU final output timeout')
                timing.mark('output_copy')
                stream = self.hold(io.BytesIO())
                _, stages['output_copy'] = self.safe_call('output.dump', lambda: outputs[0].dump(stream, 'SFB'))
                timing.mark('decode_publish_release')
                raw = stream.getvalue()
                if len(raw) != 12: raise ValueError('Expected exactly three float32 logits')
                logits = np.frombuffer(raw, dtype='<f4').copy()
                if not np.isfinite(logits).all(): raise ValueError('Non-finite NPU logits')
                self.active['output_copied'] = True
                result = dict(logits=logits.tolist(), class_index=int(logits.argmax()), tensor_sha256=tensor_hash,
                    stages_ns=stages, whole_predict_ns=self.clock()-begin, backend='board_socket_npu', hardops=list(self.active['hardops']))
                self._emit(dict(kind='prediction_complete', result=result))
                self.stop.checkpoint('predict:completed')
                self.completed_invocations += 1
            partition = timing.finish()
            if partition['enabled']:
                stages.update({'path_' + k: v for k,v in partition['phases_ns'].items()})
                stages['path_total'] = partition['total_ns']
            return result
        except BaseException as exc:
            self.measurements.fail('predict.' + timing.phase, exc, partial_phases_ns=dict(timing.parts))
            self.retain_failure(exc)
            raise

    def snapshot(self):
        result = super().snapshot()
        result['path_measurement'] = self.measurements.snapshot()
        return result


class RegisterAdapter:
    """Same register object, same addresses/order; no block read or new domain."""
    def __init__(self, owner):
        self.owner = owner
        self.read = owner.axi_registers.read
        self.write = owner.axi_registers.write
        self.read_sites = tuple(('psw.reg.read.' + hex(k), (0x400c0000+k, False)) for k in range(0, 0xcc, 4))
        self.write_sites = tuple('psw.reg.write.' + hex(k) for k in range(0, 0xcc, 4))

    def read_reg(self, offset):
        if type(offset) is not int or offset % 4 or not 0 <= offset < 0xcc:
            raise ValueError('Exact aligned frozen register offset required')
        stage, args = self.read_sites[offset//4]
        return self.owner.register_read(stage, self.read, args)

    def write_reg(self, offset, value):
        if type(offset) is not int or offset % 4 or not 0 <= offset < 0xcc:
            raise ValueError('Exact aligned frozen register offset required')
        self.owner.safe_call(self.write_sites[offset//4], lambda: self.write(0x400c0000+offset, value, False))


class ProfiledWindowRuntime(PsWindowRuntime):
    def __init__(self, *, measurements, **kwargs):
        self.measurements = measurements
        super().__init__(**kwargs)

    def _status(self):
        m = self.measurements
        begin = m.clock() if m.enabled else 0
        m.status_scans += 1
        try:
            if m.config['name'] == 'strict-p0-v1': return super()._status()
            self._check_cancel()
            values = {offset:self._read(offset) for offset in STATUS_OFFSETS}
            for offset in STICKY_OFFSETS: values[offset] = self._read(offset)
            self.retained['last_status'] = values
            if any(values[k] != v for k,v in CAPABILITIES): raise RuntimeError('PSW1 identity/capability/CMA mismatch')
            if values[4] not in (0,1) or values[0x70] or values[12] or values[8] == 3 or values[0x54] or values[0x78]:
                raise RuntimeError('PSW1 sticky fault/error/rejection')
            return values
        finally:
            if m.enabled: m.status.add(m.clock()-begin)

    def _issue(self, command, terminal_state):
        self._check_cancel()
        if self.sequence == U64-1: raise OverflowError('PSW command sequence exhausted')
        previous = self.sequence
        self.sequence += 1
        self.write_reg(0, command)
        deadline = self.monotonic() + self.timeout_seconds
        while True:
            self.measurements.polls += 1
            status = self._status()
            actual, terminal = self._seq(status,0x68), self._seq(status,0x60)
            if actual not in (previous,self.sequence) or terminal not in (previous,self.sequence):
                raise RuntimeError('PSW command/terminal sequence mismatch')
            if not status[4] and actual == terminal == self.sequence:
                if status[8] != terminal_state or status[0x74] != terminal_state: raise RuntimeError('Wrong PSW terminal state')
                return status
            if self.monotonic() >= deadline: raise TimeoutError('PSW completion deadline expired; buffer retained')
            self.measurements.sleeps += 1
            self.sleep(0.0005)

    def _bytes(self, offset, size):
        self.measurements.memory_reads += 1
        self.measurements.read_bytes += size
        return super()._bytes(offset, size)

    def submit(self, payload, consume):
        timing = Intervals(self.measurements, 'prepare_image')
        children = dict(pre_status=0, predict=0, post_status=0)
        if not self._lock.acquire(False): raise RuntimeError('Concurrent/reentrant PSW submission refused')
        try:
            if self.state != 'FREE': raise RuntimeError('Fixed PS slot unavailable')
            self._check_cancel()
            selected, identities = selected_endpoints(payload, self.identities)
            if self.request_id == U64-1 or self.generation == U64-1: raise OverflowError('Request/generation exhausted')
            self._idle(0)
            self.state = 'CPU_PREPARING'
            self.request_id += 1; self.generation += 1
            records = len(payload)//INPUT_STRIDE
            image = GUARD + payload + bytes([POISON])*(INPUT_CAPACITY-len(payload)) + GUARD + bytes([POISON])*OUTPUT_CAPACITY + GUARD
            self.retained = dict(input=payload, prepared=image, predictions=[])
            timing.mark('write_and_precheck')
            self.measurements.memory_writes += 1; self.measurements.write_bytes += len(image)
            self.ps_write(0, image)
            if self._bytes(0,ARENA_BYTES) != image: raise ValueError('PS preparation readback mismatch')
            timing.mark('descriptor')
            shadow = dict(zip(range(0x10,0x28,4),(self.arena_base+INPUT_OFFSET,len(payload),self.arena_base+OUTPUT_OFFSET,OUTPUT_CAPACITY,records,len(selected))))
            for offset,value in ((0x28,self.run_id),(0x30,self.request_id),(0x38,self.generation)):
                shadow[offset],shadow[offset+4] = value & 0xffffffff,value >> 32
            self._idle(0)
            for offset,value in shadow.items():
                self._check_cancel(); self.write_reg(offset,value)
            if any(self._read(offset) != value for offset,value in shadow.items()): raise ValueError('PSW descriptor readback mismatch')
            self.state = 'PL_INFLIGHT'
            timing.mark('issue_to_terminal')
            terminal = self._issue(1 if self.request_id == 1 else 4,2)
            counts = {0x40:records,0x44:len(selected),0x48:len(selected)*OUTPUT_STRIDE,0x4c:records*4,0x50:len(selected)*252,0x58:1}
            if any(terminal[k] != value for k,value in counts.items()): raise ValueError('PSW completion counts mismatch')
            self.state = 'ARM_COPYING'
            timing.mark('post_read_and_validate')
            actual = self._bytes(0,ARENA_BYTES); self.retained['completed'] = actual
            self._idle(2)
            end = OUTPUT_OFFSET+len(selected)*OUTPUT_STRIDE
            if actual[:OUTPUT_OFFSET] != image[:OUTPUT_OFFSET] or actual[end:] != image[end:]: raise ValueError('PS input/guard/unused-output corrupted')
            tensors = []
            for index,endpoints in enumerate(selected):
                start = OUTPUT_OFFSET+index*OUTPUT_STRIDE
                tensors.append(_output(actual[start:start+OUTPUT_STRIDE],self.run_id,self.request_id,index,self.generation,endpoints))
            self.retained['tensors'] = tensors
            self.state = 'CONSUMING'
            timing.mark('consume_all')
            predictions = self.retained['predictions']
            for index,(tensor,endpoints) in enumerate(zip(tensors,selected)):
                t = self.measurements.clock() if timing.enabled else 0
                self._idle(2)
                if timing.enabled: children['pre_status'] += self.measurements.clock()-t
                metadata = dict(run_id=self.run_id,request_id=self.request_id,generation=self.generation,window_index=index,
                    first=endpoints[0],last=endpoints[1],tensor_sha256=hashlib.sha256(tensor).hexdigest())
                t = self.measurements.clock() if timing.enabled else 0
                prediction = consume(tensor,metadata)
                if timing.enabled: children['predict'] += self.measurements.clock()-t
                predictions.append(prediction)
                t = self.measurements.clock() if timing.enabled else 0
                self._idle(2)
                if timing.enabled: children['post_status'] += self.measurements.clock()-t
            timing.mark('final_read_verify')
            if self._bytes(0,ARENA_BYTES) != actual: raise ValueError('PS slot changed while callbacks consumed tensors')
            self._idle(2)
            self.state = 'REARMING'
            timing.mark('rearm')
            rearmed = self._issue(2,0)
            if any(rearmed[k] != value for k,value in counts.items()): raise ValueError('PSW counts changed during REARM')
            result = dict(run_id=self.run_id,request_id=self.request_id,generation=self.generation,records=records,windows=len(selected),
                predictions=tuple(predictions),endpoints=tuple(selected),terminal=terminal,input_sha256=hashlib.sha256(payload).hexdigest(),
                scope='candidate protocol; actual hardware validation pending',strict_visibility_proven=False,axi_b_complete=True,
                sdk_owned_copy_checked=True,cache_contract_evidence=self.access_contract.cache_contract_evidence)
            partition = timing.finish()
            partition.update(consume_children_ns=children,request_id=self.request_id,generation=self.generation,
                issue_interval='PL service plus control sampling/wait, not pure FPGA compute')
            if partition['enabled']:
                partition['consume_other_ns'] = partition['phases_ns']['consume_all']-sum(children.values())
                if partition['consume_other_ns'] < 0: raise ValueError('Nested consume intervals overlap')
            self.measurements.complete_batch(partition)
            self.identities = identities; self.retained = {}; self.state = 'FREE'
            return result
        except BaseException as exc:
            self.measurements.fail('batch.'+timing.phase,exc,request_id=self.request_id,generation=self.generation,partial_phases_ns=dict(timing.parts))
            self._fail(exc)
            raise
        finally:
            self._lock.release()
