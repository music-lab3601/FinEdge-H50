"""Finite PSW1 reuse experiment. No device opening or deployment in this module."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.reference.runtime.ps_window_runtime import AccessContract, ARENA_BYTES, PsWindowRuntime


def process_memory():
    """One bounded Linux process observation; not SDK allocator accounting."""
    result = dict(source='/proc/self/status', available=False,
                  VmRSS_kib=None, VmHWM_kib=None, VmData_kib=None)
    try:
        text = Path('/proc/self/status').read_text(encoding='ascii')
        rows = {line.split(':',1)[0]:line.split(':',1)[1].split()
                for line in text.splitlines() if ':' in line}
        for name in ('VmRSS','VmHWM','VmData'):
            fields = rows[name]
            if len(fields)!=2 or fields[1]!='kB' or int(fields[0])<0:
                raise ValueError('Unexpected proc memory units/value')
            result[name+'_kib'] = int(fields[0])
        result['available'] = True
    except (OSError,ValueError,KeyError) as exc:
        result.update(VmRSS_kib=None,VmHWM_kib=None,VmData_kib=None,error=type(exc).__name__)
    return result


class PsWindowProbe:
    """Seven batches, 199 rows, 100 windows, one fixed PS allocation.

    Every complete arena read and every tensor is saved independently before
    advancing. This diagnostic probe is intentionally not a throughput test.
    The caller keeps this object and its owner alive after success or failure.
    """
    def __init__(self, owner, directory, contract, run_id, reg):
        self.owner, self.directory = owner, Path(directory)
        self.contract, self.run_id, self.reg = contract, run_id, reg
        self.controller = None
        self.current_batch = None
        self.results, self.snapshots = [], []
        self.memory_reads = self.predictions = 0
        self.startup_memory = process_memory()

    def read_reg(self, offset):
        value, _ = self.owner.safe_call('psw.reg.read.'+hex(offset),
            lambda: self.reg.read(0x400c0000+offset, False))
        return int(value)

    def write_reg(self, offset, value):
        self.owner.safe_call('psw.reg.write.'+hex(offset),
            lambda: self.reg.write(0x400c0000+offset, value, False))

    def read_memory(self, offset, size):
        data = self.owner.read_chunk(self.owner.ps_arena, offset, size)
        self.memory_reads += 1
        name = 'arena_%03d.bin' % self.memory_reads
        (self.directory/name).write_bytes(data)
        self.snapshots.append(dict(name=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
            request_id=self.controller.request_id, generation=self.controller.generation,
            controller_state=self.controller.state, offset=offset))
        return data

    def run(self, cfg, payload, tensorbytes, references):
        if (len(payload)!=199*256 or len(tensorbytes)!=100*16000
                or len(references)!=100 or len(cfg['batches'])!=7):
            raise ValueError('Expected fixed seven-batch/100-window vector set')
        if not self.owner.initialized or self.owner.ps_arena is None:
            raise ValueError('Same initialized owner and allocated PS arena required')
        self.controller = PsWindowRuntime(read_reg=self.read_reg, write_reg=self.write_reg,
            ps_read=self.read_memory,
            ps_write=lambda offset,data:self.owner.write_chunk(self.owner.ps_arena,offset,data),
            arena_base=self.owner.arena_metadata['address'], arena_size=ARENA_BYTES,
            run_id=self.run_id, access_contract=self.contract,
            quarantine=lambda context:self.owner.quarantine(context['error']))
        with self.owner.operation('psw.initialize'):
            self.controller.initialize()
        for expected in cfg['batches']:
            self.current_batch = expected
            start = time.perf_counter_ns()
            offset, size = expected['offset'], expected['bytes']
            packet = payload[offset:offset+size]
            if hashlib.sha256(packet).hexdigest()!=expected['input_sha256']:
                raise ValueError('Input batch hash differs')

            def consume(tensor, metadata):
                index = self.predictions
                reference = references[index]
                if (reference['window_id']!=index+1
                        or tensor!=tensorbytes[index*16000:(index+1)*16000]
                        or metadata['tensor_sha256']!=reference['tensor_sha256']):
                    raise ValueError('FPGA tensor differs from independently built CPU input')
                name = 'tensor_%03d.bin' % (index+1)
                (self.directory/name).write_bytes(tensor)
                prediction = self.owner.predict(tensor)
                a = np.asarray(prediction['logits'], dtype=np.float64)
                b = np.asarray(reference['host']['logits'], dtype=np.float64)
                error = float(np.max(np.abs(a-b)))
                if (not np.allclose(a,b,rtol=1e-3,atol=1e-4)
                        or prediction['class_index']!=int(b.argmax())):
                    # Keep the full failing prediction even though no REARM is permitted.
                    self.controller.retained['numerical_failure'] = dict(prediction=prediction,reference=reference)
                    raise ValueError('NPU differs from compiled Host; tolerance remains unchanged')
                row = dict(metadata, window_id=index+1, prediction=prediction,
                    host_max_abs=error, host_class_equal=True, tensor_file=name,
                    cpu_class_equal=prediction['class_index']==reference['cpu']['class_index'])
                with (self.directory/'predictions.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                self.predictions += 1
                return row

            with self.owner.operation('psw.batch.'+str(expected['sequence'])):
                actual = self.controller.submit(packet,consume)
                if (actual['request_id']!=expected['sequence'] or actual['records']!=expected['records']
                        or actual['windows']!=expected['windows']
                        or [[list(a),list(b)] for a,b in actual['endpoints']]!=expected['endpoints']):
                    raise ValueError('Batch accounting differs')
            metrics = dict(request_id=actual['request_id'], records=actual['records'], windows=actual['windows'],
                generation=actual['generation'], batch_ns=time.perf_counter_ns()-start,
                terminal=actual['terminal'], runtime=self.owner.snapshot(), process_memory=process_memory())
            if metrics['runtime']['active_refs'] or self.controller.state!='FREE':
                raise ValueError('Successful batch retained active scope or occupied slot')
            self.results.append(metrics)
        if self.predictions!=100 or self.owner.npu_calls!=100 or self.memory_reads!=21:
            raise ValueError('Missing or unexpected NPU calls/windows/snapshots')
        self.current_batch = None
        return dict(status='passed', scope='finite PSW1 fixed-slot functionality; not throughput or full day',
            records=199, windows=100, NPU_calls=100, warmups=0, batches=self.results,
            snapshots=self.snapshots, slot_generations=7, slot_reuse_transitions=6,
            controller_state=self.controller.state, runtime=self.owner.snapshot(),
            strict_visibility_proven=False, actual_cross_master_readbacks_verified=True,
            cache_contract_evidence=self.contract.cache_contract_evidence,
            startup_process_memory=self.startup_memory, final_process_memory=process_memory(),
            full_day_complete=False, sustained_memory_validated=False)


def startup_guard(owner):
    """Read-only PSW1 state check before Session/apply/allocation."""
    reg = owner.open_axi_register_region()
    required = {0xc0:0x50535731,0x8c:1,0x90:3,0xc4:0x30100000,0xc8:0x38100000}
    required.update({o:0 for o in (4,8,12,0x70,0x54,0x60,0x64,0x68,0x6c,
        0x74,0x78,0x40,0x44,0x48,0x4c,0x50,0x58)})
    owner.startup_state = {}
    for offset, expected in required.items():
        value, _ = owner.safe_call('startup.psw.'+hex(offset),lambda:reg.read(0x400c0000+offset,False))
        owner.startup_state[offset]=int(value)
        if int(value)!=expected:
            raise ValueError('Clean PSW1 with B-response capability required before Session/allocation')
    # Detect a fault/command arriving during the preceding multi-register read.
    # Sticky fault is sampled last; no finite check proves against a future fault.
    for offset in (4,8,12,0x54,0x60,0x64,0x68,0x6c,0x70):
        value, _ = owner.safe_call('startup.psw.final.'+hex(offset),lambda:reg.read(0x400c0000+offset,False))
        owner.startup_state[offset]=int(value)
        if int(value)!=0:
            raise ValueError('PSW1 changed during startup before Session/allocation')
