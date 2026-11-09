"""Bounded SDK PS/PL copy probe. No NPU, RTL start, free, reset or reuse.

Each test uses a distinct range in three persistent 64 KiB chunks. Equality
after a returned API is an observation, not a cross-master memory fence.
"""
from __future__ import annotations

import hashlib
import sys
import time

import numpy as np

from .udw_probe_stop import ProbeStop

ARENA = 65536
ALIGNMENT = 64
GUARD = 64
FILL = {"ps_source": 0xA3, "pl": 0x5C, "ps_return": 0xE7}
DEVICE_URL = 'axi://zg330aiu?npu=0x40000000&dma=0x80000000'
DEVICE_VERSIONS = {'device':'25122301', 'icore':'FMSHZGV3TECH-AID - 24160628'}
STARTUP = (0xC0, 0x8C, 4, 8, 12, 0x70, 0x60, 0x64, 0x68, 0x6C, 0x54)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def cases():
    result, cursor = [], 0
    for generation in (1, 2):
        for size in (64, 256, 4096, 16000):
            # Different source/destination offsets catch swapped arguments.
            slot_size = ((size + 4 * GUARD + 63) // 64) * 64
            result.append(dict(index=len(result), generation=generation, size=size,
                               slot=cursor, slot_size=slot_size,
                               ps_source=cursor+64, pl=cursor+128, ps_return=cursor+192))
            cursor += slot_size
    if cursor > ARENA:
        raise ValueError('Probe exceeds fixed allocation')
    return result


def pattern(case):
    tag = ('finedge-udma-v1:%d:%d:%d' %
           (case['index'], case['generation'], case['size'])).encode('ascii')
    return hashlib.shake_256(tag).digest(case['size'])


class TransferProbe:
    def __init__(self, stop=None, record=None, clock=time.perf_counter_ns):
        self.stop = stop or ProbeStop()
        self.stop.bind_owner(self)
        self.record = record or (lambda event: None)
        self.clock = clock
        self.blocked = False
        self.device = None
        self.regions, self.chunks, self.metadata = {}, {}, {}
        self.buffers, self.failures, self.events, self.results = [], [], [], []
        self.expected = {name: bytearray([fill])*ARENA for name, fill in FILL.items()}
        self.snapshots = {}
        self.started = False

    def quarantine(self, reason):
        self.blocked = True

    def event(self, **data):
        self.events.append(data)
        self.record(data)

    def call(self, name, operation):
        self.stop.checkpoint(name+':before')
        self.event(kind='begin', api=name)
        begin = self.clock()
        try:
            value = operation()
            elapsed = self.clock()-begin
            self.stop.hold(value)  # before cancellation, metadata or logging
            self.event(kind='return', api=name, ns=elapsed)
            self.stop.checkpoint(name+':after')
            return value, elapsed
        except BaseException as exc:
            self.failures.append((exc, sys.exc_info()[2]))
            self.quarantine(str(exc))
            self.event(kind='failure', api=name, error=type(exc).__name__+': '+str(exc))
            raise

    def initialize(self, xrt):
        if self.device is not None:
            raise ValueError('Initialize only once')
        self.xrt = xrt
        self.device, _ = self.call('Device.Open', lambda: xrt.Device.Open(DEVICE_URL))
        versions, _ = self.call('device.version', self.device.version)
        self.versions = dict(versions)
        if self.versions != DEVICE_VERSIONS:
            raise ValueError('Device/AI core version differs')
        reg, _ = self.call('defaultRegRegion', self.device.defaultRegRegion)
        startup = {}
        for offset in STARTUP:
            value, _ = self.call('startup.'+hex(offset), lambda: reg.read(0x400C0000+offset, False))
            startup[offset] = int(value)
            expected = 0x55445731 if offset == 0xC0 else 2 if offset == 0x8C else 0
            if startup[offset] != expected:
                raise ValueError('Clean UDW1/ABI2 required before any allocation')
        self.startup = startup
        for name in FILL:
            region_name = 'plddr' if name == 'pl' else 'udma'
            region, _ = self.call(name+'.region', lambda: self.device.getMemRegion(region_name))
            self.regions[name] = region
            chunk, _ = self.call(name+'.malloc', lambda: region.malloc(ARENA, False, ALIGNMENT))
            self.chunks[name] = chunk
            ptr = chunk.begin
            kind = ptr.ptype()
            if kind not in (xrt.PtrType.ADDR, xrt.PtrType.BOTH):
                raise ValueError('Expected device address; no CPTR coercion')
            address = int(ptr.addr())
            if int(chunk.byte_size) != ARENA or address % ALIGNMENT or address < 0:
                raise ValueError('Unexpected allocation extent/alignment')
            if not chunk.isOn(region):
                raise ValueError('Allocation region mismatch')
            self.metadata[name] = dict(region=region_name, address=address, bytes=ARENA,
                pointer_type='ADDR' if kind == xrt.PtrType.ADDR else 'BOTH', auto_free=False)
        a, b = (self.metadata[name]['address'] for name in ('ps_source','ps_return'))
        if max(a,b) < min(a+ARENA,b+ARENA):
            raise ValueError('PS source and return allocations overlap')
        self.event(kind='allocations', metadata=self.metadata, total_bytes=3*ARENA)

    @staticmethod
    def extent(offset, size):
        if type(offset) is not int or type(size) is not int or size <= 0 or offset < 0 or offset+size > ARENA:
            raise ValueError('Transfer outside fixed arena')

    def write(self, name, offset, data, tag):
        self.extent(offset, len(data))
        array = np.frombuffer(data, dtype=np.uint8)
        self.buffers.append(array)
        _, ns = self.call(tag, lambda: self.chunks[name].write(offset, array, len(data)))
        return ns

    def read(self, name, offset, size, tag):
        self.extent(offset, size)
        array = np.full(size, 0x39, dtype=np.uint8)
        self.buffers.append(array)
        _, ns = self.call(tag, lambda: self.chunks[name].read(array, offset, size))
        data = array.tobytes()
        self.snapshots[tag] = data
        return data, ns

    def copy(self, dst, dst_offset, src, src_offset, size, tag):
        self.extent(dst_offset, size)
        self.extent(src_offset, size)
        if (dst, src) not in (('pl','ps_source'), ('ps_return','pl')):
            raise ValueError('Only planned PS->PL and PL->PS copies allowed')
        _, ns = self.call(tag, lambda: self.chunks[dst].copyFrom(
            dst_offset, self.chunks[src], src_offset, size))
        return ns

    def verify(self, name, offset, size, tag):
        actual, ns = self.read(name, offset, size, tag)
        expected = bytes(self.expected[name][offset:offset+size])
        row = dict(kind='comparison', tag=tag, region=name, offset=offset, bytes=size,
                   expected_sha256=digest(expected), actual_sha256=digest(actual), equal=actual==expected)
        self.event(**row)
        if actual != expected:
            self.quarantine('Data or guard mismatch: '+tag)
            raise ValueError('Data or guard mismatch: '+tag)
        return ns

    def run(self):
        if self.started or set(self.chunks) != set(FILL):
            raise ValueError('Initialized, unused probe required')
        self.started = True
        try:
            for name in FILL:
                self.write(name, 0, bytes(self.expected[name]), name+'.initialize')
                self.verify(name, 0, ARENA, name+'.initial_read')
            for case in cases():
                tag = 'case%02d' % case['index']
                data = pattern(case)
                stages = {}
                begin = self.clock()
                stages['ps_write'] = self.write('ps_source',case['ps_source'],data,tag+'.ps_write')
                self.expected['ps_source'][case['ps_source']:case['ps_source']+len(data)] = data
                stages['ps_read'] = self.verify('ps_source',case['slot'],case['slot_size'],tag+'.ps_read')
                stages['ps_to_pl'] = self.copy('pl',case['pl'],'ps_source',case['ps_source'],len(data),tag+'.ps_to_pl')
                self.expected['pl'][case['pl']:case['pl']+len(data)] = data
                stages['pl_read'] = self.verify('pl',case['slot'],case['slot_size'],tag+'.pl_read')
                stages['pl_to_ps'] = self.copy('ps_return',case['ps_return'],'pl',case['pl'],len(data),tag+'.pl_to_ps')
                self.expected['ps_return'][case['ps_return']:case['ps_return']+len(data)] = data
                stages['ps_return_read'] = self.verify('ps_return',case['slot'],case['slot_size'],tag+'.ps_return_read')
                row = dict(case, payload_sha256=digest(data), stages_ns=stages,
                           whole_case_ns=self.clock()-begin, passed=True)
                self.results.append(row)
                self.event(kind='case_complete', result=row)
            for name in FILL:
                self.verify(name,0,ARENA,name+'.final_read')
            return self.report('passed')
        except BaseException as exc:
            self.failures.append((exc,sys.exc_info()[2]))
            self.quarantine(str(exc))
            raise

    def report(self, status):
        return dict(schema='udma-transfer-probe-v1', status=status, cases=self.results,
                    metadata=self.metadata, allocated_bytes=len(self.chunks)*ARENA,
                    versions=getattr(self,'versions',{}), startup=getattr(self,'startup',{}),
                    device_url=DEVICE_URL, events=self.events, owner_retained=True,
                    SDK_returns_retained=len(self.stop.retained_returns),
                    retained_host_array_bytes=sum(a.nbytes for a in self.buffers),
                    blocked=self.blocked, stop=self.stop.snapshot(),
                    NPU_calls=0, RTL_commands=0, reused_ranges=0,
                    strict_visibility_proven=False, DMA_implementation_proven=False,
                    full_day_pool_authorized=False,
                    timing_scope='API wall time; no warmup, not sustained throughput')
