"""Bounded experiment adapter: live reconstructed records to frozen UDW1.

No file of prebuilt books/windows is read here. Every eligible record is
normalized once by the existing InputBuilder, including unselected records.
The desktop and accepted low-level driver remain unchanged.
"""
from dataclasses import dataclass
from math import ceil
from time import perf_counter_ns

import numpy as np

from .ddr_transport_contract import OUTPUT_STRIDE, pack_record, unpack_output
from .stream import InputBuilder
from .udw_lab import LabFault


@dataclass(frozen=True)
class RuntimeRow:
    raw_u64: bytes
    metadata: dict
    identity: tuple
    wdm: bytes
    opportunity: bool
    selected: bool
    software_window: object
    preprocess_and_ring_ns: int
    software_pack_ns: int
    wdm_pack_ns: int


@dataclass(frozen=True)
class RuntimeBatch:
    rows: tuple

    @property
    def selected(self):
        return tuple(r for r in self.rows if r.selected)

    @property
    def capacity(self):
        return len(self.rows)*256+max(64,len(self.selected)*OUTPUT_STRIDE)+256


class RuntimeAdapter:
    def __init__(self, normalization, *, emit_stride=16):
        if type(emit_stride) is not int or emit_stride<1:
            raise ValueError('emit_stride must be a positive integer')
        self.builder=InputBuilder(normalization,packing='single_copy')
        self.emit_stride=emit_stride
        self.previous_key=None
        self.epoch=0
        self.segment_count=0
        self.epochs=[]

    def push(self, record):
        start=perf_counter_ns()
        # Existing validation/segment handling/normalization/ring insertion;
        # do not normalize a second time for WDM1.
        self.builder.push(record,materialize=False)
        pre_end=perf_counter_ns()
        if record.segment_key!=self.previous_key:
            self.previous_key=record.segment_key
            self.epoch+=1;self.segment_count=0
            self.epochs.append(dict(epoch=self.epoch,segment_key=list(record.segment_key)))
        self.segment_count+=1
        opportunity=self.builder.count==100
        selected=opportunity and (self.segment_count-100)%self.emit_stride==0
        window=self.builder.snapshot() if selected else None
        normalized=self.builder.ring[(self.builder.pointer-1)%100]
        m=dict(record.meta)
        identity=(self.epoch,int(m['source_record_row_0based']),
                  int(m['source_message_position_1based']),int(m['timestamp_ns_since_exchange_midnight']))
        pack_start=perf_counter_ns()
        wdm=pack_record(tuple(int(x) for x in normalized.view('<u4')),identity,bool(selected))
        return RuntimeRow(record.values.tobytes(),m,identity,wdm,bool(opportunity),bool(selected),
                          window,pre_end-start,0 if window is None else window.pack_ns,
                          perf_counter_ns()-pack_start)

    def batches(self, records, *, batch_size=64, record_limit=1024):
        if type(batch_size) is not int or not 1<=batch_size<=256:
            raise ValueError('batch_size must be 1..256')
        if type(record_limit) is not int or not 1<=record_limit<=10000:
            raise ValueError('record_limit must be 1..10000')
        batch=[]
        for i,record in enumerate(records):
            # The caller must close its source generator on bounded stop.
            if i>=record_limit:break
            batch.append(self.push(record))
            if len(batch)==batch_size:
                yield RuntimeBatch(tuple(batch));batch=[]
            if i+1==record_limit:break
        if batch:yield RuntimeBatch(tuple(batch))


class CapacityBudget:
    """Logical limit inside the fixed SDK arena; never asks allocator to retry."""
    def __init__(self, arena_bytes, limit_bytes, *, reserved_tail=4096):
        if not (0<reserved_tail<limit_bytes<=arena_bytes) or any(
                type(x) is not int or x%64 for x in (arena_bytes,limit_bytes,reserved_tail)):
            raise ValueError('Aligned bounded arena/quota/reserved_tail required')
        self.arena_bytes=arena_bytes;self.limit_bytes=limit_bytes;self.reserved_tail=reserved_tail

    def decision(self, used, requested):
        if any(type(x) is not int or x<0 or x%64 for x in (used,requested)) or used>self.arena_bytes:
            raise ValueError('Invalid allocator accounting')
        available=max(0,self.limit_bytes-self.reserved_tail-used)
        return dict(fits=used+requested+self.reserved_tail<=self.limit_bytes,
                    used=used,requested=requested,available=available,
                    arena_bytes=self.arena_bytes,quota_bytes=self.limit_bytes,
                    reserved_tail=self.reserved_tail)


def capacity_plan(*, record_limit=1024,batch_size=64,emit_stride=16,quota_bytes=512*1024):
    if not 1<=record_limit<=10000 or not 1<=batch_size<=256 or not 1<=emit_stride<=10000:
        raise ValueError('Invalid bounded run dimensions')
    budget=CapacityBudget(2*1024*1024,quota_bytes)
    overhead=4096+(100*256+OUTPUT_STRIDE+256)
    # Safe upper bound even when a batch starts with an already full history.
    per_batch=batch_size*256+max(64,ceil(batch_size/min(emit_stride,100))*OUTPUT_STRIDE)+256
    return dict(arena_bytes=budget.arena_bytes,quota_bytes=quota_bytes,
                reserved_tail=budget.reserved_tail,scratch_bytes=4096,
                minimum_test_bytes=overhead-4096,preflight_bytes=overhead,
                worst_full_batch_bytes=per_batch,
                full_batches_guaranteed=max(0,(quota_bytes-overhead-budget.reserved_tail)//per_batch),
                record_limit=record_limit,batch_size=batch_size,emit_stride=emit_stride,
                capacity_stop_policy='look ahead at most one batch; stop BEFORE reserve/REARM/START if it does not fit',
                reuse=False,automatic_restart=False)


def verify_runtime_output(batch, output, *, run_id, request_id, generation, index):
    selected=batch.selected[index]
    tensor,first,last=unpack_output(output,run_id=run_id,request_id=request_id,
                                  generation=generation,window_index=index)
    window=selected.software_window
    if tensor!=window.tensor.tobytes():
        raise LabFault('PL tensor differs from current software InputBuilder')
    def suffix(meta):
        return (int(meta['source_record_row_0based']),int(meta['source_message_position_1based']),
                int(meta['timestamp_ns_since_exchange_midnight']))
    if first[0]!=selected.identity[0] or first[1:]!=suffix(window.first_meta) or last!=selected.identity:
        raise LabFault('PL window endpoints differ from software identity')
    if suffix(window.last_meta)!=last[1:]:raise LabFault('Software endpoint metadata mismatch')
    return tensor,first,last
