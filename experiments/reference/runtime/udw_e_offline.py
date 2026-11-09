"""Offline fixtures/reference builder. Never imports socket or board SDK.

These timings/outputs are substitute evidence, never NPU performance results.
"""
from collections import deque
import struct
import numpy as np
from software.runtime.stream import InputBuilder
from experiments.reference.runtime.ddr_transport_contract import unpack_record, pack_record, pack_output
from experiments.reference.runtime.udw_lab import digest


def build_references(records,normalization,fixed,logits):
    builder=InputBuilder(normalization,packing='single_copy')
    previous=None;epoch=0;count=0;endpoints=deque(maxlen=100)
    windows={};minimum=[];min_infos=[];min_tensor=None
    selected_rows=[]
    for index,record in enumerate(records):
        builder.push(record,materialize=False) # independent accepted validation
        if record.segment_key!=previous:
            previous=record.segment_key;epoch+=1;count=0;endpoints.clear()
        count+=1;m=record.meta
        ident=(epoch,m['source_record_row_0based'],m['source_message_position_1based'],
               m['timestamp_ns_since_exchange_midnight'])
        endpoints.append(ident)
        selected=count>=100 and (count-100)%16==0
        selected_rows.append(ident if selected else None)
        normalized=builder.ring[(builder.pointer-1)%100].copy()
        if index<100:
            info=dict(identity=ident,first=endpoints[0],last=ident,normalized=normalized,
                      selected=index==99,opportunity=count>=100)
            minimum.append(pack_record(tuple(int(x) for x in normalized.view('<u4')),ident,index==99))
            min_infos.append(info)
        if selected:
            tensor=builder.snapshot().tensor.tobytes()
            windows[ident]=dict(first=endpoints[0],tensor=tensor)
            if index==99:min_tensor=tensor
    if min_tensor is None:raise ValueError('Need an initial contiguous 100-record minimal sample')
    used=46080;expected_records=0;expected_ids=[]
    for i in range(0,len(records),64):
        items=selected_rows[i:i+64];ids=[x for x in items if x is not None]
        size=len(items)*256+max(64,len(ids)*16128)+256
        if used+size+4096>512*1024:break
        used+=size;expected_records+=len(items);expected_ids.extend(ids)
    return dict(windows=windows,logits=logits,fixed=fixed,
                expected_records=expected_records,expected_ids=expected_ids,
                minimal=dict(wdm=minimum,infos=min_infos,tensor=min_tensor))


class OfflineTransport:
    """Integer layout emulator + saved-logit lookup, not a model simulator."""
    def __init__(self,logits,*,size=2*1024*1024):
        self.base=4096;self.size=size;self.mem=bytearray(size);self.used=0
        self.regs={o:0 for o in range(0,0xc4,4)}
        self.regs.update({0xc0:0x55445731,0x8c:2,0x58:1})
        self.trace=[];self.history=deque(maxlen=100);self.logits=logits
        self.hook=lambda op,transport:None
        self.commands=0;self.predicts=0;self.close_calls=self.free_calls=self.terminate_calls=0

    def note(self,op):
        self.trace.append(op);self.hook(op,self)

    def status(self):self.note('status');return self.regs.copy()

    def reserve(self,size):
        self.note('reserve')
        if size%64 or self.used+size>self.size:raise MemoryError('Offline arena limit')
        offset=self.used;self.used+=size;return offset

    def prepare(self,offset,data):self.mem[offset:offset+len(data)]=data;self.note('prepare')

    def read(self,offset,size):self.note('read');return bytes(self.mem[offset:offset+size])

    def registers(self,values):self.regs.update(values);self.note('registers')

    def u64(self,offset):return self.regs[offset]+(self.regs[offset+4]<<32)

    def command(self,value):
        self.trace.append('command:'+str(value));r=self.regs
        sequence=self.u64(0x68)+1
        for o in (0x60,0x68):r[o]=sequence&0xffffffff;r[o+4]=sequence>>32
        if value==2:r[8]=0;self.note('rearm');return
        self.commands+=1
        if value==1:self.history.clear()
        count=0;begin=r[0x10]-self.base;output=r[0x18]-self.base
        for index in range(r[0x20]):
            words,identity,emit=unpack_record(bytes(self.mem[begin+index*256:begin+(index+1)*256]))
            if self.history and self.history[-1][1][0]!=identity[0]:self.history.clear()
            self.history.append((words,identity))
            if len(self.history)==100 and emit:
                tensor=struct.pack('<4000I',*(item[0][c] for c in range(40) for item in self.history))
                image=pack_output(tensor,self.history[0][1],self.history[-1][1],run_id=self.u64(0x28),
                    request_id=self.u64(0x30),generation=self.u64(0x38),window_index=count)
                self.mem[output:output+len(image)]=image;output+=len(image);count+=1
        r.update({4:0,8:2,0x40:r[0x20],0x44:count,0x48:count*16128,
                  0x4c:r[0x20]*4,0x50:count*252})
        self.note('after_execution')

    def call(self,op,**values):
        if op!='predict':raise ValueError('Offline substitute supports predict only')
        self.predicts+=1;self.note('predict')
        tensor=bytes.fromhex(values['hex'])
        result=dict(ok=True,window_id=values['window_id'],logits=list(self.logits[digest(tensor)]['npu']))
        self.last_prediction=result;self.note('prediction_result')
        return result

    # Tests assert these never occur; driver/runner never call these methods.
    def close(self):self.close_calls+=1
    def free(self):self.free_calls+=1
    def terminate(self):self.terminate_calls+=1
