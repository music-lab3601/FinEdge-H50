"""Experimental shared-input candidate, not enabled in any board service.

Readback observations are not device memory fences. All allocations, Tensor
handles and failed leases remain retained. There is no release/reset/reuse API.
"""
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps
import io
import hashlib
import time

import numpy as np

from experiments.reference.runtime.ddr_transport_contract import OUTPUT_STRIDE
from experiments.reference.runtime.udw_board_bridge import BoardBridge, ARENA, QUOTA, selected_endpoints


def uint(value, bits, label):
    if type(value) is not int or not 0 <= value < 2**bits:
        raise ValueError(label+' must be uint'+str(bits))


@dataclass(frozen=True)
class LeaseKey:
    boot_id: str
    allocation_id: str
    run_id: int
    generation: int
    request_id: int
    window_index: int


def fail_closed(method):
    @wraps(method)
    def checked(self,*args,**kwargs):
        try:return method(self,*args,**kwargs)
        except Exception as error:
            self.stop(error)
            raise
    return checked


class LeaseLedger:
    """Single cold session, monotonic slots, no reusable-completion state."""
    def __init__(self, *, boot_id, allocation_id, base, size, run_id, generation,
                 quota=QUOTA, retained_owner=None):
        if not isinstance(boot_id,str) or not boot_id or not isinstance(allocation_id,str) or not allocation_id:
            raise ValueError('Cold boot and allocation identities required')
        for name,value in [('base',base),('size',size),('run_id',run_id),('generation',generation),('quota',quota)]:
            uint(value,64,name)
        if not size or base%64 or size%64 or base+size>2**32 or not 4096<quota<=size:
            raise ValueError('Invalid aligned uint32 allocation or finite quota')
        self.boot_id,self.allocation_id=boot_id,allocation_id
        self.base,self.size,self.run_id,self.generation,self.quota=base,size,run_id,generation,quota
        self.owner=retained_owner
        self.used=self.request_id=0
        self.blocked=self.finished=False
        self.views={}
        self.handles=[]
        self.calls=[]
        self.forward_outputs=[]
        self.events=[]

    def active(self):
        if self.blocked or self.finished: raise RuntimeError('Lease session stopped; new cold allocation required')

    @fail_closed
    def reserve_batch(self, payload_bytes, endpoints, request_id):
        self.active();uint(request_id,64,'request_id')
        if request_id!=self.request_id+1 or type(payload_bytes) is not int or not 0<payload_bytes<=65536 or payload_bytes%256:
            raise ValueError('Fresh sequential request and complete WDM records required')
        if len(endpoints)>payload_bytes//256:raise ValueError('Too many output windows')
        capacity=max(64,len(endpoints)*OUTPUT_STRIDE)
        size=payload_bytes+capacity+256
        if self.used+size+4096>self.quota:raise ValueError('Finite quota exhausted before any device command')
        pending=[]
        for index,(first,last) in enumerate(endpoints):
            if len(first)!=4 or len(last)!=4:raise ValueError('Invalid endpoint identity')
            for value in (*first,*last):uint(value,64,'endpoint')
            if first[0]!=last[0] or last[1]-first[1]!=99 or first[2]>=last[2] or first[3]>last[3]:
                raise ValueError('Invalid contiguous window endpoints')
            key=LeaseKey(self.boot_id,self.allocation_id,self.run_id,self.generation,request_id,index)
            header=self.used+payload_bytes+192+index*OUTPUT_STRIDE
            tensor=header+128
            if tensor%64 or tensor+16000>self.used+size-64:raise ValueError('Tensor overlaps slot guard')
            pending.append((key,dict(state='reserved',first=tuple(first),last=tuple(last),
                slot_offset=self.used,slot_size=size,header_offset=header,tensor_offset=tensor,
                logical_bytes=16000,storage_bytes=None,strict_visibility_proven=False)))
        for key,view in pending:self.views[key]=view
        self.used+=size;self.request_id=request_id
        self.events.append(dict(event='batch_reserved',request_id=request_id,used=self.used,windows=len(pending)))
        return [key for key,_ in pending]

    def view(self,key,state):
        self.active()
        if type(key) is not LeaseKey or key not in self.views:raise ValueError('Foreign or stale lease identity')
        view=self.views[key]
        if view['state']!=state:raise ValueError('Invalid lease transition: '+view['state']+' expected '+state)
        return view

    @fail_closed
    def pl_may_start(self,keys):
        self.active()
        for key in keys:self.view(key,'reserved')
        for key in keys:self.views[key]['state']='pl_may_be_writing'

    @fail_closed
    def sdk_write_may_start(self,keys):
        self.active()
        for key in keys:self.view(key,'reserved')
        for key in keys:
            self.views[key].update(state='sdk_may_be_writing',producer='SDK_known_input')

    @fail_closed
    def observe_protected_readback(self,key,tensor):
        state='sdk_may_be_writing' if self.views.get(key,{}).get('producer')=='SDK_known_input' else 'pl_may_be_writing'
        view=self.view(key,state)
        if len(tensor)!=16000 or not np.isfinite(np.frombuffer(tensor,dtype='<f4')).all():
            raise ValueError('Invalid protected readback tensor')
        view.update(state='readback_observed_unproven',tensor_sha256=hashlib.sha256(tensor).hexdigest(),
            observation=('SDK_known_input_full_slot_readback' if state=='sdk_may_be_writing'
                         else 'inherited_BoardBridge_checked_full_region_and_WDO_identity'),
            strict_visibility_proven=False)

    @fail_closed
    def bind(self,key,*,tensor_factory,dtype,chunk,region,allow_experimental_readback=False):
        view=self.view(key,'readback_observed_unproven')
        if allow_experimental_readback is not True:
            raise RuntimeError('No strict PL-to-NPU visibility fence; experimental opt-in required')
        storage=int(dtype.bytes())
        if storage!=16000 or view['tensor_offset']+storage>view['header_offset']+OUTPUT_STRIDE:
            raise ValueError('SDK storage size differs from protected WDO tensor extent')
        if int(chunk.byte_size)!=self.size or int(chunk.begin.addr())!=self.base:
            raise ValueError('SDK allocation identity/extent differs')
        # Retain even a Tensor whose metadata check subsequently fails.
        tensor=tensor_factory(dtype,chunk,view['tensor_offset']);self.handles.append(tensor)
        if (int(tensor.offset())!=view['tensor_offset'] or int(tensor.dtype().bytes())!=storage
            or int(tensor.chunk().begin.addr())!=self.base or int(tensor.chunk().byte_size)!=self.size
            or int(tensor.data().addr())!=self.base+view['tensor_offset'] or not tensor.isOn(region)):
            raise ValueError('SDK changed shared Tensor chunk, address, dtype or region')
        view.update(state='bound_experimental',storage_bytes=storage)
        return tensor

    @fail_closed
    def retain_cast(self,adapted,*,source,key=None):
        # No SDK metadata/getters before this strong reference. Cast may have
        # allocated a buffer independent of the arena or software input.
        self.handles.append(adapted)
        self.active()
        if source not in ('shared','software') or (source=='shared')!=(key is not None):
            raise ValueError('Explicit Cast source/lease required')
        call_id=len(self.calls)+1
        self.calls.append(dict(call_id=call_id,source=source,key=key,state='cast_returned_retained'))
        return call_id

    def call(self,call_id,state):
        self.active()
        if type(call_id) is not int or not 1<=call_id<=len(self.calls):
            raise ValueError('Invalid input consumption identity')
        call=self.calls[call_id-1]
        if call['state']!=state:raise ValueError('Invalid input consumption state')
        return call

    @fail_closed
    def npu_may_start(self,call_id):
        call=self.call(call_id,'cast_returned_retained')
        if call['key'] is not None:self.view(call['key'],'bound_experimental')['state']='npu_may_be_reading'
        call['state']='npu_may_be_reading' # before forward, including ambiguous errors

    @fail_closed
    def npu_completed(self,call_id):
        call=self.call(call_id,'npu_may_be_reading')
        if call['key'] is not None:self.view(call['key'],'npu_may_be_reading')['state']='completed_retained'
        call['state']='completed_retained' # completion never authorizes reuse

    def stop(self,error):
        self.blocked=True
        for view in self.views.values():view['state']='quarantined_retained'
        for call in self.calls:
            if call['state']!='quarantined_retained':call['state_before_quarantine']=call['state']
            call['state']='quarantined_retained'
        self.events.append(dict(event='quarantined',reason=str(error),owner_retained=True))

    @fail_closed
    def finish(self):
        self.active()
        if any(v['state']!='completed_retained' for v in self.views.values()):
            raise RuntimeError('Incomplete/ambiguous lease cannot finish')
        if any(c['state']!='completed_retained' for c in self.calls):
            raise RuntimeError('Incomplete/ambiguous input consumption cannot finish')
        self.finished=True
        return dict(leases=len(self.views),input_calls=len(self.calls),used=self.used,owner_retained=True,
            strict_visibility_proven=False,zero_copy=False,reuse_allowed=False)


class SharedBoardBridge(BoardBridge):
    """Explicit future board candidate; existing service still uses BoardBridge.

    Inherits all whole-slot reads, WDO/guard checks, final audit and HardOp
    completion callback. Only the input Tensor construction/write is replaced.
    Cast can still copy internally; actual device behavior remains unverified.
    """
    def __init__(self):
        super().__init__()
        self.ledger=None
        self.pending=None
        self.pending_index=0
        self.explicit_reads=[]

    def read_local(self,offset,size):
        row=dict(offset=offset,requested_bytes=size,completed_bytes=None)
        self.explicit_reads.append(row)
        data=super().read_local(offset,size)
        row['completed_bytes']=len(data)
        return data

    def initialize(self,*args,**kwargs):
        info=super().initialize(*args,**kwargs)
        self.ledger=LeaseLedger(boot_id=info['boot_id'],allocation_id=str(self.run_id)+':'+str(self.base),
            base=self.base,size=ARENA,run_id=self.run_id,generation=self.generation,retained_owner=self.chunk)
        info.update(mode='shared_Tensor_binding_candidate_unverified',zero_copy=False,
                    strict_visibility_proven=False,device_shared_binding_accepted=False)
        return info

    def ensure_active(self):
        if self.blocked or self.failed or self.finished:raise RuntimeError('Bridge stopped; no retry or reuse')
        if self.ledger is None:raise RuntimeError('Shared candidate not initialized')
        self.ledger.active()

    def quarantine(self,error):
        self.blocked=True
        if self.ledger is not None:self.ledger.stop(error)

    def submit_local(self,payload,continuation):
        self.ensure_active()
        try:
            if type(continuation) is not bool:raise ValueError('Explicit continuation required')
            if self.pending is not None:raise RuntimeError('Concurrent request refused')
            selected,_=selected_endpoints(payload,self.identities if continuation else ())
            if self.ledger.used!=self.used:raise ValueError('Lease and physical slot accounting differ')
            keys=self.ledger.reserve_batch(len(payload),selected,self.request_id+1)
            self.ledger.pl_may_start(keys)
            self.pending,self.pending_index=keys,0
            read_begin=len(self.explicit_reads)
            result=super().submit_local(payload,continuation)
            if self.pending_index!=len(keys):raise ValueError('Missing shared-window consumption')
            result.update(shared_binding_candidate=True,zero_copy=False,strict_visibility_proven=False)
            result['explicit_arena_read_requests']=self.explicit_reads[read_begin:]
            result['SDK_internal_transfer_bytes']=None
            return result
        except Exception as error:
            self.quarantine(error);raise
        finally:self.pending=None

    def software_predict(self,tensor):
        self.ensure_active()
        try:return super().software_predict(tensor)
        except Exception as error:self.quarantine(error);raise

    def predict_local(self,tensor):
        self.ensure_active()
        try:return self._predict_retained(tensor)
        except Exception as error:self.quarantine(error);raise

    def _predict_retained(self,tensor):
        if len(tensor)!=16000 or not np.isfinite(np.frombuffer(tensor,dtype='<f4')).all():
            raise ValueError('Invalid logical NPU tensor')
        begin=time.perf_counter_ns()
        key=None
        if self.pending is None:
            self.input.write(0,np.frombuffer(tensor,dtype=np.uint8),len(tensor))
            source=self.input
        else:
            if self.pending_index>=len(self.pending):raise ValueError('Unexpected extra window')
            key=self.pending[self.pending_index];self.pending_index+=1
            self.ledger.observe_protected_readback(key,tensor)
            source=self.ledger.bind(key,tensor_factory=self.xrt.Tensor,dtype=self.software_type,
                chunk=self.chunk,region=self.device.defaultMemRegion(),allow_experimental_readback=True)
        bind_end=time.perf_counter_ns()
        # Preserve the frozen software-layout -> model-layout Cast exactly.
        adapted=self.host.Cast(source,self.software_type,self.input_type,self.converter)
        call_id=self.ledger.retain_cast(adapted,source='shared' if key else 'software',key=key)
        cast_end=time.perf_counter_ns()
        if int(adapted.dtype().bytes())!=16000:raise ValueError('Unexpected adapted storage size')
        if not hasattr(self,'adapted_info'):
            self.adapted_info=dict(chunk_bytes=int(adapted.chunk().byte_size),
                                   offset=int(adapted.offset()),dtype=str(adapted.dtype()))
        adapted_on_device=bool(adapted.isOn(self.device.defaultMemRegion()))
        binding=dict(allocation_id=key.allocation_id if key else None,
            tensor_offset=source.offset() if key else None,
            bound_address=int(source.data().addr()) if key else None,storage_bytes=16000,
            adapted_is_on_same_device_region=adapted_on_device,
            adapted_address=int(adapted.data().addr()) if adapted_on_device else None,
            adapted_offset=int(adapted.offset()),same_buffer_proven_for_HardOp=False,
            explicit_host_input_write_bytes=0 if key else 16000,protective_readback_removed=False,
            source='shared' if key else 'software',retained_call_id=call_id)
        self.ledger.npu_may_start(call_id)
        outputs=self.session.forward([adapted])
        self.ledger.forward_outputs.append(outputs)
        forward_end=time.perf_counter_ns()
        if len(outputs)!=1:raise ValueError('Wrong NPU output count')
        outputs[0].setReady(False)
        if not outputs[0].waitForReady(timedelta(seconds=10),timedelta(milliseconds=1)):
            raise TimeoutError('Shared-input NPU completion timeout')
        ready_end=time.perf_counter_ns()
        stream=io.BytesIO();outputs[0].dump(stream,'SFB')
        logits=np.frombuffer(stream.getvalue(),dtype='<f4')
        if logits.shape!=(3,) or not np.isfinite(logits).all():raise ValueError('Invalid shared-input logits')
        end=time.perf_counter_ns()
        self.ledger.npu_completed(call_id);self.call_count+=1
        return dict(logits=logits.tolist(),class_index=int(logits.argmax()),npu_call=self.call_count,
            tensor_sha256=hashlib.sha256(tensor).hexdigest(),shared_input=binding,
            stages_ns=dict(shared_bind=bind_end-begin,host_input_cast=cast_end-bind_end,
                forward=forward_end-cast_end,output_wait=ready_end-forward_end,
                output_read=end-ready_end,npu_total=end-begin))

    def finish(self,directory):
        self.ensure_active()
        try:
            if any(v['state']!='completed_retained' for v in self.ledger.views.values()):
                raise RuntimeError('Cannot audit unfinished shared-input leases')
            result=super().finish(directory)
            result['shared_leases']=self.ledger.finish()
            return result
        except Exception as error:self.quarantine(error);raise
