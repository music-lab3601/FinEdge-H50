"""Cancel-aware owner used only by the independent shared-input probe.

The accepted L02 BoardBridge and its default service are not changed.
"""
from datetime import timedelta
import io
import os
from pathlib import Path
import sys
import time

import numpy as np

from .ddr_transport_contract import Descriptor,OUTPUT_STRIDE,unpack_output
from .udw_board_bridge import ARENA,QUOTA,REG,STARTUP,LocalTransport,require_clean,selected_endpoints
from .udw_shared_tensor import SharedBoardBridge,LeaseLedger
from .udw_lab import UdwLab,digest


class GuardedRegisters:
    def __init__(self,raw,stop):self.raw,self.stop=raw,stop
    def read(self,address,*args):
        return self.stop.call('reg.read.'+hex(address),lambda:self.raw.read(address,*args))
    def write(self,address,*args):
        return self.stop.call('reg.write.'+hex(address),lambda:self.raw.write(address,*args))


class ProbeBridge(SharedBoardBridge):
    def __init__(self,stop):
        super().__init__();self.stop_control=stop;stop.bind_owner(self)

    def sdk(self,stage,operation,retain=None):return self.stop_control.call(stage,operation,retain)
    def own(self,name,operation):
        return self.sdk('init.'+name,operation,lambda value:setattr(self,name,value))
    def ensure_active(self):
        self.stop_control.checkpoint('owner.entry');super().ensure_active()

    def initialize(self,model,expected_boot,run_id,control='axi_registers'):
        self.stop_control.checkpoint('initialize.before')
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=expected_boot:raise ValueError('Boot changed')
        sys.path.insert(0,'/usr/lib/aarch64-linux-gnu')
        import xir,xrt,host_backend,zg330backend
        for module in (xir,xrt,host_backend,zg330backend):
            if Path(module.__file__).parent!=Path('/usr/lib/aarch64-linux-gnu'):raise ValueError('Wrong SDK module')
        self.stop_control.checkpoint('initialize.imports_done')
        self.xrt,self.host=xrt,host_backend
        self.own('device',lambda:xrt.Device.Open('socket://zg330aiu@127.0.0.1:9981'))
        reg=self.sdk('init.register_region',lambda:self.device.defaultRegRegion())
        self.reg=GuardedRegisters(reg,self.stop_control)
        startup={k:int(self.reg.read(REG+k,False)) for k in STARTUP};require_clean(startup)
        if control!='axi_registers':raise ValueError('Probe scope requires AXI registers')
        self.own('control_device',lambda:xrt.Device.Open('axi://zg330aiu'))
        reg=self.sdk('init.local_register_region',lambda:self.control_device.defaultRegRegion())
        direct=GuardedRegisters(reg,self.stop_control)
        observed={k:int(direct.read(REG+k,False)) for k in STARTUP};require_clean(observed)
        if observed!=startup:raise ValueError('Register views differ')
        self.reg=direct
        self.own('network',lambda:xir.Network.CreateFromJsonFile(str(model)))
        self.sdk('init.params',lambda:self.network.loadParamsFromFile(str(model.with_suffix('.raw'))))
        view=self.sdk('init.view',lambda:self.network.view(0))
        inputs=self.sdk('init.inputs',lambda:view.inputs())
        self.own('input_type',lambda:inputs[0].tensorType())
        if list(self.input_type.shape)!=[1,40,100]:raise ValueError('Model shape differs')
        host_device=self.sdk('init.host_device',lambda:xrt.HostDevice.Default())
        self.own('session',lambda:xrt.Session.Create([zg330backend.ZG330Backend,host_backend.HostBackend],
            view,[self.device,host_device]))
        self.sdk('init.session_apply',lambda:self.session.apply())
        def await_hardop(session,op,backend,tensors):
            self.stop_control.hold(tensors) # callback outputs may never reach forward's caller
            self.stop_control.checkpoint('HardOp.callback')
            if 'ZG330' in backend.typeKey() and 'HardOp' in op.typeKey():
                for tensor in tensors:
                    self.sdk('HardOp.set_not_ready',lambda:tensor.setReady(False))
                    if not self.sdk('HardOp.wait',lambda:tensor.waitForReady(timedelta(seconds=10),timedelta(milliseconds=1))):
                        raise TimeoutError('HardOp completion timeout')
        self.callback=await_hardop
        self.sdk('init.callback',lambda:self.session.setPostCallBack(self.callback))
        forwards=self.sdk('init.forwards',lambda:self.session.getForwards())
        bindings=[dict(op_id=op.op_id,op=op.typeKey(),backend=backend.typeKey()) for op,backend,*_ in forwards]
        if not any('HardOp' in b['op'] and 'ZG330' in b['backend'] for b in bindings):raise ValueError('NPU binding absent')
        host_backend_handle=next(b for op,b,*_ in forwards if 'HostBackend' in b.typeKey())
        self.own('converter',lambda:host_backend.HostBackend(host_backend_handle))
        fp32=self.sdk('init.FP32_type',lambda:xir.FloatType.FP32())
        self.own('software_type',lambda:xir.TensorType(fp32,[1,40,100],self.input_type.layout))
        self.own('input',lambda:xrt.Tensor(self.software_type))
        host_region=self.sdk('init.host_region',lambda:xrt.HostDevice.MemRegion())
        self.sdk('init.host_input_malloc',lambda:self.input.mallocOn(host_region))
        region=self.sdk('init.memory_region',lambda:self.device.defaultMemRegion())
        self.own('chunk',lambda:region.malloc(ARENA,False,64))
        base=int(self.sdk('init.arena_address',lambda:self.chunk.begin.addr()))
        UdwLab.__init__(self,LocalTransport(self),base,ARENA,run_id=run_id,generation=1,
                        journal=self.events.append,timeout=5)
        self.ledger=LeaseLedger(boot_id=expected_boot,allocation_id=str(run_id)+':'+str(base),
            base=base,size=ARENA,run_id=run_id,generation=1,retained_owner=self.chunk)
        self.stop_control.bind_ledger(self.ledger)
        versions=dict(self.sdk('init.device_versions',lambda:self.device.version()))
        self.info=dict(startup=startup,boot_id=expected_boot,base=base,arena_bytes=ARENA,quota_bytes=QUOTA,
            bindings=bindings,versions=versions,input_type=str(self.input_type),pid=os.getpid(),
            mode='cancel_aware_shared_probe',control_transport=control,strict_visibility_proven=False,zero_copy=False)
        self.stop_control.checkpoint('initialize.after')
        return self.info

    def read_local(self,offset,size):
        return self.sdk('arena.read',lambda:super(ProbeBridge,self).read_local(offset,size))

    def _predict_retained(self,tensor):
        if len(tensor)!=16000 or not np.isfinite(np.frombuffer(tensor,dtype='<f4')).all():raise ValueError('Invalid tensor')
        begin=time.perf_counter_ns();key=None
        if self.pending is None:
            self.sdk('input.write',lambda:self.input.write(0,np.frombuffer(tensor,dtype=np.uint8),len(tensor)))
            source=self.input
        else:
            if self.pending_index>=len(self.pending):raise ValueError('Extra window')
            key=self.pending[self.pending_index];self.pending_index+=1
            self.ledger.observe_protected_readback(key,tensor)
            region=self.sdk('input.region',lambda:self.device.defaultMemRegion())
            source=self.ledger.bind(key,tensor_factory=lambda *a:self.sdk('input.bind',lambda:self.xrt.Tensor(*a)),
                dtype=self.software_type,chunk=self.chunk,region=region,allow_experimental_readback=True)
        bind_end=time.perf_counter_ns();call_ids=[]
        adapted=self.sdk('input.Cast',lambda:self.host.Cast(source,self.software_type,self.input_type,self.converter),
            lambda value:call_ids.append(self.ledger.retain_cast(value,source='shared' if key else 'software',key=key)))
        call_id=call_ids[0];cast_end=time.perf_counter_ns()
        self.stop_control.checkpoint('input.metadata')
        if int(adapted.dtype().bytes())!=16000:raise ValueError('Unexpected adapted storage size')
        self.adapted_info=dict(chunk_bytes=int(adapted.chunk().byte_size),offset=int(adapted.offset()),dtype=str(adapted.dtype()))
        region=self.sdk('input.adapted_region',lambda:self.device.defaultMemRegion())
        on_device=bool(adapted.isOn(region))
        binding=dict(source='shared' if key else 'software',allocation_id=key.allocation_id if key else None,
            tensor_offset=int(source.offset()) if key else None,bound_address=int(source.data().addr()) if key else None,
            adapted_is_on_same_device_region=on_device,adapted_address=int(adapted.data().addr()) if on_device else None,
            adapted_offset=int(adapted.offset()),retained_call_id=call_id,same_buffer_proven_for_HardOp=False,
            explicit_host_input_write_bytes=0 if key else 16000,protective_readback_removed=False,storage_bytes=16000)
        self.stop_control.checkpoint('NPU.before_mark');self.ledger.npu_may_start(call_id)
        outputs=self.sdk('NPU.forward',lambda:self.session.forward([adapted]),self.ledger.forward_outputs.append)
        forward_end=time.perf_counter_ns()
        if len(outputs)!=1:raise ValueError('Wrong NPU output count')
        self.sdk('NPU.output_not_ready',lambda:outputs[0].setReady(False))
        if not self.sdk('NPU.output_wait',lambda:outputs[0].waitForReady(timedelta(seconds=10),timedelta(milliseconds=1))):
            raise TimeoutError('NPU output timeout')
        ready_end=time.perf_counter_ns();stream=io.BytesIO()
        self.sdk('NPU.output_dump',lambda:outputs[0].dump(stream,'SFB'))
        logits=np.frombuffer(stream.getvalue(),dtype='<f4');end=time.perf_counter_ns()
        if logits.shape!=(3,) or not np.isfinite(logits).all():raise ValueError('Invalid logits')
        self.ledger.npu_completed(call_id);self.call_count+=1
        self.stop_control.checkpoint('NPU.completed')
        return dict(logits=logits.tolist(),class_index=int(logits.argmax()),npu_call=self.call_count,
            tensor_sha256=digest(tensor),shared_input=binding,
            stages_ns=dict(shared_bind=bind_end-begin,host_input_cast=cast_end-bind_end,forward=forward_end-cast_end,
                output_wait=ready_end-forward_end,output_read=end-ready_end,npu_total=end-begin))

    def submit_local(self,payload,continuation):
        self.ensure_active()
        try:
            if type(continuation) is not bool or self.pending is not None:raise ValueError('Invalid concurrent/continuation request')
            selected,identities=selected_endpoints(payload,self.identities if continuation else ())
            if self.ledger.used!=self.used:raise ValueError('Lease accounting differs')
            keys=self.ledger.reserve_batch(len(payload),selected,self.request_id+1);self.ledger.pl_may_start(keys)
            self.pending,self.pending_index=keys,0
            read_begin=len(self.explicit_reads)
            result=self._submit_guarded(payload,continuation,selected,identities)
            if self.pending_index!=len(keys):raise ValueError('Missing shared window')
            result['explicit_arena_read_requests']=self.explicit_reads[read_begin:]
            result['SDK_internal_transfer_bytes']=None
            return result
        except BaseException as error:self.quarantine(error);raise
        finally:self.pending=None

    def _submit_guarded(self,payload,continuation,selected,identities):
        begin=time.perf_counter_ns();records=len(payload)//256
        capacity=max(64,len(selected)*OUTPUT_STRIDE);size=len(payload)+capacity+256
        if self.used+size+4096>QUOTA:raise ValueError('Finite quota exhausted')
        before=self.guard('before_batch')
        if before[8]==2:self.issue(2,0)
        elif before[8]!=0:raise ValueError('Invalid PL state')
        offset=self.used;self.used+=size
        output_offset=offset+len(payload)+192
        image=self.GUARD+payload+self.GUARD*2+bytes([self.POISON])*capacity+self.GUARD
        self.sdk('arena.write',lambda:self.chunk.write(offset,np.frombuffer(image,dtype=np.uint8),len(image)))
        if self.read_local(offset,size)!=image:raise ValueError('Preparation readback differs')
        prepared=time.perf_counter_ns();request=self.request_id+1
        descriptor=Descriptor(self.base+offset+64,len(payload),self.base+output_offset,capacity,
            records,len(selected),self.run_id,request,self.generation)
        shadow=descriptor.shadow_registers(address_limit_exclusive=self.base+self.size)
        self.guard('before_descriptor')
        for k,v in shadow.items():self.reg.write(REG+k,v,False)
        actual_shadow=self.guard('after_descriptor')
        if any(actual_shadow[k]!=v for k,v in shadow.items()):raise ValueError('Shadow differs')
        self.stop_control.checkpoint('PL.before_command')
        terminal=self.issue(4 if continuation else 1,2)
        counts={0x40:records,0x44:len(selected),0x48:len(selected)*OUTPUT_STRIDE,0x4c:records*4,0x50:len(selected)*252,0x58:1}
        if any(terminal.get(k)!=v for k,v in counts.items()):raise ValueError('Terminal counts differ')
        pl_end=time.perf_counter_ns();self.guard('before_local_read');actual=self.read_local(offset,size);self.guard('after_local_read')
        relative=output_offset-offset;valid=len(selected)*OUTPUT_STRIDE
        if actual[:relative]!=image[:relative] or actual[relative+valid:]!=image[relative+valid:]:raise ValueError('Guard/input changed')
        tensors=[]
        for i,(first,last) in enumerate(selected):
            tensor,a,b=unpack_output(actual[relative+i*OUTPUT_STRIDE:relative+(i+1)*OUTPUT_STRIDE],
                run_id=self.run_id,request_id=request,window_index=i,generation=self.generation)
            if a!=first or b!=last:raise ValueError('WDO endpoints differ')
            tensors.append(tensor)
        read_end=time.perf_counter_ns();predictions=[]
        for tensor,(first,last) in zip(tensors,selected):
            self.guard('before_local_npu');result=self.predict_local(tensor);self.guard('after_local_npu')
            result.update(first=list(first),last=list(last));predictions.append(result)
        npu_end=time.perf_counter_ns()
        if self.read_local(offset,size)!=actual:raise ValueError('Current slot changed')
        self.guard('after_batch');self.retained.append((offset,actual));self.identities,self.request_id=identities,request
        self.stop_control.checkpoint('PL.batch_done');end=time.perf_counter_ns()
        return dict(request_id=request,records=records,predictions=predictions,used=self.used,
            descriptor=descriptor.__dict__,terminal=terminal,input_sha256=digest(payload),strict_visibility_proven=False,zero_copy=False,
            stages_ns=dict(validate_prepare=prepared-begin,descriptor_PL=pl_end-prepared,local_read_verify=read_end-pl_end,
                npu_with_health=npu_end-read_end,final_check=end-npu_end,board_total=end-begin))

    def finish(self,directory):
        self.stop_control.checkpoint('audit.before')
        result=super().finish(directory)
        self.stop_control.checkpoint('audit.after')
        return result
