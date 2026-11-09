"""Supported ZG330 userConnectNetwork input binding, before Session.apply.

No Host Cast, Tensor.to, input.write or per-window input allocation. Requires
the separate ZCW1 FPGA image targeting the NPU's default PL-DDR region.
All physical-binding evidence must pass; there is no copy fallback.
"""
from __future__ import annotations

from datetime import timedelta
import io
from pathlib import Path
import threading

import numpy as np

from experiments.reference.runtime.ps_npu_runtime import BoardNpuRuntime, DEVICE_URL, VERSIONS
from experiments.reference.runtime.shared_input_contract import MAGIC, ABI, CAPABILITIES, TENSOR_BYTES, SLOT_BYTES, INPUT_BYTES, REG_BASE, Lease, validate_extent


class SharedInputNpu(BoardNpuRuntime):
    def initialize(self,*args,**kwargs):
        raise RuntimeError('Shared input requires initialize_shared; legacy initialization disabled')

    def predict(self,*args,**kwargs):
        raise RuntimeError('Copied-input prediction disabled; an owned shared lease is required')

    def allocate_ps_arena(self,*args,**kwargs):
        raise RuntimeError('PS udma allocation disabled in the PL-DDR shared runtime')

    def initialize_shared(self, modules, model):
        if self.initialize_attempted: raise RuntimeError('Initialize once after a fresh physical boot')
        self.initialize_attempted = True
        with self.operation('shared.initialize'):
            self.xir, self.xrt = modules['xir'], modules['xrt']
            self.host, self.zg = modules['host_backend'], modules['zg330backend']
            self.device,_ = self.safe_call('device.open',lambda:self.xrt.Device.Open(DEVICE_URL))
            versions,_ = self.safe_call('device.version',self.device.version)
            self.versions = dict(versions)
            if self.versions != VERSIONS: raise ValueError('Device/core version differs')
            self.open_axi_register_region()
            expected = {0xC0:MAGIC,0x8C:ABI,0x90:CAPABILITIES}
            for offset,value in expected.items():
                if self.reg_read(offset) != value: raise ValueError('New ZCW1 firmware required before allocation')
            for offset in (4,8,12,0x40,0x44,0x48,0x4C,0x50,0x54,0x58,0x60,0x64,0x68,0x6C,0x70,0x74,0x78):
                if self.reg_read(offset): raise ValueError('Fresh clean FPGA state required')
            self.network,_ = self.safe_call('network',lambda:self.xir.Network.CreateFromJsonFile(str(model)))
            self.safe_call('parameters',lambda:self.network.loadParamsFromFile(str(Path(model).with_suffix('.raw'))))
            # Exclude only the original Host Input node. The first retained
            # op is the compiled NPU FP32->TF32 Cast, not a removed math op.
            self.view,_ = self.safe_call('network.view',lambda:self.network.view(1))
            self.input_type = self.view.inputs()[0].tensorType()
            if list(self.input_type.shape) != [1,40,100] or int(self.input_type.bytes()) != TENSOR_BYTES:
                raise ValueError('Frozen dense 16000-byte model input required')
            self.host_device,_ = self.safe_call('host.device',self.xrt.HostDevice.Default)
            self.session,_ = self.safe_call('session.create',lambda:self.xrt.Session.Create(
                [self.zg.ZG330Backend,self.host.HostBackend],self.view,[self.device,self.host_device]))
            self.backend = self.zg.ZG330Backend(self.session.backends[0])
            self.npu_region,_ = self.safe_call('shared.region',self.device.defaultMemRegion)
            self.region_type,_ = self.safe_call('shared.region_type',self.npu_region.typeKey)
            if 'plddr' not in str(self.region_type).lower():
                raise ValueError('NPU default region must be PL DDR, never the PS udma exchange')
            self.manager,_ = self.safe_call('shared.manager',self.npu_region.memManager)
            existing,_ = self.safe_call('shared.inventory',self.manager.getAllMemChunk)
            # SDK owns address allocation; never import or fabricate MemPtr.
            self.device_allocation_attempts += 1
            self.shared_chunk,_ = self.safe_call('shared.malloc',lambda:self.npu_region.malloc(SLOT_BYTES,False,4096))
            self.device_allocation_returns += 1
            self.device_allocation_attempts += 1
            self.records_chunk,_ = self.safe_call('records.malloc',lambda:self.npu_region.malloc(INPUT_BYTES,False,4096))
            self.device_allocation_returns += 1
            for chunk,size in ((self.shared_chunk,SLOT_BYTES),(self.records_chunk,INPUT_BYTES)):
                if chunk.begin.ptype() not in (self.xrt.PtrType.ADDR,self.xrt.PtrType.BOTH) or not chunk.isOn(self.npu_region):
                    raise ValueError('Owned addressable PL-DDR allocation required')
                validate_extent(int(chunk.begin.addr()),int(chunk.byte_size),size)
                for old in existing:
                    a,b = int(chunk.begin.addr()),int(old.begin.addr())
                    if max(a,b) < min(a+int(chunk.byte_size),b+int(old.byte_size)):
                        raise ValueError('Shared allocation overlaps an existing SDK owner')
            self.shared_address = int(self.shared_chunk.begin.addr())
            self.records_address = int(self.records_chunk.begin.addr())
            if max(self.shared_address,self.records_address) < min(self.shared_address+SLOT_BYTES,self.records_address+INPUT_BYTES):
                raise ValueError('Input allocations overlap')
            self.input_id = int(self.view.inputs()[0].v_id)
            self.safe_call('shared.connect',lambda:self.backend.userConnectNetwork(self.shared_chunk,self.input_id))
            self.safe_call('session.apply',self.session.apply)
            self.input,_ = self.safe_call('shared.tensor',lambda:self.xrt.Tensor(self.input_type,self.shared_chunk,0))
            self.safe_call('shared.input.not_ready',lambda:self.input.setReady(False))
            self.forwards,_ = self.safe_call('session.forwards',self.session.getForwards)
            self.bindings = tuple((int(op.op_id),op.typeKey(),backend.typeKey()) for op,backend,*_ in self.forwards)
            if (len(self.bindings) != 2 or 'HardOp' not in self.bindings[0][1]
                    or 'ZG330' not in self.bindings[0][2] or 'Output' not in self.bindings[1][1]):
                raise ValueError('Expected original merged NPU HardOp and Output only')
            self.hardop_ids = (self.bindings[0][0],)
            self.callback = self._await_hardop
            self.safe_call('session.callback',lambda:self.session.setPostCallBack(self.callback))
            self.binding_evidence = self.inspect_binding()
            self.initialized = True

    def inspect_binding(self):
        info = self.backend.forward_info
        entry = info.memchunk_map[self.input_id]
        value = info.value_map[self.input_id]
        chunk = entry.memChunk
        address = int(chunk.begin.addr())
        self.last_binding_observation=dict(user_used=bool(entry.user_used),staged_chunks=len(entry.staged_chunk),
            registered_chunk_address=address,physical_address=int(value.phy_addr),value_bytes=int(value.byte_size),
            tensor_address=int(self.input.data().addr()),tensor_offset=int(self.input.offset()),
            real=bool(value.real),is_host=bool(value.is_host),is_ocm=bool(value.is_ocm))
        if (entry.user_used is not True or len(entry.staged_chunk) != 0
                or address != self.shared_address or int(value.phy_addr) != self.shared_address
                or value.real is not True or value.is_host or value.is_ocm
                or int(value.byte_size) != TENSOR_BYTES
                or not chunk.isOn(self.npu_region) or not self.input.isOn(self.npu_region)
                or int(self.input.data().addr()) != address or int(self.input.offset()) != 0):
            raise ValueError('Backend did not retain direct same-address input binding')
        return dict(api='ZG330Backend.userConnectNetwork',input_value_id=self.input_id,
            region_type=str(self.region_type),
            fpga_output_address=address,backend_input_address=int(value.phy_addr),tensor_address=int(self.input.data().addr()),
            bytes=TENSOR_BYTES,allocation_bytes=int(chunk.byte_size),staged_input_chunks=0,
            user_used=True,explicit_input_copy_bytes=0,hidden_backend_transfer_bytes=None,
            physical_bus_trace_validated=False)

    def snapshot(self):
        result=super().snapshot()
        result.update(shared_input_binding=getattr(self,'last_binding_observation',None),
            shared_address=getattr(self,'shared_address',None),records_address=getattr(self,'records_address',None),
            shared_region_type=str(getattr(self,'region_type','not_created')),
            explicit_host_tensor_input_write_bytes=0,SDK_internal_input_transfer_bytes=None)
        return result

    def reg_read(self, offset):
        value,_ = self.safe_call('zcw.reg.read',lambda:self.axi_registers.read(REG_BASE+offset,False))
        return int(value)

    def bind_controller(self,controller):
        from experiments.reference.runtime.shared_input_runtime import SharedInputController
        if (not self.initialized or getattr(self,'shared_controller',None) is not None
                or not isinstance(controller,SharedInputController) or not controller.initialized
                or controller.shared_base!=self.shared_address or controller.input_base!=self.records_address):
            raise ValueError('Bind this initialized shared allocation to exactly one controller')
        self.shared_controller=controller

    def reg_write(self, offset, value):
        self.safe_call('zcw.reg.write',lambda:self.axi_registers.write(REG_BASE+offset,value,False))

    def write_records(self, payload):
        if type(payload) is not bytes or not payload or len(payload)%256 or len(payload)>INPUT_BYTES:
            raise ValueError('1..32 WDM records required')
        with self.operation('shared.records'):
            array = self.hold(np.frombuffer(payload,dtype=np.uint8))
            self.safe_call('records.write',lambda:self.records_chunk.write(0,array,len(payload)))
            # Only the newly arrived records are checked, normally 256 bytes.
            # This does not read/copy the 16000-byte output tensor.
            received = self.hold(np.empty(len(payload),dtype=np.uint8))
            self.safe_call('records.verify',lambda:self.records_chunk.read(received,0,len(payload)))
            if received.tobytes()!=payload:raise ValueError('New-record transport integrity mismatch')

    def read_shared(self, offset, size):
        if type(offset) is not int or type(size) is not int or size<=0 or offset<0 or offset+size>SLOT_BYTES:
            raise ValueError('Shared read outside owned input allocation')
        with self.operation('shared.read'):
            array = self.hold(np.empty(size,dtype=np.uint8))
            self.safe_call('shared.read',lambda:self.shared_chunk.read(array,offset,size))
            return array.tobytes()

    def predict_shared(self, lease, metadata):
        if not self.initialized or not isinstance(lease,Lease): raise RuntimeError('Initialized shared owner required')
        controller=getattr(self,'shared_controller',None)
        if controller is None or controller.owner.state!='NPU_READING' or controller.owner.lease!=lease:
            raise RuntimeError('Controller has not transferred this shared lease to NPU')
        if metadata.get('request_id') != lease.request_id or not metadata.get('hardware_read_compare'):
            raise ValueError('Producer verification/identity missing')
        with self.operation('shared.predict',isolated=True):
            parent = self.active['parent']
            while parent is not None:
                if parent['prediction_started']: raise RuntimeError('Reentrant shared prediction')
                parent = parent['parent']
            self.active['prediction_started'] = True
            begin = self.clock()
            self.inspect_binding()
            # SDK readiness is metadata, not a cache/write-completion barrier.
            # The controller already verified the external producer's state.
            self.safe_call('shared.input.ready',lambda:self.input.setReady(True))
            self.active['npu_may_be_reading'] = True
            self.npu_calls += 1
            outputs,forward_ns = self.safe_call('shared.forward',lambda:self.session.forward([self.input]))
            if tuple(self.active['hardops']) != self.hardop_ids or len(outputs) != 1:
                raise ValueError('Original HardOp completion missing')
            self.safe_call('output.not_ready',lambda:outputs[0].setReady(False))
            ready,wait_ns = self.safe_call('output.wait',lambda:outputs[0].waitForReady(timedelta(seconds=10),timedelta(milliseconds=1)))
            if not ready: raise TimeoutError('Shared NPU output completion timeout')
            stream = self.hold(io.BytesIO())
            _,copy_ns = self.safe_call('output.dump',lambda:outputs[0].dump(stream,'SFB'))
            raw = stream.getvalue()
            if len(raw) != 12: raise ValueError('Expected three float32 logits')
            logits = np.frombuffer(raw,dtype='<f4')
            if not np.isfinite(logits).all(): raise ValueError('Invalid shared logits')
            self.inspect_binding()  # catches staged/new input chunks after actual forward
            self.safe_call('shared.input.not_ready',lambda:self.input.setReady(False))
            self.active['output_copied'] = True
            self.stop.checkpoint('shared.predict:completed')
            self.completed_invocations += 1
            return dict(logits=logits.tolist(),class_index=int(logits.argmax()),backend='board_shared_plddr_npu',
                input_consumed=True,whole_predict_ns=self.clock()-begin,
                stages_ns=dict(forward_including_hardop_wait=forward_ns,output_wait=wait_ns,output_copy=copy_ns),
                input_binding=dict(self.binding_evidence),explicit_input_copy_bytes=0)
