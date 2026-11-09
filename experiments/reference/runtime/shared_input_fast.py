"""Latency-first resident-instruction path, separate from the accepted audit path.

The initialized SDK owns weights/instructions/buffers. Hot path uses cached
addresses and a tiny MMIO control loop; no per-window binding maps or readback.
"""
from collections import deque
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import ctypes
import io
import struct
import time
import numpy as np
from experiments.reference.runtime.shared_input_sdk import SharedInputNpu
from experiments.reference.runtime.ps_window_runtime import selected_endpoints


class NativeControl:
    def __init__(self,path):
        self.lib=ctypes.CDLL(str(path));self.lib.sf_open.restype=ctypes.c_int
        self.lib.sf_read.argtypes=[ctypes.c_uint32];self.lib.sf_read.restype=ctypes.c_uint32
        self.lib.sf_write.argtypes=[ctypes.c_uint32,ctypes.c_uint32]
        self.lib.sf_window.argtypes=[ctypes.POINTER(ctypes.c_uint32),ctypes.c_uint32,ctypes.c_uint32]
        self.lib.sf_icore.argtypes=[ctypes.c_uint32]*4
        self.lib.sf_udma_open.argtypes=[ctypes.c_uint32]
        self.lib.sf_put_records.argtypes=[ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32]
        self.lib.sf_get_logits.argtypes=[ctypes.c_uint32,ctypes.c_void_p]
        self.lib.sf_dma_selftest.argtypes=[ctypes.c_uint32]
        self.lib.sf_count_done.argtypes=[ctypes.c_uint32]*3
        if (self.lib.sf_count_done(195,195,195) or not self.lib.sf_count_done(390,195,195)
                or not self.lib.sf_count_done(94,0xffffff9b,195)):
            raise RuntimeError('Native cumulative completion self-test failed')
        if self.lib.sf_open():raise RuntimeError('MMIO mapping failed')
    def command(self,desc,op,seq):
        values=(ctypes.c_uint32*12)(*struct.unpack('<12I',desc)) if desc else None
        if self.lib.sf_window(values,op,seq):raise RuntimeError('FPGA command failed or timed out')


class FastNpu(SharedInputNpu):
    def safe_call(self,stage,operation):
        if stage=='session.apply' and getattr(self,'native_output_binding',False):
            # Device DMA operates in 64-byte blocks. Explicitly reserve an
            # aligned output allocation; do not over-read a 12-byte SDK chunk.
            self.fast_output_chunk=self.npu_region.malloc(4096,False,4096)
            self.backend.userConnectNetwork(self.fast_output_chunk,569)
        return super().safe_call(stage,operation)

    def initialize_fast(self,modules,model):
        # Explicit process-local transport choice. Reuse the already accepted
        # allocation/binding implementation without changing its frozen source.
        # This process has one thread/owner during initialization.
        from experiments.reference.runtime import shared_input_sdk
        self.native_output_binding=True
        prior=shared_input_sdk.DEVICE_URL
        shared_input_sdk.DEVICE_URL='socket://zg330aiu@127.0.0.1:9981'
        try:self.initialize_shared(modules,model)
        finally:shared_input_sdk.DEVICE_URL=prior
        self.fast_device_transport='SDK socket initialization; native MMIO and DMA hot path'

    def _await_hardop(self,session,op,backend,tensors):
        if 'HardOp' not in op.typeKey():return
        self.hard_tensor=tensors[0]
        self.hard_tensor.setReady(False)
        if not self.hard_tensor.waitForReady(timedelta(seconds=2),timedelta(0)):raise TimeoutError('Startup completion')
        self.active['hardops'].append(int(op.op_id))

    def configure_fast(self,native):
        self.native=native;self.output_array=np.empty(3,dtype='<f4')
        info=self.backend.forward_info
        self.cached_values=info.value_map;self.cached_chunks=info.memchunk_map
        self.cached_hardops=info.hardop_map
        hop=self.cached_hardops[self.hardop_ids[0]]
        self.resident_meta=dict(instruction_address=int(hop.instr_phy_addr),instruction_bytes=int(hop.instr_size),
            sync_index=int(hop.sync_idx[0]),completion_layers=int(hop.sync_idx[1]))
        self.local_core=self.xrt.ZG330Device(self.axi_device)
        self.master=int(self.local_core.master_cid) if hasattr(self.local_core,'master_cid') else 1
        self.input_state=(0,0)
        self.started=False;self.calls=[];self.last_outputs=None
        # Keep the device output before the zero-instruction Host IO cast.
        self.output_id=569
        v=self.cached_values[self.output_id]
        self.raw_output_meta=dict(id=self.output_id,address=int(v.phy_addr),bytes=int(v.byte_size),
            is_host=bool(v.is_host),is_ocm=bool(v.is_ocm),real=bool(v.real))
        if v.is_host or v.is_ocm or int(v.byte_size)<12:raise ValueError('Expected external 3-class output')
        self.output_offset=int(v.phy_addr)
        # Resolve aliases through the owning SDK chunks once, never per-window.
        self.output_chunk=None
        for entry in self.cached_chunks.values():
            chunk=entry.memChunk;start=int(chunk.begin.addr())
            if chunk.isOn(self.npu_region) and start<=self.output_offset and self.output_offset+12<=start+int(chunk.byte_size):
                self.output_chunk=chunk;self.output_offset-=start;break
        if self.output_chunk is None:raise ValueError('No owning output chunk')
        if self.output_offset+64>int(self.output_chunk.byte_size):raise ValueError('Output DMA alignment exceeds owned chunk')
        sysfs=Path('/sys/class/u-dma-buf/udmabuf0')
        observed={n:int((sysfs/n).read_text().strip(),0) for n in ('phys_addr','size','sync_mode')}
        if observed!={'phys_addr':0x30100000,'size':134217728,'sync_mode':1}:raise ValueError('UDMA mapping differs')
        self.dma_region=self.device.getMemRegion('udma')
        self.dma_chunk=self.dma_region.malloc(16384,False,4096)
        self.dma_address=int(self.dma_chunk.begin.addr())
        if native.lib.sf_udma_open(self.dma_address):raise RuntimeError('UDMA map failed')
        if native.lib.sf_dma_selftest(self.records_address):raise RuntimeError('Native DMA roundtrip failed')
        self.dma_metadata=dict(address=self.dma_address,bytes=16384,sysfs=observed,
            mapping='O_SYNC /dev/udmabuf0',roundtrip_selftest_passed=True)

    def write_fast_records(self,payload):
        if self.native.lib.sf_put_records(self.records_address,ctypes.c_char_p(payload),len(payload)):
            raise RuntimeError('Native record DMA failed')

    def read_fast_logits(self):
        if self.native.lib.sf_get_logits(self.raw_output_meta['address'],self.output_array.ctypes.data):
            raise RuntimeError('Native output DMA failed')

    def _warm(self):
        # Startup only: full SDK initializes launch address, timer and IO handling.
        with self.operation('fast.warm',isolated=True):
            self.active['prediction_started']=True
            self.input.setReady(True)
            self.last_outputs=self.session.forward([self.input])
            out=self.last_outputs[0];out.setReady(False)
            if not out.waitForReady(timedelta(seconds=2),timedelta(0)):raise TimeoutError('Warm output')
            stream=io.BytesIO();out.dump(stream,'SFB')
            return np.frombuffer(stream.getvalue(),dtype='<f4').copy()

    def predict_fast(self,reference=None):
        if not self.started:
            # Profile the original suspect once outside the timed hot path.
            begin=time.perf_counter_ns();self.inspect_binding();binding_ns=time.perf_counter_ns()-begin
            self.startup_profile={'single_binding_inspection_ns':binding_ns,
                'value_map_entries':len(self.cached_values),'memchunk_map_entries':len(self.cached_chunks)}
            for i in range(11):
                golden=self._warm()
                if reference is not None and not np.allclose(golden,reference,rtol=1e-3,atol=1e-4):raise ValueError('Startup known output differs')
            self.output_chunk.read(self.output_array,self.output_offset,12)
            if not np.array_equal(self.output_array,golden):
                raise ValueError('Raw output representation differs: raw='+repr(self.output_array.view('uint32').tolist())+' expected='+repr(golden.view('uint32').tolist()))
            self.read_fast_logits()
            if not np.array_equal(self.output_array,golden):raise ValueError('Native DMA output differs from SDK')
            # Configure the local SDK launch cache once using the SAME resident
            # instruction image. It configures registers and starts one self-check.
            self.native.lib.sf_write(0x40000+((6<<7)+self.input_state[0]+10)*4,self.input_state[1])
            layer_register=0x40000+((6<<7)+25)*4
            before=int(self.native.lib.sf_read(layer_register))
            self.local_core.icoreCalc(self.resident_meta['instruction_address'],self.resident_meta['instruction_bytes'])
            until=time.monotonic()+2
            while not self.native.lib.sf_count_done(self.native.lib.sf_read(layer_register),before,self.resident_meta['completion_layers']):
                if time.monotonic()>until:raise TimeoutError('Resident setup')
            self.output_chunk.read(self.output_array,self.output_offset,12)
            if not np.array_equal(self.output_array,golden):raise ValueError('Resident setup output differs')
            self.startup_profile.update(warmups=11,resident_self_checks=1,raw_output_matches_sdk=True,
                counter_is_cumulative=True,completion_policy='wait_for_this_launch_layer_delta',native_counter_selftests=3)
            self.started=True
        begin=time.perf_counter_ns()
        if self.native.lib.sf_icore(self.master,*self.input_state,self.resident_meta['completion_layers']):raise TimeoutError('Resident NPU timeout')
        done=time.perf_counter_ns()
        self.read_fast_logits()
        end=time.perf_counter_ns();self.npu_calls+=1;self.completed_invocations+=1
        return dict(logits=self.output_array.tolist(),class_index=int(self.output_array.argmax()),input_consumed=True,
            backend='resident_npu_mmio',whole_predict_ns=end-begin,
            stages_ns={'resident_launch_and_wait':done-begin,'output_read':end-done},
            explicit_input_copy_bytes=0,input_binding=self.binding_evidence)


class FastController:
    def __init__(self,owner,native,run_id,reference):
        self.npu,self.native,self.run_id,self.reference=owner,native,run_id,reference
        self.history=deque(maxlen=100);self.sequence=self.request=self.windows=0
        self.owner=SimpleNamespace(state='FREE')
    def submit(self,payload):
        if self.owner.state!='FREE':raise RuntimeError('One shared slot at a time')
        self.owner.state='FPGA_WRITING'
        begin=time.perf_counter_ns()
        try:
            expected,history=selected_endpoints(payload,self.history)
            if len(expected)>1:raise ValueError('One output per native command')
            self.request+=1
            self.npu.write_fast_records(payload)
            written=time.perf_counter_ns()
            desc=struct.pack('<6I3Q',self.npu.records_address,len(payload),self.npu.shared_address,16384,
                len(payload)//256,len(expected),self.run_id,self.request,self.request)
            self.sequence+=1;self.native.command(desc,1 if self.request==1 else 4,self.sequence)
            produced=time.perf_counter_ns();windows=[]
            if expected:
                self.owner.state='NPU_READING'
                included_startup=not self.npu.started
                row=self.npu.predict_fast(self.reference)
                predicted=time.perf_counter_ns();self.windows+=1
                first,last=expected[0]
                row.update(window_id=self.windows,metadata={'first':list(first),'last':list(last),'request_id':self.request},sampled=False,
                    frontend_stages_ns={'new_record_write':written-begin,'fpga_window_and_wait':produced-written},
                    input_ready_to_result_ns=predicted-produced,startup_included=included_startup)
                windows=[row]
            self.owner.state='NPU_COMPLETE';self.sequence+=1;self.native.command(None,2,self.sequence)
            self.owner.state='FREE';self.history=history
            end=time.perf_counter_ns()
            for row in windows:row['record_to_result_and_rearm_ns']=end-begin
            return {'request_id':self.request,'records':len(payload)//256,'windows':windows}
        except BaseException:
            self.owner.state='FAILED';raise
    def summary(self):
        return dict(state=self.owner.state,request_id=self.request,windows=self.windows,
            explicit_shared_read_bytes=0,full_tensor_sample_reads=0,metadata_reads=0,
            explicit_full_tensor_input_copy_bytes=0,hardware_read_compare=True,
            policy='latency_first_native_mmio_no_per_window_readback_or_binding_checks')
