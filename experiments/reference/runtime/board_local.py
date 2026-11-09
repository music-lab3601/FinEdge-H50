"""ARM resident Icraft prototype; validated loopback, gated AXI diagnostic."""
from datetime import timedelta
import io
import os
import sys
from pathlib import Path
import time
import numpy as np
from software.runtime.backends import prediction, validate_tensor
from shared.model_paths import ARTIFACTS


class BoardLocalBackend:
    name='icraft_local'

    def __init__(self, url='socket://zg330aiu@127.0.0.1:9981', allow_unverified_axi=False):
        if os.name=='nt':raise RuntimeError('Local prototype must run on the board')
        if url not in ('axi://zg330aiu','socket://zg330aiu@127.0.0.1:9981'):
            raise ValueError('Only documented local AXI or explicit loopback comparison is supported')
        if url.startswith('axi:') and not allow_unverified_axi:
            raise RuntimeError('Direct AXI failed numerical validation; use loopback or an explicit diagnostic')
        # The official BOOT also contains an older /usr/local/.../icraft Python
        # package. Use the extensions owned by the installed matching .deb.
        sdk_dir=Path('/usr/lib/aarch64-linux-gnu')
        if str(sdk_dir) not in sys.path:sys.path.insert(0,str(sdk_dir))
        import xir,xrt,host_backend,zg330backend
        for module in (xir,xrt,host_backend,zg330backend):
            if Path(module.__file__).parent!=sdk_dir:raise RuntimeError('Mismatched board Python SDK module')
        self.xrt,self.host=xrt,host_backend
        begin=time.perf_counter_ns()
        self.network=xir.Network.CreateFromJsonFile(str(ARTIFACTS/'icraft/tcn_ed4_ZG.json'))
        self.network.loadParamsFromFile(str(ARTIFACTS/'icraft/tcn_ed4_ZG.raw'))
        view=self.network.view(0)
        if len(view.inputs())!=1 or len(view.outputs())!=1:raise ValueError('Wrong external IO count')
        self.input_type=view.inputs()[0].tensorType()
        if list(self.input_type.shape)!=[1,40,100]:raise ValueError('Changed input shape')
        if url.startswith('socket:'):self.name='icraft_loopback'
        self.device=xrt.Device.Open(url)
        self.session=xrt.Session.Create([zg330backend.ZG330Backend,host_backend.HostBackend],
                                        view,[self.device,xrt.HostDevice.Default()])
        self.session.apply()
        def await_hardop(session,op,backend,tensors):
            if 'ZG330' in backend.typeKey() and 'HardOp' in op.typeKey():
                for tensor in tensors:
                    tensor.setReady(False)
                    if not tensor.waitForReady(timedelta(seconds=10),timedelta(milliseconds=1)):
                        raise TimeoutError('HardOp completion timed out before Host output copy')
        self._post_callback=await_hardop
        self.session.setPostCallBack(self._post_callback)
        bindings=[{'op_id':op.op_id,'op_type':op.typeKey(),'backend':backend.typeKey()}
                  for op,backend,*_ in self.session.getForwards()]
        if not any('HardOp' in b['op_type'] and 'ZG330' in b['backend'] for b in bindings):
            raise RuntimeError('No NPU HardOp binding; refusing fallback')
        # ARM HostBackend cannot simulate a compiled HardOp. Reuse the existing
        # Session Host adapter, which is bound only to supported input/output ops.
        self.converter=host_backend.HostBackend(next(backend for op,backend,*_ in self.session.getForwards()
                                                     if 'HostBackend' in backend.typeKey()))
        self.software_type=xir.TensorType(xir.FloatType.FP32(),[1,40,100],self.input_type.layout)
        self.input=xrt.Tensor(self.software_type)
        self.input.mallocOn(xrt.HostDevice.MemRegion())
        self.info={'backend':self.name,'url':url,'npu_bound':True,
                   'bindings':bindings,'versions':dict(self.device.version()),
                   'python_modules':{m.__name__:m.__file__ for m in (xir,xrt,host_backend,zg330backend)},
                   'load_ns':time.perf_counter_ns()-begin,'interface':'installed Icraft Python bindings'}
        self.info['completion_policy']='HardOp post callback invalidates cached readiness and waits BEFORE Host Output copy'

    def predict(self,window_id,tensor):
        tensor=validate_tensor(tensor)
        begin=time.perf_counter_ns()
        self.input.write(0,tensor,tensor.nbytes)
        adapted=self.host.Cast(self.input,self.software_type,self.input_type,self.converter)
        cast_end=time.perf_counter_ns()
        outputs=self.session.forward([adapted])
        forward_end=time.perf_counter_ns()
        if len(outputs)!=1:raise RuntimeError('Wrong output count')
        outputs[0].setReady(False)
        if not outputs[0].waitForReady(timedelta(seconds=10),timedelta(milliseconds=1)):
            raise TimeoutError('Local NPU completion timed out')
        ready_end=time.perf_counter_ns()
        buffer=io.BytesIO();outputs[0].dump(buffer,'SFB')
        logits=np.frombuffer(buffer.getvalue(),dtype='<f4').copy()
        end=time.perf_counter_ns()
        return prediction(window_id,logits,self.name,end-begin,device_call_ns=end-begin,
            stages={'input_cast_ns':cast_end-begin,'forward_submit_ns':forward_end-cast_end,
                    'completion_wait_ns':ready_end-forward_end,'output_read_ns':end-ready_end})

    def close(self):
        # Ownership release is local to this process; never reset the board/FPGA.
        self.session=None
        self.device=None
