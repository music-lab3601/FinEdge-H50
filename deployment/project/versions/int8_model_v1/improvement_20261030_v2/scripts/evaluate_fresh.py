import subprocess
import numpy as np
import torch
import onnxruntime as ort
from common2 import ROOT,BASE,PROJECT,read,save,sha,env


def main():
    frozen=read(ROOT/'contracts/frozen_candidate.json')
    if frozen['model_sha256']!=sha(ROOT/frozen['model_path']):raise ValueError('Candidate changed')
    inputs=np.load(ROOT/'data/inputs.npy')
    out=ROOT/'artifacts/fresh';out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model=torch.jit.load(str(PROJECT/'vendor/ed4_model/model_cpu.ts'),map_location='cpu').eval()
    opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
    onnx=ort.InferenceSession(str(BASE/'model/tcn_polyphase.onnx'),opts,providers=['CPUExecutionProvider'])
    a=[];b=[]
    with torch.inference_mode():
        for x in inputs:
            a.append(model(torch.from_numpy(x[None])).numpy()[0]);b.append(onnx.run(None,{'lob':x[None]})[0][0])
    a,b=np.array(a,dtype='<f4'),np.array(b,dtype='<f4')
    np.save(out/'original_fp32.npy',a);np.save(out/'same_graph_fp32.npy',b)
    if not np.allclose(a,b,rtol=1e-4,atol=1e-5) or not np.array_equal(a.argmax(1),b.argmax(1)):
        raise ValueError('Same-graph FP32 equivalence failed on fresh data')
    save(out/'fp32_equivalence.json',dict(status='passed',samples=len(inputs),max_abs=float(abs(a-b).max()),class_changes=0))
    for kind,net in [('ordinary_int8',BASE/'artifacts/compile/int8/tcn_d_ZG.json'),
                     ('tf32',BASE/'artifacts/compile/tf32/tcn_d_ZG.json'),
                     ('mixed',ROOT/frozen['model_path'])]:
        folder=out/kind;folder.mkdir()
        cmd=[str(BASE/'build/Release/d_host_tools.exe'),'--mode','forward','--json',str(net),
             '--raw',str(net.with_suffix('.raw')),'--inputs',str(ROOT/'data/inputs.f32'),'--out',str(folder)]
        with (folder/'host.log').open('wb') as log:
            p=subprocess.run(cmd,cwd=folder,env=env(),stdout=log,stderr=subprocess.STDOUT)
        save(folder/'command.json',dict(command=cmd,exit_code=p.returncode,model_sha256=sha(net),params_sha256=sha(net.with_suffix('.raw')),board_accessed=False))
        if p.returncode:raise RuntimeError('Fresh Host failed: '+kind)
        values=np.fromfile(folder/'logits.f32',dtype='<f4').reshape(len(inputs),3)
        if not np.isfinite(values).all():raise ValueError('Nonfinite fresh output')
        np.save(folder/'logits.npy',values)
        print('Fresh evaluated',kind,flush=True)

if __name__=='__main__':main()
