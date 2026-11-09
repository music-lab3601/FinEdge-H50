import numpy as np
import torch
import onnxruntime as ort
from deployment.icraft.native.common3 import ROOT, BASE, V2, PROJECT, save
from deployment.icraft.native.prepare_holdout3 import frozen_check
from deployment.icraft.native.screen3 import run
from deployment.icraft.native.codec3 import encode, decode

def main():
    f=frozen_check();x=np.load(ROOT/'data/inputs.npy');out=ROOT/'artifacts/fresh';out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    m=torch.jit.load(str(PROJECT/'models/cpu/model_cpu.ts'),map_location='cpu').eval()
    opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
    ortm=ort.InferenceSession(str(BASE/'model/tcn_polyphase.onnx'),opts,providers=['CPUExecutionProvider'])
    a=[];b=[]
    with torch.inference_mode():
        for v in x:
            a.append(m(torch.from_numpy(v[None])).numpy()[0]);b.append(ortm.run(None,{'lob':v[None]})[0][0])
    a,b=np.array(a,dtype='<f4'),np.array(b,dtype='<f4')
    np.save(out/'original_fp32.npy',a);np.save(out/'same_structure_fp32.npy',b)
    assert np.allclose(a,b,rtol=1e-4,atol=1e-5) and np.array_equal(a.argmax(1),b.argmax(1))
    save(out/'fp32_equivalence.json',dict(status='passed',samples=len(x),max_abs=float(abs(a-b).max()),class_changes=0))
    decoded=np.array([decode(encode(v[None]))[0] for v in x],dtype='<f4');decoded.tofile(out/'dequantized.f32')
    entries=[('full_tf32',BASE/'artifacts/compile/tf32/tcn_d_ZG.json',ROOT/'data/inputs.f32',False),
             ('accepted_mixed',V2/'artifacts/mixed_tf32_compat/tcn_d_ZG.json',ROOT/'data/inputs.f32',False),
             ('dequantized_mixed',V2/'artifacts/mixed_tf32_compat/tcn_d_ZG.json',out/'dequantized.f32',False)]
    entries += [(key,ROOT/m['json'],ROOT/'data/inputs.i8',True) for key,m in f['models'].items()]
    for key,model,inputs,integer in entries:
        run('forward',inputs,out/key,model,integer)
        v=np.fromfile(out/key/'logits.f32','<f4').reshape(len(x),3);assert np.isfinite(v).all()
        np.save(out/key/'logits.npy',v);print('Evaluated',key,flush=True)

if __name__=='__main__':main()
