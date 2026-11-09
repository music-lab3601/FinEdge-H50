"""Exact causal dependency pruning; keep learned weights and FP32 semantics.

After each block, only the phase containing the final time index can affect
later power-of-two dilations. Decimate that phase and halve future dilation.
The final block needs only the last five inputs, with two valid k3 convolutions.
"""
from shared.paths import resolve_path
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import collections
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh,TensorProto as T

from shared.model_paths import ARTIFACTS
ROOT=ARTIFACTS.parent
from shared.paths import ROOT as REPOSITORY, work_path


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--compile',action='store_true')
    p.add_argument('--preserve-taps',action='store_true');a=p.parse_args()
    if not a.label.replace('_','').isalnum():raise ValueError('Safe unique label')
    out=work_path('runtime/subhalf')/a.label;out.mkdir(parents=True,exist_ok=False)
    old=onnx.load(ARTIFACTS/'export/tcn_ed4.onnx');params={v.name:nh.to_array(v) for v in old.graph.initializer}
    original={v.name:v for v in old.graph.node};nodes=[];inits=[];weights=[]
    def const(name,array):
        inits.append(nh.from_array(np.asarray(array),name));return name
    def node(op,inputs,name,**attrs):nodes.append(h.make_node(op,inputs,[name],name=name,**attrs));return name
    def conv(x,w,b,name,left=0,stride=1):
        wn=const(name+'_w',w.astype(np.float32));bn=const(name+'_b',b.astype(np.float32))
        return node('Conv',[x,wn,bn],name,kernel_shape=[1,w.shape[-1]],pads=[0,left,0,0],strides=[1,stride],dilations=[1,1],group=1)
    def sliced(x,start,end,name,step=1):
        inputs=[x,const(name+'_start',np.array([start],np.int64)),const(name+'_end',np.array([end],np.int64)),const(name+'_axes',np.array([3],np.int64))]
        if step!=1:inputs.append(const(name+'_step',np.array([step],np.int64)))
        return node('Slice',inputs,name)
    def taps(x,w,b,name,length,left=2):
        if left:
            x=node('Pad',[x,const(name+'_pad',np.array([0,0,0,left,0,0,0,0],np.int64)),const(name+'_zero',np.array(0.,np.float32))],name+'_padded',mode='constant')
        count=length+left-2
        ys=[]
        for k in range(3):
            part=sliced(x,k,k+count,name+'_tap%d_slice'%k)
            ys.append(conv(part,w[:,:,:,k:k+1],np.zeros(w.shape[0],np.float32),name+'_tap%d'%k))
        y=node('Add',[ys[0],ys[1]],name+'_sum01');y=node('Add',[y,ys[2]],name+'_sum012')
        return node('Add',[y,const(name+'_bias',b.reshape(1,-1,1,1).astype(np.float32))],name+'_biased')
    def temporal(i,j):
        prefix='/features/%d/temporal/%d'%(i,j)
        if i==0:
            w=params['features.0.temporal.%d.conv.weight'%j];b=params['features.0.temporal.%d.conv.bias'%j]
        else:
            taps=[params[original[prefix+'/Conv'+suffix].input[1]][:,:,0] for suffix in ('','_1','_2')]
            w=np.stack(taps,axis=-1);b=params[original[prefix+'/Add_4'].input[1]].reshape(-1)
        weights.append(dict(block=i,convolution=j,shape=list(w.shape),sha256=hashlib.sha256(w.tobytes()+b.tobytes()).hexdigest()))
        return w[:,:,None,:],b
    x=node('Reshape',['lob',const('input_shape',np.array([1,40,1,100],np.int64))],'input_4d');length=100;lengths=[]
    for i in range(5):
        w0,b0=temporal(i,0);w1,b1=temporal(i,2)
        if a.preserve_taps:
            if i==4:
                tail=sliced(x,length-5,length,'last_block_tail')
                y=node('Relu',[taps(tail,w0,b0,'b4_conv0',5,left=0)],'b4_relu0')
                y=node('Relu',[taps(y,w1,b1,'b4_conv1',3,left=0)],'b4_relu1')
                residual=sliced(x,length-1,length,'last_residual');length=1
            else:
                if i==0:
                    y=node('Relu',[conv(x,w0,b0,'b0_conv0',left=2)],'b0_relu0')
                    y=node('Relu',[conv(y,w1,b1,'b0_conv1',left=2)],'b0_relu1')
                    residual=conv(x,params['features.0.shortcut.weight'][:,:,None,:],params['features.0.shortcut.bias'],'b0_residual')
                else:
                    y=node('Relu',[taps(x,w0,b0,'b%d_conv0'%i,length)],'b%d_relu0'%i)
                    y=node('Relu',[taps(y,w1,b1,'b%d_conv1'%i,length)],'b%d_relu1'%i)
                    residual=x
            x=node('Relu',[node('Add',[y,residual],'b%d_add'%i)],'b%d_out'%i)
            if i<4:
                phase=(length-1)%2;x=sliced(x,phase,length,'b%d_keep_phase'%i,step=2);length=(length-phase+1)//2
            lengths.append(length);continue
        elif i==4:
            tail=sliced(x,length-5,length,'last_block_tail')
            y=node('Relu',[conv(tail,w0,b0,'b4_conv0')],'b4_relu0')
            y=node('Relu',[conv(y,w1,b1,'b4_conv1')],'b4_relu1')
            residual=sliced(x,length-1,length,'last_residual')
            length=1
        else:
            phase=(length-1)%2
            y=node('Relu',[conv(x,w0,b0,'b%d_conv0'%i,left=2)],'b%d_relu0'%i)
            y=node('Relu',[conv(y,w1,b1,'b%d_conv1'%i,left=2-phase,stride=2)],'b%d_relu1'%i)
            channels=40 if i==0 else 32
            selector=np.zeros((32,channels,1,phase+1),np.float32)
            if i==0:
                selector[:,:,0,phase]=params['features.0.shortcut.weight'][:,:,0]
                bias=params['features.0.shortcut.bias']
            else:
                selector[np.arange(32),np.arange(32),0,phase]=1.;bias=np.zeros(32,np.float32)
            residual=conv(x,selector,bias,'b%d_residual_select'%i,stride=2)
            length=(length-phase+1)//2
        x=node('Relu',[node('Add',[y,residual],'b%d_add'%i)],'b%d_out'%i);lengths.append(length)
    flat=node('Flatten',[x],'flat',axis=1)
    node('Gemm',[flat,const('classifier_w',params['classifier.weight']),const('classifier_b',params['classifier.bias'])],'logits',transB=1)
    graph=h.make_graph(nodes,'tcn_exact_polyphase',[h.make_tensor_value_info('lob',T.FLOAT,[1,40,100])],
        [h.make_tensor_value_info('logits',T.FLOAT,[1,3])],initializer=inits)
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',11)]);model.ir_version=8
    model=onnx.shape_inference.infer_shapes(model);onnx.checker.check_model(model);onnx.save(model,out/'polyphase.onnx')
    import onnxruntime as ort
    so=ort.SessionOptions();so.intra_op_num_threads=so.inter_op_num_threads=1
    base=ort.InferenceSession(str(ARTIFACTS/'export/tcn_ed4.onnx'),so,providers=['CPUExecutionProvider'])
    candidate=ort.InferenceSession(str(out/'polyphase.onnx'),so,providers=['CPUExecutionProvider'])
    golden=np.fromfile(REPOSITORY/'models/cpu/golden/inputs.f32',dtype='<f4').reshape(32,40,100)
    rng=np.random.default_rng(20261027);samples=list(golden)+list(rng.normal(size=(64,40,100)).astype(np.float32))
    xs=np.asarray(samples);old_y=[];new_y=[]
    for x0 in xs:
        old_y.append(base.run(None,{'lob':x0[None]})[0][0]);new_y.append(candidate.run(None,{'lob':x0[None]})[0][0])
    old_y,new_y=np.asarray(old_y),np.asarray(new_y)
    passed=bool(np.allclose(old_y,new_y,rtol=1e-4,atol=1e-5) and np.array_equal(old_y.argmax(1),new_y.argmax(1)))
    report=dict(status='onnx_equivalence_passed' if passed else 'failed',preserve_original_tap_accumulation=a.preserve_taps,tested=len(xs),golden=32,
        max_abs=float(np.max(abs(new_y-old_y))),class_mismatches=int(np.sum(old_y.argmax(1)!=new_y.argmax(1))),
        lengths=lengths,original_nodes=len(old.graph.node),candidate_nodes=len(model.graph.node),weight_sources=weights,
        original_onnx_sha256=hashlib.sha256((ARTIFACTS/'export/tcn_ed4.onnx').read_bytes()).hexdigest(),
        candidate_onnx_sha256=hashlib.sha256((out/'polyphase.onnx').read_bytes()).hexdigest())
    (out/'equivalence.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if not passed:raise RuntimeError('Causal polyphase equivalence failed')
    if a.compile:
        output=out/'icraft';output.mkdir()
        cfg=(ARTIFACTS/'icraft/tcn_ed4_tf32.toml').read_text(encoding='utf-8').replace('tcn_ed4','polyphase')
        cfg=cfg.replace((ARTIFACTS/'export/polyphase.onnx').as_posix(),(out/'polyphase.onnx').as_posix())
        cfg=cfg.replace((ARTIFACTS/'icraft').as_posix(),output.as_posix())
        (out/'compile.toml').write_text(cfg,encoding='utf-8')
        stages=[]
        for stage in ('parse','optimize','quantize','adapt','generate'):
            with (out/(stage+'.log')).open('wb') as f:
                r=subprocess.run([str(resolve_path('tool://icraft/bin/icraft.exe')),stage,str(out/'compile.toml')],cwd=out,stdout=f,stderr=subprocess.STDOUT)
            stages.append(dict(stage=stage,exit_code=r.returncode))
            if r.returncode:raise RuntimeError('Icraft '+stage+' failed')
        generated=json.loads((output/'polyphase_ZG.json').read_text(encoding='utf-8'))
        report['compiled']=dict(stages=stages,ops=len(generated['ops']),layers=sum(o.get('layer_count',0) for o in generated['ops']),
            names=dict(collections.Counter(o.get('name','') for o in generated['ops'])))
        (out/'equivalence.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
