"""Prepend supported Cast/Transpose to a loaded frozen quantized body.

The only body change is replacing its Input op with a real low-bit Input.
New prefix has no Params, so the existing raw parameter stream remains exact.
"""
import argparse,copy,shutil,subprocess,time
import numpy as np
from deployment.icraft.native.common3 import ROOT, BASE, V2, SDK, read, save, sha, env

def main():
    p=argparse.ArgumentParser();p.add_argument('--bits',type=int,choices=[8,16],default=8)
    p.add_argument('--physical-features',type=int,choices=[40,48,64],default=40);p.add_argument('--symmetric',action='store_true')
    p.add_argument('--fold-first-layout',action='store_true');a=p.parse_args()
    label=f'int{a.bits}' if a.physical_features==40 else f'int{a.bits}_f{a.physical_features}'
    if a.symmetric:label+='_sym'
    if a.fold_first_layout:label+='_fold'
    out=ROOT/f'artifacts/{label}';out.mkdir(exist_ok=False)
    x=np.load(V2/'data/inputs.npy')
    lo=np.minimum(x.min(axis=(0,2)),0.).astype(np.float64)
    hi=np.maximum(x.max(axis=(0,2)),0.).astype(np.float64)
    qmin,qmax=-(1<<(a.bits-1)),(1<<(a.bits-1))-1
    scale=np.where(hi>lo,(hi-lo)/(qmax-qmin),1.).astype('<f4')
    zp=np.clip(np.rint(qmin-lo/scale),qmin,qmax).astype(np.int64)
    if a.symmetric:
        scale=np.where(np.maximum(abs(lo),abs(hi))>0,np.maximum(abs(lo),abs(hi))/qmax,1.).astype('<f4')
        zp=np.zeros(40,dtype=np.int64)
    source=V2/'artifacts/mixed_tf32_compat/tcn_d_quantized.json';net=read(source)
    maximum=max(v['v_id'] for o in net['ops'] for v in o['inputs']+o['outputs'])
    opmax=max(o['op_id'] for o in net['ops'])
    old=net['ops'][0]['outputs'][0];input_type=copy.deepcopy(old['dtype']);input_type['shape']=[1,100,a.physical_features]
    physical_scale=np.concatenate((scale,np.ones(a.physical_features-40,dtype='<f4')))
    physical_zp=np.concatenate((zp,np.zeros(a.physical_features-40,dtype=np.int64)))
    literal='@scales([axis=-1]'+','.join(format(float(v),'.9g') for v in physical_scale)+')'
    input_type['element_dtype']=dict(_type_key='icraft::xir::QuantizedType',expressed_dtype='@fp(32)',
        storage_dtype=f'@sint({a.bits})',scale=literal,zero_points=[] if a.symmetric else physical_zp.tolist())
    integer=copy.deepcopy(old);integer.update(v_id=maximum+1,dtype=input_type)
    bridge_target=copy.deepcopy(net['ops'][2]['outputs'][0] if a.fold_first_layout else old)
    decoded=copy.deepcopy(bridge_target);decoded.update(v_id=maximum+2);decoded['dtype']['shape']=[1,100,a.physical_features]
    input_op=copy.deepcopy(net['ops'][0]);input_op['outputs']=[integer]
    cast=dict(_type_key='icraft::xir::Cast',compile_target='@zhuget(330)',inputs=[integer],outputs=[decoded],
        name='external_integer_decode',op_id=opmax+1,tags={'lowbit_decode':True},target_dtype=decoded['dtype'])
    prefix=[input_op,cast]
    sliced=decoded
    if a.physical_features>40:
        sliced=copy.deepcopy(bridge_target);sliced.update(v_id=maximum+3);sliced['dtype']['shape']=[1,100,40]
        if a.fold_first_layout:sliced=copy.deepcopy(bridge_target)
        prefix.append(dict(_type_key='icraft::xir::Slice',compile_target='@zhuget(330)',inputs=[decoded],outputs=[sliced],
            name='drop_transport_padding',op_id=opmax+3,tags={'lowbit_decode':True},begin=[0,0,0],end=[1,100,40],stride=[1,1,1]))
    trans=dict(_type_key='icraft::xir::Transpose',compile_target='@zhuget(330)',inputs=[sliced],outputs=[old],
        name='restore_feature_time',op_id=opmax+2,tags={'lowbit_decode':True},dims=[0,2,1],layout=old['dtype']['layout'])
    if not a.fold_first_layout:prefix.append(trans)
    elif a.physical_features==40:
        cast['outputs']=[bridge_target];cast['target_dtype']=bridge_target['dtype']
    original_body=copy.deepcopy(net['ops'][3:] if a.fold_first_layout else net['ops'][1:]);net['ops']=prefix+original_body
    net['name']=f'tcn_d_entry_int{a.bits}';net['tags']['D_external_entry']='affine integer input decoded before unchanged mixed body'
    source_out=out/f'tcn_d_entry_int{a.bits}_quantized.json';save(source_out,net)
    shutil.copyfile(source.with_suffix('.raw'),source_out.with_suffix('.raw'))
    save(ROOT/f'contracts/quantization_{label}.json',dict(schema='D_fixed_input_quantization_v3',bits=a.bits,
        signed=True,scale=scale.tolist(),scale_float32_bits=scale.view('<u4').tolist(),zero_point=zp.tolist(),
        qmin=qmin,qmax=qmax,scale_axis='feature',time_invariant=True,fit_inputs=str(V2/'data/inputs.npy'),
        fit_inputs_sha256=sha(V2/'data/inputs.npy'),fit_uses_labels=False,
        logical_float_shape=[1,40,100],external_shape=[1,100,a.physical_features],external_axis_order=['batch','time','padded_feature'],
        physical_scale=physical_scale.tolist(),physical_zero_point=physical_zp.tolist(),padding_value=0,
        external_dense_bytes=100*a.physical_features*a.bits//8,
        formula='q_f = clip(round_ties_even(x_f / float32_scale_f) + zero_point_f, qmin, qmax)',
        decode='x_f = float32(q_f - zero_point_f) * float32_scale_f',
        original_body_ops_identical=net['ops'][len(prefix):]==original_body,
        source_body_start_index=3 if a.fold_first_layout else 1,
        folded_original_io_cast_and_transpose=a.fold_first_layout,
        original_body_json_sha256=sha(source),body_params_sha256=sha(source.with_suffix('.raw')),
        new_quantized_params_sha256=sha(source_out.with_suffix('.raw')),
        prefix_ops=[opmax+1,opmax+2],input_value_id=maximum+1))
    cfg=f'''[adapt]
target = "zhuge"
json = "{source_out.as_posix()}"
raw = "{source_out.with_suffix('.raw').as_posix()}"
jr_path = "{out.as_posix()}"

[generate]
json = "{(out/f'tcn_d_entry_int{a.bits}_adapted.json').as_posix()}"
raw = "{(out/f'tcn_d_entry_int{a.bits}_adapted.raw').as_posix()}"
jr_path = "{out.as_posix()}"
'''
    cfg_path=out/'compile.toml';cfg_path.write_text(cfg,encoding='utf-8')
    stages=[]
    for stage in ('adapt','generate'):
        cmd=[str(SDK/'bin/icraft.exe'),stage,str(cfg_path)];start=time.perf_counter()
        with (out/(stage+'.log')).open('wb') as stream:
            r=subprocess.run(cmd,cwd=out,env=env(),stdout=stream,stderr=subprocess.STDOUT)
        stages.append(dict(stage=stage,command=cmd,exit_code=r.returncode,seconds=time.perf_counter()-start))
        save(out/'stage_results.json',dict(status='running' if r.returncode==0 else 'failed',stages=stages,board_accessed=False))
        if r.returncode:raise RuntimeError('Integer entry '+stage+' failed; preserve evidence')
    model=out/f'tcn_d_entry_int{a.bits}_ZG.json'
    save(out/'stage_results.json',dict(status='compiled',stages=stages,model_sha256=sha(model),
        params_sha256=sha(model.with_suffix('.raw')),board_accessed=False))
    print('Compiled',model)

if __name__=='__main__':main()
