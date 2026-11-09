"""Exact algebra x=S*q: W*x = (W*S)*q; only first HWIO input axis scales.

Additional TF32 weight rounding is measured and included in quality evaluation.
"""
import copy,subprocess,time
import numpy as np
from deployment.icraft.native.common3 import ROOT, BASE, V2, SDK, read, save, sha, env

def main():
    out=ROOT/'artifacts/int8_folded';out.mkdir(exist_ok=False)
    # Preserve the same per-feature symmetric rule; zero-point 0 avoids causal-pad bias changes.
    q=read(ROOT/'contracts/quantization_int8_f64_sym_fold.json')
    s=np.array(q['scale'],dtype='<f4');s.tofile(out/'scales.f32')
    source=V2/'artifacts/mixed_tf32_compat/tcn_d_quantized.json';net=read(source)
    external=copy.deepcopy(net['ops'][0]['outputs'][0]);external['dtype']['element_dtype']='@sint(8)'
    net['ops'][0]['outputs']=[external]
    for op in net['ops'][1:]:
        op['inputs']=[copy.deepcopy(external) if v['v_id']==external['v_id'] else v for v in op['inputs']]
    net['name']='tcn_d_integer_folded';temporary=out/'before_weight_fold.json';save(temporary,net)
    model=out/'tcn_d_integer_folded_quantized.json'
    cmd=[str(ROOT/'build/Release/d_fold_scale.exe'),str(temporary),str(source.with_suffix('.raw')),
         str(out/'scales.f32'),str(model),str(model.with_suffix('.raw'))]
    with (out/'weight_fold.log').open('wb') as log:
        p=subprocess.run(cmd,cwd=out,env=env(),stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError('SDK parameter fold failed')
    save(ROOT/'contracts/quantization_int8_folded.json',dict(q,
        external_shape=[1,40,100],external_axis_order=['batch','feature','time'],external_dense_bytes=4000,
        physical_scale=q['scale'],physical_zero_point=[0]*40,padding_value=None,
        codec_zero_point=[0]*40,scale_folding=dict(parameters=[6,20],layout='HWIO',input_axis=2,
            formula='W_new[h,w,f,o] = TF32_RNE(W_old[h,w,f,o] * scale[f])',zero_point=0,
            bias_changed=False,training=False,original_params_sha256=sha(source.with_suffix('.raw')),
            folded_params_sha256=sha(model.with_suffix('.raw'))),
        note='Input is raw signed integer codes. Per-feature dequantization is folded into first convolution and shortcut; do not dequantize on ARM.'))
    config=f'''[adapt]
target = "zhuge"
json = "{model.as_posix()}"
raw = "{model.with_suffix('.raw').as_posix()}"
jr_path = "{out.as_posix()}"

[generate]
json = "{(out/'tcn_d_integer_folded_adapted.json').as_posix()}"
raw = "{(out/'tcn_d_integer_folded_adapted.raw').as_posix()}"
jr_path = "{out.as_posix()}"
'''
    cfg=out/'compile.toml';cfg.write_text(config,encoding='utf-8');stages=[]
    for stage in ('adapt','generate'):
        command=[str(SDK/'bin/icraft.exe'),stage,str(cfg)];start=time.perf_counter()
        with (out/(stage+'.log')).open('wb') as log:
            p=subprocess.run(command,cwd=out,env=env(),stdout=log,stderr=subprocess.STDOUT)
        stages.append(dict(stage=stage,exit_code=p.returncode,command=command,seconds=time.perf_counter()-start))
        save(out/'stage_results.json',dict(status='running' if p.returncode==0 else 'failed',stages=stages))
        if p.returncode:raise RuntimeError('Folded integer '+stage+' failed')
    generated=out/'tcn_d_integer_folded_ZG.json'
    save(out/'stage_results.json',dict(status='compiled',stages=stages,model_sha256=sha(generated),params_sha256=sha(generated.with_suffix('.raw')),
        expected_layers_per_forward=sum(o.get('layer_count',0) for o in read(generated)['ops']),board_accessed=False))
    print('Compiled real folded integer entry',generated)

if __name__=='__main__':main()
