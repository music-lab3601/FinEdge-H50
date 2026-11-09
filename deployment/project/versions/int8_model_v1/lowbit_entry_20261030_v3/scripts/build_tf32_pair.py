"""Same encoded input and same folded first weights, all-TF32 body reference.

This is a paired numerical reference, not another searched mixed candidate.
"""
import copy,shutil,subprocess,time
from common3 import ROOT,BASE,SDK,read,save,sha,env

def main():
    out=ROOT/'artifacts/int8_entry_tf32_body';out.mkdir(exist_ok=False)
    source=BASE/'artifacts/compile/tf32/tcn_d_quantized.json'
    n=read(source);n['name']='d3t'
    entry=copy.deepcopy(n['ops'][0]['outputs'][0]);entry['dtype']['element_dtype']='@sint(8)'
    n['ops'][0]['outputs']=[entry]
    for o in n['ops'][1:]:o['inputs']=[copy.deepcopy(entry) if v['v_id']==entry['v_id'] else v for v in o['inputs']]
    save(out/'before_fold.json',n)
    scales=ROOT/'artifacts/int8_folded/scales.f32';model=out/'d3t_quantized.json'
    commands=[[str(ROOT/'build/Release/d_fold_scale.exe'),str(out/'before_fold.json'),str(source.with_suffix('.raw')),str(scales),str(model),str(model.with_suffix('.raw'))],
              [str(SDK/'bin/icraft.exe'),'adapt','--target','zhuge','--json',str(model),'--raw',str(model.with_suffix('.raw')),'--jr_path',str(out)]]
    stages=[]
    for i,cmd in enumerate(commands):
        with (out/f'stage{i}.log').open('wb') as f:r=subprocess.run(cmd,cwd=out,env=env(),stdout=f,stderr=subprocess.STDOUT)
        stages.append(dict(command=cmd,exit_code=r.returncode));save(out/'stages.json',stages)
        if r.returncode:raise RuntimeError('TF32 pair preparation failed')
    adapted=out/'d3t_adapted.json';n=read(adapted)
    first=next(i for i,o in enumerate(n['ops']) if o['_type_key'].endswith('Conv2d'))
    decoded=copy.deepcopy(n['ops'][first]['inputs'][0]);entry=copy.deepcopy(decoded);entry['v_id']=0
    entry['dtype']['element_dtype']=dict(_type_key='icraft::xir::QuantizedType',expressed_dtype='@fp(32)',storage_dtype='@sint(8)',scale='@scales([axis=-1]48*1)',zero_points=[])
    remaining=n['ops'][first:];produced={v['v_id'] for o in remaining for v in o['outputs']}
    missing=sorted({v['v_id'] for o in remaining for v in o['inputs'] if v['_type_key'].endswith('Value')}-produced)
    assert missing==[decoded['v_id']],missing
    inp=copy.deepcopy(n['ops'][0]);inp['outputs']=[entry]
    cast=dict(_type_key='icraft::xir::Cast',compile_target='@zhuget(330)',inputs=[entry],outputs=[decoded],op_id=9000,name='decode',tags={},target_dtype=copy.deepcopy(decoded['dtype']))
    n['ops']=[inp,cast]+remaining;n['name']='d3tp';physical=out/'d3tp_adapted.json';save(physical,n);shutil.copy2(adapted.with_suffix('.raw'),physical.with_suffix('.raw'))
    cmd=[str(SDK/'bin/icraft.exe'),'generate','--json',str(physical),'--raw',str(physical.with_suffix('.raw')),'--jr_path',str(out)]
    with (out/'generate.log').open('wb') as f:r=subprocess.run(cmd,cwd=out,env=env(),stdout=f,stderr=subprocess.STDOUT)
    stages.append(dict(command=cmd,exit_code=r.returncode));save(out/'stages.json',stages)
    if r.returncode:raise RuntimeError('TF32 pair generation failed')
    print('TF32-body low-bit pair compiled')

if __name__=='__main__':main()
