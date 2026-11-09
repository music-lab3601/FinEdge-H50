"""Successful native-input path, without replaying failed compiler experiments.

Run only in a new sibling D directory containing cpp/, scripts/ and .tmp/.
All output dirs must be absent. Does not read fresh evaluation data or labels.
"""
import copy,subprocess
import numpy as np
from common3 import ROOT,V2,SDK,read,save,env

def main():
    out=ROOT/'artifacts/int8_folded';out.mkdir(parents=True,exist_ok=False)
    x=np.load(V2/'data/inputs.npy')
    lo=np.minimum(x.min((0,2)),0.).astype('f8');hi=np.maximum(x.max((0,2)),0.).astype('f8')
    s=(np.maximum(abs(lo),abs(hi))/127).astype('<f4');s.tofile(out/'scales.f32')
    save(ROOT/'contracts/quantization_int8_folded.json',dict(scale=s.tolist(),zero_point=[0]*40,time_invariant=True))
    source=V2/'artifacts/mixed_tf32_compat/tcn_d_quantized.json';n=read(source)
    e=copy.deepcopy(n['ops'][0]['outputs'][0]);e['dtype']['element_dtype']='@sint(8)';n['ops'][0]['outputs']=[e]
    for o in n['ops'][1:]:o['inputs']=[copy.deepcopy(e) if v['v_id']==e['v_id'] else v for v in o['inputs']]
    n['name']='tcn_d_integer_folded';save(out/'before_weight_fold.json',n)
    model=out/'tcn_d_integer_folded_quantized.json'
    commands=[[str(ROOT/'build/Release/d_fold_scale.exe'),str(out/'before_weight_fold.json'),str(source.with_suffix('.raw')),str(out/'scales.f32'),str(model),str(model.with_suffix('.raw'))],
              [str(SDK/'bin/icraft.exe'),'adapt','--target','zhuge','--json',str(model),'--raw',str(model.with_suffix('.raw')),'--jr_path',str(out)]]
    for i,cmd in enumerate(commands):
        with (out/f'rebuild_stage{i}.log').open('wb') as f:r=subprocess.run(cmd,cwd=out,env=env(),stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('Native input preparation failed')
    from physical_entry import main as physical
    physical()
    from build_tf32_pair import main as pair
    pair()
    print('Successful compile path rebuilt; no holdout or board access')

if __name__=='__main__':main()
