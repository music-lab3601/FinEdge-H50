import argparse
import subprocess
import numpy as np
from deployment.icraft.base.common import ROOT, PACKAGE, read, save, sha, local_env


def run(kind,group,stage='ZG'):
    inputs=(np.fromfile(PACKAGE/'golden/inputs.f32',dtype='<f4').reshape(32,40,100)
            if group=='golden' else np.load(ROOT/f'data/{group}.inputs.npy'))
    out=ROOT/f'artifacts/host/{kind}/{stage}/{group}';out.mkdir(parents=True,exist_ok=False)
    inputs.tofile(out/'inputs.f32')
    model=ROOT/f'artifacts/compile/{kind}/tcn_d_{stage}.json'
    cmd=[str(ROOT/'build/Release/d_host_tools.exe'),'--mode','forward','--json',str(model),
         '--raw',str(model.with_suffix('.raw')),'--inputs',str(out/'inputs.f32'),'--out',str(out)]
    with (out/'host.log').open('wb') as log:
        result=subprocess.run(cmd,cwd=out,env=local_env(),stdout=log,stderr=subprocess.STDOUT)
    save(out/'command.json',dict(command=cmd,exit_code=result.returncode,model_sha256=sha(model),
            params_sha256=sha(model.with_suffix('.raw')),inputs_sha256=sha(out/'inputs.f32'),board_accessed=False))
    if result.returncode:raise RuntimeError('Host failed; inspect '+str(out/'host.log'))
    values=np.fromfile(out/'logits.f32',dtype='<f4').reshape(-1,3)
    if len(values)!=len(inputs) or not np.isfinite(values).all():raise ValueError('Host output count/nonfinite')
    np.save(out/'logits.npy',values)
    print(kind,stage,group,len(values),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['tf32','int8','mixed'],required=True)
    p.add_argument('--group',choices=['golden','calibration','evaluation'],required=True)
    p.add_argument('--stage',choices=['quantized','adapted','ZG'],default='ZG')
    a=p.parse_args();run(a.kind,a.group,a.stage)
