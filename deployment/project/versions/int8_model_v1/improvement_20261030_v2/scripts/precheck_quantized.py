import subprocess
import numpy as np
from common2 import ROOT,BASE,read,save,sha,env

def main():
    out=ROOT/'artifacts/quantized_calibration_precheck';out.mkdir(exist_ok=False)
    inputs=np.load(BASE/'data/calibration.inputs.npy');inputs.astype('<f4').tofile(out/'inputs.f32')
    model=ROOT/'artifacts/mixed/tcn_d_quantized.json'
    command=[str(BASE/'build/Release/d_host_tools.exe'),'--mode','forward','--json',str(model),
        '--raw',str(model.with_suffix('.raw')),'--inputs',str(out/'inputs.f32'),'--out',str(out)]
    with (out/'host.log').open('wb') as f:
        p=subprocess.run(command,cwd=out,env=env(),stdout=f,stderr=subprocess.STDOUT)
    save(out/'command.json',dict(command=command,exit_code=p.returncode))
    if p.returncode:raise RuntimeError('Mixed quantized precheck failed')
    values=np.fromfile(out/'logits.f32',dtype='<f4').reshape(256,3)
    original=np.load(BASE/'artifacts/reference/calibration.torch.npy')
    save(out/'diagnostic.json',dict(status='completed',scope='calibration only, pre-codegen model not deployment acceptance',
        class_changes=int((values.argmax(1)!=original.argmax(1)).sum()),
        logits_max_abs=float(abs(values-original).max()),logits_mean_abs=float(abs(values-original).mean()),
        model_sha256=sha(model),board_accessed=False))
    print(read(out/'diagnostic.json'))
if __name__=='__main__':main()
