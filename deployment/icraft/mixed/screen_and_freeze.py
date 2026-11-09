import subprocess
import numpy as np
from deployment.icraft.mixed.common2 import ROOT, BASE, read, save, sha, env


def main():
    out=ROOT/'artifacts/calibration_screen';out.mkdir(exist_ok=False)
    inputs=np.load(BASE/'data/calibration.inputs.npy');inputs.astype('<f4').tofile(out/'inputs.f32')
    model=ROOT/'artifacts/mixed_tf32_compat/tcn_d_ZG.json'
    command=[str(BASE/'build/Release/d_host_tools.exe'),'--mode','forward','--json',str(model),
             '--raw',str(model.with_suffix('.raw')),'--inputs',str(out/'inputs.f32'),'--out',str(out)]
    with (out/'host.log').open('wb') as log:
        p=subprocess.run(command,cwd=out,env=env(),stdout=log,stderr=subprocess.STDOUT)
    save(out/'command.json',dict(command=command,exit_code=p.returncode,board_accessed=False))
    if p.returncode:raise RuntimeError('Mixed Host calibration failed')
    values=np.fromfile(out/'logits.f32',dtype='<f4').reshape(256,3)
    np.save(out/'logits.npy',values)
    original=np.load(BASE/'artifacts/reference/calibration.torch.npy')
    ordinary=np.load(BASE/'artifacts/host/int8/ZG/calibration/logits.npy')
    diff=values-original
    result=dict(status='frozen_before_fresh_evaluation',model_sha256=sha(model),params_sha256=sha(model.with_suffix('.raw')),
        study_plan_sha256=sha(ROOT/'contracts/study_plan.json'),mix_sha256=sha(ROOT/'contracts/mix_tf32_compat.csv'),
        precision='INT8 middle blocks plus TF32 sensitive boundaries',
        model_path='artifacts/mixed_tf32_compat/tcn_d_ZG.json',
        failed_fp16_candidate_compile=True,
        calibration_class_changes_int8=int((ordinary.argmax(1)!=original.argmax(1)).sum()),
        calibration_class_changes_mixed=int((values.argmax(1)!=original.argmax(1)).sum()),
        calibration_logits_max_abs_int8=float(abs(ordinary-original).max()),
        calibration_logits_max_abs_mixed=float(abs(diff).max()),
        calibration_logits_mean_abs_int8=float(abs(ordinary-original).mean()),
        calibration_logits_mean_abs_mixed=float(abs(diff).mean()),
        new_candidate_count=1,board_accessed=False,
        selection='Single evidence-based mixed candidate, no search. Fresh evaluation not generated/observed before this freeze.',
        final_acceptance_data='later same-day fresh 512 nonoverlapping footprints per study_plan')
    save(ROOT/'contracts/frozen_candidate.json',result)
    print(result)

if __name__=='__main__':main()
