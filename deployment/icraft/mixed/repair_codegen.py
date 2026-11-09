"""Single documented codegen optimization fallback, unchanged mixed precision graph."""
import subprocess
import time
from deployment.icraft.mixed.common2 import ROOT, SDK, read, save, sha, env

def main():
    source=ROOT/'artifacts/mixed'
    out=ROOT/'artifacts/codegen_conservative';out.mkdir(exist_ok=False)
    cmd=[str(SDK/'bin/icraft.exe'),'generate','--json',str(source/'tcn_d_adapted.json'),
        '--raw',str(source/'tcn_d_adapted.raw'),'--jr_path',str(out),
        '--xlmopt','0','--klmopt','0','--icropt','0','--ocmopt','0','--no_mergeops']
    start=time.perf_counter()
    with (out/'generate.log').open('wb') as log:
        p=subprocess.run(cmd,cwd=out,env=env(),stdout=log,stderr=subprocess.STDOUT)
    result=dict(status='compiled' if p.returncode==0 else 'failed',command=cmd,exit_code=p.returncode,
        seconds=time.perf_counter()-start,adapted_json_sha256=sha(source/'tcn_d_adapted.json'),
        adapted_raw_sha256=sha(source/'tcn_d_adapted.raw'),
        explanation='Default generator crashed 0xC0000409. Same precision/weights/calibration graph retried once with documented memory/instruction optimization disabled.',
        new_precision_candidate=False,board_accessed=False)
    save(out/'result.json',result)
    print(result)
if __name__=='__main__':main()
