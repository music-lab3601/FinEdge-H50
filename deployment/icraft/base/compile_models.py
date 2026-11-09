"""Finite, same-structure TF32/INT8 compilation. No device access."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time
from deployment.icraft.base.common import ROOT, ICRAFT, read, save, sha, local_env


def path(p): return Path(p).as_posix()


def configuration(kind):
    shared=ROOT/'artifacts/compile/shared'; out=ROOT/f'artifacts/compile/{kind}'
    common=f'''[parse]
net_name = "tcn_d"
framework = "onnx"
target = "zhuge"
network = "{path(ROOT/'model/tcn_polyphase.onnx')}"
jr_path = "{path(shared)}"
inputs = "1,40,100"
inputs_layout = "FD"
inputs_dtype = "FP32"
pre_method = "nop"
pre_mean = "nop"
pre_scale = "nop"
channel_swap = "nop"

[optimize]
target = "zhuge"
json = "{path(shared/'tcn_d_parsed.json')}"
raw = "{path(shared/'tcn_d_parsed.raw')}"
jr_path = "{path(shared)}"

[quantize]
target = "zhuge"
qdtype = "{'tf32' if kind=='tf32' else 'int8'}"
no_transinput = true
json = "{path(shared/'tcn_d_optimized.json')}"
raw = "{path(shared/'tcn_d_optimized.raw')}"
jr_path = "{path(out)}"
'''
    if kind!='tf32':
        common+=f'''forward_dir = "{path(ROOT/'data/calibration_bin')}"
forward_list = "{path(ROOT/'data/forward_list.txt')}"
saturation = "kld"
per = "channel"
batch = 1
'''
    if kind=='mixed': common+=f'mix_precision = "{path(ROOT/"contracts/mix.csv")}"\n'
    common+=f'''
[adapt]
target = "zhuge"
json = "{path(out/'tcn_d_quantized.json')}"
raw = "{path(out/'tcn_d_quantized.raw')}"
jr_path = "{path(out)}"

[generate]
json = "{path(out/'tcn_d_adapted.json')}"
raw = "{path(out/'tcn_d_adapted.raw')}"
jr_path = "{path(out)}"
'''
    return common


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['tf32','int8','mixed'],required=True)
    parser.add_argument('--shared',action='store_true'); args=parser.parse_args()
    if args.kind=='mixed' and not (ROOT/'contracts/mixed_candidate_justification.json').is_file():
        raise ValueError('A mixed candidate requires layer evidence and a frozen justification')
    run=ROOT/f'artifacts/compile/{args.kind}';run.mkdir(parents=True,exist_ok=True)
    if (run/'stage_results.json').exists():raise ValueError('Candidate already attempted; preserve evidence, do not retune/retry silently')
    (ROOT/'artifacts/compile/shared').mkdir(parents=True,exist_ok=True)
    cfg=run/'compile.toml';cfg.write_text(configuration(args.kind),encoding='utf-8')
    report=dict(kind=args.kind,config_sha256=sha(cfg),graph_sha256=sha(ROOT/'model/tcn_polyphase.onnx'),
                calibration_manifest_sha256=sha(ROOT/'data/manifest.json') if (ROOT/'data/manifest.json').is_file() else None,board_accessed=False,stages=[],status='running')
    stages=(['parse','optimize'] if args.shared else [])+['quantize','adapt','generate']
    for stage in stages:
        log=run/(stage+'.log'); start=time.perf_counter()
        cmd=[str(ICRAFT/'bin/icraft.exe'),stage,str(cfg)]
        with log.open('wb') as stream:
            proc=subprocess.run(cmd,cwd=run,env=local_env(),stdout=stream,stderr=subprocess.STDOUT)
        report['stages'].append(dict(stage=stage,command=cmd,exit_code=proc.returncode,
            seconds=time.perf_counter()-start,log_sha256=sha(log)))
        save(run/'stage_results.json',report)
        print(args.kind,stage,proc.returncode,flush=True)
        if proc.returncode:
            report.update(status='failed',failed_stage=stage);save(run/'stage_results.json',report)
            raise SystemExit(proc.returncode)
    report.update(status='compiled',generated_sha256=sha(run/'tcn_d_ZG.json'),
                  params_sha256=sha(run/'tcn_d_ZG.raw'))
    save(run/'stage_results.json',report)


if __name__=='__main__':main()
