import argparse,csv,io,subprocess,time
from deployment.icraft.mixed.common2 import ROOT, BASE, SDK, read, save, sha, env


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tf32-compat',action='store_true');args=parser.parse_args()
    high='tf32' if args.tf32_compat else 'fp16'
    out=ROOT/('artifacts/mixed_tf32_compat' if args.tf32_compat else 'artifacts/mixed');out.mkdir(exist_ok=False)
    report=read(ROOT/'artifacts/layer_diagnostics/report.json')
    rows={r['op_id']:r for r in report['rows']}
    # Keep boundaries coherent; don't quantize then immediately widen early input.
    selected={45,46,3,4,5,6,8,9,10,32,33,34,35,36,37,38,39,41,42}
    diagnostic_facts={str(k):rows[k] for k in (3,4,5,6,35,42)}
    justification=ROOT/('contracts/tf32_compat_justification.json' if args.tf32_compat else 'contracts/mixed_candidate_justification.json')
    save(justification,dict(
        candidate=f'sensitive_boundary_{high}_middle_int8',candidate_design_count=1,
        selected_using='original calibration tensors only',
        layer_diagnostic_sha256=sha(ROOT/'artifacts/layer_diagnostics/report.json'),
        evidence=diagnostic_facts,
        interpretation='Early clipping/error appears before subsequent blocks; last temporal convolution has high propagated relative error. Preserve first/last blocks and classifier, including adjoining layout and nonlinear operations.',
        protected_precision=high,protected_op_ids=sorted(selected),unchanged_calibration=True,unchanged_weights=True,
        unchanged_normalization=True,unchanged_onnx_structure=True,
        fresh_holdout_not_read=True,previous_512_evaluation_not_used_for_layer_selection=True))
    source=BASE/'artifacts/compile/int8/.icraft/logs/tcn_d/tcn_d_quantize_mix_config.csv'
    records=list(csv.DictReader(source.read_text(encoding='utf-8-sig').splitlines()))
    for r in records:
        if int(r['op_id']) in selected:
            r['dtype']=high;r['value_range_in_bits']='32;32' if args.tf32_compat else '16;16'
    mix=ROOT/('contracts/mix_tf32_compat.csv' if args.tf32_compat else 'contracts/mix.csv')
    with mix.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    common=BASE/'artifacts/compile/shared'; cal=BASE/'data'
    cfg=f'''[quantize]
target = "zhuge"
qdtype = "int8"
no_transinput = true
json = "{(common/'tcn_d_optimized.json').as_posix()}"
raw = "{(common/'tcn_d_optimized.raw').as_posix()}"
jr_path = "{out.as_posix()}"
forward_dir = "{(cal/'calibration_bin').as_posix()}"
forward_list = "{(cal/'forward_list.txt').as_posix()}"
saturation = "kld"
per = "channel"
batch = 1
mix_precision = "{mix.as_posix()}"

[adapt]
target = "zhuge"
json = "{(out/'tcn_d_quantized.json').as_posix()}"
raw = "{(out/'tcn_d_quantized.raw').as_posix()}"
jr_path = "{out.as_posix()}"

[generate]
json = "{(out/'tcn_d_adapted.json').as_posix()}"
raw = "{(out/'tcn_d_adapted.raw').as_posix()}"
jr_path = "{out.as_posix()}"
'''
    path=out/'compile.toml';path.write_text(cfg,encoding='utf-8')
    result=dict(status='running',stages=[],graph_sha256=sha(BASE/'model/tcn_polyphase.onnx'),
        optimized_sha256=sha(common/'tcn_d_optimized.json'),mix_sha256=sha(mix),
        calibration_manifest_sha256=sha(cal/'manifest.json'),board_accessed=False)
    for stage in ('quantize','adapt','generate'):
        command=[str(SDK/'bin/icraft.exe'),stage,str(path)];start=time.perf_counter()
        with (out/(stage+'.log')).open('wb') as log:
            p=subprocess.run(command,cwd=out,env=env(),stdout=log,stderr=subprocess.STDOUT)
        result['stages'].append(dict(stage=stage,command=command,exit_code=p.returncode,seconds=time.perf_counter()-start))
        save(out/'stage_results.json',result)
        if p.returncode:
            result.update(status='failed',failed_stage=stage);save(out/'stage_results.json',result);raise RuntimeError('Mixed compile failed')
    q=read(out/'tcn_d_quantized.json')
    ops={o['op_id']:o for o in q['ops']}
    observed={str(k):[v['dtype']['element_dtype'] for v in ops[k]['outputs']] for k in selected if k in ops}
    result.update(status='compiled',generated_sha256=sha(out/'tcn_d_ZG.json'),params_sha256=sha(out/'tcn_d_ZG.raw'),selected_output_types=observed)
    save(out/'stage_results.json',result)
    print('Mixed candidate compiled')

if __name__=='__main__':main()
