"""Compile the integer decoder separately, preserving accepted body instructions."""
import copy,subprocess,time
from deployment.icraft.native.common3 import ROOT, V2, SDK, read, save, sha, env

def main():
    out=ROOT/'artifacts/decoder_int8';out.mkdir(exist_ok=False)
    src=read(ROOT/'artifacts/int8_f64_sym_fold/tcn_d_entry_int8_quantized.json')
    prefix=copy.deepcopy(src['ops'][:3]) # Input, Cast(INT8 -> TF32), Slice
    old=prefix[-1]['outputs'][0]
    fi=copy.deepcopy(old);fi.update(v_id=10001);fi['dtype']['shape']=[1,40,100]
    trans=dict(_type_key='icraft::xir::Transpose',op_id=10001,compile_target='@zhuget(330)',inputs=[old],outputs=[fi],
        name='feature_major_output',tags={},dims=[0,2,1],layout=fi['dtype']['layout'])
    fp=copy.deepcopy(fi);fp.update(v_id=10002);fp['dtype']['element_dtype']='@fp(32)'
    cast=dict(_type_key='icraft::xir::Cast',op_id=10002,compile_target='@zhuget(330)',inputs=[fi],outputs=[fp],
        name='decoder_fp32_output',tags={'io_process':True},target_dtype=fp['dtype'])
    output=dict(_type_key='icraft::xir::Output',op_id=10003,compile_target='@hostt',inputs=[fp],outputs=[],name='',tags={})
    net={k:v for k,v in src.items() if k!='ops'}
    net.update(name='tcn_d_int8_decoder',ops=prefix+[trans,cast,output],params_bytes=0,
               params_md5=__import__('hashlib').md5(b'').hexdigest())
    q=out/'tcn_d_int8_decoder_quantized.json';save(q,net);q.with_suffix('.raw').write_bytes(b'')
    config=f'''[adapt]
target = "zhuge"
json = "{q.as_posix()}"
raw = "{q.with_suffix('.raw').as_posix()}"
jr_path = "{out.as_posix()}"

[generate]
json = "{(out/'tcn_d_int8_decoder_adapted.json').as_posix()}"
raw = "{(out/'tcn_d_int8_decoder_adapted.raw').as_posix()}"
jr_path = "{out.as_posix()}"
'''
    cfg=out/'compile.toml';cfg.write_text(config,encoding='utf-8');stages=[]
    for stage in ('adapt','generate'):
        cmd=[str(SDK/'bin/icraft.exe'),stage,str(cfg)];start=time.perf_counter()
        with (out/(stage+'.log')).open('wb') as log:
            p=subprocess.run(cmd,cwd=out,env=env(),stdout=log,stderr=subprocess.STDOUT)
        stages.append(dict(stage=stage,exit_code=p.returncode,seconds=time.perf_counter()-start,command=cmd))
        save(out/'stage_results.json',dict(status='running' if not p.returncode else 'failed',stages=stages))
        if p.returncode:raise RuntimeError('Decoder '+stage+' failed')
    model=out/'tcn_d_int8_decoder_ZG.json'
    save(out/'stage_results.json',dict(status='compiled',stages=stages,model_sha256=sha(model),
        params_sha256=sha(model.with_suffix('.raw')),expected_layers=sum(o.get('layer_count',0) for o in read(model)['ops']),board_accessed=False))
    print('Compiled integer decoder')
if __name__=='__main__':main()
