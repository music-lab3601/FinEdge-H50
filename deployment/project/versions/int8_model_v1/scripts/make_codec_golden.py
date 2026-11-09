"""SDK-referenced logical integer encoding, explicitly not a device input ABI."""
import subprocess
import numpy as np
from common import ROOT, PACKAGE, read, save, sha, local_env


def main():
    contract=read(ROOT/'input_contract.json');q=contract['internal_integer_codec']
    # Literal repeats denote repeated scales, not multiplication of the step.
    literal=q['compiled_dtype']['element_dtype']['scale']
    s=np.float32(literal.split('*')[-1].rstrip(')'))
    targets=np.array([-200,-129,-128.5,-128,-127.5,-127,-126.5,-3.5,-2.5,-1.5,-.5,-.1,0,.1,.5,1.5,2.5,3.5,126.5,127,127.5,128,128.5,129,200],dtype=np.float32)*s
    special=np.concatenate((targets,np.nextafter(targets,np.float32(-np.inf)),
                            np.nextafter(targets,np.float32(np.inf)),
                            np.array([-0.,0.,np.finfo(np.float32).tiny,-np.finfo(np.float32).tiny],dtype=np.float32)))
    synthetic=np.stack([np.resize(np.roll(special,i),4000).reshape(40,100) for i in range(32)]).astype('<f4')
    golden=np.fromfile(PACKAGE/'golden/inputs.f32',dtype='<f4').reshape(32,40,100)
    values=np.concatenate((golden,synthetic))
    out=ROOT/'golden';out.mkdir(exist_ok=False)
    values.tofile(out/'inputs.f32');np.save(out/'inputs.npy',values)
    model=ROOT/'artifacts/compile/int8/tcn_d_quantized.json'
    prefix=f"{q['preceding_cast']['value_id']},{q['value_id']}"
    cmd=[str(ROOT/'build/Release/d_host_tools.exe'),'--mode','encode','--json',str(model),
         '--raw',str(model.with_suffix('.raw')),'--value',str(q['value_id']),
         '--prefix-values',prefix,'--inputs',str(out/'inputs.f32'),'--out',str(out)]
    with (out/'sdk_codec.log').open('wb') as log:
        proc=subprocess.run(cmd,cwd=out,env=local_env(),stdout=log,stderr=subprocess.STDOUT)
    save(out/'command.json',dict(command=cmd,exit_code=proc.returncode,board_accessed=False))
    if proc.returncode:raise RuntimeError('SDK codec failed, see golden/sdk_codec.log')
    meta=read(out/'summary.json'); actual=meta['cast_chain'][-1]['scales']
    if not actual or not np.all(np.array(actual,dtype=np.float32)==s):
        raise ValueError('Compiler scale literal did not reproduce exact SDK F32 scale')
    save(out/'manifest.json',dict(schema='D_sdk_logical_codec_goldens_v1',samples=len(values),
        sources=['32 delivered golden tensors','32 deterministic saturation/tie/nextafter tensors'],
        codec_scope='quantized graph Cast prefix; not the generated graph external ABI',
        direct_device_binding=False,logical_input_shape=[1,40,100],prefix_value_ids=prefix,
        sdk_summary=meta,files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file()}))
    print('SDK codec goldens',len(values),meta['sqb_bytes_per_tensor'],meta['hqb_bytes_per_tensor'])


if __name__=='__main__':main()
