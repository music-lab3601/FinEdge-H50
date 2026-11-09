"""Codec boundary verification and old observed-data screen; never fresh data."""
import subprocess
import numpy as np
from common3 import ROOT,BASE,V2,PROJECT,read,save,sha,env
from codec3 import encode,decode,sdk_source,SCALE

MODEL=ROOT/'artifacts/int8_physical/d3p_ZG.json'

def run(mode,inputs,out,model=MODEL,integer=True):
    out.mkdir(exist_ok=False)
    exe=ROOT/'build/Release/d_integer_host.exe' if integer else BASE/'build/Release/d_host_tools.exe'
    cmd=[str(exe),'--mode',mode,'--json',str(model),'--raw',str(model.with_suffix('.raw')),'--inputs',str(inputs),'--out',str(out)]
    with (out/'host.log').open('wb') as log:p=subprocess.run(cmd,cwd=out,env=env(),stdout=log,stderr=subprocess.STDOUT)
    save(out/'command.json',dict(command=cmd,exit_code=p.returncode,model_sha256=sha(model),params_sha256=sha(model.with_suffix('.raw')),board_accessed=False))
    if p.returncode:raise RuntimeError('Host failed: '+str(out))

def main():
    gold=ROOT/'golden_v2';gold.mkdir(exist_ok=False)
    vendor=PROJECT/'vendor/ed4_model/golden/inputs.f32'
    known=np.fromfile(vendor,'<f4').reshape(32,1,40,100)
    # Construct in normalized domain; physical arithmetic/rounding is verified
    # after the actual float32 division, not assumed exact at nominal ties.
    edges=np.array([-1000,-129,-128.5,-128,-127.5,-126.5,-2.5,-1.5,-.5,0,.5,1.5,2.5,126.5,127,127.5,128,1000],dtype='<f4')
    cases=[x.copy() for x in known]
    for i in range(32):
        v=np.resize(np.roll(edges,i),4000).reshape(1,40,100)*SCALE[None,:,None]
        if i%3==1:v=np.nextafter(v,np.float32(np.inf))
        if i%3==2:v=np.nextafter(v,np.float32(-np.inf))
        cases.append(v.astype('<f4'))
    values=np.array(cases,dtype='<f4');np.save(gold/'normalized.npy',values)
    blob=b''.join(encode(x) for x in values);(gold/'encoded.bin').write_bytes(blob)
    (gold/'encoded.hex').write_text('\n'.join(f'{b:02x}' for b in blob)+'\n',encoding='ascii')
    np.array([sdk_source(x) for x in values],dtype='<f4').tofile(gold/'sdk_source.f32')
    run('encode',gold/'sdk_source.f32',gold/'sdk')
    actual=(gold/'sdk/sdk_encoded.bin').read_bytes()
    if actual!=blob:
        save(gold/'verification.json',dict(status='failed',different_bytes=sum(a!=b for a,b in zip(actual,blob)),expected_bytes=len(blob),actual_bytes=len(actual)))
        raise RuntimeError('Python vs SDK physical codec mismatch')
    save(gold/'verification.json',dict(status='passed',samples=64,physical_elements=307200,encoded_bytes=len(blob),sdk_bytes_equal=True,
        vendor_samples=32,boundary_samples=32,rounding='clip(floor(float32(float32(x/scale)+0.5f)),-128,127)',
        fixed_scale_per_feature=True,padding_codes=0,source_sha256=sha(vendor),encoded_sha256=sha(gold/'encoded.bin'),board_accessed=False))
    print('64 physical encoder goldens match SDK bytes',flush=True)
    out=ROOT/'artifacts/observed_screen';out.mkdir(exist_ok=False)
    x=np.load(V2/'data/inputs.npy');(out/'inputs.i8').write_bytes(b''.join(encode(v[None]) for v in x))
    decoded=np.array([decode(encode(v[None]))[0] for v in x],dtype='<f4');decoded.tofile(out/'dequantized.f32')
    run('forward',out/'inputs.i8',out/'integer')
    run('forward',out/'dequantized.f32',out/'dequantized_mixed',V2/'artifacts/mixed_tf32_compat/tcn_d_ZG.json',False)
    print('Observed-data screen completed',flush=True)

if __name__=='__main__':main()
