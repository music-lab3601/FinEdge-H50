"""Offline later 1024 holdout, generated only after numerical candidate freeze."""
import subprocess
import numpy as np
from common3 import ROOT,BASE,V2,PROJECT,read,save,sha,env
from codec3 import encode

ROW=np.dtype([('identity','<u8',(4,)),('raw','<u8',(40,)),('normalized','<f4',(40,))])

def frozen_check():
    f=read(ROOT/'contracts/frozen_candidate.json')
    assert f['input_contract_sha256']==sha(ROOT/'input_contract.json')
    assert f['codec_sha256']==sha(ROOT/'scripts/codec3.py')
    # codec3 reads this historical build-time carrier for the 40 constants.
    # Bind the actual constants to the independently frozen public contract.
    public=np.array(read(ROOT/'input_contract.json')['numerical_quantization']['scale'],dtype='<f4')
    carrier=np.array(read(ROOT/'contracts/quantization_int8_folded.json')['scale'],dtype='<f4')
    assert np.array_equal(public,carrier)
    assert np.array_equal(public,np.fromfile(ROOT/'artifacts/int8_folded/scales.f32','<f4'))
    for m in f['models'].values():
        assert sha(ROOT/m['json'])==m['json_sha256']
        assert sha(ROOT/m['raw'])==m['raw_sha256']
    return f

def main():
    f=frozen_check();out=ROOT/'data';out.mkdir(exist_ok=False)
    endpoints=np.linspace(250099,450210,1024,dtype=np.int64)
    assert np.diff(endpoints).min()>=150
    raw=PROJECT/'versions/network_fast_v1/artifacts/raw_mbo56.bin'
    assert sha(raw)=='f837fe8870cde83feed2f0ae21c73d778586249790219f976e53d60e0484f9d6'
    rows_path=out/'offline_rebuilt_rows.bin'
    cmd=[str(V2/'build/Release/d_fresh_book.exe'),str(raw),str(rows_path),str(int(endpoints.max())+51)]
    with (out/'extract.log').open('wb') as log:r=subprocess.run(cmd,cwd=out,env=env(),stdout=log,stderr=subprocess.STDOUT)
    save(out/'extract_command.json',dict(command=cmd,exit_code=r.returncode,mode='offline_not_realtime'))
    if r.returncode:raise RuntimeError('Holdout extract failed')
    rows=np.memmap(rows_path,mode='r',dtype=ROW)
    prefix=np.memmap(V2/'data/offline_rebuilt_rows.bin',mode='r',dtype=ROW)
    assert np.array_equal(rows[:len(prefix)],prefix)
    norm=read(PROJECT/'vendor/ed4_model/normalization.json')
    selected=[];labels=[];ids=[];physical=[];items=[]
    for i,end in enumerate(endpoints):
        part=rows[end-99:end+51];identity=part['identity']
        assert len(part)==150 and np.all(identity[:,0]==identity[0,0]) and np.all(np.diff(identity[:,1].astype('i8'))==1),int(end)
        x=part['raw'][:100].astype('f8');mid=(x[:,0]+x[:,2])/2
        x[:,0::2]=(x[:,0::2]-mid[:,None])/mid[:,None]
        x=((x-np.array(norm['mean']))/np.array(norm['std'])).astype('<f4')
        assert np.array_equal(x,part['normalized'][:100])
        selected.append(x.T.copy());ids.append(identity.copy());physical.append(part['raw'].copy())
        a=int(part['raw'][99,0])+int(part['raw'][99,2]);b=int(part['raw'][149,0])+int(part['raw'][149,2])
        labels.append(2 if b>a else 0 if b<a else 1)
        items.append(dict(index=i,endpoint=int(end),footprint=[int(end)-99,int(end)+50]))
    x=np.array(selected,dtype='<f4');y=np.array(labels,dtype='<i8')
    np.save(out/'inputs.npy',x);x.tofile(out/'inputs.f32');(out/'inputs.i8').write_bytes(b''.join(encode(v[None]) for v in x))
    for name,val in [('labels',y),('endpoints',endpoints),('identities',np.array(ids,dtype='<u8')),('raw_footprints',np.array(physical,dtype='<u8'))]:np.save(out/(name+'.npy'),val)
    save(out/'manifest.json',dict(schema='D_lowbit_holdout_v3',date='2025-12-15',symbol='AAPL',samples=len(y),class_counts=np.bincount(y,minlength=3).tolist(),
        source_sha256='0e69b858781b8c7d82e4563b5098f70ff45b02a17c8c3a88aa8880c430ba2fd7',raw_cache_sha256=sha(raw),
        source_reconstructed_records=len(rows),rows_sha256=sha(rows_path),known_prefix_records_checked=len(prefix),prefix_bit_equal=True,
        candidate_frozen_sha256=sha(ROOT/'contracts/frozen_candidate.json'),new_first_input_index=250000,last_label_index=450260,
        old_max_input_or_label_index=150260,unused_gap_records=99739,full_footprints_nonoverlapping=True,min_gap_records_between_footprints=int(np.diff(endpoints).min()-150),
        independent_normalization_recomputed=True,normalization_refitted=False,items=items,
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file() and p!=rows_path},
        limitations=['same-day later engineering holdout, not unseen trading date','reconstruction not aligned to teammate validation cache','nonoverlap is not proof of statistical market independence']))
    print('Prepared fresh1024',np.bincount(y,minlength=3).tolist())

if __name__=='__main__':main()
