"""Build untouched later holdout only after the single candidate is frozen."""
import subprocess
import numpy as np
from common2 import ROOT,BASE,PROJECT,read,save,sha,env

ROW=np.dtype([('identity','<u8',(4,)),('raw','<u8',(40,)),('normalized','<f4',(40,))])
assert ROW.itemsize==512


def main():
    frozen=read(ROOT/'contracts/frozen_candidate.json')
    if frozen['model_sha256']!=sha(ROOT/frozen['model_path']):raise ValueError('Candidate changed')
    plan=read(ROOT/'contracts/study_plan.json')['fresh_evaluation']
    endpoints=np.linspace(50099,150210,512,dtype=np.int64)
    if np.any(np.diff(endpoints)<150):raise ValueError('Fresh footprints overlap')
    out=ROOT/'data';out.mkdir(exist_ok=False)
    raw=PROJECT/'versions/network_fast_v1/artifacts/raw_mbo56.bin'
    raw_meta=read(PROJECT/'versions/network_fast_v1/artifacts/raw_cache.json')
    if sha(raw)!=raw_meta['raw_sha256']:raise ValueError('Raw cache changed')
    rows_path=out/'offline_rebuilt_rows.bin'
    command=[str(ROOT/'build/Release/d_fresh_book.exe'),str(raw),str(rows_path),str(int(endpoints.max())+51)]
    with (out/'extract.log').open('wb') as log:
        p=subprocess.run(command,cwd=out,env=env(),stdout=log,stderr=subprocess.STDOUT)
    save(out/'extract_command.json',dict(command=command,exit_code=p.returncode,kind='offline reconstruction not realtime benchmark'))
    if p.returncode:raise RuntimeError('Fresh extraction failed')
    rows=np.memmap(rows_path,mode='r',dtype=ROW)
    old_ids=np.load(BASE/'data/identities.u64.npy');old_raw=np.load(BASE/'data/raw.u64.npy');old_norm=np.load(BASE/'data/rows.f32.npy')
    n=len(old_ids)
    if not(np.array_equal(rows['identity'][:n],old_ids) and np.array_equal(rows['raw'][:n],old_raw)
           and np.array_equal(rows['normalized'][:n],old_norm)):
        raise ValueError('Reconstructed known prefix differs from frozen v1 trace')
    norm=read(PROJECT/'vendor/ed4_model/normalization.json')
    selected=[];labels=[];identities=[];raw_footprints=[];items=[]
    for i,end in enumerate(endpoints):
        part=rows[end-99:end+51]
        ids=part['identity']
        if len(part)!=150 or not np.all(ids[:,0]==ids[0,0]) or not np.all(np.diff(ids[:,1].astype(np.int64))==1):
            raise ValueError('Fresh fixed footprint crosses invalid profile segment: '+str(int(end)))
        # Independently reconstruct all FP32 inputs using frozen float64 semantics.
        physical=part['raw'][:100].astype(np.float64)
        mid=(physical[:,0]+physical[:,2])/2
        physical[:,0::2]=(physical[:,0::2]-mid[:,None])/mid[:,None]
        values=((physical-np.array(norm['mean']))/np.array(norm['std'])).astype('<f4')
        if not np.array_equal(values,part['normalized'][:100]):raise ValueError('Fresh normalization differs')
        x=values.T.copy();selected.append(x);identities.append(ids.copy());raw_footprints.append(part['raw'].copy())
        a=int(part['raw'][99,0])+int(part['raw'][99,2]);b=int(part['raw'][149,0])+int(part['raw'][149,2])
        labels.append(2 if b>a else 0 if b<a else 1)
        items.append(dict(index=i,endpoint_accepted_index=int(end),footprint_accepted_range=[int(end)-99,int(end)+50],
                          first_identity=ids[0].tolist(),last_identity=ids[99].tolist(),future_identity=ids[149].tolist()))
    inputs=np.array(selected,dtype='<f4'); y=np.array(labels,dtype='<i8')
    np.save(out/'inputs.npy',inputs);inputs.tofile(out/'inputs.f32')
    np.save(out/'labels.npy',y);np.save(out/'endpoints.npy',endpoints)
    np.save(out/'identities.npy',np.array(identities,dtype='<u8'))
    np.save(out/'raw_footprints.npy',np.array(raw_footprints,dtype='<u8'))
    save(out/'manifest.json',dict(schema='D_fresh_later_holdout_v2',status='prepared_after_candidate_freeze',
        purpose='previously unused later same-day engineering evaluation',date='2025-12-15',symbol='AAPL',
        source_sha256=raw_meta['source_sha256'],raw_cache_sha256=raw_meta['raw_sha256'],
        known_prefix_records_checked=n,known_prefix_bit_equal=True,
        source_reconstructed_records=len(rows),source_rows_sha256=sha(rows_path),samples=len(y),
        class_counts=np.bincount(y,minlength=3).tolist(),footprints_nonoverlapping=True,
        min_gap_records_between_footprints=int(np.diff(endpoints).min()-150),
        old_max_input_or_label_index=10098,new_first_input_index=50000,
        isolated_from_previous_holdout_and_calibration=True,
        normalized_using_original_parameters=True,normalization_refitted=False,
        candidate_frozen_sha256=sha(ROOT/'contracts/frozen_candidate.json'),
        candidate_model_sha256=frozen['model_sha256'],items=items,
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file() and p!=rows_path},
        limitations=['same day and engineering reconstruction, not an independent trading date',
                     'nonoverlapping records do not prove market observations are statistically independent']))
    print('Fresh holdout prepared',len(y),np.bincount(y,minlength=3).tolist())

if __name__=='__main__':main()
