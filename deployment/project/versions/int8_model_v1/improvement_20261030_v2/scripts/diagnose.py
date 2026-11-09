"""Only calibration tensors select sensitive layer regions."""
import subprocess
import numpy as np
from common2 import ROOT,BASE,read,save,sha,env

def main():
    out=ROOT/'artifacts/layer_diagnostics';out.mkdir(exist_ok=False)
    cal=np.load(BASE/'data/calibration.inputs.npy');indices=np.linspace(0,len(cal)-1,16,dtype=int)
    cal[indices].astype('<f4').tofile(out/'inputs.f32')
    sources={'fp32':BASE/'artifacts/compile/shared/tcn_d_optimized.json',
             'int8':BASE/'artifacts/compile/int8/tcn_d_quantized.json'}
    mapping={}
    for name,model in sources.items():
        folder=out/name;folder.mkdir()
        cmd=[str(ROOT/'build/Release/d_layer_dump.exe'),str(model),str(model.with_suffix('.raw')),str(out/'inputs.f32'),str(folder)]
        with (folder/'log.txt').open('wb') as log:
            p=subprocess.run(cmd,cwd=folder,env=env(),stdout=log,stderr=subprocess.STDOUT)
        save(folder/'command.json',dict(command=cmd,exit_code=p.returncode))
        if p.returncode:raise RuntimeError('Layer dump failed: '+str(folder))
        mapping[name]={(r['sample'],r['op_id'],r['output']):r for line in (folder/'tensors.jsonl').read_text().splitlines() for r in [__import__('json').loads(line)]}
    operators={o['op_id']:o for o in read(sources['fp32'])['ops']}
    rows=[]
    for opid,op in operators.items():
        aa=[];bb=[]
        for sample in range(len(indices)):
            key=(sample,opid,0)
            if key not in mapping['fp32'] or key not in mapping['int8']:continue
            a=np.fromfile(out/'fp32'/mapping['fp32'][key]['file'],dtype='<f4')
            b=np.fromfile(out/'int8'/mapping['int8'][key]['file'],dtype='<f4')
            if a.shape!=b.shape:raise ValueError('Common-op output shape differs')
            aa.append(a);bb.append(b)
        if not aa:continue
        a=np.concatenate(aa).astype(float);b=np.concatenate(bb).astype(float);diff=a-b
        den=np.linalg.norm(a)*np.linalg.norm(b)
        rows.append(dict(op_id=opid,type=op['_type_key'].split('::')[-1],samples=len(aa),
            max_abs=float(abs(diff).max()),rmse=float(np.sqrt(np.mean(diff*diff))),
            relative_l2=float(np.linalg.norm(diff)/max(np.linalg.norm(a),1e-12)),
            cosine=float(np.dot(a,b)/den) if den else None))
    save(out/'report.json',dict(status='completed',selection_scope='16 existing calibration windows only',
        calibration_indices=indices.tolist(),calibration_inputs_sha256=sha(BASE/'data/calibration.inputs.npy'),
        methods='Match original op IDs; dequantize outputs with SDK Cast. Error includes upstream propagation, not isolated causal attribution.',
        rows=rows,board_accessed=False))
    print(rows)
if __name__=='__main__':main()
