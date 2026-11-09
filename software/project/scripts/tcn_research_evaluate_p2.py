"""Research B fixed-checkpoint evaluation and real input preflight; no optimizer."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from fractions import Fraction
import time

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'artifacts/tcn_research_p2_2026-10-14'
OLD_MANIFEST = Path('@EXTERNAL_ROOT@/manifest.json')
OLD_MANIFEST_SHA = 'e8402f0ea730c0dbdb1727f8c300e8b453ca68e3e8fd80fa9698fb7ffdf27bad'
OLD_CP = Path('@EXTERNAL_ROOT@/best.pt')
OLD_CP_SHA = '8d36f62e8a6839acfd557a1d8178f6052b3aca76436f1a2c017f95bbee05d846'
OLD_CONFIG_SHA = 'f4fe394a5f81b1383727227d2fb9741a0c2e93d08d2e432a4bfa70ccc867c342'
MODEL_SHA = 'acc9e0b6edfc10793df25c69173d94cd9c6c76c74d73294e327dd91b7c3c45fc'

def require(value, message):
    if not value: raise ValueError(message)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def metrics(matrix):
    require(isinstance(matrix,list) and len(matrix)==3 and all(isinstance(r,list) and len(r)==3 for r in matrix),'3x3 matrix required')
    require(all(type(x)is int and x>=0 for r in matrix for x in r),'Integer nonnegative counts required')
    support=[sum(r) for r in matrix]; predicted=[sum(matrix[i][j] for i in range(3)) for j in range(3)]
    total=sum(support); require(total>0,'Nonempty coverage required')
    recall=[matrix[i][i]/support[i] if support[i] else 0. for i in range(3)]
    precision=[matrix[i][i]/predicted[i] if predicted[i] else 0. for i in range(3)]
    f1=[2*matrix[i][i]/(support[i]+predicted[i]) if support[i]+predicted[i] else 0. for i in range(3)]
    return dict(samples=total,accuracy=sum(matrix[i][i] for i in range(3))/total,macro_f1=sum(f1)/3,recall=recall,precision=precision,f1=f1,support=support,confusion_true_rows_predicted_columns=matrix)

def majority_reference(support,majority):
    require(type(majority)is int and 0<=majority<3,'Invalid TRAIN majority')
    require(isinstance(support,list) and len(support)==3 and all(type(v)is int and v>=0 for v in support),'Invalid support')
    c=[[0]*3 for _ in range(3)]
    for i,n in enumerate(support): c[i][majority]=n
    return metrics(c)

def quality(matrix,majority):
    m=metrics(matrix); b=majority_reference(m['support'],majority)
    def exact(c):
        s=[sum(r) for r in c]; p=[sum(c[i][j] for i in range(3)) for j in range(3)]
        return Fraction(sum(c[i][i] for i in range(3)),sum(s)),sum((Fraction(2*c[i][i],s[i]+p[i]) if s[i]+p[i] else Fraction(0) for i in range(3)),Fraction(0))/3
    acc,f1=exact(matrix); ba,bf=exact(b['confusion_true_rows_predicted_columns'])
    recall=all(s>0 and Fraction(matrix[i][i],s)>=Fraction(1,5) for i,s in enumerate(m['support']))
    return dict(eligible_on_tuning_only=acc>ba and f1>bf and recall,accuracy_strictly_above_own_TRAIN_majority=acc>ba,macro_f1_strictly_above_own_TRAIN_majority=f1>bf,each_class_recall_at_least_point_two=recall,majority_reference=b)

def runtime():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    require(torch.cuda.is_available(),'CUDA runtime unavailable; explicit CPU fallback must be registered')
    return np,torch

def state_digest(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(json.dumps([name,str(t.dtype),list(t.shape)],separators=(',',':')).encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def baseline_metadata():
    require(digest(ROOT/'scripts/tcn_candidate.py')==MODEL_SHA,'Historical model code changed')
    require(digest(ROOT/'configs/tcn01_train.json')==OLD_CONFIG_SHA,'Historical config changed')
    require(digest(OLD_MANIFEST)==OLD_MANIFEST_SHA,'Original two-day normalizer identity changed')
    require(digest(OLD_CP)==OLD_CP_SHA,'Selected checkpoint identity changed')
    old=json.loads(OLD_MANIFEST.read_text(encoding='utf-8-sig'))
    require(old['train_class_counts']==[1448237,1379387,1427062] and old['majority_class_from_train']==0,'Original TRAIN majority changed')
    return old

def load_baseline(torch):
    from tcn_candidate import TemporalConvCandidate
    # This project-created, exact SHA-bound historical full checkpoint contains
    # NumPy RNG state. Its trusted origin is checked before deserialization.
    require(digest(OLD_CP)==OLD_CP_SHA,'Historical checkpoint changed before load')
    cp=torch.load(OLD_CP,map_location='cpu',weights_only=False)
    require(cp['epoch']==1 and cp['experiment']=='TCN01','Selected epoch identity changed')
    require(cp['identity']['manifest_sha256']==OLD_MANIFEST_SHA,'Checkpoint origin differs')
    require(cp['identity']['files_sha256']['configs\\tcn01_train.json']==OLD_CONFIG_SHA,'Checkpoint config differs')
    require(cp['config']==json.loads((ROOT/'configs/tcn01_train.json').read_text(encoding='utf-8-sig')),'Checkpoint full config differs')
    model=TemporalConvCandidate(); model.load_state_dict(cp['model'],strict=True)
    require(sum(p.numel() for p in model.parameters())==33219,'Parameter count differs')
    require(all(torch.isfinite(v).all().item() for v in model.state_dict().values()),'Nonfinite checkpoint')
    return model,cp

def checked_output(out):
    p=Path(out); require(p.is_absolute() and '..' not in p.parts and p.is_relative_to(STAGE),'Only new P2 workspace output allowed')
    resolved=p.resolve(); require(resolved.is_relative_to(STAGE.resolve()),'Output resolves outside P2 stage')
    require(all(not q.is_symlink() and not (getattr(q.stat(),'st_file_attributes',0)&0x400) for q in p.parents if q.exists()),'Output parent cannot be a symlink or junction')
    require(not p.exists(),'Output already exists; use saved evidence')
    p.mkdir(parents=True); return p

def admitted_data(path,expected_sha256):
    # The data module performs admission and validates source/sidecar identities.
    import tcn_research_data_p2 as data
    return data, data.load_prepared(path,expected_sha256=expected_sha256)

def reference_samples(data,m):
    """Independently construct boundary/middle windows without the new iterator."""
    import numpy as np
    dtype=np.dtype([('sequence','<u8'),('timestamp_ns','<u8'),('segment_id','<u4'),('flags','<u4'),('bid_levels','<u4'),('ask_levels','<u4'),('features','<u8',(40,))])
    checked=0
    for role in ('train','tuning_validation'):
        ds=data.ResearchWindows(m,role,m['normalization'])
        try:
            base=0
            for r in ds.entries:
                n=r['endpoint_stop_exclusive']-r['endpoint_start']; local=[0,n//2,n-1]
                actual,y=ds.batch(np.asarray([base+i for i in local],dtype=np.int64))
                raw=np.memmap(r['records_path'],dtype=dtype,mode='r')
                try:
                    for j,offset in enumerate(local):
                        end=r['endpoint_start']+offset
                        values=np.array(raw['features'][end-99:end+1],dtype=np.float64); values[:,0::2]/=10000.
                        expected=((values-np.asarray(m['normalization']['mean']))/np.asarray(m['normalization']['std'])).T.astype(np.float32)
                        now=int(raw['features'][end][0])+int(raw['features'][end][2]); future=int(raw['features'][end+50][0])+int(raw['features'][end+50][2]); label=0 if future<now else 2 if future>now else 1
                        require(np.array_equal(actual[j],expected) and int(y[j])==label,'Independent raw-window/label reference differs')
                        checked+=1
                finally: raw._mmap.close()
                base+=n
        finally: ds.close()
    return dict(samples_checked=checked,boundary_middle_last_per_range=True,input_tensor_exact=True,integer_H50_label_exact=True)

def preflight(data_path,out,expected_sha256):
    started=time.perf_counter(); data,m=admitted_data(data_path,expected_sha256); old=baseline_metadata(); out=checked_output(out)
    np,torch=runtime()
    reference=reference_samples(data,m)
    from tcn_candidate import TemporalConvCandidate
    torch.manual_seed(42); model=TemporalConvCandidate().eval(); before=state_digest(model)
    gpu=TemporalConvCandidate().eval(); gpu.load_state_dict(model.state_dict()); gpu.to('cuda')
    checks=[]; maximum=0.; forwards=0; backwards=0
    for role in ('train','tuning_validation'):
        for norm_name,norm in [('five_day',m['normalization']),('old_two_day',old['normalization'])]:
            ds=data.ResearchWindows(m,role,norm)
            try:
                for size in (1,16,256):
                    ids=np.linspace(0,len(ds)-1,size,dtype=np.int64); x,y=ds.batch(ids)
                    require(x.shape==(size,40,100) and x.dtype==np.float32 and np.isfinite(x).all(),'Real input layout invalid')
                    with torch.inference_mode():
                        a=model(torch.from_numpy(x)); b=gpu(torch.from_numpy(x).to('cuda')).cpu(); forwards+=2
                    err=float((a-b).abs().max()); maximum=max(maximum,err)
                    ok=torch.allclose(a,b,rtol=1e-5,atol=1e-6); arg=torch.equal(a.argmax(1),b.argmax(1))
                    require(ok and arg,'CPU/GPU real-input output mismatch')
                    checks.append(dict(role=role,normalization=norm_name,batch_size=size,max_abs_error=err,allclose=ok,argmax_equal=arg,input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),label_counts=np.bincount(y,minlength=3).tolist()))
            finally: ds.close()
    # Two bounded real TRAIN backward calls validate differentiability, without an optimizer.
    ds=data.ResearchWindows(m,'train',m['normalization'])
    try: x,y=ds.batch(np.linspace(0,len(ds)-1,16,dtype=np.int64))
    finally: ds.close()
    gradient_checks=[]
    for net,device in [(model,'cpu'),(gpu,'cuda')]:
        net.train(); logits=net(torch.from_numpy(x).to(device)); forwards+=1
        loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(y).to(device)); loss.backward(); backwards+=1
        gradients=[p.grad for p in net.parameters()]; require(all(g is not None and torch.isfinite(g).all().item() for g in gradients),'Nonfinite/missing gradients')
        norm=float(sum((g.detach().double().square().sum().item() for g in gradients))**.5); require(norm>0,'All gradients zero')
        gradient_checks.append(dict(device=device,loss=float(loss.detach().cpu()),gradient_tensors=len(gradients),gradient_l2_norm=norm))
        net.zero_grad(set_to_none=True); net.eval()
    require(before==state_digest(model)==state_digest(gpu),'Preflight changed model state')
    oldmodel,cp=load_baseline(torch); old_before=state_digest(oldmodel); oldmodel.to('cuda').eval()
    ds=data.ResearchWindows(m,'tuning_validation',old['normalization'])
    try: x,y=ds.batch(np.linspace(0,len(ds)-1,256,dtype=np.int64))
    finally: ds.close()
    with torch.inference_mode(): z=oldmodel(torch.from_numpy(x).to('cuda')); forwards+=1
    require(torch.isfinite(z).all().item() and state_digest(oldmodel)==old_before,'Old model preflight failed')
    result=dict(version='TCN-research-real-input-preflight-1.0',passed=True,runtime_preflight_passed=True,data_manifest_path=str(Path(data_path).resolve()),data_manifest_sha256=digest(data_path),model_code_sha256=MODEL_SHA,evaluator_source_sha256=digest(__file__),checks=checks,gradient_checks=gradient_checks,max_cpu_gpu_abs_error=maximum,model_state_unchanged=True,all_gradients_cleared=True,model_constructions=3,forward_calls=forwards,backward_calls=backwards,optimizer_constructions=0,optimizer_updates=0,training_epochs=0,training_runs=0,old_checkpoint_sha256=OLD_CP_SHA,old_selected_epoch=cp['epoch'],sealed_labels_or_scores_opened=False,source_flag64_cleared=False,seconds=time.perf_counter()-started,runtime=dict(torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,cpu_threads=4,precision='float32',tf32=False,deterministic_algorithms=True))
    result['independent_real_window_reference']=reference
    result['original_checkpoint_sha256_after']=digest(OLD_CP)
    require(result['original_checkpoint_sha256_after']==OLD_CP_SHA,'Checkpoint changed during preflight')
    write(out/'preflight.json',result); return result

def evaluate(data_path,out,expected_sha256):
    started=time.perf_counter(); data,m=admitted_data(data_path,expected_sha256); old=baseline_metadata(); out=checked_output(out); np,torch=runtime()
    model,cp=load_baseline(torch); before=state_digest(model); model.to('cuda').eval()
    source_before={d:digest(m['sources'][d]['records_path']) for d in ('2025-12-15','2025-12-16')}
    require(all(source_before[d]==m['sources'][d]['records_sha256'] for d in source_before),'Tuning source changed before evaluation')
    matrices={d:np.zeros((3,3),dtype=np.int64) for d in ('2025-12-15','2025-12-16')}; losses={d:0. for d in matrices}; counts={d:0 for d in matrices}; batches={d:0 for d in matrices}
    predictions={d:(out/(d+'.predictions.u8')).open('wb') for d in matrices}
    label_hash={d:hashlib.sha256() for d in matrices}; endpoint_hash={d:hashlib.sha256() for d in matrices}
    forward_seconds=0.; total_batches=0; seen=0
    try:
        with torch.inference_mode():
            for batch in data.iter_batches(m,'tuning_validation',old['normalization'],256):
                x=batch['X']; y=batch['y']; ends=batch['endpoints']; d=str(batch['date'][0])
                require(d in matrices and np.all(batch['date']==d),'Ordered evaluation batch crosses days')
                tx=torch.from_numpy(x).to('cuda'); ty=torch.from_numpy(y).to('cuda'); torch.cuda.synchronize(); mark=time.perf_counter()
                logits=model(tx); torch.cuda.synchronize(); forward_seconds+=time.perf_counter()-mark
                require(torch.isfinite(logits).all().item(),'Nonfinite outputs')
                ce=torch.nn.functional.cross_entropy(logits,ty,reduction='none'); require(torch.isfinite(ce).all().item(),'Nonfinite CE')
                pred=logits.argmax(1).cpu().numpy().astype('u1'); predictions[d].write(pred.tobytes()); label_hash[d].update(y.astype('u1').tobytes()); endpoint_hash[d].update(ends.astype('<u8').tobytes())
                matrices[d]+=np.bincount(3*y+pred,minlength=9).reshape(3,3); losses[d]+=float(ce.double().sum().cpu()); counts[d]+=len(y); batches[d]+=1; total_batches+=1; seen+=len(y)
                if total_batches==1 or total_batches%1000==0: print(json.dumps(dict(evaluation='old_TCN01_research_B',processed=seen,expected=6612655,batches=total_batches)),flush=True)
    finally:
        for f in predictions.values(): f.close()
    per_day={}
    for d in matrices:
        r=metrics(matrices[d].tolist()); r['loss']=losses[d]/counts[d]; r['cross_entropy_sum']=losses[d]; r['batches']=batches[d]
        r['predictions']=dict(path=str(out/(d+'.predictions.u8')),sha256=digest(out/(d+'.predictions.u8')),bytes=counts[d]); r['ordered_labels_sha256']=label_hash[d].hexdigest(); r['ordered_endpoints_sha256']=endpoint_hash[d].hexdigest(); r['quality']=quality(matrices[d].tolist(),old['majority_class_from_train']); per_day[d]=r
        report=m['day_label_reports'][d]
        require(r['samples']==report['window_count'] and r['support']==report['class_counts'],'Per-day label coverage differs')
        require(r['ordered_labels_sha256']==report['label_sha256'] and r['ordered_endpoints_sha256']==report['endpoint_stream_sha256'],'Ordered endpoint/label identity differs')
    matrix=sum(matrices.values()); aggregate=metrics(matrix.tolist()); aggregate['loss']=sum(losses.values())/seen
    require(seen==m['sample_counts']['tuning_validation'] and seen==6612655,'Incomplete tuning coverage')
    require(before==state_digest(model) and all(p.grad is None for p in model.parameters()),'Evaluation mutated weights/gradients')
    source_after={d:digest(m['sources'][d]['records_path']) for d in source_before}
    require(source_before==source_after,'Tuning sources changed during evaluation')
    result=dict(version='TCN-research-fixed-old-baseline-evaluation-1.0',source_kind='accepted_real_research_B',checkpoint_path=str(OLD_CP),checkpoint_sha256=OLD_CP_SHA,selected_epoch=cp['epoch'],old_model_config_sha256=OLD_CONFIG_SHA,old_origin_manifest_sha256=OLD_MANIFEST_SHA,old_normalization_sha256=canonical(old['normalization']),normalization_fit_dates=['2025-12-08','2025-12-09'],own_TRAIN_class_counts=old['train_class_counts'],own_TRAIN_majority=old['majority_class_from_train'],data_manifest_path=str(Path(data_path).resolve()),data_manifest_sha256=digest(data_path),model_code_sha256=MODEL_SHA,evaluator_source_sha256=digest(__file__),aggregate=aggregate,per_day=per_day,quality=quality(matrix.tolist(),old['majority_class_from_train']),candidate_majority_reference=majority_reference(aggregate['support'],m['majority_class_from_train']),model_state_sha256_before=before,model_state_sha256_after=state_digest(model),model_state_unchanged=True,all_gradients_none=True,forward_calls=total_batches,backward_calls=0,optimizer_updates=0,training_epochs=0,training_runs=0,new_reservations=0,sealed_labels_or_scores_opened=False,source_control_verified_for_new_days=False,coverage_unknown_for_new_days=True,original_flag64_cleared=False,forward_seconds_excludes_input_transfer_and_loss=forward_seconds,total_seconds=time.perf_counter()-started,device='cuda',precision='float32',batch_size=256,score_scope='Fixed historical TCN01 on common 15-16 research B tuning windows; no five-day model trained',result_wording='基于当前可观察数据和既定筛选规则的表现')
    result['source_records_sha256_before']=source_before; result['source_records_sha256_after']=source_after
    result['original_checkpoint_sha256_after']=digest(OLD_CP)
    require(result['original_checkpoint_sha256_after']==OLD_CP_SHA,'Checkpoint file changed during evaluation')
    result['protocol_sha256']=m['protocol_sha256']; result['data_sources_identity_sha256']=canonical(m['sources']); result['tuning_ranges_sha256']=canonical([r for r in m['ranges'] if r['role']=='tuning_validation'])
    write(out/'evaluation.json',result); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['preflight','evaluate']); p.add_argument('--data',required=True); p.add_argument('--data-sha',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    r=preflight(a.data,a.output,a.data_sha) if a.command=='preflight' else evaluate(a.data,a.output,a.data_sha)
    print(json.dumps({k:v for k,v in r.items() if k in ['passed','aggregate','forward_calls','backward_calls','total_seconds','seconds']},ensure_ascii=True))
if __name__=='__main__': main()
