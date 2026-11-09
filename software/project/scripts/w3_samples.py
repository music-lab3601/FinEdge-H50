"""Candidate label diagnostics and gated W3 preparation; no held-out model selection."""
import argparse
from bisect import bisect_right
import datetime as dt
import json
from pathlib import Path
import sys
import numpy as np
from w1 import ROOT, config, save, sha256, stamp
from w2 import DTYPE
from w3 import draft, pointer

WINDOW=100


def physical(features):
    x=np.asarray(features,dtype=np.float64).copy()
    x[...,0::2]/=10000.0
    return x


def endpoints(start, stop, horizon):
    if not isinstance(horizon,int) or isinstance(horizon,bool) or horizon<1:
        raise ValueError('H must be a positive integer number of book-update records')
    if start<0 or stop<start: raise ValueError('Invalid segment boundaries')
    return range(start+WINDOW-1,max(start+WINDOW-1,stop-horizon))


def labels(records, ends, horizon, threshold):
    if not np.isfinite(threshold) or threshold<0: raise ValueError('Threshold must be finite and nonnegative')
    ends=np.asarray(ends,dtype=np.int64)
    current=records['features'][ends][:,[0,2]].astype(np.float64).sum(axis=1)
    future=records['features'][ends+horizon][:,[0,2]].astype(np.float64).sum(axis=1)
    if np.any(current<=0): raise ValueError('Invalid midpoint')
    difference=future-current
    return np.where(difference>current*threshold,2,np.where(difference < -current*threshold,0,1)).astype(np.int64)


def fit_stats(chunks):
    count=0; mean=np.zeros(40); m2=np.zeros(40)
    for x in chunks:
        x=physical(x)
        if len(x)==0: continue
        if not np.all(np.isfinite(x)): raise ValueError('Non-finite feature')
        n=len(x); avg=x.mean(axis=0); delta=avg-mean
        m2+=((x-avg)**2).sum(axis=0)+delta**2*count*n/(count+n)
        mean+=delta*n/(count+n); count+=n
    if not count: raise ValueError('No training records')
    std=np.sqrt(m2/count); zero=std==0; std[zero]=1
    return {'record_count':count,'mean':mean.tolist(),'std':std.tolist(),'zero_variance_columns':np.flatnonzero(zero).tolist(),
            'fit_scope':'unique valid records from training dates only; population std'}


class RingWindow:
    """CPU reference: reset at every segment/date boundary; raw input only."""
    def __init__(self): self.reset()
    def reset(self): self.buffer=np.empty((WINDOW,40),dtype=np.float64);self.count=0
    def push(self, record):
        self.buffer[self.count%WINDOW]=record; self.count+=1
        if self.count<WINDOW:return None
        start=self.count%WINDOW
        return np.concatenate((self.buffer[start:],self.buffer[:start]),axis=0)


class Windows:
    """On-demand windows; labels never appear in model input. Worker-safe lazy maps."""
    def __init__(self, manifest, split):
        self.manifest=manifest;self.entries=[r for r in manifest['ranges'] if r['split']==split]
        self.cumulative=np.cumsum([r['end_stop']-r['end_start'] for r in self.entries]).tolist()
        self.maps={};self.mean=np.asarray(manifest['normalization']['mean']);self.std=np.asarray(manifest['normalization']['std'])
    def __len__(self):return self.cumulative[-1] if self.cumulative else 0
    def __getstate__(self):return {**self.__dict__,'maps':{}}
    def close(self):
        for array in self.maps.values():array._mmap.close()
        self.maps.clear()
    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        j=bisect_right(self.cumulative,index);r=self.entries[j]
        end=r['end_start']+index-(self.cumulative[j-1] if j else 0)
        path=r['records_path']
        if path not in self.maps:self.maps[path]=np.memmap(path,dtype=DTYPE,mode='r')
        a=self.maps[path]
        x=((physical(a['features'][end-99:end+1])-self.mean)/self.std).T.astype(np.float32)
        y=labels(a,[end],self.manifest['horizon'],self.manifest['threshold'])[0]
        return x,y


def get_run(date, recheck=False):
    from w3_engine import check
    folder=Path(json.loads(pointer(date).read_text(encoding='utf-8'))['folder'])
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    if report['status']!='passed_full_day_engineering_audit' or not report['full_day']:
        raise ValueError(f'{date} full audit has not passed')
    if report['trading_date']!=date or report['symbol']!=draft()['symbol']:
        raise ValueError('Date/symbol mismatch')
    pinned=draft().get('source_versions',{}).get(date)
    if pinned:
        outputs=json.loads((folder/'output_hashes.json').read_text(encoding='utf-8'))
        if report['source_sha256']!=pinned['raw_sha256'] or outputs['records.bin']!=pinned['records_sha256']:
            raise ValueError('Data version differs from the jointly approved snapshot')
    if recheck:check(folder)
    a=np.memmap(folder/'records.bin',dtype=DTYPE,mode='r')
    segments=json.loads((folder/'segments.json').read_text(encoding='utf-8'))
    return folder,report,a,segments


def count_labels(a,segments,h,t):
    counts=np.zeros(3,dtype=np.int64)
    for s in segments:
        ends=endpoints(s['start_row'],s['end_row']+1,h)
        for begin in range(ends.start,ends.stop,100000):
            y=labels(a,np.arange(begin,min(begin+100000,ends.stop)),h,t)
            counts+=np.bincount(y,minlength=3)
    return counts


def profile():
    d=draft(); result={'scope':'candidate diagnostics on proposed TRAIN dates only; not frozen; no model accuracy', 'results':[]}
    for date,role in d['date_roles_proposal'].items():
        if role!='train':continue
        folder,report,a,segments=get_run(date,True)
        for h in (20,50,100):
            for bp in (0,1,2,5):
                counts=count_labels(a,segments,h,bp/10000)
                result['results'].append({'date':date,'H':h,'threshold_basis_points':bp,'counts_down_flat_up':counts.tolist(),
                                          'total':int(counts.sum())})
    save(ROOT/'artifacts/w3_2026-10-03/train_label_candidates.json',result)
    print(json.dumps(result,indent=2))


def validate_protocol(d):
    if not d.get('formal_training_protocol_frozen') or not d.get('shared_comparison_confirmed'):
        raise ValueError('STOP: data/label/split protocol and joint comparison are not confirmed')
    h=d['label_horizon'];t=d['label_threshold'];endpoints(0,200,h)
    if t is None or not np.isfinite(t) or t<0:raise ValueError('Threshold not frozen/invalid')
    expected={'feature_order_proposal':'ask_price,ask_size,bid_price,bid_size for each level 1..10',
              'label_formula_proposal':'future endpoint mid-price relative to current mid-price; no smoothing',
              'trigger_proposal':'every target visible-book change, including unchanged top ten',
              'segment_policy_proposal':'full depth, regular Q-to-M system phase, stock trading; never cross invalid flags or state transitions',
              'normalization_proposal':'per-feature mean/std fitted on training records only; zero variance std=1'}
    if any(d.get(k)!=v for k,v in expected.items()) or d.get('class_mapping_proposal')!={'down':0,'stationary':1,'up':2}:
        raise ValueError('Protocol differs from supported implementation; review adapter first')
    if d['feature_count']!=40 or d['window_length']!=100:raise ValueError('40 x 100 required')
    dates=d['dates'];roles=d['date_roles_proposal'];order={'train':0,'tuning_validation':1,'comparison_validation':2,'final_test':3}
    if dates!=sorted(set(dates)) or set(roles)!=set(dates):raise ValueError('Date partitions missing/duplicated')
    if set(roles.values())!=set(order) or [order[roles[x]] for x in dates]!=sorted(order[roles[x]] for x in dates):
        raise ValueError('Four chronological disjoint roles required')
    if not set(dates)<=set(d.get('approved_acquisition_dates',[])):raise ValueError('Date scope not approved')


def prepare():
    d=draft();validate_protocol(d);h=d['label_horizon'];t=d['label_threshold']
    runs={date:get_run(date,True) for date in d['dates']}
    def chunks():
        for date,(_,_,a,segments) in runs.items():
            if d['date_roles_proposal'][date]!='train':continue
            for s in segments:
                for begin in range(s['start_row'],s['end_row']+1,100000):
                    yield a['features'][begin:min(begin+100000,s['end_row']+1)]
    norm=fit_stats(chunks());ranges=[];sources=[];counts=np.zeros(3,dtype=np.int64)
    for date,(folder,report,a,segments) in runs.items():
        role=d['date_roles_proposal'][date]
        hashes=json.loads((folder/'output_hashes.json').read_text(encoding='utf-8'))
        sources.append({'date':date,'split':role,'folder':str(folder),'raw_sha256':report['source_sha256'],
                        'records_sha256':hashes['records.bin'],'report_sha256':sha256(folder/'report.json')})
        for s in segments:
            ends=endpoints(s['start_row'],s['end_row']+1,h)
            if len(ends):ranges.append({'date':date,'split':role,'segment_id':s['segment_id'],
                'records_path':str(folder/'records.bin'),'segment_start':s['start_row'],'segment_stop':s['end_row']+1,
                'end_start':ends.start,'end_stop':ends.stop})
        if role=='train':counts+=count_labels(a,segments,h,t)
    totals={role:sum(r['end_stop']-r['end_start'] for r in ranges if r['split']==role) for role in set(d['date_roles_proposal'].values())}
    if any(n==0 for n in totals.values()) or np.any(counts==0):raise ValueError('Empty split or missing training class; review before training')
    majority=int(counts.argmax())
    # Only tuning validation is scored here; comparison and final labels remain unopened.
    val=np.zeros(3,dtype=np.int64)
    for date,(_,_,a,segments) in runs.items():
        if d['date_roles_proposal'][date]=='tuning_validation':val+=count_labels(a,segments,h,t)
    result={'version':'W3-samples-1.1','created_utc':stamp(),'scope':'software data preparation; hardware NOT verified',
            'preparation_code_sha256':sha256(Path(__file__)),
            'protocol':d,'horizon':h,'threshold':t,'normalization':norm,'sources':sources,'ranges':ranges,
            'sample_counts':totals,'train_class_counts':counts.tolist(),'majority_class_from_train':majority,
            'tuning_majority_accuracy':float(val[majority]/val.sum()),
            'tuning_majority_macro_f1':float((2*val[majority]/(val.sum()+val[majority]))/3),
            'comparison_and_final_labels_inspected':False}
    destination=Path(config()['data_root'])/'derived/w3'/('prepared_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(parents=True,exist_ok=False)
    save(destination/'manifest.json',result)
    save(destination/'protocol.json',d)
    save(ROOT/'artifacts/w3_2026-10-03/prepared_latest.json',{'manifest':str(destination/'manifest.json'),'sha256':sha256(destination/'manifest.json')})
    print('PREPARED:',destination/'manifest.json');print(json.dumps(totals,indent=2))


def loss_smoke():
    import torch
    pointer_path=ROOT/'artifacts/w3_2026-10-03/prepared_latest.json'
    meta=json.loads(pointer_path.read_text(encoding='utf-8'))
    if sha256(meta['manifest'])!=meta['sha256']:raise ValueError('Prepared manifest changed')
    m=json.loads(Path(meta['manifest']).read_text(encoding='utf-8'));validate_protocol(m['protocol'])
    for source in m['sources']:
        if source['split']=='train' and sha256(Path(source['folder'])/'records.bin')!=source['records_sha256']:
            raise ValueError('Training records changed')
    ds=Windows(m,'train');n=min(96,len(ds));items=[ds[i] for i in np.linspace(0,len(ds)-1,n,dtype=int)]
    x=torch.from_numpy(np.stack([z[0] for z in items]));y=torch.tensor([int(z[1]) for z in items])
    torch.manual_seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu';x=x.to(device);y=y.to(device)
    # Disposable plumbing CNN; not the W4 model selection or a saved predictor.
    net=torch.nn.Sequential(torch.nn.Conv1d(40,16,3,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool1d(1),torch.nn.Flatten(),torch.nn.Linear(16,3)).to(device)
    opt=torch.optim.Adam(net.parameters(),lr=0.003);loss_fn=torch.nn.CrossEntropyLoss()
    before=float(loss_fn(net(x),y).detach().cpu())
    for _ in range(80):
        opt.zero_grad();loss=loss_fn(net(x),y)
        if not torch.isfinite(loss):raise ValueError('Non-finite loss')
        loss.backward();opt.step()
    after=float(loss_fn(net(x),y).detach().cpu())
    report={'scope':'training-fragment plumbing test only; no generalization/accuracy or hardware claim; no weights saved',
            'device':device,'samples':n,'class_counts':torch.bincount(y,minlength=3).cpu().tolist(),
            'loss_before':before,'loss_after':after,'passed':after<before,'manifest_sha256':meta['sha256'],
            'version':'W3-plumbing-CNN-1.0','seed':42,'steps':80,'learning_rate':0.003,
            'model_architecture':str(net),'code_sha256':sha256(Path(__file__)),'checked_utc':stamp()}
    save(ROOT/'artifacts/w3_2026-10-03/loss_smoke.json',report);print(json.dumps(report,indent=2))
    if not report['passed']:raise ValueError('Loss did not decrease; diagnose, do not train W4 yet')


def check_prepared():
    """Independent boundary/numeric checks; never evaluates comparison/final labels."""
    meta=json.loads((ROOT/'artifacts/w3_2026-10-03/prepared_latest.json').read_text(encoding='utf-8'))
    path=Path(meta['manifest'])
    if sha256(path)!=meta['sha256']:raise ValueError('Prepared manifest changed')
    m=json.loads(path.read_text(encoding='utf-8'));validate_protocol(m['protocol'])
    if m['protocol']!=draft():raise ValueError('Current protocol differs from prepared version')
    mean=np.array(m['normalization']['mean']);std=np.array(m['normalization']['std'])
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std<=0):raise ValueError('Invalid normalization')
    total=np.zeros(40);centered=np.zeros(40);n=0;checked_ranges=0;train_checks=0
    ds=Windows(m,'train');train_offset=0
    try:
        for source in m['sources']:
            f=Path(source['folder']);records=f/'records.bin'
            if sha256(records)!=source['records_sha256']:raise ValueError('Source records changed')
            if sha256(f/'report.json')!=source['report_sha256']:raise ValueError('Audit report changed')
            a=np.memmap(records,dtype=DTYPE,mode='r')
            segments=json.loads((f/'segments.json').read_text(encoding='utf-8'))
            try:
                ranges=[r for r in m['ranges'] if r['date']==source['date']]
                for r in ranges:
                    s=next(s for s in segments if s['segment_id']==r['segment_id'])
                    if r['split']!=source['split'] or r['records_path']!=str(records):raise ValueError('Split/path mismatch')
                    if r['segment_start']!=s['start_row'] or r['segment_stop']!=s['end_row']+1:raise ValueError('Segment boundary mismatch')
                    if r['end_start']!=s['start_row']+99 or r['end_stop']!=s['end_row']+1-m['horizon']:raise ValueError('Window/label boundary mismatch')
                    count=r['end_stop']-r['end_start'];checked_ranges+=1
                    if source['split']=='train':
                        # Read actual first/second/last train windows only, with integer label reference.
                        for offset in sorted({0,min(1,count-1),count-1}):
                            end=r['end_start']+offset;ring=RingWindow()
                            for row in a['features'][end-99:end+1]:raw=ring.push(row)
                            ref=raw.copy();ref[:,0::2]/=10000
                            ref=((ref-mean)/std).T.astype(np.float32)
                            x,y=ds[train_offset+offset]
                            if not np.array_equal(x,ref):raise ValueError('CPU window reference mismatch')
                            current=int(a['features'][end,0])+int(a['features'][end,2])
                            future=int(a['features'][end+m['horizon'],0])+int(a['features'][end+m['horizon'],2])
                            expected=2 if future-current>current*m['threshold'] else 0 if future-current < -current*m['threshold'] else 1
                            if y!=expected:raise ValueError('Independent endpoint label mismatch')
                            if not np.all(np.isfinite(x)):raise ValueError('Non-finite tensor')
                            train_checks+=1
                        train_offset+=count
                if source['split']=='train':
                    for s in segments:
                        for begin in range(s['start_row'],s['end_row']+1,50000):
                            x=a['features'][begin:min(begin+50000,s['end_row']+1)].astype(np.float64)
                            x[:,0::2]/=10000
                            total+=x.sum(axis=0);centered+=((x-mean)**2).sum(axis=0);n+=len(x)
            finally:a._mmap.close()
        reference_mean=total/n;reference_std=np.sqrt(centered/n)
        zero=m['normalization']['zero_variance_columns'];reference_std[zero]=1
        if n!=m['normalization']['record_count']:raise ValueError('Normalization fitted record count mismatch')
        if not np.allclose(mean,reference_mean,rtol=1e-10,atol=1e-8):raise ValueError('Train-only mean mismatch')
        if not np.allclose(std,reference_std,rtol=1e-7,atol=1e-8):raise ValueError('Train-only standard deviation mismatch')
        totals={role:sum(r['end_stop']-r['end_start'] for r in m['ranges'] if r['split']==role) for role in m['sample_counts']}
        if totals!=m['sample_counts']:raise ValueError('Sample counts mismatch')
        result={'status':'passed','checked_utc':stamp(),'manifest':str(path),'manifest_sha256':meta['sha256'],
                'checked_ranges':checked_ranges,'training_window_label_checks':train_checks,'normalization_train_records':n,
                'sample_counts':totals,'comparison_and_final_labels_inspected':False,'hardware_verified':False,
                'scope':'Source hashes, all range boundaries/counts, train CPU windows/labels and independently recomputed train mean/std'}
        save(ROOT/'artifacts/w3_2026-10-03/prepared_check.json',result);print(json.dumps(result,indent=2))
    finally:ds.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['profile','prepare','check','loss-smoke']);a=p.parse_args()
    {'profile':profile,'prepare':prepare,'check':check_prepared,'loss-smoke':loss_smoke}[a.action]()

if __name__=='__main__':
    try:main()
    except Exception as e:print('ERROR:',e);sys.exit(1)
