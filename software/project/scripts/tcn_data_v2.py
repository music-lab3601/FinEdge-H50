"""Versioned data-only preparation. No training, model imports or account access."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np

from databento_mbo_v2 import Adapter, CONTROL_UNVERIFIED, VERSION
from itch_book import ROW
from w3_samples import physical, fit_stats, endpoints, labels

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/tcn_data_v2_2026-10-09'
CFG = ROOT/'configs/tcn_data_protocol_v2.json'
E_ROOT = Path('@EXTERNAL_ROOT@/tcn_data_v2')
DTYPE = np.dtype([('sequence','<u8'),('timestamp_ns','<u8'),('segment_id','<u4'),
                  ('flags','<u4'),('bid_levels','<u4'),('ask_levels','<u4'),('features','<u8',(40,))])
assert DTYPE.itemsize == ROW.size == 352


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(8*1024**2):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def config():
    c = read(CFG)
    if not c['user_approved'] or c['training_authorized_in_this_task']:
        raise ValueError('Wrong preparation authorization')
    if c['date_roles'] != roles():
        raise ValueError('Unexpected approved split')
    if c['feature_count'] != 40 or c['window_length'] != 100 or c['label_horizon'] != 50 or c['label_threshold'] != 0:
        raise ValueError('Frozen feature/label interface changed')
    if sha(ROOT/'configs/w3_protocol_frozen_v1.json') != c['parent_protocol_sha256']:
        raise ValueError('Original frozen protocol modified')
    return c


def roles():
    return {**{f'2025-12-{d:02}':'train' for d in range(8,13)},
            '2025-12-15':'tuning_validation','2025-12-16':'tuning_validation',
            '2025-12-17':'comparison_validation','2025-12-18':'final_test','2025-12-19':'final_test'}


def identity():
    paths = set()
    for folder in ('configs','scripts','tests'):
        paths.update(p for p in (ROOT/folder).glob('*') if p.is_file()
                     and 'data_v2' not in p.name and 'mbo_v2' not in p.name)
    paths.update((ROOT/'state').glob('tcn*plan*.json'))
    paths.add(ROOT/'artifacts/w3_2026-10-03/prepared_latest.json')
    for folder in ('artifacts/tcn_training_2026-10-06','artifacts/tcn02_training_2026-10-07'):
        paths.update(p for p in (ROOT/folder).glob('*.json') if p.is_file())
    return {str(p):sha(p) for p in sorted(paths)}


def register():
    if CFG.exists() or (OUT/'registration.json').exists():
        c = config()
        print(json.dumps({'status':'already_registered','version':c['version']},ensure_ascii=False))
        return
    oldpath = ROOT/'configs/w3_protocol_frozen_v1.json'
    old = read(oldpath)
    c = {
        'version':'TCN-data-protocol-2.0','registered_utc':now(),
        'status':'approved_split_pending_control_state_adapter_acceptance',
        'user_approved':True,'jointly_confirmed':False,'cross_repository_alignment_verified':False,
        'authorization_quote':'可以 那就按照你的步骤来吧 你先帮我完成新数据适配与新版切分登记',
        'authorization_scope':'New data adapter and V2 split registration only; no model training or new run reservation.',
        'training_authorized_in_this_task':False,
        'date_roles':roles(),'symbol':'AAPL','dataset':'XNAS.ITCH',
        'parent_protocol':str(oldpath),'parent_protocol_sha256':sha(oldpath),
        'old_test_roles_retired_only_in_v2':['2025-12-11','2025-12-12'],
        'old_tuning_role_retired_only_in_v2':['2025-12-10'],
        'old_source_versions':old['source_versions'],
        'feature_count':40,'window_length':100,'feature_order':old['feature_order_proposal'],
        'trigger':old['trigger_proposal'],'segment_policy':old['segment_policy_proposal'],
        'label_formula':old['label_formula_proposal'],'label_horizon':50,'label_threshold':0.0,
        'class_mapping':{'down':0,'stationary':1,'up':2},
        'normalization':'Five TRAIN dates unique valid records only; population std; zero variance std=1',
        'price_units':'raw Price(4); physical prices USD /10000; quantities shares',
        'new_mbo_gate':'control_state_unverified: fail closed; no fixed-clock substitution',
        'sealed_dates':['2025-12-17','2025-12-18','2025-12-19'],
        'sealed_access':'Engineering decode/book/metadata/hash only; no labels, normalization fitting, readable feature previews or model scores',
        'training_ready':False,'model_forward_count':0,'optimizer_updates':0,
    }
    protected = identity()
    save(CFG,c)
    save(OUT/'registration.json',{'registered_utc':now(),'config_path':str(CFG),'config_sha256':sha(CFG),
                                 'date_roles':roles(),'protected_before':protected,'training_started':False})
    save(OUT/'dependency_receipt.json',{'checked_utc':now(),'databento_dbn':version('databento-dbn'),
        'numpy':version('numpy'),'torch':version('torch'),'install_policy':'Pinned databento-dbn==0.71.0 --no-deps --only-binary; no dependency upgrades',
        'api_keys_collected':False,'network_download_requests_created':0})
    print(json.dumps({'status':'V2_roles_registered','training_ready':False},ensure_ascii=False),flush=True)


def train_sources():
    c = config()
    if (OUT/'train_preparation.json').exists():
        raise ValueError('Existing accepted train preparation; use its receipt, do not repeat')
    runs = []
    for day in (d for d,r in roles().items() if r == 'train'):
        pin = c['old_source_versions'][day]
        folder = Path(pin['engineering_folder'])
        path = folder/'records.bin'
        if sha(folder/'report.json') != pin['audit_report_sha256'] or sha(path) != pin['records_sha256']:
            raise ValueError('Old source identity failed')
        report, segs = read(folder/'report.json'), read(folder/'segments.json')
        if report['symbol'] != 'AAPL' or report['trading_date'] != day or not report['full_day']:
            raise ValueError('Old source identity context failed')
        a = np.memmap(path,dtype=DTYPE,mode='r')
        valid = 0
        for seg in segs:
            start, stop = seg['start_row'], seg['end_row']+1
            if start < 0 or stop > len(a) or stop-start != seg['rows'] or seg['segment_id'] <= 0:
                raise ValueError('Old segment geometry failed')
            for begin in range(start,stop,50000):
                chunk = a[begin:min(begin+50000,stop)]
                if np.any(chunk['flags']) or np.any(chunk['segment_id'] != seg['segment_id']):
                    raise ValueError('Invalid old row in accepted segment')
                f = chunk['features']
                if np.any(chunk['bid_levels']<10) or np.any(chunk['ask_levels']<10) or np.any(f<=0):
                    raise ValueError('Depth/positive price/size check failed')
                if np.any(f[:,0::4][:,1:]<=f[:,0::4][:,:-1]) or np.any(f[:,2::4][:,1:]>=f[:,2::4][:,:-1]) or np.any(f[:,2]>=f[:,0]):
                    raise ValueError('Sorted uncrossed book invariant failed')
                if np.any(np.diff(chunk['timestamp_ns'].astype(np.int64))<0):
                    raise ValueError('Timestamp regression in old valid segment')
            valid += stop-start
        if valid != pin['valid_records']:
            raise ValueError('Old valid count differs')
        runs.append((day,path,a,segs))
        print(json.dumps({'old_date_verified':day,'valid_records':valid},ensure_ascii=False),flush=True)
    def chunks():
        for _,_,a,segs in runs:
            for s in segs:
                for b in range(s['start_row'],s['end_row']+1,50000):
                    yield a['features'][b:min(b+50000,s['end_row']+1)]
    norm = fit_stats(chunks())
    n, sum_x, centered = 0,np.zeros(40),np.zeros(40)
    mean = np.asarray(norm['mean'])
    for f in chunks():
        x = physical(f)
        n += len(x)
        sum_x += x.sum(axis=0)
        centered += ((x-mean)**2).sum(axis=0)
    std2 = np.sqrt(centered/n)
    zero = std2==0
    std2[zero] = 1
    if n != norm['record_count'] or not np.allclose(sum_x/n,mean,rtol=1e-10,atol=1e-8) or not np.allclose(std2,norm['std'],rtol=1e-7,atol=1e-8):
        raise ValueError('Independent normalization pass differs')
    ranges, sources, class_counts = [],{},np.zeros(3,dtype=np.int64)
    for day,path,a,segs in runs:
        sources[day] = {'records_path':str(path),'records_sha256':sha(path),'rows':len(a),
                        'segments_path':str(path.parent/'segments.json'),'segments_sha256':sha(path.parent/'segments.json')}
        if sources[day]['records_sha256'] != c['old_source_versions'][day]['records_sha256']:
            raise ValueError('Old records changed during preparation')
        for s in segs:
            ends = endpoints(s['start_row'],s['end_row']+1,50)
            ranges.append({'date':day,'split':'train','records_path':str(path),'segment_id':s['segment_id'],
                           'segment_start':s['start_row'],'segment_stop':s['end_row']+1,'end_start':ends.start,'end_stop':ends.stop})
            for begin in range(ends.start,ends.stop,50000):
                ids = np.arange(begin,min(begin+50000,ends.stop))
                y = labels(a,ids,50,0)
                # Independent integer H50 classification of this exact data slice.
                current = a['features'][ids,0].astype(np.int64)+a['features'][ids,2].astype(np.int64)
                future = a['features'][ids+50,0].astype(np.int64)+a['features'][ids+50,2].astype(np.int64)
                ref = np.where(future>current,2,np.where(future<current,0,1))
                if not np.array_equal(y,ref):
                    raise ValueError('Training labels disagree with independent integer reference')
                class_counts += np.bincount(y,minlength=3)
        a._mmap.close()
    receipt = {'prepared_utc':now(),'status':'five_day_train_prepared','date_roles':roles(),
        'normalization':norm,'sources':sources,'ranges':ranges,'train_windows':sum(r['end_stop']-r['end_start'] for r in ranges),
        'train_class_counts_down_flat_up':class_counts.tolist(),'fit_dates':[d for d,r in roles().items() if r=='train'],
        'train_labels_independently_verified':True,'new_dates_labels_inspected':False,
        'training_ready':False,'model_forward_count':0,'optimizer_updates':0}
    save(OUT/'normalization_review.json',{'independent_record_count':n,'sum_features':sum_x.tolist(),
        'sum_squared_centered_at_fit_mean':centered.tolist(),'root_actual_second_pass':True,
        'zero_variance_columns':np.flatnonzero(zero).tolist(),
        'max_mean_abs_difference':float(np.max(np.abs(sum_x/n-mean))),
        'max_std_abs_difference':float(np.max(np.abs(std2-norm['std'])))})
    save(OUT/'train_preparation.json',receipt)
    print(json.dumps({'status':receipt['status'],'train_windows':receipt['train_windows'],'unique_train_records':n}),flush=True)


def adapt(day):
    import databento_dbn as dbn
    c = config()
    if day not in c['date_roles'] or c['date_roles'][day] == 'train':
        raise ValueError('Only downloaded new dates may be adapted')
    # The normalized download receipt is a stable local source, never a signed URL.
    receipt = read(ROOT/'artifacts/tcn_databento_download_2026-10-09/download_receipt.json')
    filename = f'xnas-itch-{day.replace("-","")}.mbo.dbn.zst'
    native_root = Path('@EXTERNAL_ROOT@/XNAS-20261009-VB8DJSWBRE')
    source = native_root/filename
    entry = next((x for x in receipt['files'] if x['path']==filename),None)
    if entry is None:
        raise ValueError('Source absent from original delivery manifest')
    expected_sha = entry['sha256']
    expected_size = entry['bytes']
    if expected_sha is None or expected_size is None or sha(source)!=expected_sha or source.stat().st_size!=expected_size:
        raise ValueError('New delivered source size/hash failed')
    if (OUT/f'adapter_{day}.json').exists():
        raise ValueError('Existing accepted engineering receipt: use it, never repeat')
    folder = E_ROOT/day/('engineering_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    decoder = dbn.DBNDecoder(compression=dbn.Compression.ZSTD)
    adapter, metadata = None,None
    types = Counter()
    with source.open('rb') as f, (folder/'records.bin').open('xb',buffering=4*1024**2) as target:
        def handle(decoded):
            nonlocal adapter, metadata
            for r in decoded:
                types[type(r).__name__] += 1
                if isinstance(r,dbn.Metadata):
                    if metadata is not None or r.dataset!='XNAS.ITCH' or str(r.schema)!='mbo' or r.symbols != ['AAPL']:
                        raise ValueError('DBN metadata context mismatch')
                    metadata = {'version':r.version,'dataset':r.dataset,'schema':str(r.schema),
                        'start':r.start,'end':r.end,'symbols':r.symbols,
                        'mappings':{symbol:[{**interval,'start_date':interval['start_date'].isoformat(),
                            'end_date':interval['end_date'].isoformat()} for interval in intervals]
                            for symbol,intervals in r.mappings.items()}}
                    ids = []
                    for interval in r.mappings.get('AAPL',[]):
                        if interval['start_date'] <= __import__('datetime').date.fromisoformat(day) < interval['end_date']:
                            ids.append(int(interval['symbol']))
                    adapter = Adapter(day,ids,target.write)
                elif isinstance(r,dbn.MBOMsg):
                    if adapter is None:raise ValueError('MBO before metadata')
                    adapter.consume(r)
                else:
                    raise ValueError('Unexpected non-MBO record: '+type(r).__name__)
        while block := f.read(1024**2):
            handle(decoder.write_and_decode(block))
            save(OUT/f'progress_{day}.json',{'date':day,'compressed_bytes_read':f.tell(),
                'source_bytes':source.stat().st_size,'decoded_records':types['MBOMsg'],
                'rows':adapter.rows if adapter else 0,'elapsed_seconds':time.monotonic()-start,'training_ready':False})
        handle(decoder.decode())
        if decoder.buffer():raise ValueError('Undecoded trailing DBN data at EOF')
        report = adapter.finish() if adapter else None
    if report is None:raise ValueError('No data decoded')
    if sha(source)!=expected_sha:raise ValueError('Delivered source changed during adaptation')
    records_path = folder/'records.bin'
    if records_path.stat().st_size != report['rows']*352:raise ValueError('Output row byte size differs')
    # Independent data-only scan of every generated row; no label access.
    a = np.memmap(records_path,dtype=DTYPE,mode='r')
    counts = Counter()
    for b in range(0,len(a),50000):
        chunk = a[b:b+50000]
        if np.any(chunk['segment_id']) or np.any((chunk['flags']&CONTROL_UNVERIFIED)==0):
            raise ValueError('Unverified control row admitted')
        values,frequencies = np.unique(chunk['flags'],return_counts=True)
        counts.update({str(int(v)):int(n) for v,n in zip(values,frequencies)})
    a._mmap.close()
    if dict(counts)!=report['row_flag_counts']:raise ValueError('Independent row flag recount differs')
    report.update({'date':day,'role':c['date_roles'][day],'status':'book_engineering_passed_control_gate_blocked',
        'source_path':str(source),'source_sha256':expected_sha,'source_bytes':source.stat().st_size,
        'records_path':str(records_path),'records_sha256':sha(records_path),'records_bytes':records_path.stat().st_size,
        'decoded_types':dict(types),'metadata':metadata,'independent_row_scan_passed':True,
        'elapsed_seconds':time.monotonic()-start,'finished_utc':now()})
    save(folder/'segments.json',[])
    save(folder/'report.json',report)
    save(OUT/f'adapter_{day}.json',report)
    print(json.dumps({'date':day,'rows':report['rows'],'status':report['status'],'seconds':report['elapsed_seconds']}),flush=True)


def finalize():
    c = config()
    train = read(OUT/'train_preparation.json')
    row_review = read(OUT/'engineering_row_review.json')
    if row_review['status']!='passed':raise ValueError('Full emitted-row review missing')
    adapters = {d:read(OUT/f'adapter_{d}.json') for d,r in roles().items() if r!='train'}
    protected_before = read(OUT/'registration.json')['protected_before']
    protected_after = {p:sha(p) for p in protected_before}
    if protected_before != protected_after:raise ValueError('Protected old files modified')
    manifest = {'version':'TCN-data-manifest-2.0','protocol_path':str(CFG),'protocol_sha256':sha(CFG),
        'date_roles':roles(),'horizon':50,'threshold':0.0,'window_length':100,'feature_count':40,
        'normalization':train['normalization'],'ranges':train['ranges'],'sources':train['sources'],
        'sample_counts':{'train':train['train_windows'],'tuning_validation':0,'comparison_validation':0,'final_test':0},
        'diagnostic_new_sources':{d:{k:v for k,v in r.items() if k!='metadata'} for d,r in adapters.items()},
        'training_ready':False,'blocker':'MBO lacks verified original Q/M/H/h control state; all new rows excluded',
        'sealed_dates':c['sealed_dates'],'new_dates_labels_inspected':False,
        'optimizer_updates':0,'model_forward_count':0,'new_training_reservations':0,'jointly_confirmed':False}
    dest = E_ROOT/'prepared_v2'
    dest.mkdir(parents=True,exist_ok=False)
    save(dest/'manifest.json',manifest)
    save(dest/'normalization.json',train['normalization'])
    save(OUT/'manifest.json',manifest)
    save(OUT/'prepared_latest.json',{'manifest_path':str(dest/'manifest.json'),'manifest_sha256':sha(dest/'manifest.json'),
        'training_ready':False,'version':manifest['version']})
    catalog_path = Path('@EXTERNAL_ROOT@/data_preparation_v2_20261009.json')
    if catalog_path.exists():raise ValueError('Existing V2 catalog: inspect, never overwrite')
    prepared_catalog = {'version':'FinEdge-data-preparation-catalog-2.0','date_roles':roles(),
        'manifest_path':str(dest/'manifest.json'),'manifest_sha256':sha(dest/'manifest.json'),
        'training_ready':False,'train_sources':train['sources'],
        'new_diagnostic_sources':manifest['diagnostic_new_sources'],'blocker':manifest['blocker']}
    save(catalog_path,prepared_catalog)
    save(OUT/'catalog_receipt.json',{'path':str(catalog_path),'sha256':sha(catalog_path),
        'bytes':catalog_path.stat().st_size,'old_catalog_overwritten':False})
    result = {'completed_utc':now(),'status':'split_and_engineering_prepared_blocked_on_control_state',
        'split_registered':True,'five_day_train_prepared':True,'five_new_days_book_reconstructed':True,
        'new_rows':sum(x['rows'] for x in adapters.values()),'new_valid_rows':0,
        'train_windows':train['train_windows'],'unique_train_records':train['normalization']['record_count'],
        'training_ready':False,'blocker':manifest['blocker'],
        'old_files_sha_unchanged':True,'protected_before':protected_before,'protected_after':protected_after,
        'optimizer_updates':0,'model_forward_count':0,'new_training_reservations':0,'new_dates_labels_inspected':False,
        'outputs':{str(p):sha(p) for p in sorted(OUT.glob('*.json')) if p.name!='completion_acceptance.json'}}
    save(OUT/'completion_acceptance.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('outputs','protected_before','protected_after')},ensure_ascii=False),flush=True)


def check_rows():
    """Independent emitted-row geometry checks, no source replay/labels/model."""
    config()
    reviews = {}
    for day,role in roles().items():
        if role=='train':continue
        r = read(OUT/f'adapter_{day}.json')
        path = Path(r['records_path'])
        if sha(path)!=r['records_sha256']:raise ValueError('New records identity changed')
        a = np.memmap(path,dtype=DTYPE,mode='r')
        count, flags_count, previous_sequence = 0,Counter(),None
        for begin in range(0,len(a),50000):
            chunk = a[begin:begin+50000]
            f, flags = chunk['features'],chunk['flags']
            ask, bid = f[:,0::4],f[:,2::4]
            aq,bq = f[:,1::4],f[:,3::4]
            pos = np.arange(10)[None,:]
            ask_present = pos < np.minimum(chunk['ask_levels'],10)[:,None]
            bid_present = pos < np.minimum(chunk['bid_levels'],10)[:,None]
            if np.any((ask>0)!=ask_present) or np.any((bid>0)!=bid_present) or np.any((aq>0)!=ask_present) or np.any((bq>0)!=bid_present):
                raise ValueError('Depth and zero-padding inconsistency')
            if np.any((ask[:,1:]<=ask[:,:-1]) & ask_present[:,1:]) or np.any((bid[:,1:]>=bid[:,:-1]) & bid_present[:,1:]):
                raise ValueError('Price levels not strictly ordered')
            both = ask_present[:,0]&bid_present[:,0]
            depth_bad = (chunk['ask_levels']<10)|(chunk['bid_levels']<10)
            if np.any(((flags&2)!=0)!=depth_bad) or np.any(((flags&4)!=0)!=(both&(ask[:,0]==bid[:,0]))) or np.any(((flags&8)!=0)!=(both&(bid[:,0]>ask[:,0]))):
                raise ValueError('Book diagnostic flags differ from emitted geometry')
            if np.any(chunk['segment_id']) or np.any((flags&81)!=81) or np.any(chunk['timestamp_ns']>=86400*10**9):
                raise ValueError('Control exclusion/day-boundary inconsistency')
            seq = chunk['sequence'].astype(np.int64)
            if np.any(np.diff(seq)<=0) or previous_sequence is not None and int(seq[0])<=previous_sequence:
                raise ValueError('Repeated/non-increasing native sampled sequence')
            previous_sequence = int(seq[-1])
            vals,ns=np.unique(flags,return_counts=True)
            flags_count.update({str(int(v)):int(n) for v,n in zip(vals,ns)})
            count += len(chunk)
        a._mmap.close()
        if count!=r['rows'] or dict(flags_count)!=r['row_flag_counts']:
            raise ValueError('Full emitted-row recount differs')
        reviews[day]={'rows':count,'record_sha256':r['records_sha256'],'all_rows_scanned':True,
                      'geometry_depth_flags_sequence_control_passed':True,'labels_inspected':False}
    save(OUT/'engineering_row_review.json',{'checked_utc':now(),'status':'passed','dates':reviews,
        'scope':'Every emitted row scanned; no independent raw event replay; no labels/model',
        'total_rows':sum(r['rows'] for r in reviews.values())})
    print(json.dumps({'emitted_row_review':'passed','rows':sum(r['rows'] for r in reviews.values())}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['register','prepare-train','adapt','check-rows','finalize'])
    parser.add_argument('--date')
    args = parser.parse_args()
    if args.action=='register':register()
    elif args.action=='prepare-train':train_sources()
    elif args.action=='adapt':adapt(args.date)
    elif args.action=='check-rows':check_rows()
    else:finalize()


if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'last_error.json',{'at_utc':now(),'error_type':type(e).__name__,'message':str(e),
                                 'training_ready':False,'optimizer_updates':0})
        raise
