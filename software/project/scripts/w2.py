"""W2 offline audit. Commands: preflight, smoke, full, check, status.
Each run gets a new directory. W2 outputs are diagnostic, never training-ready.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import datetime as dt
import gzip
import json
from pathlib import Path
import shutil
import struct
import sys
import time

import numpy as np
from itch_book import VERSION, FLAGS, ROW, MUTATIONS, decode, iter_frames, Book, snapshot, order_digest
from w1 import ROOT, save, sha256, stamp
from w2_reference import replay

REPORTS=ROOT/'artifacts/w2_2026-10-03'
DTYPE=np.dtype([('sequence','<u8'),('timestamp_ns','<u8'),('segment_id','<u4'),('flags','<u4'),
                ('bid_levels','<u4'),('ask_levels','<u4'),('features','<u8',(40,))])
assert DTYPE.itemsize==ROW.size

def settings():
    return (json.loads((ROOT/'configs/w1_data.json').read_text(encoding='utf-8')),
            json.loads((ROOT/'configs/w2_audit.json').read_text(encoding='utf-8')))

def preflight():
    data,cfg=settings();path=Path(data['data_root'])/'raw'/data['sample']['filename']
    if not path.exists() or path.stat().st_size!=data['sample']['expected_bytes']:raise ValueError('W1 complete raw file missing/size differs')
    result={'checked_utc':stamp(),'source':str(path),'source_bytes':path.stat().st_size,'python':sys.executable,
            'numpy':np.__version__,'data_drive_free_gib':round(shutil.disk_usage(path.parent).free/1024**3,2),
            'sample_symbol':cfg['symbol'],'parser_version':VERSION,'scope':'W2 audit only','training_ready':False}
    if result['data_drive_free_gib']<5:raise ValueError('Need at least 5 GiB spare space before audit')
    save(REPORTS/'preflight.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
    return path,data,cfg

def execute(mode,output_root=None):
    source,data,cfg=preflight()
    print('Checking raw source SHA256 before parsing...',flush=True)
    digest=sha256(source)
    if digest!=data['sample']['reference_sha256']:raise ValueError('Source differs from W1 reference')
    signature=(source.stat().st_size,source.stat().st_mtime_ns)
    destination=Path(output_root) if output_root else Path(data['data_root'])/'derived/w2'
    folder=destination/(mode+'_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000))
    folder.mkdir(parents=True,exist_ok=False)
    save(folder/'config.json',cfg)
    save(folder/'schema.json',{'version':VERSION,'row_bytes':ROW.size,'dtype':DTYPE.descr,'feature_order':cfg['feature_order_candidate'],
        'price_divisor':10000,'size_unit':'shares','timestamp_unit':'ns since exchange midnight','exchange_timezone':cfg['exchange_timezone'],
        'flags':FLAGS,'invalid_rows':'Preserved; missing depth is zero-filled ONLY in diagnostic rows with nonzero flags. Never train these rows.',
        'valid_segment':'flags=0 and same positive segment_id; any bad book-state row or relevant state transition breaks continuity',
        'sequence':'one-based position in source file, NOT exchange transport sequence','training_protocol_frozen':False})
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'scripts/itch_book.py',ROOT/'scripts/w2.py',ROOT/'scripts/w2_reference.py',ROOT/'configs/w2_audit.json']}
    report={'run_started_utc':stamp(),'mode':mode,'parser_version':VERSION,'source':str(source),'source_sha256':digest,
        'source_sha_provenance':data['sample']['reference_sha256_source'],'code_hashes':hashes,'folder':str(folder),
        'symbol':cfg['symbol'],'trading_date':cfg['trading_date'],'training_ready':False,'full_day':False,'status':'running'}
    save(folder/'report.json',report)
    counts=[0]*256;target_counts=Counter();reason_counts=Counter();regular_reasons=Counter();examples={}
    directories={};directory_duplicates=[];target=None;book=Book();phase=None;trade_state=None;halt=False
    system_events=[];stock_states=[];segments=[];active=None;seg_id=0;rows=0;regular_rows=0;valid=0;longest=0
    seq=0;last_target_time=None;regressions=0;target_total=0;first_time=None;last_time=None
    first_seen=set();last_checkpoint=None;last_traced=0;peak_orders=0;max_depth={'B':0,'S':0};changed_types=Counter()
    limit=cfg['smoke_message_limit'] if mode=='smoke' else None
    started=time.monotonic();last_progress=started;failure=None
    def close_segment():
        nonlocal active,longest
        if active:
            longest=max(longest,active['rows']);segments.append(active);active=None
    try:
        with ExitStack() as stack:
            raw=stack.enter_context(gzip.open(source,'rb'))
            binary=stack.enter_context((folder/'records.bin').open('wb'))
            trace=stack.enter_context((folder/'selected_messages.bin').open('wb'))
            checkpoints=stack.enter_context((folder/'checkpoints.jsonl').open('w',encoding='utf-8'))
            preview=stack.enter_context((folder/'preview_1000.csv').open('w',encoding='utf-8-sig',newline=''))
            writer=csv.writer(preview);writer.writerow(['sequence','timestamp_ns','segment_id']+[f'{field}_{level}' for level in range(1,11) for field in ['ask_price_int','ask_size','bid_price_int','bid_size']])
            def checkpoint(sequence):
                nonlocal last_checkpoint
                if sequence==last_checkpoint:return
                book.validate()
                item={'sequence':sequence,'active_orders':len(book.orders),'top':book.top(),'order_sha256':order_digest(book.orders)}
                checkpoints.write(json.dumps(item,separators=(',',':'))+'\n');last_checkpoint=sequence
            for seq,offset,payload in iter_frames(raw):
                typ=chr(payload[0]);counts[payload[0]]+=1
                if seq==1:first_time=int.from_bytes(payload[5:11],'big')
                locate=(payload[1]<<8)|payload[2]
                event=None
                if typ=='R':
                    event=decode(payload);stock=event['stock']
                    if locate in directories:
                        previous=directories[locate]
                        directory_duplicates.append({'sequence':seq,'locate':locate,'previous':previous,'current':event})
                        if previous['stock']!=stock:raise ValueError('Directory locate remapped to a different stock')
                    directories[locate]=event
                    if stock==cfg['symbol']:
                        if target is not None and target!=locate:raise ValueError('Target has multiple locate codes')
                        target=locate
                selected=(target is not None and locate==target)
                global_control=typ in ('S','W','V')
                if selected or global_control:
                    event=event or decode(payload)
                    trace.write(struct.pack('<QH',seq,len(payload))+payload)
                    last_traced=seq
                    if len(examples.get(typ,[]))<3:
                        examples.setdefault(typ,[]).append({'sequence':seq,'uncompressed_offset':offset,'raw_hex':payload.hex(),'decoded':event})
                    if selected:
                        target_total+=1;target_counts[typ]+=1
                        if 'stock' in event and event['stock']!=cfg['symbol']:raise ValueError('Target symbol does not match locate')
                        backwards=last_target_time is not None and event['timestamp_ns']<last_target_time
                        if backwards:regressions+=1;close_segment()
                        last_target_time=event['timestamp_ns']
                    else:backwards=False
                    if typ=='S':
                        if seq!=1 and not system_events:raise ValueError('Missing initial system event')
                        system_events.append({'sequence':seq,**event});phase=event['event'];close_segment()
                    elif typ=='W':
                        raise ValueError('Market-wide circuit breaker event requires case-specific review')
                    elif selected and typ=='H':
                        if event['trading_state'] not in ('H','P','Q','T'):raise ValueError('Unknown trading state')
                        if trade_state!=event['trading_state']:close_segment()
                        trade_state=event['trading_state'];stock_states.append({'sequence':seq,**event})
                    elif selected and typ=='h' and event['market_code']=='Q':
                        if event['halt_action'] not in ('H','T'):raise ValueError('Unknown operational halt action')
                        halt=event['halt_action']=='H';close_segment();stock_states.append({'sequence':seq,**event})
                    if selected and book.apply(event):
                        changed_types[typ]+=1;rows+=1
                        peak_orders=max(peak_orders,len(book.orders))
                        for side in 'BS':max_depth[side]=max(max_depth[side],len(book.prices[side]))
                        flags,features=snapshot(book,phase=='Q',trade_state=='T' and not halt,backwards)
                        if phase=='Q':regular_rows+=1
                        for bit,reason in FLAGS.items():
                            if flags&bit:
                                reason_counts[reason]+=1
                                if phase=='Q':regular_reasons[reason]+=1
                        if flags:
                            close_segment();segment=0
                        else:
                            valid+=1
                            if active is None:
                                seg_id+=1;active={'segment_id':seg_id,'start_row':rows-1,'start_sequence':seq,'start_timestamp_ns':event['timestamp_ns'],'rows':0}
                            active.update(end_row=rows-1,end_sequence=seq,end_timestamp_ns=event['timestamp_ns']);active['rows']+=1;segment=seg_id
                            if valid<=1000:writer.writerow([seq,event['timestamp_ns'],segment]+features)
                        binary.write(ROW.pack(seq,event['timestamp_ns'],segment,flags,len(book.prices['B']),len(book.prices['S']),*features))
                        if rows%10000==0 or typ not in first_seen:
                            checkpoint(seq);first_seen.add(typ)
                if seq%1_000_000==0 and time.monotonic()-last_progress>=10:
                    print(f'PARSE {seq:,} messages; target={target_total:,}; snapshots={rows:,}; valid={valid:,}; elapsed={time.monotonic()-started:.1f}s',flush=True)
                    last_progress=time.monotonic()
                if limit and seq>=limit:
                    last_time=int.from_bytes(payload[5:11],'big');break
            else:
                report['full_day']=True
                last_time=int.from_bytes(payload[5:11],'big') if seq else None
            close_segment()
            if last_traced:checkpoint(last_traced)
        if (source.stat().st_size,source.stat().st_mtime_ns)!=signature:raise ValueError('Source changed during audit')
        if target is None:raise ValueError('Target absent from directory')
        if not system_events or system_events[0]['event']!='O' or system_events[0]['sequence']!=1:
            raise ValueError('No valid beginning-of-day state')
        if report['full_day']:
            codes=[x['event'] for x in system_events]
            if codes!=list('OSQMEC') or system_events[-1]['sequence']!=seq:raise ValueError(f'Unexpected system lifecycle {codes}')
            if seq!=cfg['expected_sample_messages'] or target_total!=cfg['expected_sample_target_messages']:
                raise ValueError('Full-day counts differ from previous independent audit; review before acceptance')
        report['reference']=replay(folder)
        if report['reference']['mutations_replayed']!=rows:raise ValueError('Reference mutation count differs')
        if regressions:raise ValueError('Target timestamp regressions require review')
        report['status']='passed_full_day_engineering_audit' if report['full_day'] else 'passed_smoke_only'
        if report['full_day'] and longest<1000:report['status']='review_required_insufficient_contiguous_depth'
    except Exception as exc:
        failure=str(exc);report['status']='failed';report['error']=failure
    report.update(finished_utc=stamp(),seconds=round(time.monotonic()-started,2),messages_read=seq,
        message_counts={chr(k):n for k,n in enumerate(counts) if n},target_message_counts=dict(target_counts),target_messages=target_total,
        target_locate=target,stock_directory_entries=len(directories),directory_duplicates=directory_duplicates,
        system_events=system_events,stock_states=stock_states,first_timestamp_ns=first_time,last_timestamp_ns=last_time,
        book_mutations=rows,mutation_types=dict(changed_types),peak_active_orders=peak_orders,max_depth=max_depth,
        regular_session_candidate_rows=regular_rows,diagnostic_valid_rows=valid,regular_candidate_valid_ratio=valid/regular_rows if regular_rows else None,
        invalid_flag_counts=dict(reason_counts),regular_invalid_flag_counts=dict(regular_reasons),timestamp_regressions=regressions,
        valid_segments=len(segments),longest_valid_segment=longest,book_fatal_anomalies=1 if failure else 0,
        unexpected_semantics_are_fatal=True,formal_training_protocol_frozen=False)
    save(folder/'report.json',report);save(folder/'raw_examples.json',examples);save(folder/'segments.json',segments)
    if not failure:
        files=['records.bin','selected_messages.bin','checkpoints.jsonl','preview_1000.csv','schema.json','segments.json','raw_examples.json','config.json']
        if longest>=1000:
            chosen=next(s for s in segments if s['rows']>=1000)
            records=np.memmap(folder/'records.bin',dtype=DTYPE,mode='r')
            with (folder/'contiguous_1000.csv').open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(['sequence','timestamp_ns','segment_id']+[f'{field}_{level}' for level in range(1,11) for field in ['ask_price_int','ask_size','bid_price_int','bid_size']])
                for row in records[chosen['start_row']:chosen['start_row']+1000]:
                    writer.writerow([int(row['sequence']),int(row['timestamp_ns']),int(row['segment_id'])]+row['features'].tolist())
            del records
            files.append('contiguous_1000.csv')
        save(folder/'output_hashes.json',{name:sha256(folder/name) for name in files})
    save(REPORTS/('latest_'+mode+'.json'),{'folder':str(folder),'status':report['status'],'finished_utc':stamp()})
    print(json.dumps({k:report.get(k) for k in ['folder','status','messages_read','target_messages','book_mutations','diagnostic_valid_rows','longest_valid_segment','seconds','error']},ensure_ascii=False,indent=2))
    if failure:raise RuntimeError(f'{failure}; partial outputs preserved at {folder}')
    return folder

def check(folder):
    folder=Path(folder);report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    if report['status'] not in ('passed_smoke_only','passed_full_day_engineering_audit'):raise ValueError('Run not passed; inspect report.json')
    for name,digest in json.loads((folder/'output_hashes.json').read_text(encoding='utf-8')).items():
        if sha256(folder/name)!=digest:raise ValueError(f'Output hash mismatch: {name}')
    path=folder/'records.bin'
    if path.stat().st_size!=report['book_mutations']*ROW.size:raise ValueError('Record count/size mismatch')
    a=np.memmap(path,dtype=DTYPE,mode='r') if path.stat().st_size else np.empty(0,dtype=DTYPE)
    if len(a)>1 and np.any(a['sequence'][1:]<=a['sequence'][:-1]):raise ValueError('Source order not preserved')
    valid_count=0
    for start in range(0,len(a),100000):
        part=a[start:start+100000];v=part[part['flags']==0];valid_count+=len(v)
        f=v['features'].reshape(-1,10,4)
        if len(v) and (np.any(f==0) or np.any(f[:,1:,0]<=f[:,:-1,0]) or np.any(f[:,1:,2]>=f[:,:-1,2]) or np.any(f[:,0,2]>=f[:,0,0]) or np.any(v['segment_id']==0)):
            raise ValueError('Valid row has bad depth/price/segment')
    segments=json.loads((folder/'segments.json').read_text(encoding='utf-8'))
    for segment in segments:
        p=a[segment['start_row']:segment['end_row']+1]
        if len(p)!=segment['rows'] or np.any(p['flags']!=0) or np.any(p['segment_id']!=segment['segment_id']):
            raise ValueError('Segment crosses invalid records')
    if valid_count!=report['diagnostic_valid_rows'] or sum(s['rows'] for s in segments)!=valid_count:raise ValueError('Valid count mismatch')
    result={'checked_utc':stamp(),'status':'passed','scope':report['status'],'rows':len(a),'valid_rows':valid_count,
        'segments':len(segments),'training_ready':False,'hardware_verified':False,'folder':str(folder)}
    save(folder/'output_check.json',result);print(json.dumps(result,ensure_ascii=False,indent=2));return result

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['preflight','smoke','full','check','status'])
    p.add_argument('--output-root');p.add_argument('--run');args=p.parse_args()
    REPORTS.mkdir(parents=True,exist_ok=True)
    if args.action=='preflight':preflight()
    elif args.action in ('smoke','full'):
        folder=execute(args.action,args.output_root);check(folder)
    elif args.action=='check':
        folder=args.run or json.loads((REPORTS/'latest_full.json').read_text(encoding='utf-8'))['folder'];check(folder)
    else:
        for mode in ('smoke','full'):
            path=REPORTS/f'latest_{mode}.json'
            print(path.read_text(encoding='utf-8') if path.exists() else f'{mode}: not run')

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('Interrupted. Audit is NOT accepted. Keep this run for diagnosis; rerun creates a new folder.',file=sys.stderr);sys.exit(130)
    except Exception as exc:
        save(REPORTS/'last_error.json',{'at_utc':stamp(),'error':str(exc)})
        print(f'ERROR: {exc}',file=sys.stderr);sys.exit(1)
