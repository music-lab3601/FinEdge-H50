"""Audit hardware bytes and report same-PC end-to-end latency, never clock mixing."""
from pathlib import Path
import argparse, csv, json, struct, sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];P=ROOT.parents[1];G=P/'versions/realtime_integration_v1';V=G/'queued_v2'
from shared.io import read, atomic_json, sha

TRACE=np.dtype([(n,'<u8') for n in ['window','ordinal','slot','generation','request_ns','npu_ns','board_ticks','service_ticks']]+[('logits','<f4',(3,)),('class_index','<u4'),('count_before','<u4'),('count_after','<u4')])
AUDIT=np.dtype([('window','<u8'),('ordinal','<u8'),('generation','<u8'),('slot','<u8'),('input','u1',(4800,)),('logits','<f4',(3,)),('class_index','<u4')])
ROW=np.dtype([('segment','<u8'),('row','<u8'),('ordinal','<u8'),('event_ns','<u8'),('values','<u8',(40,))])

def stats(ns):
    a=np.asarray(ns,dtype=np.float64)
    if not len(a):return dict(count=0)
    return dict(count=len(a),mean_us=float(np.mean(a)/1000),p50_us=float(np.percentile(a,50)/1000),p95_us=float(np.percentile(a,95)/1000),p99_us=float(np.percentile(a,99)/1000),max_us=float(np.max(a)/1000))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--oracle',type=Path);a=ap.parse_args();out=a.directory
    m=read(out/'run_manifest.json');b=read(out/'board_result.json');s=read(out/'sender/result.json')
    traces=np.fromfile(out/'traces.bin',dtype=TRACE);blob=(out/'sender/predictions.bin').read_bytes()
    if len(blob)%200:raise ValueError('Partial prediction packet')
    packets=[]
    for offset in range(0,len(blob),200):
        stamp=struct.unpack_from('<Q',blob,offset)[0];h=struct.unpack_from('<IIQQQ3fIQQII',blob,offset+8)
        if h[:2]!=(0x32524d46,2 if 'slots' in m else 1) or h[2]!=m['session'] or h[-1]!=125000000:raise ValueError('Packet ABI/session/clock')
        packets.append((stamp,h,blob[offset+8+32:offset+8+44]))
    terminal={int(t['window']):t for t in traces}
    measured=[p for p in packets if m['fixture']['first_raw_ordinal']<=p[1][4]<=m['fixture']['last_raw_ordinal']]
    packet_windows=[p[1][3] for p in measured]
    if len(set(packet_windows))!=len(packet_windows) or set(packet_windows)!=set(terminal):raise ValueError('Missing/duplicate network results or trace identity mismatch')
    for stamp,h,raw_logits in measured:
        t=terminal[h[3]]
        if int(t['ordinal'])!=h[4] or int(t['class_index'])!=h[8] or t['logits'].tobytes()!=raw_logits or int(t['board_ticks'])!=h[9] or int(t['service_ticks'])!=h[10]:raise ValueError('GP/network result differs')
    ok=traces[(traces['class_index']>>24)==0]
    if np.any(((ok['count_after']-ok['count_before'])&0xffffffff)!=30):raise ValueError('Actual NPU layer completion differs')
    pubs=np.fromfile(out/'sender/source/publication.bin',dtype='<u8').reshape(-1,8) if (out/'sender/source/publication.bin').exists() else np.zeros((0,8),dtype='<u8')
    due={};submitted={}
    for p in pubs:
        for ordinal in range(int(p[1]),int(p[1]+p[2])):due[ordinal]=int(p[3]);submitted[ordinal]=int(p[5])
    event_latency=[];submitted_latency=[];selected_lateness=[]
    for stamp,h,_ in measured:
        if h[8]>>24==0 and h[4] in due:
            event_latency.append(stamp-due[h[4]]);submitted_latency.append(stamp-submitted[h[4]])
            selected_lateness.append(submitted[h[4]]-due[h[4]])
    if any(x<0 for x in event_latency+submitted_latency):raise ValueError('PC timebase/order mismatch: negative end-to-end interval')
    metrics=b['metrics'];bm=b['book_metrics']
    session_opportunities=int(metrics['2']);total_rows=int(metrics['1']);measured_rows=int(bm['0x60'])
    if session_opportunities!=max(0,total_rows-99):raise ValueError('Frozen single-contiguous-segment opportunity profile differs')
    prefix_rows=total_rows-measured_rows
    if prefix_rows<0:raise ValueError('Measured rows exceed session rows')
    opportunities=session_opportunities-max(0,prefix_rows-99)
    full_measurement=opportunities==session_opportunities
    within_5ms=sum(x<=5000000 for x in event_latency)
    reasons={str(code):int(np.count_nonzero((traces['class_index']>>24)==code)) for code in range(5)}
    rejected=sum(int(metrics[str(i)]) for i in (6,7,8))
    if not m['functional'] and len(event_latency)!=len(ok):raise ValueError('A completed prediction lacks a source publication timestamp')
    if sum(reasons.values())!=len(traces):raise ValueError('Unknown terminal reason')
    if int(metrics['3'])!=int(metrics['5']) or (full_measurement and len(traces)!=int(metrics['5'])):raise ValueError('Admitted requests were not all retired and recorded')
    if session_opportunities!=int(metrics['5'])+rejected:raise ValueError('Opportunity accounting does not balance')
    if len(traces)>opportunities:raise ValueError('Recorded terminals exceed measured opportunities')
    if np.any(ok['board_ticks']<ok['service_ticks']):raise ValueError('Board launch interval exceeds full board interval')
    result=dict(status='passed',raw_events=int(m['fixture']['measured_events']),raw_committed=int(bm['0x58']),measured_rows=int(bm['0x60']),predictions=len(ok),terminals=len(traces),cancellations=len(traces)-len(ok),network_results=len(measured),all_GP_network_result_bytes_equal=True,
        session_records=int(metrics['1']),session_opportunities=int(metrics['2']),session_admitted=int(metrics['3']),session_retired=int(metrics['5']),session_copy_busy_rejected=int(metrics['6']),session_full_rejected=int(metrics['7']),session_epoch_rejected=int(metrics['8']),signal_gate_skipped=int(metrics['9']),
        queue_slots=m.get('slots',2),max_queue_age_us=m.get('max_age_us'),H50_record_limit=m.get('h50_records'),
        five_ms_completed=within_5ms,five_ms_opportunities=opportunities,five_ms_completion_ratio=within_5ms/opportunities if opportunities else None,
        completed_prediction_coverage=len(ok)/opportunities if opportunities else None,terminal_reasons=reasons,
        measured_opportunities=opportunities,bootstrap_rows=prefix_rows,measurement_covers_full_session=full_measurement,
        measured_unaccepted_opportunities=opportunities-len(traces),
        opportunity_accounting=dict(session_total=session_opportunities,retired=int(metrics['5']),rejected=rejected,balanced=True,terminal_record_scope='complete session' if full_measurement else 'measured subset; bootstrap included in session counters',measured_total=opportunities,recorded_predicted=len(ok),recorded_expired=reasons['1'],recorded_other_cancelled=sum(reasons[str(i)] for i in (2,3,4))),
        scratch_released=int(metrics.get('26',0)),DDR_queue_peak=int(metrics.get('27',0)),expired_total=int(metrics.get('29',0)),expired_H50=int(metrics.get('30',0)),expired_age=int(metrics.get('31',0)),
        computed_H50_late=int(metrics.get('32',0)),computed_age_late=int(metrics.get('33',0)),late_started=int(metrics.get('35',0)),
        shared_instruction_stream=b.get('shared_instruction_stream',False),
        measured_interval_span_seconds=m['fixture'].get('event_span_ns',0)/1e9/m['speed'],
        event_due_to_prediction_PC=stats(event_latency),API_submission_to_prediction_PC=stats(submitted_latency),
        raw_accept_to_result_board=stats(ok['board_ticks'].astype('f8')*8),launch_to_result_board=stats(ok['service_ticks'].astype('f8')*8),NPU_launch_wait=stats(ok['npu_ns']),
        raw_accept_to_launch_board=stats((ok['board_ticks']-ok['service_ticks']).astype('f8')*8),
        source_lateness=stats([int(p[5])-int(p[3]) for p in pubs]),
        source_lateness_selected_predictions=stats(selected_lateness),predictions_over_5ms=sum(x>5000000 for x in event_latency),
        board_peak_queue_events=int(bm['0x130']),board_peak_backlog_events=int(bm['0x138']),board_max_input_age_us=int(bm['0x140'])*.008,
        input_bytes_per_prediction=4800,ARM_runtime_input_copy=False,ARM_runtime_output_copy=False,session_input_write_bytes=int(metrics['10']),session_output_read_bytes=int(metrics['11']),session_output_marker_bytes=int(metrics['24']),output_visibility_retries=int(metrics['25']),
        source_clock='same PC QPC scheduled due and UDP receive',board_clock_Hz=125000000,audit_inputs_enabled=bool(m['audit_inputs']),
        firmware_BOOT_sha256=m['loaded_firmware']['boot_sha256'],trace_sha256=sha(out/'traces.bin'),prediction_packets_sha256=sha(out/'sender/predictions.bin'))
    if result['raw_events']!=result['raw_committed'] or b['status']!='finished_parked' or s['status']!='all_events_committed':result['status']='failed'
    if len(pubs)>1:
        result['actual_source_events_s']=(sum(int(p[2]) for p in pubs)-int(pubs[-1][2]))*1e9/(int(pubs[-1][5])-int(pubs[0][5]))
        last_result=max((stamp for stamp,h,_ in measured if h[8]>>24==0),default=int(pubs[-1][6]))
        result['measured_prediction_rate_s']=len(ok)*1e9/(max(int(pubs[-1][6]),last_result)-int(pubs[0][3]))
    if a.oracle:
        result['input_audit']=dict(status='not_in_submission',count=0)
        if result['input_audit']['status']=='failed':result['status']='failed'
        if len(pubs) and not m['rate']:
            rows=np.fromfile(a.oracle,dtype=ROW);indices={int(v):i for i,v in enumerate(rows['ordinal'])}
            raw=Path(read(m['fixture_path'])['raw_source'])
            with raw.open('rb') as f:f.seek((int(pubs[0,1])-1)*56);first_event=struct.unpack_from('<Q',f.read(56),8)[0]
            resolved=missed=0;details=[];deadlines={}
            for stamp,h,_ in measured:
                if h[8]>>24:continue
                i=indices[h[4]]
                if i+50<len(rows) and rows[i]['segment']==rows[i+50]['segment'] and int(rows[i+50]['row'])-int(rows[i]['row'])==50:
                    deadline=int(pubs[0,3])+(int(rows[i+50]['event_ns'])-first_event)//m['speed']
                    resolved+=1;missed+=stamp>deadline;deadlines[h[4]]=deadline
                    details.append(dict(window=h[3],ordinal=h[4],receive_ns=stamp,deadline_ns=deadline,H50_timely=stamp<=deadline,within_5ms=stamp-due.get(h[4],stamp)<=5000000,event_latency_us=(stamp-due.get(h[4],stamp))/1000))
            result.update(H50_resolved=resolved,H50_deadline_misses=missed,H50_timely=resolved-missed,H50_timely_per_prediction=(resolved-missed)/resolved if resolved else None,H50_timely_per_opportunity=(resolved-missed)/opportunities if opportunities else None,H50_scope='Prediction PC receive before mapped event time of same-segment row+50; no accuracy claim')
            if details:
                with (out/'window_deadlines.csv').open('w',encoding='utf-8',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    if 'slots' in m and result['scratch_released']*4800!=int(metrics['10']):raise ValueError('Verified DDR payload bytes and scratch release differ')
    if 'slots' in m:result['cancelled_without_DDR']=int(metrics['19'])-result['scratch_released']
    if 'slots' in m and full_measurement and result['expired_total']!=reasons['1']:raise ValueError('Expired reason accounting mismatch')
    if 'slots' in m:
        if int(metrics['28']) or result['DDR_queue_peak']>m['slots']:raise ValueError('Queue did not drain or exceeded configured capacity')
        if full_measurement and (np.count_nonzero(ok['class_index']&256)!=result['computed_H50_late'] or np.count_nonzero(ok['class_index']&512)!=result['computed_age_late']):raise ValueError('Completion expiry flags/counters differ')
        expired=traces[(traces['class_index']>>24)==1]
        if full_measurement and (np.count_nonzero(expired['class_index']&256)!=result['expired_H50'] or np.count_nonzero(expired['class_index']&512)!=result['expired_age']):raise ValueError('Cancellation expiry flags/counters differ')
        result.update(measured_expired_total=len(expired),measured_expired_H50=int(np.count_nonzero(expired['class_index']&256)),measured_expired_age=int(np.count_nonzero(expired['class_index']&512)),window_capture_or_FIFO_credit_rejected_session=int(metrics['7']),DDR_capacity_reached=result['DDR_queue_peak']==m['slots'])
    captured=int(metrics['26'])
    result.update(transport_version='window_stream_v1',full_snapshot_RAM_copies=0,
        FIFO_peak_beats=int(metrics['36']),FIFO_capacity_beats=512,
        last_capture_cycles_125MHz=int(metrics['37']),
        last_transfer_cycles_200MHz=int(metrics['38']),
        total_capture_cycles_125MHz=int(metrics['39']),
        mean_capture_us=int(metrics['39'])*.008/captured if captured else None,
        last_capture_us=int(metrics['37'])*.008,
        last_transfer_us=int(metrics['38'])*.005,
        DDR_input_visibility_retries=int(metrics['40']),
        capture_release_scope='Window fully captured in FIFO; DDR READY only after complete 75-beat readback',
        GP_stream_statistics_only_sampled_after_native_drain=True)
    result['performance_metrics_applicable']=not m['functional']
    if m['functional']:
        for key in ('five_ms_completed','five_ms_opportunities','five_ms_completion_ratio','completed_prediction_coverage'):
            result[key]=None
        result['input_audit_scope']='First startup windows during bootstrap; ordinals and exact reference hashes recorded separately'
    atomic_json(out/'analysis.json',result);print(json.dumps(result,ensure_ascii=True));return int(result['status']!='passed')

if __name__=='__main__':raise SystemExit(main())
