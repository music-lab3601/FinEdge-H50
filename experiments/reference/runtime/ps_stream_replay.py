"""Lossless raw ITCH -> fixed PSW batch -> board NPU replay.

The shadow CPU ring consumes already normalized WDM words. It is a reference,
never a second preprocessing path or an input sent to the board. One batch is
in flight, and market observations/H50 tracking remain ordered per record.
"""
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
import hashlib
import json
import math
import os
import struct
import time
import uuid

import numpy as np

from software.runtime.backends import CpuBackend, prediction
from software.runtime.delivery import read_json, sha256, write_json
from experiments.reference.runtime.demo_rate import validate_rate
from experiments.reference.runtime.fpga_streaming import BatchAccounting, WdmBatcher, LatestDisplay
from experiments.reference.runtime.market import DelayedEvaluation, HORIZON, TARGET_ID
from shared.model_paths import ARTIFACTS, PACKAGE, MANIFEST_SHA
from experiments.reference.runtime.raw_itch import RawItchSource
from software.runtime.stream import PREPROCESS_ID

EXPECTED_DAY = dict(frames=1316324, records=1126705, windows=1126606)


def host_rss_kib():
    """Read current working set; unavailable remains null, never fabricated zero."""
    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
                    (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
                    'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
            value=Counters();value.cb=ctypes.sizeof(value)
            get_process=ctypes.windll.kernel32.GetCurrentProcess
            get_process.restype=wintypes.HANDLE
            get_memory=ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
            if not get_memory(get_process(),ctypes.byref(value),value.cb):return None
            return int(value.WorkingSetSize//1024)
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])
    except (OSError,ValueError,AttributeError):pass
    return None


class OnlineTiming:
    """Constant-size logarithmic histogram; exact count/sum/min/max.

    Quantiles are upper bin bounds, with 16 bins per power of two (<=4.5%
    bin width). They are labelled approximate, never exact sample quantiles.
    """
    def __init__(self):
        self.bins = [0] * 1025
        self.count = self.total = self.maximum = 0
        self.minimum = None

    def add(self, ns):
        if type(ns) is not int or not 0 <= ns <= (1 << 64):
            raise ValueError('Invalid timing sample')
        index = 0 if ns == 0 else min(1024, math.ceil(math.log2(ns)*16))
        self.bins[index] += 1
        self.count += 1
        self.total += ns
        self.maximum = max(self.maximum, ns)
        self.minimum = ns if self.minimum is None else min(self.minimum, ns)

    def summary(self):
        result = dict(count=self.count,total_ns=self.total,min_ns=self.minimum,max_ns=self.maximum,
                      mean_ns=self.total/self.count if self.count else None,
                      quantile_method='log2 histogram upper bound, 16 bins/octave')
        for name,q in (('p50_upper_ns',.5),('p95_upper_ns',.95),('p99_upper_ns',.99)):
            target = math.ceil(self.count*q)
            cumulative = 0
            value = None
            for index,count in enumerate(self.bins):
                cumulative += count
                if self.count and cumulative >= target:
                    value = 0 if index == 0 else math.ceil(2**(index/16))
                    break
            result[name] = value
        return result


class StreamJournal:
    """Bounded buffered JSONL writers; no in-memory history."""
    def __init__(self, directory):
        self.handles = {name:(directory/(name+'.jsonl')).open('x',encoding='utf-8',buffering=65536)
                        for name in ('events','predictions','market','outcomes','batches')}
        self.counts = {name:0 for name in self.handles}

    def write(self, kind, value):
        self.handles[kind].write(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
        self.counts[kind] += 1

    def flush(self):
        for stream in self.handles.values(): stream.flush()

    def close(self):
        for stream in self.handles.values(): stream.close()


class WdmShadow:
    """Independent 100-record ring built from WDM float32 bytes, no normalize."""
    def __init__(self):
        self.rows = np.empty((100,40),dtype='<f4')
        self.identities = deque(maxlen=100)
        self.metadata = deque(maxlen=100)
        self.pointer = 0

    def push(self, packet, record):
        if len(packet) != 256: raise ValueError('Incomplete shadow WDM record')
        identity = struct.unpack_from('<4Q',packet)
        emit = struct.unpack_from('<I',packet,32)[0]
        if emit not in (0,1) or any(packet[36:64]) or any(packet[224:256]):
            raise ValueError('Invalid shadow WDM fields')
        if self.identities and identity[0] != self.identities[-1][0]:
            self.identities.clear(); self.metadata.clear(); self.pointer=0
        if self.identities:
            old = self.identities[-1]
            if identity[1] != old[1]+1 or identity[2] <= old[2] or identity[3] < old[3]:
                raise ValueError('Shadow WDM continuity failure')
        fields = ('source_record_row_0based','source_message_position_1based','timestamp_ns_since_exchange_midnight')
        if identity[1:] != tuple(record.meta[k] for k in fields):
            raise ValueError('WDM identity differs from source record')
        values = np.frombuffer(packet,dtype='<f4',count=40,offset=64)
        if not np.isfinite(values).all(): raise ValueError('Nonfinite WDM shadow')
        self.rows[self.pointer] = values
        self.pointer = (self.pointer+1)%100
        self.identities.append(identity); self.metadata.append(dict(record.meta))
        if bool(emit) != (len(self.identities)==100):
            raise ValueError('Full-day WDM dropped or premature window')
        if not emit: return None
        tensor = np.empty((1,40,100),dtype='<f4')
        tail = 100-self.pointer
        tensor[0,:,:tail] = self.rows[self.pointer:].T
        if self.pointer: tensor[0,:,tail:] = self.rows[:self.pointer].T
        return dict(tensor=tensor,start=list(self.identities[0]),end=list(identity),
                    first=dict(self.metadata[0]),last=dict(record.meta),
                    tensor_sha256=hashlib.sha256(tensor.tobytes()).hexdigest())


def split_batch(batch, window_limit):
    """Split only an already normalized bounded batch, preserving every row."""
    if batch.window_count <= window_limit: return batch,None
    if window_limit <= 0: raise ValueError('Positive remaining checkpoint budget required')
    seen=0
    for index in range(len(batch.records)):
        seen += struct.unpack_from('<I',batch.payload,index*256+32)[0]
        if seen == window_limit:
            cut=index+1; break
    left=replace(batch,payload=batch.payload[:cut*256],records=batch.records[:cut],
                 endpoints=batch.endpoints[:window_limit])
    right=replace(batch,continuation=True,payload=batch.payload[cut*256:],records=batch.records[cut:],
                  endpoints=batch.endpoints[window_limit:],first_window_id=batch.first_window_id+window_limit)
    return left,right


def run_ps_stream(session_path=None, *, on_event=None, display=None, paused=None, cancelled=None,
                  get_rate=None, max_windows=None, source=None, source_factory=RawItchSource,
                  client=None, client_factory=None, cpu=None, cpu_factory=CpuBackend,
                  journal=None, display_factory=None, output_directory=None, normalization=None,
                  expected_counts=None, cpu_warmups=10, sample_window_ids=(1,100,10000,1126606),
                  checkpoint_callback=None, prefix_references=None, clock=time.perf_counter_ns):
    """Run once; max_windows is a pause checkpoint, not a truncation/full-day claim.

    Clearing ``paused`` after ``awaiting_continuation`` resumes the SAME source,
    connection, Device owner and ring. Cancellation finishes a partial stream.
    Injected source/client/cpu/journal/display permit offline tests without SDK.
    No retry, reconnect, implicit loop, reset or CPU fallback exists here.
    """
    if max_windows is not None and (type(max_windows) is not int or max_windows <= 0):
        raise ValueError('Checkpoint must be a positive window count')
    if type(cpu_warmups) is not int or not 0 <= cpu_warmups <= 100:
        raise ValueError('Bounded CPU warmup count required')
    paused=paused or Event(); cancelled=cancelled or Event()
    get_rate=get_rate or (lambda:0)
    validate_rate(get_rate())
    on_event=on_event or (lambda event:None)
    run_id='ps_stream_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    directory=Path(output_directory) if output_directory else ARTIFACTS/'runs'/run_id
    directory.mkdir(parents=True,exist_ok=False)
    journal=journal or StreamJournal(directory)
    if display_factory is None:
        from experiments.reference.runtime.ps_stream_display import PsStreamDisplay
        display_factory=PsStreamDisplay
    view=display_factory(mailbox=display or LatestDisplay())
    expected=dict(EXPECTED_DAY if expected_counts is None else expected_counts)
    stats=dict(records=0,windows=0,batches=0,errors=0,rounds_started=1,cpu_npu_class_mismatches=0,
               cpu_npu_max_abs_error=0.,npu_mismatches=0,conversion_class_mismatches=0,
               conversion_max_abs_error=0.,npu_host_checked_windows=0,npu_max_abs_error=0.,
               max_pending_h50=0,max_batch_records=0,max_shadow_windows=0)
    context=dict(run_id=run_id,status='initializing',mode='ps_stream',directory=str(directory),
                 session_path=str(session_path) if session_path else None,stats=stats,
                 host_simulation_enabled=False,cpu_reference='every window from normalized WDM ring',
                 preprocessing=PREPROCESS_ID,model_manifest_sha256=MANIFEST_SHA,
                 strict_visibility_proven=False,zero_copy=False,performance_claim=False,
                 full_day_complete=False,raw_day_processing_complete=False,expected_counts=expected,
                 max_windows_checkpoint=max_windows,cpu_warmups=0,market_target_id=TARGET_ID,
                 input_source='selected original AAPL ITCH; runtime book reconstruction',
                 tensor_capture='fixed sample IDs and at most one failed batch',
                 npu_host_comparison='frozen first-100 reference only; no Host simulation on hot path',
                 conversion_interpretation='CPU versus compiled NPU differences are reported, not treated as board equivalence failures')
    evaluator=DelayedEvaluation(); accounting=BatchAccounting(); shadow=WdmShadow()
    timing={key:OnlineTiming() for key in ('source_and_normalize_batch','shadow_and_cpu_batch','board_batch_rtt',
                                          'batch_total','torch_cpu','board_predict','board_forward')}
    iterator=None; raw_records=None; pending=None; batch=None; response=None; shadows=[]
    start=clock(); checkpoint_done=False; cpu_warmed=False; finished=False; connected=False
    last_output_clock=None; observed_rate=None; last_memory={}; prefix={}
    def phase(kind, **data):
        item=dict(type=kind,run_id=run_id,**data)
        journal.write('events',item); journal.flush()
        view.consume(item); on_event(item)
    def emit_outcomes(items):
        for outcome in items:
            item=dict(type='outcome',run_id=run_id,**outcome,evaluation=evaluator.summary())
            journal.write('outcomes',item); view.consume(item)
    def pause_boundary():
        announced=False
        while paused.is_set() and not cancelled.is_set():
            if not announced:
                phase('phase',phase='paused',message='批次边界暂停，保持原连接和窗口状态。',stats=dict(stats))
                announced=True
            view.publish(); cancelled.wait(.05)
        if announced and not cancelled.is_set(): phase('phase',phase='running',stats=dict(stats))
    def save_tensor(window_id,tensor,label):
        path=directory/('tensor_%s_%09d.f32'%(label,window_id))
        if not path.exists():
            with path.open('xb') as stream: stream.write(tensor.tobytes())
    try:
        context['model_sha256']=sha256(PACKAGE/'model_cpu.ts')
        context['normalization_sha256']=sha256(PACKAGE/'normalization.json')
        source=source if source is not None else source_factory()
        source.cancelled=cancelled.is_set
        context['source_provenance']=dict(source.provenance)
        context['runtime_book_reconstruction']=isinstance(source,RawItchSource)
        normalization=normalization if normalization is not None else read_json(PACKAGE/'normalization.json')
        batcher=WdmBatcher(normalization,32)
        cpu=cpu if cpu is not None else cpu_factory()
        if cancelled.is_set(): context['status']='partial'; return context
        if client is None:
            if client_factory is None:
                from experiments.reference.runtime.ps_stream_session import client_factory
            client=client_factory(session_path,directory/'wire.jsonl')
        if prefix_references is None and getattr(client,'hardware',False):
            if not session_path: raise ValueError('Hardware prefix reference requires frozen session path')
            cfg=read_json(session_path)
            reference_path=Path(cfg['bundle_path']).parent/'host_prefix_reference.json'
            if sha256(reference_path)!=cfg['host_prefix_reference_sha256']:
                raise ValueError('Frozen Host prefix reference hash mismatch')
            prefix_references=read_json(reference_path)
            if prefix_references.get('model_manifest_sha256')!=MANIFEST_SHA:
                raise ValueError('Frozen Host prefix model manifest mismatch')
            context['host_prefix_reference_sha256']=cfg['host_prefix_reference_sha256']
        if prefix_references is not None:
            rows=prefix_references['comparisons'] if isinstance(prefix_references,dict) else prefix_references
            if not isinstance(rows,list) or not 1<=len(rows)<=100:
                raise ValueError('Expected at most 100 frozen Host prefix windows')
            if getattr(client,'hardware',False) and len(rows)!=100:
                raise ValueError('Hardware stream requires 100 frozen Host prefix windows')
            for number,row in enumerate(rows,1):
                values=np.asarray(row['host']['logits'],dtype=np.float64)
                if (row['window_id']!=number or values.shape!=(3,) or not np.isfinite(values).all()
                        or row['host']['class_index']!=int(values.argmax())):
                    raise ValueError('Invalid frozen Host prefix reference')
                prefix[number]=row
            context['host_prefix_reference_windows']=len(prefix)
        context['board_info']=client.hello(); connected=True
        context['execution_kind']='hardware_candidate' if getattr(client,'hardware',False) else 'local_test_substitute'
        context['status']='running'
        write_json(directory/'manifest.json',context)
        phase('ready',directory=str(directory),backends=['CPU参考','PSW FPGA组窗 → 板载NPU'],
              execution_kind=context['execution_kind'],
              expected_windows=expected['windows'],stats=dict(stats))
        raw_records=source.iter_records()
        iterator=iter(batcher.batches(raw_records))
        while not cancelled.is_set():
            pause_boundary()
            if cancelled.is_set(): break
            if max_windows is not None and not checkpoint_done and stats['windows']==max_windows:
                checkpoint_done=True; paused.set()
                phase('phase',phase='awaiting_continuation',message='检查点完成；继续将沿用本次连接处理余下交易日。',
                      windows=stats['windows'],stats=dict(stats),evaluation=evaluator.summary())
                if checkpoint_callback is not None: checkpoint_callback(dict(stats),paused,cancelled)
                pause_boundary()
                if cancelled.is_set(): break
            batch_start=clock(); response=None; shadows=[]
            if pending is not None: batch,pending=pending,None
            else:
                began=clock()
                try: batch=next(iterator)
                except StopIteration: break
                timing['source_and_normalize_batch'].add(clock()-began)
            if cancelled.is_set(): break
            if max_windows is not None and not checkpoint_done:
                batch,pending=split_batch(batch,max_windows-stats['windows'])
            batch=replace(batch,sequence=accounting.next_sequence)
            accounting.submit(batch)
            began=clock(); window_id=batch.first_window_id
            for index,record in enumerate(batch.records):
                item=shadow.push(batch.payload[index*256:(index+1)*256],record)
                if item is None: continue
                if not cpu_warmed:
                    for _ in range(cpu_warmups): cpu.predict(0,item['tensor'])
                    context['cpu_warmups']=cpu_warmups; cpu_warmed=True
                item['window_id']=window_id; item['record_index']=index
                if window_id in prefix and prefix[window_id]['tensor_sha256']!=item['tensor_sha256']:
                    save_tensor(window_id,item['tensor'],'failure')
                    raise ValueError('Frozen Host prefix tensor source/hash mismatch')
                item['cpu']=cpu.predict(window_id,item['tensor'])
                cpu_values=np.asarray(item['cpu'].get('logits'),dtype=np.float64)
                if (item['cpu'].get('window_id')!=window_id or item['cpu'].get('backend')!='torch_cpu'
                        or cpu_values.shape!=(3,) or not np.isfinite(cpu_values).all()
                        or item['cpu'].get('class_index')!=int(cpu_values.argmax())):
                    raise ValueError('Invalid CPU reference logits/class')
                timing['torch_cpu'].add(int(item['cpu']['elapsed_ns']))
                shadows.append(item); window_id+=1
            timing['shadow_and_cpu_batch'].add(clock()-began)
            expected_windows=[{k:item[k] for k in ('window_id','start','end','tensor_sha256')} for item in shadows]
            if tuple((tuple(item['start']),tuple(item['end'])) for item in shadows)!=batch.endpoints:
                raise ValueError('Shadow windows differ from WDM batch identities')
            began=clock()
            response=client.submit(batch.payload,expected_windows=expected_windows)
            board_rtt=clock()-began; timing['board_batch_rtt'].add(board_rtt)
            actual=response['windows']
            acknowledged=[dict(window_id=p['window_id'],first=p['start'],last=p['end']) for p in actual]
            # Validate the complete response before publishing any result from this batch.
            if len(actual)!=len(shadows): raise ValueError('Missing or extra returned windows')
            for got,item in zip(actual,shadows):
                if any(got.get(k)!=item[k] for k in ('window_id','start','end','tensor_sha256')):
                    raise ValueError('Board window identity/tensor hash mismatch')
                values=np.asarray(got.get('logits'),dtype=np.float64)
                if values.shape!=(3,) or not np.isfinite(values).all() or got.get('class_index')!=int(values.argmax()):
                    raise ValueError('Invalid board logits/class')
                if got.get('backend')!='board_socket_npu': raise ValueError('Expected actual board NPU backend')
                item['host_max_abs_error']=None
                if item['window_id'] in prefix:
                    reference=prefix[item['window_id']]['host']
                    host_values=np.asarray(reference['logits'],dtype=np.float64)
                    error=float(np.max(np.abs(values-host_values)))
                    if not np.allclose(values,host_values,rtol=1e-3,atol=1e-4) or got['class_index']!=reference['class_index']:
                        stats['npu_mismatches']+=1
                        raise ValueError('Board NPU differs from frozen Host prefix reference')
                    item['host_max_abs_error']=error
            accounting.acknowledge(response['seq'],acknowledged)
            completed_at=clock()
            if last_output_clock is not None and batch.window_count:
                observed_rate=batch.window_count*1e9/max(1,completed_at-last_output_clock)
            if batch.window_count: last_output_clock=completed_at
            returned=iter(zip(actual,shadows)); pair=next(returned,None)
            for index,record in enumerate(batch.records):
                stats['records']+=1
                mid2=int(record.values[0])+int(record.values[2])
                market=dict(type='market',run_id=run_id,round=1,last=record.meta,mid2=mid2)
                journal.write('market',market); view.consume(market)
                emit_outcomes(evaluator.observe(record.meta,mid2,1))
                if pair is None or pair[1]['record_index']!=index: continue
                got,item=pair
                npu=prediction(got['window_id'],got['logits'],'board_socket_npu',got['whole_predict_ns'],
                               stages_ns=got.get('stages_ns',{}),batch_rtt_ns=board_rtt)
                outputs={'torch_cpu':item['cpu'],'board_socket_npu':npu}
                delta=float(np.max(np.abs(np.asarray(item['cpu']['logits'])-np.asarray(got['logits']))))
                same=item['cpu']['class_index']==got['class_index']
                host_checked=item['window_id'] in prefix
                host_error=item['host_max_abs_error']
                conversion_error=conversion_match=None
                if host_checked:
                    host=prefix[item['window_id']]['host']
                    conversion_error=float(np.max(np.abs(np.asarray(item['cpu']['logits'])-np.asarray(host['logits']))))
                    conversion_match=item['cpu']['class_index']==host['class_index']
                    stats['npu_host_checked_windows']+=1
                    stats['npu_max_abs_error']=max(stats['npu_max_abs_error'],host_error)
                    stats['conversion_max_abs_error']=max(stats['conversion_max_abs_error'],conversion_error)
                    stats['conversion_class_mismatches']+=int(not conversion_match)
                stats['windows']+=1
                stats['cpu_npu_class_mismatches']+=int(not same)
                stats['cpu_npu_max_abs_error']=max(stats['cpu_npu_max_abs_error'],delta)
                evaluator.track(got['window_id'],1,record.meta,outputs)
                stats['max_pending_h50']=max(stats['max_pending_h50'],len(evaluator.pending))
                if len(evaluator.pending)>HORIZON: raise ValueError('H50 pending bound exceeded')
                timing['board_predict'].add(got['whole_predict_ns'])
                if 'forward_including_hardop_wait' in got.get('stages_ns',{}):
                    timing['board_forward'].add(got['stages_ns']['forward_including_hardop_wait'])
                row=dict(type='prediction',run_id=run_id,window_id=got['window_id'],round=1,
                         first=item['first'],last=item['last'],first_identity=got['start'],last_identity=got['end'],
                         input_sha256=item['tensor_sha256'],outputs=outputs,origin_mid2=mid2,
                         target_source_row=record.meta['source_record_row_0based']+HORIZON,
                         batch_roundtrip_ns=board_rtt,
                         book=record.values.tolist(),fpga_tensor_verified=True,
                         cpu_npu_max_abs_error=delta,cpu_npu_class_match=same,
                         cpu_npu_comparison_includes_conversion=True,
                         conversion_max_abs_error=conversion_error,conversion_class_match=conversion_match,
                         conversion_comparison_available=host_checked,npu_max_abs_error=host_error,
                         host_reference_checked=host_checked,host_max_abs_error=host_error,
                         npu_within_tolerance=True if host_checked else None,
                         compiled_reference_checked=host_checked,stats=dict(stats),evaluation=evaluator.summary(),
                         observed_windows_per_second=observed_rate,target_windows_per_second=validate_rate(get_rate()),
                         elapsed_seconds=(clock()-start)/1e9,expected_windows=expected['windows'])
                journal.write('predictions',row); view.consume(row)
                if got['window_id'] in sample_window_ids: save_tensor(got['window_id'],item['tensor'],'sample')
                pair=next(returned,None)
            if pair is not None: raise ValueError('Unobserved prediction endpoint')
            stats['batches']+=1
            stats['max_batch_records']=max(stats['max_batch_records'],len(batch.records))
            stats['max_shadow_windows']=max(stats['max_shadow_windows'],len(shadows))
            data=response.get('data',{})
            last_memory=data.get('process_memory',data.get('memory',{}))
            journal.write('batches',dict(type='batch',sequence=batch.sequence,records=len(batch.records),
                windows=batch.window_count,first_window_id=batch.first_window_id,board_rtt_ns=board_rtt,
                board_data=data,stats=dict(stats),source_metrics=dict(source.metrics)))
            journal.flush(); view.publish()
            view.consume(dict(type='metrics',stats=dict(stats),memory=dict(host_rss_kib=host_rss_kib(),
                board_rss_kib=last_memory.get('VmRSS_kib')),observed_windows_per_second=observed_rate,
                target_windows_per_second=validate_rate(get_rate()),elapsed_seconds=(clock()-start)/1e9))
            timing['batch_total'].add(clock()-batch_start)
            # No catch-up burst: a requested rate covers compute+logs for this batch.
            while not cancelled.is_set() and not paused.is_set():
                rate=validate_rate(get_rate())
                remaining=batch.window_count/rate-(clock()-batch_start)/1e9 if rate else 0
                if remaining<=0: break
                cancelled.wait(min(.05,remaining))
            shadows=[]; response=None; batch=None
        totals=accounting.finish()
        context['accounting']=totals
        if cancelled.is_set(): context['status']='partial'
        else:
            metrics=source.metrics
            if (not batcher.exhausted or pending is not None or not metrics.get('completed_full_stream')
                    or metrics.get('frames_read')!=expected['frames']
                    or metrics.get('eligible_records')!=expected['records']
                    or stats['records']!=expected['records'] or stats['windows']!=expected['windows']):
                raise ValueError('EOF/count verification failed; full-day completion refused')
            context['status']='completed'
            context['raw_day_processing_complete']=isinstance(source,RawItchSource) and expected==EXPECTED_DAY
        context['board_final']=client.finish(); finished=True
        context['full_day_complete']=bool(context['raw_day_processing_complete'] and
                                         context['execution_kind']=='hardware_candidate' and context['status']=='completed')
        context['batcher']=batcher.summary()
    except Exception as exc:
        stats['errors']+=1
        context.update(status='failed',error=str(exc),full_day_complete=False)
        if response is not None:
            journal.write('batches',dict(type='failed_response',response=response))
        for item in shadows: save_tensor(item['window_id'],item['tensor'],'failure')
        phase('error',message=str(exc),directory=str(directory),stats=dict(stats))
    finally:
        if iterator is not None:
            try: iterator.close()
            except Exception as exc: context['source_close_error']=str(exc)
        if raw_records is not None and hasattr(raw_records,'close'):
            try: raw_records.close()
            except Exception as exc: context['raw_source_close_error']=str(exc)
        emit_outcomes(evaluator.finish('end_of_source' if context['status']=='completed' else context['status']))
        if client is not None and hasattr(client,'disconnect'):
            try: client.disconnect()
            except Exception as exc: context['disconnect_error']=str(exc)
        if cpu is not None:
            try: cpu.close()
            except Exception as exc: context['cpu_close_error']=str(exc)
        context.update(stats=dict(stats),source_metrics=dict(getattr(source,'metrics',{})),
                       market_evaluation=evaluator.summary(),evaluation=evaluator.summary(),
                       timings={k:v.summary() for k,v in timing.items()},elapsed_seconds=(clock()-start)/1e9,
                       board_finish_acknowledged=finished,connected=connected,
                       observed_windows_per_second=observed_rate,expected_windows=expected['windows'],
                       memory=dict(host_rss_kib=host_rss_kib(),board_rss_kib=last_memory.get('VmRSS_kib')),
                       last_board_memory=last_memory,pending_batch=accounting.pending is not None,
                       log_counts=dict(getattr(journal,'counts',{})))
        write_json(directory/'manifest.json',context)
        phase('finished',directory=str(directory),summary=context,status=context['status'],
              stats=dict(stats),market_evaluation=evaluator.summary())
        journal.close()
    return context
