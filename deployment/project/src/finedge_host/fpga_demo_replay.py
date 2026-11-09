"""Live finite ITCH -> PC book/normalization -> FPGA window -> board NPU -> GUI.

The PC window is an independent diagnostic reference; it is never sent to NPU
during the live phase. No Host instruction simulation runs on the hot path.
"""
from contextlib import ExitStack, closing
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import uuid

import numpy as np

from .backends import CpuBackend, prediction
from .delivery import read_json, write_json, sha256
from .paths import ARTIFACTS, PACKAGE
from .raw_itch import RawItchSource
from .replay import ReplayControl
from .stream import InputBuilder, PREPROCESS_ID
from .udw_e_frontend import CandidateFrontend
from .udw_board_bridge import digest
from .market import DelayedEvaluation, HORIZON, TARGET_ID
from .datasets import dataset_info
from .fpga_demo_session import DemoClient, golden_input, golden_payload, collect_board_logs
from .demo_rate import validate_rate, remaining_period, OutputRate


def checked_prediction(result, tensor, first=None, last=None):
    if result.get('tensor_sha256') != digest(tensor.tobytes()):
        raise ValueError('FPGA tensor checksum differs from independent CPU window')
    values = np.asarray(result.get('logits'),dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all() or result.get('class_index') != int(values.argmax()):
        raise ValueError('Invalid NPU logits/class')
    if first is not None and (result.get('first') != list(first) or result.get('last') != list(last)):
        raise ValueError('Returned FPGA window identity mismatch')


def check_cached_reference(cached, window, got):
    if (cached['input_sha256'] != digest(window.tensor.tobytes()) or
        cached['first'] != window.first_meta or cached['last'] != window.last_meta):
        raise ValueError('Live window differs from frozen compiled reference identity')
    error = float(np.max(np.abs(np.array(got['logits'])-cached['host']['logits'])))
    ok = bool(np.allclose(got['logits'],cached['host']['logits'],rtol=1e-3,atol=1e-4)
              and got['class_index']==cached['host']['class_index'])
    if not ok: raise ValueError('Actual NPU differs from cached compiled Host reference')
    return error


def run_fpga_demo(session_path, control=None, emit=None, *, source_factory=RawItchSource,
                  cpu_factory=CpuBackend, client_factory=DemoClient, collect=collect_board_logs,
                  output_directory=None):
    control = control or ReplayControl(20)
    validate_rate(control.rate)
    output_rate = OutputRate()
    emit = emit or (lambda item: None)
    run_id = 'fpga_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
    directory = Path(output_directory) if output_directory else ARTIFACTS/'runs'/run_id
    directory.mkdir(parents=True,exist_ok=False)
    stats = dict(records=0,windows=0,rounds_started=1,cpu_npu_class_mismatches=0,
                 cpu_npu_max_abs_error=0.0,npu_mismatches=0,conversion_class_mismatches=0,errors=0)
    context = dict(run_id=run_id,status='initializing',mode='fpga',stats=stats,
                   execution_kind='hardware_candidate' if client_factory is DemoClient else 'local_test_substitute',
                   initial_target_windows_per_second=control.rate,
                   actual_rate_scope='last up to 32 emission timestamps; includes pacing and pause; excludes GUI drawing',
                   data_source='selected_original_AAPL_ITCH_frames',runtime_book_reconstruction=True,
                   preprocessing=PREPROCESS_ID,host_simulation_enabled=False,live_window_limit=50,
                   no_PC_window_return=True,PC_window_role='independent diagnostic only',
                   strict_visibility_proven=False,zero_copy=False,performance_claim=False,
                   market_target_id=TARGET_ID,model_sha256=sha256(PACKAGE/'model_cpu.ts'))
    evaluation = DelayedEvaluation()
    client = cpu = source = None
    finished = False
    stack = ExitStack()
    streams = {name:stack.enter_context((directory/(name+'.jsonl')).open('x',encoding='utf-8',buffering=1))
               for name in ('events','predictions','market','outcomes','records')}
    tensors = stack.enter_context((directory/'reference_inputs.f32').open('xb'))
    def event(kind, **data):
        item = dict(type=kind,run_id=run_id,**data)
        streams['events'].write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
        emit(item)
    def outcomes(items):
        for item in items:
            streams['outcomes'].write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
            event('outcome',**item,evaluation=evaluation.summary())
    def paused():
        last_ping = time.monotonic()
        while control.pause.is_set() and not control.stop.is_set():
            if control.reset.is_set(): raise ValueError('Finite FPGA demo cannot reset or reuse DDR')
            if client is not None and not finished and time.monotonic()-last_ping>=10:
                client.call('ping')
                last_ping = time.monotonic()
            control.stop.wait(.05)
        if control.reset.is_set(): raise ValueError('Finite FPGA demo cannot reset or reuse DDR')
    def observe(record, tail=False):
        stats['records'] += 1
        streams['records'].write(json.dumps(dict(values=record.values.tolist(),meta=record.meta,tail_only=tail))+'\n')
        mid2 = int(record.values[0])+int(record.values[2])
        item = dict(type='market',run_id=run_id,round=1,last=record.meta,mid2=mid2)
        streams['market'].write(json.dumps(item)+'\n')
        emit(item)
        outcomes(evaluation.observe(record.meta,mid2,1))
        return mid2
    try:
        if not session_path: raise ValueError('请使用“启动FPGA闭环演示.cmd”准备一次新的板卡会话。')
        cfg = read_json(session_path)
        reference_rows = None
        if context['execution_kind'] == 'hardware_candidate':
            reference_path = Path(cfg['bundle_path']).parent/'reference.json'
            if sha256(reference_path) != cfg['remote_hashes']['reference.json']:
                raise ValueError('Cached compiled reference hash differs')
            reference_rows = read_json(reference_path)['windows']
            if len(reference_rows) != 50: raise ValueError('Expected fifty frozen reference windows')
            context['cached_Host_reference_sha256'] = sha256(reference_path)
        context.update(package_sha256=cfg['package_sha256'],boot_id=cfg['boot_id'],
                       session_path=str(session_path),compiled_model_hashes={k:v for k,v in cfg['remote_hashes'].items()
                                                                            if k.startswith('tcn_ed4_ZG.')})
        write_json(directory/'manifest.json',context)
        # Integrity check and raw prefix preparation happen without any hardware allocations here.
        source = source_factory()
        source.cancelled = control.stop.is_set
        source.on_progress = lambda m: emit(dict(type='source_progress',source=m,run_id=run_id))
        context['source_provenance'] = source.provenance
        normalization = read_json(PACKAGE/'normalization.json')
        front = CandidateFrontend(normalization,'H')
        reference = InputBuilder(normalization)
        cpu = cpu_factory()
        golden = golden_input()
        for _ in range(10): cpu.predict(0,golden)
        if control.stop.is_set():
            context['status']='stopped'
            return context
        client = client_factory(session_path,directory/'wire.jsonl')
        context['board_info'] = client.call('info')
        info = context['board_info']
        if info['boot_id']!=cfg['boot_id'] or info.get('package_sha256')!=cfg['package_sha256']:
            raise ValueError('Connected service identity differs')
        if info.get('control_transport')!='axi_registers' or not any(
            'ZG330' in b['backend'] and 'HardOp' in b['op'] for b in info.get('bindings',[])):
            raise ValueError('Actual FPGA/NPU path absent')
        context['warmup'] = []
        for _ in range(10):
            if control.stop.is_set(): break
            got = client.call('software',golden.tobytes())
            checked_prediction(got,golden)
            if not np.allclose(got['logits'],cfg['golden_logits'],rtol=1e-3,atol=1e-4) or got['class_index']!=int(np.argmax(cfg['golden_logits'])):
                raise ValueError('NPU compiled golden check failed')
            context['warmup'].append(got)
        if not control.stop.is_set():
            result = client.call('batch',golden_payload(golden),continuation=False)
            if len(result['predictions'])!=1: raise ValueError('Golden PL output count')
            checked_prediction(result['predictions'][0],golden)
            if not np.allclose(result['predictions'][0]['logits'],cfg['golden_logits'],rtol=1e-3,atol=1e-4):
                raise ValueError('FPGA -> NPU compiled golden check failed')
            context['golden_fpga'] = result
        context['status']='running'
        event('ready',directory=str(directory),source_count=None,backends=['CPU参考','FPGA组窗 → 板载NPU（真实运行）'],
              dataset=dataset_info('itch20190730'))
        event('round',round=1,reset_window=True)
        with closing(source.iter_records()) as records:
            batch=[]
            while stats['windows']<50 and not control.stop.is_set():
                paused()
                if control.stop.is_set(): break
                began=time.perf_counter_ns()
                try: record=next(records)
                except StopIteration: raise ValueError('Raw source ended before finite 50-window demo')
                mid2=observe(record)
                prep_start=time.perf_counter_ns()
                item=front.push(record)
                item['selected']=item['opportunity']
                # Keep the initial demo within one segment. Never silently cross or skip a gap.
                if front.continuity.epoch!=1: raise ValueError('Segment changed; finite demo stopped')
                prep_end=time.perf_counter_ns()
                batch.append(front.pack(item))
                packed=time.perf_counter_ns()
                window=reference.push(record)
                if window is None:
                    if stats['records']%10==0: event('warming_window',count=stats['records'],round=1)
                    continue
                window_id=stats['windows']+1
                payload=b''.join(batch)
                # CPU reference runs independently; only WDM records enter the live board call.
                cpu_result=cpu.predict(window_id,window.tensor)
                response=client.call('batch',payload,continuation=bool(stats['windows']))
                batch=[]
                if response['records'] != (100 if window_id==1 else 1) or len(response['predictions'])!=1:
                    raise ValueError('FPGA record / window count mismatch')
                got=response['predictions'][0]
                # Save the raw returned response before validating it (also in wire.jsonl).
                checked_prediction(got,window.tensor,item['first'],item['last'])
                host_error = host_ok = conversion_error = None
                if reference_rows is not None:
                    cached = reference_rows[window_id-1]
                    try: host_error = check_cached_reference(cached,window,got)
                    except ValueError:
                        stats['npu_mismatches']+=1
                        raise
                    host_ok = True
                    conversion_error = float(np.max(np.abs(np.array(cpu_result['logits'])-cached['host']['logits'])))
                npu=prediction(window_id,got['logits'],'fpga_board_npu',response['rtt_ns'],
                               stages_ns=got['stages_ns'],board_stages_ns=response['stages_ns'])
                outputs=dict(torch_cpu=cpu_result,icraft_socket=npu)
                delta=float(np.max(np.abs(np.array(cpu_result['logits'])-np.array(npu['logits']))))
                matched=cpu_result['class_index']==npu['class_index']
                stats['windows']+=1
                stats['cpu_npu_class_mismatches']+=int(not matched)
                stats['cpu_npu_max_abs_error']=max(stats['cpu_npu_max_abs_error'],delta)
                row=dict(run_id=run_id,window_id=window_id,round=1,first=window.first_meta,last=window.last_meta,
                         input_sha256=digest(window.tensor.tobytes()),outputs=outputs,
                         preprocess_ns=prep_end-prep_start,pack_ns=packed-prep_end,
                         validation_total_ns=time.perf_counter_ns()-began,
                         target_windows_per_second=validate_rate(control.rate),
                         observed_windows_per_second=output_rate.observe(time.perf_counter_ns()),
                         conversion_max_abs_error=conversion_error,conversion_comparison_available=reference_rows is not None,
                         cpu_npu_max_abs_error=delta,cpu_npu_class_match=matched,
                         npu_max_abs_error=host_error,npu_within_tolerance=host_ok,
                         compiled_reference_checked=reference_rows is not None,
                         origin_mid2=mid2,market_target_id=TARGET_ID,
                         target_source_row=record.meta['source_record_row_0based']+HORIZON,
                         fpga_tensor_verified=True,fpga_response=response,
                         live_hardware=context['execution_kind']=='hardware_candidate')
                tensors.write(window.tensor.tobytes()); tensors.flush()
                streams['predictions'].write(json.dumps(row,allow_nan=False)+'\n')
                evaluation.track(window_id,1,record.meta,outputs)
                event('prediction',**{k:v for k,v in row.items() if k!='run_id'},book=record.values.tolist(),
                      stats=dict(stats),evaluation=evaluation.summary())
                # Period includes inference/reference/logging, with no catch-up bursts.
                wait=remaining_period(control.rate,time.perf_counter_ns()-began)
                control.stop.wait(wait)
            context['board_final']=client.call('finish')
            finished=True
            if not context['board_final'].get('audit_passed'):
                raise ValueError('Final board retention audit failed')
            # Resolve causally: consume 50 more raw records for chart/evaluation only.
            # No additional FPGA or NPU requests are issued in this tail.
            if stats['windows']==50 and not control.stop.is_set():
                event('demo_phase',message='50个硬件窗口已完成；继续观察50条行情兑现H50，不再提交硬件。')
                for _ in range(HORIZON):
                    paused()
                    if control.stop.is_set(): break
                    try: record=next(records)
                    except StopIteration: break
                    observe(record,tail=True)
                    control.stop.wait(.04)
        context['status']='completed' if stats['windows']==50 and not control.stop.is_set() else 'stopped'
    except Exception as exc:
        context.update(status='failed',error=str(exc))
        stats['errors']+=1
        event('error',message=str(exc),directory=str(directory))
        # Never continue, retry, reconnect or fall back to CPU after an ambiguous board call.
    finally:
        if client is not None:
            try: client.disconnect()
            except Exception as exc: context['disconnect_error']=str(exc)
            try: collect(session_path,directory)
            except Exception as exc:
                context['evidence_collection_error']=str(exc)
                if context['status']=='completed': context['status']='incomplete_evidence'
        if cpu is not None: cpu.close()
        outcomes(evaluation.finish(context['status']))
        context['evaluation']=evaluation.summary()
        context['market_evaluation']=evaluation.summary()
        context['source_metrics']=getattr(source,'metrics',{})
        context['stats']=dict(stats)
        write_json(directory/'manifest.json',context)
        event('finished',directory=str(directory),summary=context)
        stack.close()
    return context
