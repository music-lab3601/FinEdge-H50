"""Product replay: raw events -> FPGA windows -> board NPU, no model reference.

Metadata history below contains identities only, never a software model tensor.
The admitted transport retains its independent opaque-byte integrity checking.
"""
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
import json
import re
import struct
import time
import uuid

import numpy as np

from .backends import prediction
from .delivery import read_json, sha256, write_json
from .demo_rate import validate_rate
from .fpga_streaming import BatchAccounting, LatestDisplay, WdmBatcher
from .market import DelayedEvaluation, HORIZON, TARGET_ID
from .paths import ARTIFACTS, MANIFEST_SHA, PACKAGE
from .ps_stream_display import PsStreamDisplay
from .ps_stream_replay import EXPECTED_DAY, OnlineTiming, StreamJournal, split_batch
from .raw_itch import RawItchSource
from .databento_mbo import DatabentoMboSource
from .stream import PREPROCESS_ID

BACKEND = 'board_socket_npu'
NPU_TIME_SCOPE = ('Board whole_predict_ns: input write/cast, forward including completion '
                  'wait, output read; excludes FPGA window building and PC network roundtrip; '
                  'not pure NPU core compute time. Startup self-check/warmups excluded.')


class WindowMetadata:
    """Bounded source identity ring; does not store or construct input tensors."""
    def __init__(self):
        self.rows = deque(maxlen=100)

    def batch(self, batch):
        emitted = []
        for index, record in enumerate(batch.records):
            packet = batch.payload[index*256:(index+1)*256]
            identity = struct.unpack_from('<4Q', packet)
            if self.rows and self.rows[-1][0][0] != identity[0]:
                self.rows.clear()
            if self.rows:
                old = self.rows[-1][0]
                if identity[1] != old[1]+1 or identity[2] <= old[2] or identity[3] < old[3]:
                    raise ValueError('Window metadata continuity failure')
            fields = ('source_record_row_0based', 'source_message_position_1based',
                      'timestamp_ns_since_exchange_midnight')
            if identity[1:] != tuple(record.meta[k] for k in fields):
                raise ValueError('Packet/source identity mismatch')
            self.rows.append((identity, dict(record.meta)))
            emit = struct.unpack_from('<I', packet, 32)[0]
            if emit != int(len(self.rows) == 100):
                raise ValueError('Premature or missing window')
            if emit:
                emitted.append(dict(window_id=batch.first_window_id+len(emitted),
                    start=list(self.rows[0][0]), end=list(identity),
                    first=dict(self.rows[0][1]), last=dict(record.meta), record_index=index))
        if tuple((tuple(p['start']), tuple(p['end'])) for p in emitted) != batch.endpoints:
            raise ValueError('Metadata differs from batch endpoints')
        return emitted


def validate_result(got, metadata, *, native=False):
    if any(got.get(k) != metadata[k] for k in ('window_id', 'start', 'end')):
        raise ValueError('Board window identity mismatch')
    values = np.asarray(got.get('logits'), dtype=np.float64)
    if (values.shape != (3,) or not np.isfinite(values).all()
            or type(got.get('class_index')) is not int or got['class_index'] != int(values.argmax())):
        raise ValueError('Invalid NPU logits/class')
    if got.get('backend') != BACKEND:
        raise ValueError('Expected actual board NPU backend; fallback forbidden')
    if native:
        if (got.get('integrity_mode')!='native_identity_and_completion_v2' or got.get('tensor_sha256') is not None
                or type(got.get('board_pipeline_ns')) is not int or got['board_pipeline_ns']<0
                or type(got.get('startup_included')) is not bool):
            raise ValueError('Native completion/timing evidence missing')
    elif not re.fullmatch('[0-9a-f]{64}', got.get('tensor_sha256', '')):
        raise ValueError('Missing FPGA tensor checksum')
    ns = got.get('whole_predict_ns')
    if type(ns) is not int or not 0 <= ns <= (1 << 64):
        raise ValueError('Missing or invalid board NPU call timing')


def run_showcase(session_path=None, *, display=None, on_event=None, paused=None, cancelled=None,
                 get_rate=None, max_windows=10000, source=None, source_factory=None,
                 client=None, client_factory=None, journal=None, display_factory=PsStreamDisplay,
                 output_directory=None, normalization=None, expected_counts=None,
                 checkpoint_callback=None, clock=time.perf_counter_ns):
    """One session, one in-flight batch, no CPU/Host inference or fallback.

    Injection is for automated local tests only; the public launcher always uses
    the frozen real hardware client. No retry/reset/reallocation lives here.
    """
    if max_windows is not None and (type(max_windows) is not int or max_windows <= 0):
        raise ValueError('Positive checkpoint required')
    paused = paused or Event(); cancelled = cancelled or Event()
    get_rate = get_rate or (lambda: 20)
    validate_rate(get_rate())
    on_event = on_event or (lambda event: None)
    run_id = 'showcase_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    directory = Path(output_directory) if output_directory else ARTIFACTS/'runs'/run_id
    directory.mkdir(parents=True, exist_ok=False)
    journal = journal or StreamJournal(directory)
    view = display_factory(mailbox=display or LatestDisplay())
    expected = dict(EXPECTED_DAY if expected_counts is None else expected_counts)
    stats = dict(records=0, windows=0, batches=0, errors=0, max_pending_h50=0,
                 npu_call_total_ns=0, npu_call_last_ns=None, max_batch_records=0,
                 board_pipeline_count=0,board_pipeline_total_ns=0,board_pipeline_last_ns=None,
                 batch_roundtrip_count=0,batch_roundtrip_total_ns=0,batch_roundtrip_last_ns=None)
    context = dict(run_id=run_id, mode='showcase_fpga_npu', status='initializing',
        directory=str(directory), product='FinEdge 智能盘口分析系统',
        session_path=str(session_path) if session_path else None,
        cpu_reference_enabled=False, cpu_model_calls=0, host_simulation_enabled=False,
        prediction_backends=[BACKEND], preprocessing=PREPROCESS_ID,
        model_manifest_sha256=MANIFEST_SHA, npu_timing_scope=NPU_TIME_SCOPE,
        window_builder='FPGA', host_byte_integrity_check=True,
        input_source='AAPL original ITCH; runtime order book reconstruction on PC',
        market_target_id=TARGET_ID, expected_counts=expected, expected_windows=expected['windows'],
        max_windows_checkpoint=max_windows, strict_visibility_proven=False, zero_copy=False,
        full_day_complete=False, raw_day_processing_complete=False, performance_claim=False)
    evaluator = DelayedEvaluation(); accounting = BatchAccounting(); metadata = WindowMetadata()
    timer = OnlineTiming(); board_timer=OnlineTiming();roundtrip_timer=OnlineTiming()
    iterator = raw_records = pending = response = None
    started = clock(); connected = finished = checkpoint_done = False
    last_output_clock = observed_rate = None
    frozen_data = None

    def phase(kind, **data):
        item = dict(type=kind, run_id=run_id, **data)
        journal.write('events', item); journal.flush(); view.consume(item); on_event(item)

    def outcomes(items):
        for result in items:
            item = dict(type='outcome', run_id=run_id, **result, evaluation=evaluator.summary())
            journal.write('outcomes', item); view.consume(item)

    def pause_boundary():
        announced = False
        while paused.is_set() and not cancelled.is_set():
            if not announced:
                phase('phase', phase='paused', message='已在批次边界暂停，继续使用原会话。', stats=dict(stats))
                announced = True
            view.publish(); cancelled.wait(.05)
        if announced and not cancelled.is_set():
            phase('phase', phase='running', stats=dict(stats))

    try:
        # Product-specific contract validation happens before any board access.
        if client is None and client_factory is None:
            contract_path=Path(session_path).parent/'showcase_contract.json'
            contract=read_json(contract_path) if contract_path.is_file() else {}
            if contract.get('schema')=='native-showcase-v2':
                from .native_showcase import native_showcase_client
                for name,digest in contract['source_hashes'].items():
                    from .paths import ROOT
                    if sha256(ROOT/name)!=digest:raise ValueError('Frozen showcase source changed')
                client_factory=native_showcase_client
                context['showcase_contract_sha256']=sha256(Path(session_path).parent/'showcase_contract.json')
            else:
                from .showcase_session import validate_showcase_session, showcase_client
                binding = validate_showcase_session(session_path)
                context['showcase_contract_sha256'] = binding['contract_sha256']
                client_factory = showcase_client
                contract=read_json(contract_path)
            frozen_data = contract['data_source']
            context['dataset_binding'] = frozen_data
        context['normalization_sha256'] = sha256(PACKAGE/'normalization.json')
        if source is None:
            if source_factory is not None: source = source_factory()
            else:
                from .showcase_data import data_binding, open_source
                source = open_source(frozen_data or data_binding())
        if expected_counts is None and getattr(source,'expected_counts',None) is not None:
            expected = dict(source.expected_counts)
        if frozen_data is not None and expected != frozen_data['expected_counts']:
            raise ValueError('Runtime counts differ from frozen dataset')
        context.update(expected_counts=expected,expected_windows=expected['windows'])
        source.cancelled = cancelled.is_set
        context['source_provenance'] = dict(source.provenance)
        real_source = isinstance(source,(RawItchSource,DatabentoMboSource))
        context['runtime_book_reconstruction'] = real_source
        if isinstance(source,DatabentoMboSource):
            context['input_source'] = '2025-12-15 AAPL original Databento MBO; runtime book reconstruction'
            context['original_validation_alignment'] = False
        elif not real_source: context['input_source'] = 'injected local test source'
        batcher = WdmBatcher(normalization if normalization is not None else
                             read_json(PACKAGE/'normalization.json'), 32)
        if cancelled.is_set():
            context['status'] = 'partial'
            return context
        if client is None:
            client = client_factory(session_path, directory/'wire.jsonl')
        native=bool(getattr(client,'native',False))
        if native:
            context.update(native_stream=True,model_mode=client.model_mode,zero_copy=True,
                zero_copy_scope='full FPGA window is the NPU-bound PL DDR input; new record/output still use DMA',
                host_byte_integrity_check=False,full_tensor_readback=False,
                npu_timing_scope='Resident NPU launch/wait plus 64-byte output DMA; no per-window SDK adaptation',
                board_pipeline_timing_scope='complete received record -> binary result plus REARM; excludes network/GUI/logging',
                numeric_policy='fused TF32 experimental, old logits tolerance not passed' if client.model_mode=='fused' else 'qualified old outputs preserved')
        context['board_info'] = client.hello(); connected = True
        context['execution_kind'] = 'hardware_candidate' if getattr(client, 'hardware', False) else 'local_test_substitute'
        context['window_builder'] = 'FPGA' if getattr(client, 'hardware', False) else 'local_test_substitute'
        context['status'] = 'running'
        write_json(directory/'manifest.json', context)
        phase('ready', directory=str(directory), backends=['FPGA组窗 → 板载NPU'],
              execution_kind=context['execution_kind'], expected_windows=expected['windows'], stats=dict(stats))
        raw_records = source.iter_records(); iterator = iter(batcher.batches(raw_records))
        while not cancelled.is_set():
            pause_boundary()
            if cancelled.is_set(): break
            if max_windows is not None and not checkpoint_done and stats['windows'] == max_windows:
                checkpoint_done = True; paused.set()
                phase('phase', phase='awaiting_continuation', stats=dict(stats),
                      message='已完成首段演示；点击继续，将沿用原会话处理后续行情。')
                if checkpoint_callback: checkpoint_callback(dict(stats), paused, cancelled)
                pause_boundary()
                if cancelled.is_set(): break
            batch_start = clock(); response = None
            if pending is not None: batch, pending = pending, None
            else:
                try: batch = next(iterator)
                except StopIteration: break
            if cancelled.is_set(): break
            if max_windows is not None and not checkpoint_done:
                batch, pending = split_batch(batch, max_windows-stats['windows'])
            batch = replace(batch, sequence=accounting.next_sequence)
            accounting.submit(batch)
            endpoints = metadata.batch(batch)
            # Transport independently checks tensor bytes, protocol and ledger.
            # It never sends its check bytes to the NPU or runs a reference model.
            response = client.submit(batch.payload)
            actual = response['windows']
            if len(actual) != len(endpoints): raise ValueError('Missing or extra NPU result')
            for got, meta in zip(actual, endpoints): validate_result(got, meta,native=native)
            rtt=response.get('data',{}).get('batch_roundtrip_ns')
            if rtt is not None:
                roundtrip_timer.add(rtt);stats.update(batch_roundtrip_last_ns=rtt,
                    batch_roundtrip_count=roundtrip_timer.count,batch_roundtrip_total_ns=roundtrip_timer.total)
            accounting.acknowledge(response['seq'], [dict(window_id=p['window_id'],
                first=p['start'], last=p['end']) for p in actual])
            now = clock()
            if last_output_clock is not None and batch.window_count:
                observed_rate = batch.window_count*1e9/max(1, now-last_output_clock)
            if batch.window_count: last_output_clock = now
            returned = iter(zip(actual, endpoints)); pair = next(returned, None)
            for index, record in enumerate(batch.records):
                stats['records'] += 1
                mid2 = int(record.values[0])+int(record.values[2])
                market = dict(type='market', run_id=run_id, round=1, last=record.meta, mid2=mid2)
                journal.write('market', market); view.consume(market)
                outcomes(evaluator.observe(record.meta, mid2, 1))
                if pair is None or pair[1]['record_index'] != index: continue
                got, meta = pair
                outputs = {BACKEND: prediction(got['window_id'], got['logits'], BACKEND, got['whole_predict_ns'])}
                stats['windows'] += 1
                stats['npu_call_last_ns'] = got['whole_predict_ns']
                stats['npu_call_total_ns'] += got['whole_predict_ns']; timer.add(got['whole_predict_ns'])
                if native and not got['startup_included']:
                    ns=got['board_pipeline_ns'];board_timer.add(ns)
                    stats.update(board_pipeline_last_ns=ns,board_pipeline_count=board_timer.count,board_pipeline_total_ns=board_timer.total)
                evaluator.track(got['window_id'], 1, record.meta, outputs)
                stats['max_pending_h50'] = max(stats['max_pending_h50'], len(evaluator.pending))
                if len(evaluator.pending) > HORIZON: raise ValueError('H50 pending bound exceeded')
                row = dict(type='prediction', run_id=run_id, window_id=got['window_id'], round=1,
                    first=meta['first'], last=meta['last'], first_identity=got['start'], last_identity=got['end'],
                    input_sha256=got['tensor_sha256'], outputs=outputs, origin_mid2=mid2,
                    board_pipeline_ns=got.get('board_pipeline_ns'),batch_roundtrip_ns=got.get('batch_roundtrip_ns'),
                    stages_ns=got.get('stages_ns'),
                    startup_included=got.get('startup_included',False),model_mode=getattr(client,'model_mode','legacy'),
                    integrity_mode=got.get('integrity_mode','full_tensor_readback'),
                    target_source_row=record.meta['source_record_row_0based']+HORIZON,
                    book=record.values.tolist(), fpga_tensor_verified=not native, stats=dict(stats),
                    evaluation=evaluator.summary(), observed_windows_per_second=observed_rate,
                    target_windows_per_second=validate_rate(get_rate()),
                    elapsed_seconds=(clock()-started)/1e9, expected_windows=expected['windows'])
                journal.write('predictions', row); view.consume(row)
                pair = next(returned, None)
            if pair is not None: raise ValueError('Unobserved prediction endpoint')
            stats['batches'] += 1
            stats['max_batch_records'] = max(stats['max_batch_records'], len(batch.records))
            journal.write('batches', dict(type='batch', sequence=batch.sequence, records=len(batch.records),
                windows=batch.window_count, first_window_id=batch.first_window_id,
                board_data=response.get('data', {}), stats=dict(stats), source_metrics=dict(source.metrics)))
            journal.flush()
            view.consume(dict(type='metrics', stats=dict(stats), observed_windows_per_second=observed_rate,
                target_windows_per_second=validate_rate(get_rate()), elapsed_seconds=(clock()-started)/1e9))
            view.publish()
            while not cancelled.is_set() and not paused.is_set():
                rate = validate_rate(get_rate())
                remaining = batch.window_count/rate-(clock()-batch_start)/1e9 if rate else 0
                if remaining <= 0: break
                cancelled.wait(min(.05, remaining))
            response = None
        context['accounting'] = accounting.finish()
        if cancelled.is_set(): context['status'] = 'partial'
        else:
            metrics = source.metrics
            if (not batcher.exhausted or pending is not None or not metrics.get('completed_full_stream')
                    or metrics.get('frames_read') != expected['frames']
                    or metrics.get('eligible_records') != expected['records']
                    or stats['records'] != expected['records'] or stats['windows'] != expected['windows']):
                raise ValueError('EOF/count verification failed; full-day completion refused')
            context['status'] = 'completed'
            admitted = source.expected_counts if isinstance(source,DatabentoMboSource) else EXPECTED_DAY
            context['raw_day_processing_complete'] = real_source and expected == admitted
        context['board_final'] = client.finish(); finished = True
        context['full_day_complete'] = bool(context['raw_day_processing_complete'] and
            context['execution_kind'] == 'hardware_candidate' and context['status'] == 'completed')
        context['batcher'] = batcher.summary()
    except Exception as exc:
        stats['errors'] += 1
        context.update(status='failed', error=str(exc), full_day_complete=False)
        if response is not None:
            try:
                json.dumps(response, allow_nan=False)
                journal.write('batches', dict(type='failed_response', response=response))
            except (ValueError, TypeError):
                # Preserve malformed test/transport evidence without allowing NaN
                # or an unserializable response to mask the original failure.
                (directory/'failed_response.txt').write_text(repr(response), encoding='utf-8')
        if accounting.pending is not None:
            (directory/'failed_batch.wdm').write_bytes(accounting.pending.payload)
        phase('error', message=str(exc), directory=str(directory), stats=dict(stats))
    finally:
        for label, handle in (('iterator', iterator), ('raw_source', raw_records)):
            if handle is not None and hasattr(handle, 'close'):
                try: handle.close()
                except Exception as exc: context[label+'_close_error'] = str(exc)
        outcomes(evaluator.finish('end_of_source' if context['status'] == 'completed' else context['status']))
        if client is not None and hasattr(client, 'disconnect'):
            try: client.disconnect()
            except Exception as exc: context['disconnect_error'] = str(exc)
        context.update(stats=dict(stats), source_metrics=dict(getattr(source, 'metrics', {})),
            market_evaluation=evaluator.summary(), evaluation=evaluator.summary(),
            timings={'npu_call': timer.summary(),'board_pipeline':board_timer.summary(),'batch_roundtrip':roundtrip_timer.summary()}, elapsed_seconds=(clock()-started)/1e9,
            board_finish_acknowledged=finished, connected=connected,
            observed_windows_per_second=observed_rate, pending_batch=accounting.pending is not None,
            log_counts=dict(getattr(journal, 'counts', {})))
        write_json(directory/'manifest.json', context)
        phase('finished', directory=str(directory), summary=context, status=context['status'],
              stats=dict(stats), market_evaluation=evaluator.summary())
        journal.close()
    return context


def export_showcase_csv(directory, destination, *, limits=None):
    """Bounded streaming merge of a fixed prefix; no reference result columns."""
    import csv
    from contextlib import closing
    from .paths import CLASS_CN
    paths = {name: Path(directory)/(name+'.jsonl') for name in ('predictions', 'outcomes')}
    limits = limits or {name: path.stat().st_size if path.is_file() else 0 for name, path in paths.items()}

    def rows(name):
        if not paths[name].is_file(): return
        with paths[name].open('rb') as handle:
            remaining = limits[name]
            while remaining > 0:
                line = handle.readline(remaining)
                if not line: break
                remaining -= len(line)
                if not line.endswith(b'\n'): break
                yield json.loads(line)

    count = 0
    with closing(rows('outcomes')) as outcomes, closing(rows('predictions')) as predictions, \
            open(destination, 'w', encoding='utf-8-sig', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(('run_id', 'window_id', 'date', 'source_row', 'exchange_ns',
                         'midpoint_usd', 'npu_prediction', 'logit_down', 'logit_flat', 'logit_up',
                         'npu_call_ms', 'input_sha256', 'outcome_status', 'actual_direction',
                         'actual_midpoint_usd', 'prediction_correct', 'unresolved_reason',
                         'board_pipeline_ms','batch_roundtrip_ms','startup_included','model_mode','integrity_mode'))
        outcome = next(outcomes, None)
        for row in predictions:
            while outcome is not None and outcome['window_id'] < row['window_id']:
                outcome = next(outcomes, None)
            result = outcome if outcome is not None and outcome['window_id'] == row['window_id'] else {}
            npu = row['outputs'][BACKEND]; meta = row['last']
            writer.writerow((row['run_id'], row['window_id'], meta['date'], meta['source_record_row_0based'],
                meta['timestamp_ns_since_exchange_midnight'], row['origin_mid2']/20000,
                CLASS_CN[npu['class_index']], *npu['logits'], npu['elapsed_ns']/1e6, row['input_sha256'],
                result.get('status', 'pending'), CLASS_CN[result['actual_class']] if 'actual_class' in result else '',
                result['actual_mid2']/20000 if 'actual_mid2' in result else '',
                result.get('matches', {}).get(BACKEND, ''), result.get('reason', ''),
                row['board_pipeline_ns']/1e6 if row.get('board_pipeline_ns') is not None else '',
                row['batch_roundtrip_ns']/1e6 if row.get('batch_roundtrip_ns') is not None else '',
                row.get('startup_included',False),row.get('model_mode','legacy'),row.get('integrity_mode','full_tensor_readback')))
            count += 1
    return count
