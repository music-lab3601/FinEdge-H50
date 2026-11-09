from datetime import datetime, timezone
from contextlib import closing
from importlib.metadata import version
import hashlib
import json
import platform
import threading
import time
import uuid

import numpy as np

from .backends import CpuBackend, OnnxBackend, WorkerBackend, warmup
from .delivery import read_json, sha256, write_json
from .paths import ARTIFACTS, MANIFEST_SHA, PACKAGE, ROOT
from .stream import InputBuilder, PREPROCESS_ID, RawSource
from .datasets import dataset_info, open_source
from .market import DelayedEvaluation, HORIZON, TARGET_ID
from .source_pump import SourcePump


class ReplayControl:
    def __init__(self, rate=10):
        self.stop = threading.Event()
        self.pause = threading.Event()
        self.reset = threading.Event()
        self.rate = rate


def run_replay(mode='socket', rounds=1, duration=0, control=None, emit=None,
               dataset='delivery1024', max_windows=0, repeat=False, queue_capacity=8):
    """Sequential reference comparison; no GUI calls occur on this worker thread."""
    if mode not in ('cpu', 'onnx', 'host', 'socket', 'socket-verify'):
        raise ValueError('Unknown replay mode')
    if queue_capacity < 1:
        raise ValueError('Source queue capacity must be positive')
    control = control or ReplayControl(rate=0)
    emit = emit or (lambda event: None)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:6]
    directory = ARTIFACTS / 'runs' / run_id
    directory.mkdir(parents=True)
    context = {'run_id': run_id, 'mode': mode, 'status': 'initializing', 'model_manifest_sha256': MANIFEST_SHA,
               'application_version': version('finedge-host-demo'),
               'preprocess_id': PREPROCESS_ID, 'python': platform.python_version(),
               'market_accuracy_evaluated': False, 'source_role': 'train_engineering_snippet',
               'comparison_note': 'PC CPU, Host instruction simulation, socket includes network; not a same-board speedup benchmark.'}
    context.update(host_simulation_enabled=mode in ('host', 'socket-verify'),
                   pacing='period_includes_preprocessing_inference_and_logging; no catch-up bursts',
                   initial_target_windows_per_second=control.rate)
    context.update(dataset_info(dataset), source_role=dataset_info(dataset)['role'], max_windows=max_windows, repeat=repeat)
    write_json(directory / 'manifest.json', context)
    handles = []
    stats = {'records': 0, 'windows': 0, 'rounds_started': 0, 'npu_mismatches': 0,
             'conversion_class_mismatches': 0, 'conversion_max_abs_error': 0.0,
             'npu_max_abs_error': 0.0, 'errors': 0, 'npu_host_checked_windows': 0,
             'cpu_npu_class_mismatches': 0, 'cpu_npu_max_abs_error': 0.0, 'rate_deadline_misses': 0}
    latencies = {}
    stage_times = {'preprocess_ns': [], 'pack_ns': [], 'validation_total_ns': []}
    pending_window = None
    source = pump = None
    pump_reports = []
    evaluation = DelayedEvaluation()
    started = None
    results_file = (directory / 'predictions.jsonl').open('x', encoding='utf-8', buffering=1)
    events_file = (directory / 'events.jsonl').open('x', encoding='utf-8', buffering=1)
    market_file = (directory / 'market.jsonl').open('x', encoding='utf-8', buffering=1)
    outcomes_file = (directory / 'outcomes.jsonl').open('x', encoding='utf-8', buffering=1)
    context.update(market_target_id=TARGET_ID, horizon_records=HORIZON,
                   candle_source='observed eligible top10 book midpoint; not trade OHLC',
                   evaluation_scope='observed replay sample; overlapping windows and repeated rounds are not independent samples')
    def publish_outcomes(items):
        for payload in items:
            item = {'type': 'outcome', 'run_id': run_id, **payload}
            outcomes_file.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
            emit({**item, 'evaluation': evaluation.summary()})
    def event(kind, **payload):
        item = {'type': kind, 'run_id': run_id, **payload}
        events_file.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
        emit(item)
    try:
        load_start = time.perf_counter_ns()
        # Keep the original factory for existing baseline tests and callers.
        source = RawSource() if dataset == 'delivery1024' else open_source(dataset)
        raw_stream = hasattr(source, 'iter_records')
        context['runtime_book_reconstruction'] = raw_stream
        if raw_stream:
            source.cancelled = control.stop.is_set
            source.on_progress = lambda metrics: emit({'type':'source_progress','run_id':run_id, 'source':metrics})
            context.update(source_provenance=source.provenance, source_queue_capacity=queue_capacity,
                           warmup_source_scan='separate fresh prefix scan; not published as replay records',
                           source_flow_control='lossless file backpressure, no event-time pacing',
                           queued_latency_note='queue/pacing/pause may increase validation_total_ns; not a live latency benchmark')
        context['file_read_and_integrity_ns'] = time.perf_counter_ns() - load_start
        normalization = read_json(PACKAGE / 'normalization.json')
        if read_json(PACKAGE / 'train_config.json')['horizon'] != HORIZON:
            raise ValueError('Model horizon differs from the replay evaluation contract')
        warm_builder = InputBuilder(normalization)
        first = None
        warm_input_started = time.perf_counter_ns()
        with closing(iter(source)) as warm_records:
            for record in warm_records:
                if control.stop.is_set(): break
                first = warm_builder.push(record)
                if first:
                    break
        context['warmup_input_build_ns'] = time.perf_counter_ns()-warm_input_started
        if first is None:
            if control.stop.is_set():
                context['status'] = 'stopped'
                return context
            raise ValueError('Data has no complete 100-record window')
        load_start = time.perf_counter_ns()
        cpu = CpuBackend(); handles.append(cpu)
        if mode == 'onnx':
            handles.append(OnnxBackend())
        if mode in ('host', 'socket-verify'):
            handles.append(WorkerBackend('host', log_path=directory / 'host.log', timeout=45))
        if mode in ('socket', 'socket-verify'):
            handles.append(WorkerBackend('socket', log_path=directory / 'socket.log', timeout=45))
        context['backend_load_and_connection_ns'] = time.perf_counter_ns() - load_start
        context['backend_info'] = {b.name: getattr(b, 'info', {'backend': b.name}) for b in handles}
        versions_path = ARTIFACTS / 'provenance/software_versions.json'
        if mode in ('host', 'socket', 'socket-verify') and versions_path.is_file():
            versions = read_json(versions_path)
            context['software_versions_snapshot'] = versions
            # The snapshot is usable only while the executable and SDK files
            # still match it. Device versions are also reported live above.
            context['software_versions_snapshot_matches_files'] = (
                all(sha256(path) == digest for path, digest in versions['files'].items())
                and sha256(ROOT / 'build/Release/finedge_worker.exe')
                == versions['worker_sha256'])
        context['warmup'] = {b.name: warmup(b, first.tensor) for b in handles}
        context['raw_manifest_sha256'] = sha256(getattr(source, 'manifest_path', source.path / 'manifest.json'))
        context['model_sha256'] = sha256(PACKAGE / 'model_cpu.ts')
        if mode in ('host', 'socket', 'socket-verify'):
            context['compiled_json_sha256'] = sha256(ARTIFACTS / 'icraft/tcn_ed4_ZG.json')
            context['compiled_raw_sha256'] = sha256(ARTIFACTS / 'icraft/tcn_ed4_ZG.raw')
        context['status'] = 'running'
        write_json(directory / 'manifest.json', context)
        event('ready', directory=str(directory), source_count=None if raw_stream else len(source), backends=[b.name for b in handles],
              dataset=dataset_info(dataset))
        started = time.perf_counter()
        round_index = 0
        next_cycle_deadline = None
        pacing_rate = None
        while not control.stop.is_set():
            if duration and time.perf_counter() - started >= duration:
                break
            if max_windows and stats['windows'] >= max_windows:
                break
            if not duration and not repeat and round_index >= rounds:
                break
            round_index += 1
            stats['rounds_started'] = round_index
            round_start_records = stats['records']
            builder = InputBuilder(normalization)
            next_cycle_deadline = None
            event('round', round=round_index, reset_window=True)
            if raw_stream:
                interrupted = lambda: (control.stop.is_set() or control.reset.is_set()
                                       or bool(duration and time.perf_counter()-started >= duration))
                pump = SourcePump(source, builder, queue_capacity, interrupted)
                records = pump
            else:
                records = source
            for entry in records:
                while control.pause.is_set() and not control.stop.is_set() and not control.reset.is_set():
                    next_cycle_deadline = None
                    time.sleep(.02)
                if control.stop.is_set() or (duration and time.perf_counter()-started >= duration):
                    break
                if max_windows and stats['windows'] >= max_windows:
                    break
                if control.reset.is_set():
                    break
                cycle_entered_ns = time.perf_counter_ns()
                record = entry.record if raw_stream else entry
                window = entry.window if raw_stream else builder.push(record)
                stats['records'] += 1
                mid2 = int(record.values[0]) + int(record.values[2])
                # Initial 99 observations are charted too. Never read labels.i8.bin
                # or future source rows to decorate the replay.
                market_item = {'type': 'market', 'run_id': run_id, 'round': round_index,
                               'last': record.meta, 'mid2': mid2}
                market_file.write(json.dumps(market_item, ensure_ascii=False, allow_nan=False) + '\n')
                emit(market_item)
                publish_outcomes(evaluation.observe(record.meta, mid2, round_index))
                if window is None:
                    if stats['records'] % 20 == 0:
                        emit({'type': 'warming_window', 'run_id': run_id, 'count': builder.count, 'round': round_index})
                    continue
                window_id = stats['windows'] + 1
                pending_window = {'window_id': window_id, 'round': round_index,
                                  'first': window.first_meta, 'last': window.last_meta,
                                  'input_sha256': hashlib.sha256(window.tensor.tobytes()).hexdigest()}
                outputs = {b.name: b.predict(window_id, window.tensor) for b in handles}
                stats['windows'] += 1
                for name, result in outputs.items():
                    latencies.setdefault(name, []).append(result['elapsed_ns'])
                baseline = outputs['torch_cpu']
                compare = outputs.get('icraft_host', outputs.get('onnx_cpu'))
                conversion_error = 0.0
                if compare:
                    conversion_error = float(np.max(np.abs(np.array(compare['logits']) - baseline['logits'])))
                    stats['conversion_max_abs_error'] = max(stats['conversion_max_abs_error'], conversion_error)
                    stats['conversion_class_mismatches'] += int(compare['class_index'] != baseline['class_index'])
                npu_error, npu_ok = None, None
                cpu_npu_error, cpu_npu_class_match = None, None
                if 'icraft_socket' in outputs:
                    actual = outputs['icraft_socket']
                    cpu_npu_error = float(np.max(np.abs(np.array(actual['logits']) - baseline['logits'])))
                    cpu_npu_class_match = actual['class_index'] == baseline['class_index']
                    stats['cpu_npu_max_abs_error'] = max(stats['cpu_npu_max_abs_error'], cpu_npu_error)
                    stats['cpu_npu_class_mismatches'] += int(not cpu_npu_class_match)
                if 'icraft_socket' in outputs and 'icraft_host' in outputs:
                    host, actual = outputs['icraft_host'], outputs['icraft_socket']
                    stats['npu_host_checked_windows'] += 1
                    npu_error = float(np.max(np.abs(np.array(actual['logits']) - host['logits'])))
                    npu_ok = bool(np.allclose(actual['logits'], host['logits'], rtol=1e-3, atol=1e-4)
                                  and actual['class_index'] == host['class_index'])
                    stats['npu_mismatches'] += int(not npu_ok)
                    stats['npu_max_abs_error'] = max(stats['npu_max_abs_error'], npu_error)
                item = {'run_id': run_id, 'window_id': window_id, 'round': round_index,
                        'first': window.first_meta, 'last': window.last_meta,
                        'input_sha256': hashlib.sha256(window.tensor.tobytes()).hexdigest(),
                        'preprocess_ns': window.preprocess_ns, 'pack_ns': window.pack_ns,
                        'outputs': outputs, 'conversion_max_abs_error': conversion_error,
                        'conversion_comparison_available': compare is not None,
                        'cpu_npu_max_abs_error': cpu_npu_error, 'cpu_npu_class_match': cpu_npu_class_match,
                        'npu_max_abs_error': npu_error, 'npu_within_tolerance': npu_ok,
                        'validation_total_ns': time.perf_counter_ns() - window.entered_ns,
                        'prediction_completed_monotonic_ns': time.perf_counter_ns()}
                if raw_stream:
                    item.update(source_metrics=entry.source_metrics,
                                source_queue_wait_ns=max(0,cycle_entered_ns-entry.published_ns))
                evaluation.track(window_id, round_index, record.meta, outputs)
                item.update(origin_mid2=mid2, market_target_id=TARGET_ID,
                            target_source_row=record.meta['source_record_row_0based']+HORIZON)
                results_file.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
                pending_window = None
                for name in stage_times:
                    stage_times[name].append(item[name])
                emit({'type': 'prediction', **item, 'book': record.values.tolist(), 'stats': dict(stats),
                      'evaluation': evaluation.summary()})
                current_rate = control.rate
                interval = 1/current_rate if current_rate > 0 else 0
                if interval:
                    now = time.perf_counter()
                    if next_cycle_deadline is None or current_rate != pacing_rate:
                        next_cycle_deadline = (cycle_entered_ns if raw_stream else window.entered_ns)/1e9 + interval
                    else:
                        next_cycle_deadline += interval
                    remaining = next_cycle_deadline - now
                    stats['rate_deadline_misses'] += int(remaining <= 0)
                    if remaining < 0:
                        next_cycle_deadline = now
                    # Absolute deadlines absorb Windows sleep granularity without
                    # accumulating drift. Overruns rebase, never skip or catch up.
                    control.stop.wait(max(0, remaining))
                    pacing_rate = current_rate
                else:
                    next_cycle_deadline = pacing_rate = None
                if (control.stop.is_set() or control.reset.is_set()
                        or bool(max_windows and stats['windows'] >= max_windows)):
                    break
            if pump is not None:
                pump_reports.append({'round':round_index, **pump.close(), 'source':dict(source.metrics),
                                     'records_published':stats['records']-round_start_records,
                                     'dequeued_unpublished_on_shutdown':pump.metrics['records_dequeued']-(stats['records']-round_start_records)})
                pump = None
            if control.reset.is_set():
                control.reset.clear()
                event('reset', round=round_index)
                if not duration:
                    rounds += 1
            # No outcome may cross manual reset, a dataset tail or a new round.
            reason = ('stopped' if control.stop.is_set() else 'round_end_or_reset')
            publish_outcomes(evaluation.finish(reason))
        context['status'] = 'failed' if stats['npu_mismatches'] else ('stopped' if control.stop.is_set() else 'passed')
    except BaseException as exc:
        stats['errors'] += 1
        context.update(status='failed', error=str(exc), failed_window=pending_window)
        event('error', message=str(exc), directory=str(directory), failed_window=pending_window)
        raise
    finally:
        if pump is not None:
            pump_reports.append({'round':stats['rounds_started'], **pump.close(),
                                 'source':dict(source.metrics),
                                 'records_published':stats['records']-round_start_records,
                                 'dequeued_unpublished_on_shutdown':pump.metrics['records_dequeued']-(stats['records']-round_start_records)})
        if pump_reports:
            context['source_pumps'] = pump_reports
        if source is not None and hasattr(source, 'metrics'):
            context['raw_source_final'] = dict(source.metrics)
        publish_outcomes(evaluation.finish('error_or_run_end'))
        context['market_evaluation'] = evaluation.summary()
        context['market_accuracy_evaluated'] = evaluation.resolved > 0
        for b in reversed(handles):
            b.close()
        results_file.close()
        events_file.close()
        market_file.close()
        outcomes_file.close()
        measured = time.perf_counter() - started if started is not None else 0
        context.update(stats=stats, elapsed_seconds=measured,
                       throughput_windows_per_second=stats['windows']/measured if measured else 0.0,
                       ten_minute_replay_verified=measured >= 600 and stats['windows'] > 0 and stats['errors'] == 0 and stats['npu_mismatches'] == 0,
                       ten_minute_npu_host_verified=measured >= 600 and stats['windows'] > 0
                       and stats['npu_host_checked_windows'] == stats['windows']
                       and stats['errors'] == 0 and stats['npu_mismatches'] == 0)
        context['latency_ms'] = {name: {'median': float(np.median(v))/1e6, 'p99': float(np.percentile(v, 99))/1e6,
                                      'samples': len(v)} for name, v in latencies.items()}
        context['stage_latency_ms'] = {name: {'median': float(np.median(v))/1e6,
                                             'p99': float(np.percentile(v, 99))/1e6, 'samples': len(v)}
                                       for name, v in stage_times.items() if v}
        write_json(directory / 'manifest.json', context)
        emit({'type': 'finished', 'run_id': run_id, 'directory': str(directory), 'summary': context})
    return context
