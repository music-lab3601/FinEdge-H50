"""Event-time file replay experiments; no exchange network transport implied.

The producer always consumes all order events. Only inference opportunities may
be replaced by latest mode. Historical deadlines are scored separately offline.
"""
import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

from software.runtime.backends import CpuBackend, OnnxBackend, WorkerBackend, warmup
from software.runtime.delivery import read_json, sha256, write_json
from shared.model_paths import ARTIFACTS, PACKAGE, ROOT
from experiments.reference.runtime.raw_itch import RawItchSource
from software.runtime.stream import InputBuilder

NS = 1_000_000_000


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    if not len(x): return {'count': 0}
    return {'count': int(len(x)), 'min': float(x.min()), 'p50': float(np.percentile(x, 50)),
            'p95': float(np.percentile(x, 95)), 'p99': float(np.percentile(x, 99)),
            'max': float(x.max()), 'mean': float(x.mean())}


class ReplayEnd(Exception): pass


class EventClock:
    def __init__(self, start_ns, duration_s, speed=1., stop=None):
        if speed <= 0 or duration_s <= 0: raise ValueError('Positive speed and duration required')
        self.start = int(start_ns)
        self.end = self.start + int(duration_s*NS)
        self.speed = float(speed)
        self.anchor = None
        self.stop = stop or threading.Event()
        self.events = 0
        self.late_ns = []
        self.first_sequence = self.last_sequence = None

    def due(self, timestamp):
        return self.anchor + int((int(timestamp)-self.start)/self.speed)

    def __call__(self, sequence, offset, payload):
        timestamp = int.from_bytes(payload[5:11], 'big')
        if timestamp < self.start: return  # Full prefix reconstructs before timing.
        if timestamp >= self.end: raise ReplayEnd()
        if self.stop.is_set(): raise ReplayEnd()
        if self.anchor is None: self.anchor = time.perf_counter_ns()
        due = self.due(timestamp)
        while True:
            remaining = (due-time.perf_counter_ns())/NS
            if remaining <= 0: break
            if self.stop.wait(min(remaining, .02)): raise ReplayEnd()
        self.late_ns.append(max(0, time.perf_counter_ns()-due))
        self.events += 1
        if self.first_sequence is None: self.first_sequence = sequence
        self.last_sequence = sequence


class WindowChannel:
    def __init__(self, policy='latest', capacity=1):
        if policy not in ('latest', 'fifo') or capacity < 1:
            raise ValueError('Invalid bounded scheduling policy')
        self.policy, self.capacity = policy, capacity
        self.items = deque()
        self.cv = threading.Condition()
        self.closed = self.cancelled = False
        self.error = None
        self.offered = self.taken = self.replaced = self.high_water = self.cancelled_pending = 0
        self.wait_ns = 0

    def put(self, item):
        with self.cv:
            self.offered += 1
            if self.policy == 'latest':
                self.replaced += len(self.items)
                self.items.clear()
            else:
                begin = time.perf_counter_ns()
                blocked = len(self.items) >= self.capacity
                while len(self.items) >= self.capacity and not self.cancelled:
                    self.cv.wait(.02)
                if blocked: self.wait_ns += time.perf_counter_ns()-begin
            if self.cancelled:
                self.cancelled_pending += 1
                return False
            self.items.append(item)
            self.high_water = max(self.high_water, len(self.items))
            self.cv.notify_all()
            return True

    def get(self):
        with self.cv:
            while not self.items and not self.closed and not self.cancelled: self.cv.wait(.02)
            if self.items:
                self.taken += 1
                item = self.items.popleft(); self.cv.notify_all()
                return item
            if self.error is not None: raise self.error
            return None

    def finish(self, error=None):
        with self.cv:
            self.closed, self.error = True, error
            self.cv.notify_all()

    def cancel(self):
        with self.cv:
            self.cancelled = True
            self.cancelled_pending += len(self.items)
            self.items.clear()
            self.cv.notify_all()

    def metrics(self):
        return {'policy': self.policy, 'capacity': self.capacity, 'offered': self.offered,
                'taken': self.taken, 'replaced': self.replaced, 'high_water': self.high_water,
                'backpressure_ns': self.wait_ns, 'cancelled_pending': self.cancelled_pending,
                'remaining': len(self.items)}


class LazyWindowChannel(WindowChannel):
    """One pending publication; snapshot latest ring atomically on selection."""
    def __init__(self,builder):
        super().__init__('latest',1)
        self.builder=builder
        self.invalidated=0

    def push_record(self,record,due):
        with self.cv:
            try:self.builder.push(record,materialize=False)
            except Exception:
                self.invalidated+=len(self.items);self.replaced+=len(self.items);self.items.clear()
                raise
            if self.builder.count<100:
                self.invalidated+=len(self.items);self.replaced+=len(self.items);self.items.clear()
                return False
            return self.put((None,record.segment_key,time.perf_counter_ns(),due))

    def get(self):
        with self.cv:
            item=super().get()
            if item is None:return None
            return (self.builder.snapshot(),)+item[1:]

    def metrics(self):
        return dict(super().metrics(),invalidated_on_segment_or_error=self.invalidated)


def run_experiment(*, start_ns, duration_s=30, speed=1., policy='latest', capacity=1,
                   backend='socket', cpu_every=0, output=None, max_age_ms=100.,lazy_windows=False):
    output = Path(output or ARTIFACTS/'stage2'/('run_'+str(time.time_ns())))
    output.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    clock = EventClock(start_ns, duration_s, speed, stop)
    if lazy_windows and (policy!='latest' or capacity!=1):raise ValueError('Lazy windows require latest capacity 1')
    builder = InputBuilder(read_json(PACKAGE/'normalization.json'),packing='single_copy' if lazy_windows else 'legacy')
    channel = LazyWindowChannel(builder) if lazy_windows else WindowChannel(policy, capacity)
    source = RawItchSource(before_event=clock, measure_stages=True,
                           feature_start_ns=int(start_ns)-60*NS)
    progress_at = [0.]
    def progress(metrics):
        now=time.perf_counter()
        if now-progress_at[0] >= 5:
            write_json(output/'progress.json', {'phase':'prefix' if clock.anchor is None else 'event_time_replay',
                'selected_events':metrics.get('selected_events'), 'eligible_records':metrics.get('eligible_records'),
                'last_timestamp_ns':metrics.get('last_timestamp_ns'), 'channel':channel.metrics()})
            progress_at[0]=now
    source.on_progress=progress
    # Warmup uses fixed saved tensors outside formal source time, never future labels.
    warm_tensor = np.array(np.load(ARTIFACTS/'stage2/benchmark_inputs.npy', mmap_mode='r')[0], copy=True)
    model = cpu = None
    stats = {'records': 0, 'windows': 0, 'preprocess_ns': [], 'pack_ns': [],
             'latest_row': None, 'latest_segment': None}
    results = []
    wall_start = time.perf_counter_ns()
    producer = None
    status = 'failed'
    error = None
    model_info = None
    timer_requested = False
    prepared = warm = None
    try:
        if backend == 'cpu': model = CpuBackend()
        elif backend == 'onnx': model = OnnxBackend()
        elif backend == 'local':
            from experiments.reference.runtime.board_local import BoardLocalBackend
            # Direct AXI failed the fixed-tensor numerical gate. Resident ARM
            # loopback is the validated same-board path; no Ethernet round trip.
            model = BoardLocalBackend(url='socket://zg330aiu@127.0.0.1:9981')
        elif backend in ('socket', 'local'):
            kwargs = {'log_path': output/'worker.log', 'timeout': 30}
            if backend == 'local':
                kwargs.update(executable=ROOT/'build/finedge_worker', url='axi://zg330aiu')
            model = WorkerBackend(backend, **kwargs)
        else: raise ValueError('Unsupported experiment backend')
        model_info = getattr(model, 'info', {'backend': model.name})
        if cpu_every:
            if backend == 'local': raise ValueError('Board CPU Torch reference is an offline PC verification')
            cpu = CpuBackend()
        warm = warmup(model, warm_tensor)
        if cpu is not None: warmup(cpu, warm_tensor)
        if os.name == 'nt':
            import ctypes
            timer_requested = ctypes.windll.winmm.timeBeginPeriod(1) == 0
        prepared = time.perf_counter_ns()
        def produce():
            prefix = deque(maxlen=99)
            initialized = False
            try:
                for record in source.iter_records(cancel=stop.is_set):
                    t = record.meta['timestamp_ns_since_exchange_midnight']
                    if t < clock.start:
                        prefix.append(record)
                        continue
                    if not initialized:
                        if len(prefix)<99:raise ValueError('Insufficient 99-record warm prefix; choose earlier feature start')
                        for earlier in prefix: builder.push(earlier)
                        initialized = True
                    w = None if lazy_windows else builder.push(record)
                    stats['records'] += 1
                    stats['latest_row'] = record.meta['source_record_row_0based']
                    stats['latest_segment'] = record.segment_key
                    if lazy_windows:
                        if channel.push_record(record,clock.due(t)):
                            stats['windows']+=1
                            stats['preprocess_ns'].append(builder.last_preprocess_ns)
                    elif w is not None:
                        stats['windows'] += 1
                        stats['preprocess_ns'].append(w.preprocess_ns)
                        stats['pack_ns'].append(w.pack_ns)
                        if not channel.put((w, record.segment_key, time.perf_counter_ns(), clock.due(t))): break
                channel.finish()
            except ReplayEnd:
                source.metrics.pop('error', None)
                source.metrics['completed_test_interval'] = True
                channel.finish()
            except BaseException as exc:
                channel.finish(exc)
        producer = threading.Thread(target=produce, name='event-clock-book-producer', daemon=True)
        producer.start()
        with (output/'predictions.jsonl').open('x', encoding='utf-8', buffering=1) as log:
            while True:
                packet = channel.get()
                if packet is None: break
                w, segment, ready, due = packet
                if lazy_windows:stats['pack_ns'].append(w.pack_ns)
                selected = time.perf_counter_ns()
                ident = int(w.last_meta['source_message_position_1based'])
                reference = cpu.predict(ident, w.tensor) if cpu is not None and (len(results)%cpu_every == 0) else None
                pred = model.predict(ident, w.tensor)
                completed = time.perf_counter_ns()
                observed_h50 = (stats['latest_segment'] == segment and stats['latest_row'] >= w.last_meta['source_record_row_0based']+50)
                item = {'window_id': ident, 'last': w.last_meta, 'first': w.first_meta,
                        'input_sha256': hashlib.sha256(w.tensor.tobytes()).hexdigest(),
                        'source_due_ns': due, 'input_ready_ns': ready, 'selected_ns': selected,
                        'completed_ns': completed, 'input_age_ns': max(0, selected-due),
                        'result_age_ns': max(0, completed-due), 'queue_wait_ns': selected-ready,
                        'source_processing_lag_ns': max(0, ready-due),
                        'preprocess_ns': w.preprocess_ns, 'pack_ns': w.pack_ns,
                        'prediction': pred, 'cpu_reference': reference,
                        'processed_source_sequence_at_completion': source.metrics.get('last_processed_source_sequence'),
                        'processed_source_time_at_completion': source.metrics.get('last_processed_timestamp_ns'),
                        'expired_by_observed_h50': bool(observed_h50),
                        'expired_by_age_cap': completed-due > max_age_ms*1e6}
                log.write(json.dumps(item, allow_nan=False)+'\n')
                results.append(item)
        status = 'passed'
    except BaseException as exc:
        error = str(exc)
        raise
    finally:
        stop.set()
        channel.cancel()
        if producer is not None:
            producer.join(10)
            if producer.is_alive(): raise RuntimeError('Event replay producer failed to stop')
        if model is not None: model.close()
        if cpu is not None: cpu.close()
        if timer_requested:
            import ctypes
            ctypes.windll.winmm.timeEndPeriod(1)
        end = time.perf_counter_ns()
        duration = (end-clock.anchor)/NS if clock.anchor is not None else None
        metrics = channel.metrics()
        assert metrics['offered'] == metrics['taken']+metrics['replaced']+metrics['cancelled_pending']+metrics['remaining']
        report = {'status': status, 'error': error, 'backend': backend, 'start_ns': int(start_ns),
                  'source_duration_s': duration_s, 'speed': speed, 'max_age_ms': max_age_ms,
                  'cpu_every': cpu_every, 'source_provenance': source.provenance,
                  'lazy_windows':lazy_windows,
                  'model_sha256': sha256(ARTIFACTS/'icraft/tcn_ed4_ZG.raw'),
                  'normalization_sha256': sha256(PACKAGE/'normalization.json'),
                  'model_info': model_info, 'channel': metrics, 'events': clock.events,
                  'warmup':warm,
                  'initialization_before_source_s':(prepared-wall_start)/NS if prepared else None,
                  'prefix_reconstruction_s':(clock.anchor-prepared)/NS if prepared and clock.anchor else None,
                  'first_sequence': clock.first_sequence, 'last_sequence': clock.last_sequence,
                  'records': stats['records'], 'windows_offered': stats['windows'],
                  'inferences': len(results), 'source_final': source.metrics,
                  'clock_anchor_ns': clock.anchor, 'replay_wall_s_including_drain': duration,
                  'wall_s_including_load_prefix_warmup': (end-wall_start)/NS,
                  'event_lateness_ms': distribution(np.asarray(clock.late_ns)/1e6),
                  'events_per_wall_s_including_drain': clock.events/duration if duration else None,
                  'inferences_per_wall_s_including_drain': len(results)/duration if duration else None,
                  'preprocess_us': distribution(np.asarray(stats['preprocess_ns'])/1e3),
                  'pack_us': distribution(np.asarray(stats['pack_ns'])/1e3),
                  'queue_wait_ms': distribution([r['queue_wait_ns']/1e6 for r in results]),
                  'input_age_ms': distribution([r['input_age_ns']/1e6 for r in results]),
                  'result_age_ms': distribution([r['result_age_ns']/1e6 for r in results]),
                  'inference_ms': distribution([r['prediction']['elapsed_ns']/1e6 for r in results]),
                  'observed_h50_expired': sum(r['expired_by_observed_h50'] for r in results),
                  'age_cap_expired': sum(r['expired_by_age_cap'] for r in results),
                  'historical_event_time_mapping': 'due=perf_counter_anchor+(event_ns-start_ns)/speed',
                  'actual_exchange_network': False, 'host_simulation': False}
        report['windows_timer_resolution_request_ms'] = 1 if timer_requested else None
        write_json(output/'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start-ns', type=int, required=True)
    p.add_argument('--duration', type=float, default=30)
    p.add_argument('--speed', type=float, default=1)
    p.add_argument('--policy', choices=('fifo', 'latest'), default='latest')
    p.add_argument('--capacity', type=int, default=1)
    p.add_argument('--backend', choices=('cpu', 'onnx', 'socket', 'local'), default='socket')
    p.add_argument('--cpu-every', type=int, default=0)
    p.add_argument('--lazy-windows',action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    r = run_experiment(start_ns=a.start_ns, duration_s=a.duration, speed=a.speed, policy=a.policy,
                       capacity=a.capacity, backend=a.backend, cpu_every=a.cpu_every, output=a.output,lazy_windows=a.lazy_windows)
    print(json.dumps({k:r[k] for k in ('status','backend','inferences','records','channel','result_age_ms')}), flush=True)


if __name__ == '__main__': main()
