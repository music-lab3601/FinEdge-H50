"""Bounded lossless file producer: raw events/book/preprocess vs inference."""
from dataclasses import dataclass
import queue
import threading
import time


@dataclass
class SourcePacket:
    record: object
    window: object
    published_ns: int
    source_metrics: dict


class SourcePump:
    def __init__(self, source, builder, capacity=8, interrupted=lambda: False):
        if capacity < 1:
            raise ValueError('Source queue capacity must be positive')
        self.source, self.builder = source, builder
        self.queue = queue.Queue(maxsize=capacity)
        self.stop = threading.Event()
        self.interrupted = interrupted
        self.metrics = {'capacity': capacity, 'queue_high_water': 0, 'records_produced': 0,
                        'records_enqueued': 0, 'records_dequeued': 0, 'backpressure_wait_ns': 0,
                        'backpressure_timeouts': 0, 'cancelled_before_enqueue': 0,
                        'producer_failed': False, 'producer_completed': False}
        self.thread = threading.Thread(target=self._produce, name='itch-book-window-producer', daemon=True)
        self.thread.start()

    def _put(self, item):
        begin = time.perf_counter_ns()
        blocked = self.queue.full()
        while not self.stop.is_set():
            try:
                # Wake immediately when a slot becomes free, rather than
                # adding a fixed polling sleep to every inference cycle.
                self.queue.put(item, timeout=.005)
                if blocked:
                    self.metrics['backpressure_wait_ns'] += time.perf_counter_ns()-begin
                self.metrics['queue_high_water'] = max(self.metrics['queue_high_water'], self.queue.qsize())
                return True
            except queue.Full:
                blocked = True
                self.metrics['backpressure_timeouts'] += 1
        return False

    def _produce(self):
        iterator = self.source.iter_records(cancel=self.stop.is_set)
        try:
            for record in iterator:
                if self.stop.is_set(): break
                window = self.builder.push(record)
                self.metrics['records_produced'] += 1
                packet = SourcePacket(record, window, time.perf_counter_ns(), dict(self.source.metrics))
                if not self._put(packet):
                    self.metrics['cancelled_before_enqueue'] += 1
                    return
                self.metrics['records_enqueued'] += 1
            if not self.stop.is_set():
                self.metrics['producer_completed'] = bool(self.source.metrics.get('completed_full_stream'))
                self._put(('end', None))
        except BaseException as exc:
            self.metrics['producer_failed'] = True
            self._put(('error', exc))
        finally:
            iterator.close()

    def __iter__(self):
        while not self.stop.is_set() and not self.interrupted():
            try:
                item = self.queue.get(timeout=.05)
            except queue.Empty:
                continue
            if isinstance(item, SourcePacket):
                self.metrics['records_dequeued'] += 1
                yield item
            elif item[0] == 'error':
                raise item[1]
            else:
                return

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('Raw source producer did not stop; refusing silent background processing')
        self.metrics['queued_records_at_shutdown'] = sum(isinstance(p, SourcePacket) for p in list(self.queue.queue))
        return dict(self.metrics)
