"""Bounded online display analytics; no model inference or device access."""
from collections import deque
import math

from .ps_stream_display import PsStreamDisplay

BACKEND = 'board_socket_npu'
CAPACITY = 1000
STAGES = ('record_DMA', 'fpga_window', 'npu_wait', 'output_DMA', 'rearm')


def softmax_scores(logits):
    if len(logits) != 3 or not all(math.isfinite(x) for x in logits):
        raise ValueError('Three finite logits required')
    shift = max(logits)
    values = [math.exp(x-shift) for x in logits]
    total = sum(values)
    return [x/total for x in values]


def quality_metrics(pairs):
    """Rows=actual, columns=predicted; macro F1 fixes all three labels."""
    matrix = [[0]*3 for _ in range(3)]
    for actual, predicted in pairs:
        matrix[actual][predicted] += 1
    actuals = [sum(row) for row in matrix]
    predicted = [sum(row[c] for row in matrix) for c in range(3)]
    count = sum(actuals)
    recalls = [matrix[c][c]/actuals[c] for c in range(3) if actuals[c]]
    f1 = [2*matrix[c][c]/(actuals[c]+predicted[c]) if actuals[c]+predicted[c] else 0 for c in range(3)]
    return dict(count=count, confusion_actual_by_predicted=matrix,
        actual_distribution=actuals, prediction_distribution=predicted,
        accuracy=sum(matrix[c][c] for c in range(3))/count if count else None,
        balanced_accuracy=sum(recalls)/len(recalls) if recalls else None,
        balanced_supported_classes=len(recalls), macro_f1=sum(f1)/3 if count else None,
        majority_baseline=max(actuals)/count if count else None,
        missing_actual_classes=[c for c in range(3) if not actuals[c]],
        fixed_class_order=['down', 'flat', 'up'])


def latency_metrics(values):
    samples = sorted(values)
    if not samples:
        return dict(count=0, mean_ns=None, p50_ns=None, p95_ns=None, p99_ns=None, max_ns=None)
    def quantile(q):
        position = (len(samples)-1)*q
        lo = int(position); hi = min(lo+1, len(samples)-1)
        return samples[lo]+(samples[hi]-samples[lo])*(position-lo)
    return dict(count=len(samples), mean_ns=sum(samples)/len(samples),
                p50_ns=quantile(.5), p95_ns=quantile(.95), p99_ns=quantile(.99), max_ns=samples[-1])


class MonitoringDisplay(PsStreamDisplay):
    def __init__(self, *args, monitor_capacity=CAPACITY, **kwargs):
        if monitor_capacity < 1:
            raise ValueError('Positive analytics capacity required')
        self.monitor_capacity = monitor_capacity
        self.resolved_pairs = deque(maxlen=monitor_capacity)
        self.latencies = {key: deque(maxlen=monitor_capacity) for key in ('npu', 'board', *STAGES)}
        self.last_resolved_id = 0
        self.last_prediction_id = 0
        self.last_latency_row = None
        super().__init__(*args, **kwargs)

    def consume(self, event):
        if event['type'] == 'prediction':
            window = event['window_id']
            if window <= self.last_prediction_id:
                raise ValueError('Monitor requires strictly increasing prediction IDs')
            self.last_prediction_id = window
            if not event.get('startup_included', False):
                self.latencies['npu'].append(event['outputs'][BACKEND]['elapsed_ns'])
                if event.get('board_pipeline_ns') is not None:
                    self.latencies['board'].append(event['board_pipeline_ns'])
                    self.last_latency_row = window
                for key in STAGES:
                    value = (event.get('stages_ns') or {}).get(key)
                    if value is not None:
                        if not math.isfinite(value) or value < 0:
                            raise ValueError('Invalid native stage timing')
                        self.latencies[key].append(value)
        elif event['type'] == 'outcome' and event.get('status') == 'resolved':
            if event['window_id'] <= self.last_resolved_id:
                raise ValueError('Duplicate or reordered resolved outcome')
            self.last_resolved_id = event['window_id']
            self.resolved_pairs.append((event['actual_class'], event['predicted_classes'][BACKEND]))
        return super().consume(event)

    def snapshot(self):
        result = super().snapshot()
        result['monitor'] = dict(capacity=self.monitor_capacity,
            quality=quality_metrics(self.resolved_pairs),
            latency={key: latency_metrics(values) for key, values in self.latencies.items()},
            last_latency_window=self.last_latency_row,
            timing_scope='Most recent hot predictions; startup excluded; board excludes PC/network',
            labels_scope='Causal endpoint midpoint H50, zero threshold; not original training labels')
        return result
