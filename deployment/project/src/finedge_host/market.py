"""Observed midpoint candles and causal H50 evaluation; no future-file reads."""
from collections import deque


HORIZON = 50
TARGET_ID = 'endpoint-midpoint-H50-threshold0-v1'
INTERVALS = {'10毫秒': 10_000_000, '100毫秒': 100_000_000,
             '1秒': 1_000_000_000, '5秒': 5_000_000_000, '1分钟': 60_000_000_000}


def exchange_time(ns, milliseconds=True):
    seconds, nanos = divmod(int(ns), 1_000_000_000)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}' + (f'.{nanos//1_000_000:03}' if milliseconds else '')


def segment_key(meta, round_index):
    return (round_index, *(meta[k] for k in ('date', 'symbol', 'snippet_id',
                                           'research_segment_id', 'native_segment_id')))


class CandleSeries:
    """OHLC of observed eligible book updates, in exact half-price units.

    Sparse buckets are left empty. The last bucket is incomplete until an
    observation in a later bucket/segment arrives. No synthetic volume.
    """
    def __init__(self, interval_ns, capacity=120):
        if interval_ns <= 0:
            raise ValueError('Positive candle interval required')
        self.interval_ns = interval_ns
        self.bars = deque(maxlen=capacity)

    def push(self, meta, mid2, round_index):
        timestamp = int(meta['timestamp_ns_since_exchange_midnight'])
        key = segment_key(meta, round_index)
        bucket = timestamp // self.interval_ns * self.interval_ns
        if self.bars and self.bars[-1]['segment_key'] == key:
            if timestamp < self.bars[-1]['last_timestamp_ns']:
                raise ValueError('Market time regressed')
        if not self.bars or (self.bars[-1]['bucket_ns'], self.bars[-1]['segment_key']) != (bucket, key):
            if self.bars:
                self.bars[-1]['closed'] = True
            self.bars.append({'bucket_ns': bucket, 'segment_key': key,
                              'date': meta['date'], 'symbol': meta['symbol'],
                              'open_mid2': mid2, 'high_mid2': mid2, 'low_mid2': mid2,
                              'close_mid2': mid2, 'samples': 1, 'closed': False,
                              'first_timestamp_ns': timestamp, 'last_timestamp_ns': timestamp})
        else:
            bar = self.bars[-1]
            bar.update(high_mid2=max(bar['high_mid2'], mid2), low_mid2=min(bar['low_mid2'], mid2),
                       close_mid2=mid2, samples=bar['samples']+1, last_timestamp_ns=timestamp)


class DelayedEvaluation:
    """Only resolve a prediction when its same-segment t+50 row is observed."""
    def __init__(self):
        self.pending = deque()
        self.current_key = None
        self.current_meta = None
        self.current_mid2 = None
        self.resolved = self.unresolved = 0
        self.backends = {}

    def observe(self, meta, mid2, round_index):
        key = segment_key(meta, round_index)
        events = []
        if self.current_key is not None and key != self.current_key:
            events.extend(self.finish('segment_boundary'))
        if self.current_key == key and self.current_meta is not None:
            if (meta['source_record_row_0based'] != self.current_meta['source_record_row_0based']+1
                    or meta['timestamp_ns_since_exchange_midnight'] < self.current_meta['timestamp_ns_since_exchange_midnight']):
                raise ValueError('Evaluation requires consecutive same-segment observed records')
        self.current_key, self.current_meta, self.current_mid2 = key, dict(meta), int(mid2)
        row = meta['source_record_row_0based']
        while self.pending and self.pending[0]['target_source_row'] <= row:
            prediction = self.pending.popleft()
            if prediction['target_source_row'] != row:
                raise ValueError('Evaluation skipped a target row')
            actual = 2 if mid2 > prediction['origin_mid2'] else 0 if mid2 < prediction['origin_mid2'] else 1
            matches = {name: cls == actual for name, cls in prediction['predicted_classes'].items()}
            for name, cls in prediction['predicted_classes'].items():
                metrics = self.backends.setdefault(name, {'count': 0, 'correct': 0,
                                                          'confusion_actual_by_predicted': [[0]*3 for _ in range(3)]})
                metrics['count'] += 1
                metrics['correct'] += int(matches[name])
                metrics['confusion_actual_by_predicted'][actual][cls] += 1
            self.resolved += 1
            events.append({**prediction, 'status': 'resolved', 'actual_class': actual,
                           'actual_mid2': int(mid2), 'actual_meta': dict(meta), 'matches': matches,
                           'midpoint_change_usd': (int(mid2)-prediction['origin_mid2'])/20000})
        return events

    def track(self, window_id, round_index, meta, outputs):
        if self.current_key != segment_key(meta, round_index) or meta != self.current_meta:
            raise ValueError('Prediction must refer to the latest observed record')
        classes = {name: int(p['class_index']) for name, p in outputs.items()}
        if any(cls not in (0, 1, 2) for cls in classes.values()):
            raise ValueError('Unknown class')
        self.pending.append({'window_id': window_id, 'round': round_index, 'target_id': TARGET_ID,
                             'origin_meta': dict(meta), 'origin_mid2': self.current_mid2,
                             'target_source_row': meta['source_record_row_0based']+HORIZON,
                             'predicted_classes': classes})

    def finish(self, reason):
        events = [{**p, 'status': 'unresolved', 'reason': reason} for p in self.pending]
        self.unresolved += len(events)
        self.pending.clear()
        self.current_key = self.current_meta = self.current_mid2 = None
        return events

    def summary(self):
        # Copy nested metrics: queued GUI events must remain immutable snapshots.
        return {'target_id': TARGET_ID, 'horizon_records': HORIZON, 'resolved': self.resolved,
                'unresolved': self.unresolved, 'pending': len(self.pending),
                'backends': {name: {**m, 'accuracy': m['correct']/m['count'],
                                   'confusion_actual_by_predicted': [r[:] for r in m['confusion_actual_by_predicted']]}
                             for name, m in self.backends.items()}}
