"""Bounded, headless projection of losslessly recorded PSW stream events.

The runner computes H50 and writes every result before calling consume().
Only complete display snapshots may be coalesced; every market observation
updates all OHLC intervals first. No SDK, socket or Tk dependency lives here.
"""
from collections import OrderedDict, deque
from copy import deepcopy
import math
import time

from .market import CandleSeries, INTERVALS, segment_key


EXPECTED_WINDOWS = 1126606
TERMINAL = frozenset(('completed', 'partial', 'passed', 'passed_full_day', 'stopped', 'failed', 'incomplete'))


class PsStreamDisplay:
    def __init__(self, mailbox, refresh_hz=10.0, clock=time.monotonic,
                 history_capacity=150, observation_capacity=20000, candle_capacity=120):
        if not math.isfinite(refresh_hz) or refresh_hz <= 0:
            raise ValueError('Positive finite display refresh rate required')
        if min(history_capacity, observation_capacity, candle_capacity) < 1:
            raise ValueError('Display capacities must be positive')
        self.mailbox, self.clock = mailbox, clock
        self.period = 1.0 / refresh_hz
        self.history_capacity = history_capacity
        self.series = {label: CandleSeries(ns, candle_capacity) for label, ns in INTERVALS.items()}
        self.observations = deque(maxlen=observation_capacity)
        self.predictions = OrderedDict()
        self.latest_market = None
        self.latest_prediction_id = self.latest_resolved_id = None
        self.current_segment = None
        self.evaluation = {}
        self.stats = {'records': 0, 'windows': 0}
        self.memory = {}
        self.state = {'status': 'initializing', 'phase': 'preparing', 'message': '',
                      'error': None, 'directory': None, 'expected_windows': EXPECTED_WINDOWS,
                      'execution_kind': 'unknown', 'full_day_complete': False,
                      'target_windows_per_second': 0, 'observed_windows_per_second': None,
                      'elapsed_seconds': 0}
        self.events_consumed = self.snapshots_published = 0
        self.last_publish = None
        self.dirty = False

    def consume(self, event):
        kind = event['type']
        self.events_consumed += 1
        if kind == 'market':
            meta, round_index = event['last'], event.get('round', 1)
            key = segment_key(meta, round_index)
            if self.current_segment is not None and key != self.current_segment:
                for series in self.series.values():
                    series.bars.clear()
                self.observations.clear()
            self.current_segment = key
            mid2 = int(event['mid2'])
            for series in self.series.values():
                series.push(meta, mid2, round_index)
            # Both leaves are builtin immutable ints. Snapshots may share
            # these immutable tuples, but never the mutable deque/list itself.
            self.observations.append((int(meta['timestamp_ns_since_exchange_midnight']), mid2))
            self.latest_market = {'last': deepcopy(meta), 'round': round_index, 'mid2': mid2}
        elif kind == 'prediction':
            # Keep only display fields; raw protocol frames, tensors and arenas
            # must never accidentally become part of the live GUI cache.
            keys = ('run_id', 'window_id', 'round', 'first', 'last', 'origin_mid2', 'book',
                    'outputs', 'cpu_npu_max_abs_error', 'cpu_npu_class_match',
                    'npu_max_abs_error', 'preprocess_ns', 'pack_ns', 'batch_roundtrip_ns',
                    'target_source_row', 'input_sha256','board_pipeline_ns','startup_included',
                    'model_mode','integrity_mode')
            row = {key: deepcopy(event[key]) for key in keys if key in event}
            row.setdefault('round', 1)
            self.predictions[row['window_id']] = row
            self.latest_prediction_id = row['window_id']
            while len(self.predictions) > self.history_capacity:
                self.predictions.popitem(last=False)
        elif kind == 'outcome':
            row = self.predictions.get(event['window_id'])
            if row is not None:
                row['outcome'] = {key: deepcopy(value) for key, value in event.items()
                                  if key not in ('evaluation', 'type')}
            if event['status'] == 'resolved':
                self.latest_resolved_id = event['window_id']
        elif kind == 'ready':
            if self.state['status'] not in TERMINAL:
                self.state.update(status='running', phase='building_book')
        elif kind == 'phase':
            if self.state['status'] not in TERMINAL:
                self.state['phase'] = event.get('phase', self.state['phase'])
        elif kind == 'error':
            self.state.update(status='failed', phase='failed', error=event.get('message', 'Unknown error'))
        elif kind == 'finished':
            summary = event.get('summary', {})
            status = summary.get('status', event.get('status', 'incomplete'))
            # A late finished or progress frame cannot erase a recorded error.
            if self.state['status'] != 'failed':
                self.state.update(status=status, phase=status)
            if summary.get('error'):
                self.state['error'] = str(summary['error'])
            self.stats.update(deepcopy(summary.get('stats', {})))
            self.evaluation = deepcopy(summary.get('market_evaluation', self.evaluation))
            self.memory.update(deepcopy(summary.get('memory', {})))
            for key in ('elapsed_seconds', 'observed_windows_per_second', 'expected_windows',
                        'execution_kind', 'full_day_complete'):
                if key in summary:
                    self.state[key] = summary[key]
        if 'evaluation' in event:
            self.evaluation = deepcopy(event['evaluation'])
        self.stats.update(deepcopy(event.get('stats', {})))
        self.memory.update(deepcopy(event.get('memory', {})))
        for key in ('directory', 'expected_windows', 'target_windows_per_second',
                    'observed_windows_per_second', 'elapsed_seconds', 'execution_kind'):
            if key in event:
                self.state[key] = event[key]
        if 'message' in event and (kind in ('error', 'finished') or self.state['status'] not in TERMINAL):
            self.state['message'] = str(event['message'])[:2000]
        if kind == 'prediction' and self.state['status'] not in TERMINAL:
            self.state['phase'] = 'running'
        self.dirty = True
        return self.publish(force=kind in ('ready', 'phase', 'error', 'finished'))

    def snapshot(self):
        """Owned bounded snapshot, useful for tests and GUI-independent inspection."""
        owned = deepcopy({'type': 'ps_stream_snapshot', 'schema': 1, **self.state,
                         'stats': self.stats, 'memory': self.memory, 'evaluation': self.evaluation,
                         'market': self.latest_market,
                         'series': {key: list(series.bars) for key, series in self.series.items()},
                         'recent_predictions': list(self.predictions.values()),
                         'latest_prediction_id': self.latest_prediction_id,
                         'latest_resolved_id': self.latest_resolved_id,
                         'events_consumed': self.events_consumed,
                         'snapshots_published': self.snapshots_published,
                         'capacities': {'predictions': self.history_capacity,
                                        'observations': self.observations.maxlen,
                                        'candles_per_interval': next(iter(self.series.values())).bars.maxlen}})
        owned['observations'] = list(self.observations)
        return owned

    def publish(self, force=False):
        now = self.clock()
        if not self.dirty or (not force and self.last_publish is not None and now-self.last_publish < self.period):
            return False
        self.snapshots_published += 1
        self.mailbox.publish_owned(self.snapshot())
        # A slow copy, preemption or a busy GUI must not consume the next
        # refresh period and trigger a fresh publication for every event.
        self.last_publish = self.clock()
        self.dirty = False
        return True
