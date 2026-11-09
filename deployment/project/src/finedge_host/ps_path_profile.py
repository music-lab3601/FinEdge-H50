"""Bounded, explicit control-path measurement; no SDK, network or file I/O."""
from __future__ import annotations

import time
import re

from .ps_stream_server import BoundedDiagnostics

SCHEMA = 'psw-stream-pathopt-v1'
PROFILES = ('strict-p0-v1', 'control-p1a-v1')
PROFILE_KEYS = {'name', 'measurement', 'sample_stride'}
BUCKETS = (1000, 4000, 16000, 64000, 256000, 1024000, 4096000,
           16384000, 65536000, 262144000, 1048576000, 4194304000)
BATCH_PHASES = ('prepare_image', 'write_and_precheck', 'descriptor',
    'issue_to_terminal', 'post_read_and_validate', 'consume_all',
    'final_read_verify', 'rearm')


def profile_config(name, measurement='phases', sample_stride=64):
    value = dict(name=name, measurement=measurement, sample_stride=sample_stride)
    validate_profile(value)
    return value


def validate_profile(value):
    if (type(value) is not dict or set(value) != PROFILE_KEYS
            or value['name'] not in PROFILES or value['measurement'] not in ('off', 'phases', 'sampled')
            or type(value['sample_stride']) is not int or value['sample_stride'] not in (64, 128, 256, 1024)):
        raise ValueError('Explicit supported path profile/measurement/stride required')


def validate_candidate_release(cfg, bundle_sha, release, release_sha, confirmation,
                               *, user_authorization=None, user_authorization_sha=None):
    """Exact-bundle permission plus fresh physical record; no device operation.

    An explicit human waiver is recorded separately from independent review.
    Neither authorization route can supply the required physical event.
    """
    is_sha = lambda v: isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) is not None
    reviewed=(release.get('authorization_mode','independent_review')=='independent_review'
              and release.get('independent_review_passed') is True)
    direct=(release.get('authorization_mode')=='direct_user'
            and release.get('independent_review_passed') is False
            and type(user_authorization) is dict
            and user_authorization.get('schema')=='psw-pathopt-direct-user-v1'
            and user_authorization.get('source')=='human_user_current_session'
            and user_authorization.get('user_authorized_execution') is True
            and user_authorization.get('independent_review_passed') is False
            and user_authorization.get('clarification_reply')=='无需审阅'
            and is_sha(user_authorization_sha)
            and release.get('user_authorization_sha256')==user_authorization_sha)
    run_windows=release.get('run_windows',10000)
    phase=release.get('experiment_phase','performance')
    if (release.get('schema') != 'psw-pathopt-execution-v1'
            or not (reviewed or direct)
            or release.get('device_experiment_approved') is not True
            or release.get('bundle_sha256') != bundle_sha or not is_sha(bundle_sha)
            or release.get('path_profile') != cfg['path_profile']
            or release.get('expected_BOOT_sha256') != cfg['expected_BOOT_sha256']
            or release.get('max_windows') != 10000 or release.get('preserve_full_checks') is not True
            or type(run_windows) is not int or run_windows not in (100,10000)
            or phase not in ('functionality','performance')
            or (phase=='functionality' and run_windows!=100)
            or not re.fullmatch('[A-Za-z0-9_]{1,64}', release.get('experiment_id', ''))
            or not release.get('previous_boot_id') or not is_sha(release.get('previous_result_sha256'))):
        raise ValueError('Exact-bundle P0/P1a release requires independent review or recorded direct user authorization')
    if (confirmation.get('source') != 'human_user_current_session'
            or confirmation.get('physical_power_removed') is not True
            or confirmation.get('linux_boot_complete') is not True
            or confirmation.get('experiment_id') != release['experiment_id']
            or confirmation.get('release_sha256') != release_sha or not is_sha(release_sha)
            or confirmation.get('bundle_sha256') != bundle_sha
            or confirmation.get('previous_boot_id') != release['previous_boot_id']
            or confirmation.get('previous_result_sha256') != release['previous_result_sha256']
            or not confirmation.get('verbatim_reply', '').strip()):
        raise ValueError('Fresh actual physical confirmation bound to candidate release required')
    return dict(experiment_id=release['experiment_id'], release_sha256=release_sha,
                previous_boot_id=release['previous_boot_id'], previous_result_sha256=release['previous_result_sha256'],
                authorization_mode='independent_review' if reviewed else 'direct_user',
                independent_review_passed=reviewed,run_windows=run_windows,experiment_phase=phase,
                user_authorization_sha256=user_authorization_sha if direct else None)


class Stat:
    __slots__ = ('count', 'total', 'low', 'high', 'hist')

    def __init__(self):
        self.count = self.total = self.high = 0
        self.low = None
        self.hist = [0] * (len(BUCKETS) + 1)

    def add(self, ns):
        if type(ns) is not int or ns < 0 or self.count == 2**64-1:
            raise ValueError('Invalid/overflowing bounded duration counter')
        self.count += 1
        self.total += ns
        if self.low is None or ns < self.low: self.low = ns
        if ns > self.high: self.high = ns
        i = 0
        while i < len(BUCKETS) and ns > BUCKETS[i]: i += 1
        self.hist[i] += 1

    def snapshot(self):
        return dict(count=self.count, total_ns=self.total, min_ns=self.low,
                    max_ns=self.high, histogram=list(self.hist))


class Intervals:
    """One bounded transaction. Adjacent marks close an exact partition."""
    def __init__(self, measurement, first):
        self.measurement = measurement
        self.enabled = measurement.enabled
        self.phase = first
        self.begin = self.last = measurement.clock() if self.enabled else 0
        self.parts = {}
        self.closed = False

    def mark(self, next_phase):
        if self.closed: raise RuntimeError('Timing partition is closed')
        if self.enabled:
            now = self.measurement.clock()
            ns = now - self.last
            if ns < 0: raise ValueError('Monotonic phase clock reversed')
            self.parts[self.phase] = self.parts.get(self.phase, 0) + ns
            self.last = now
        self.phase = next_phase

    def finish(self):
        if self.closed: raise RuntimeError('Timing partition is closed')
        self.mark('closed')
        self.closed = True
        total = self.last-self.begin
        if total != sum(self.parts.values()): raise AssertionError('Timing partition does not close')
        return dict(enabled=self.enabled, total_ns=total, phases_ns=dict(self.parts))


class Measurements:
    """Fixed counters/aggregates, one last batch, one first failure; no history."""
    def __init__(self, config, clock=time.perf_counter_ns):
        validate_profile(config)
        self.config, self.clock = dict(config), clock
        self.enabled = config['measurement'] != 'off'
        self.wrapper_enabled = config['measurement'] == 'sampled'
        self.seen = [0] * 5
        self.samples = [Stat() for _ in range(5)]
        self.sampled_bodies = [Stat() for _ in range(5)]
        self.sampled_residuals = [Stat() for _ in range(5)]
        self.failed_samples = [0] * 5
        self.phase_stats = {name: Stat() for name in BATCH_PHASES}
        self.status = Stat()
        self.status_scans = self.polls = self.sleeps = 0
        self.memory_reads = self.memory_writes = self.read_bytes = self.write_bytes = 0
        self.scalar_returns_not_retained = 0
        self.forward_sdk_ns = self.hardop_nested_sdk_ns = 0
        self.forward_calls = self.forward_child_ns = 0
        self.in_forward = False
        self.last_batch = self.first_failure = None

    @staticmethod
    def kind(stage):
        if stage.startswith('psw.reg.read.'): return 0
        if stage.startswith('psw.reg.write.'): return 1
        if stage == 'ps_read': return 2
        if stage == 'ps_write': return 3
        return 4

    def sample_begin(self, stage):
        k = self.kind(stage)
        if self.seen[k] == 2**64-1: raise OverflowError('API attempt count exhausted')
        self.seen[k] += 1
        if self.enabled and (self.seen[k]-1) % self.config['sample_stride'] == 0:
            return k, self.clock()
        return None

    def sample_end(self, token, successful=True, sdk_ns=None):
        if token is not None:
            ns = self.clock()-token[1]
            if successful:
                self.samples[token[0]].add(ns)
                if sdk_ns is not None:
                    if type(sdk_ns) is not int or not 0 <= sdk_ns <= ns:
                        raise ValueError('Sampled SDK body exceeds wrapper interval')
                    self.sampled_bodies[token[0]].add(sdk_ns)
                    self.sampled_residuals[token[0]].add(ns-sdk_ns)
            else:
                self.failed_samples[token[0]] += 1
                if self.first_failure is not None:
                    self.first_failure.setdefault('sampled_wrapper_failure_elapsed_ns', ns)

    def api(self, stage, ns):
        # Only the forward body contains these nested callback SDK calls.
        if self.in_forward and stage.startswith('hardop.'):
            self.forward_child_ns += ns
        if stage == 'npu.forward':
            if self.forward_child_ns > ns: raise ValueError('Nested SDK timing exceeds forward body')
            self.forward_calls += 1
            self.forward_sdk_ns += ns
            self.hardop_nested_sdk_ns += self.forward_child_ns

    def complete_batch(self, value):
        self.last_batch = value
        if self.enabled:
            for name, ns in value['phases_ns'].items(): self.phase_stats[name].add(ns)

    def fail(self, stage, exc, **context):
        if self.first_failure is None:
            self.first_failure = dict(stage=stage, type=type(exc).__name__,
                error=str(exc)[:2048], context=context)

    def snapshot(self):
        return dict(profile=dict(self.config), enabled=self.enabled,wrapper_sampling_enabled=self.wrapper_enabled,
            wrapper_attempt_counts=dict(zip(('reg_read','reg_write','ps_read','ps_write','other'), self.seen)),
            wrapper_sampled_stats={name: stat.snapshot() for name, stat in zip(
                ('reg_read','reg_write','ps_read','ps_write','other'), self.samples)},
            sampled_SDK_body_stats={name: stat.snapshot() for name, stat in zip(
                ('reg_read','reg_write','ps_read','ps_write','other'), self.sampled_bodies)},
            sampled_wrapper_residual_stats={name: stat.snapshot() for name, stat in zip(
                ('reg_read','reg_write','ps_read','ps_write','other'), self.sampled_residuals)},
            sampled_residual_scope='wrapper total minus same-call SDK body; includes timer/hold/diagnostic/checkpoint/context overhead; nested calls overlap',
            wrapper_failed_sample_counts=dict(zip(('reg_read','reg_write','ps_read','ps_write','other'),self.failed_samples)),
            wrapper_sampling='deterministic first then every stride by API type; samples only, no exact extrapolated total',
            histogram_upper_bounds_ns=list(BUCKETS), phase_stats={k:v.snapshot() for k,v in self.phase_stats.items()},
            status_scans=self.status_scans, status_scan_time=self.status.snapshot(), polls=self.polls, sleeps=self.sleeps,
            memory_reads=self.memory_reads, memory_writes=self.memory_writes,
            memory_read_requested_bytes=self.read_bytes, memory_write_requested_bytes=self.write_bytes,
            scalar_returns_not_retained=self.scalar_returns_not_retained,
            forward_successful_calls=self.forward_calls,forward_body_ns=self.forward_sdk_ns, hardop_nested_body_ns=self.hardop_nested_sdk_ns,
            forward_body_excluding_nested_SDK_ns=self.forward_sdk_ns-self.hardop_nested_sdk_ns,
            nested_timing_policy='HardOp SDK times are children of forward; never add them to forward',
            last_batch=self.last_batch, first_failure=self.first_failure, retained_history=0)


class FixedDiagnostics(BoundedDiagnostics):
    """Same successful API statistics, indexed storage; JSON built at snapshot."""
    def __init__(self, max_stages=128):
        super().__init__(max_stages)
        self.names, self.indices, self.values = [], {}, [[0, 0, None, 0] for _ in range(max_stages)]

    def site(self, name):
        index = self.indices.get(name)
        if index is None:
            if len(self.names) == self.max_stages: return -1
            index = len(self.names)
            self.indices[name] = index
            self.names.append(name)
        return index

    def record_api(self, index, ns):
        if type(ns) is not int or ns < 0: raise ValueError('Nonnegative SDK duration required')
        self.api_events += 1
        if index < 0:
            stat = self.other_api
            stat['count'] += 1; stat['total_ns'] += ns
            stat['max_ns'] = max(stat['max_ns'], ns)
            stat['min_ns'] = ns if stat['min_ns'] is None else min(stat['min_ns'], ns)
            return
        value = self.values[index]
        value[0] += 1; value[1] += ns
        if value[2] is None or ns < value[2]: value[2] = ns
        if ns > value[3]: value[3] = ns

    def __call__(self, event):
        if type(event) is dict and event.get('kind') == 'api':
            self.record_api(self.site(event['stage']), event['ns'])
        else: super().__call__(event)

    def snapshot(self):
        return dict(api_events=self.api_events, predictions=self.predictions, other_events=self.other_events,
            stages={name:dict(count=v[0],total_ns=v[1],min_ns=v[2],max_ns=v[3])
                    for name,v in zip(self.names,self.values)}, other_api=dict(self.other_api),
            retained_events=0, stage_bound=self.max_stages)
