"""Closed, finite paired-input benchmark; caller owns final audit and saving.

The two arms use the same accepted owner, protected slot reads and original
Cast. This module never deploys, signals, resets, retries or frees a device.
"""
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import re
import time

import numpy as np

from software.runtime.delivery import read_json, sha256
from experiments.reference.runtime.udw_readback_owner import BufferedTrace
from experiments.reference.runtime.udw_readback_probe import packet_files as readback_packet_files
from experiments.reference.runtime.udw_readback_probe import known_sdk_input, validate_measurement_plan
from experiments.reference.runtime.udw_copy_probe import DEVICE_VERSIONS, FINDINGS
from experiments.reference.runtime.udw_shared_probe import golden_records
from experiments.reference.runtime.udw_e_preflight import capture_boot
from experiments.reference.runtime.udw_probe_stop import ProbeStop


SCOPE = dict(schema='readback-paired-benchmark-v1', known_windows=1,
    warmup_windows=10, timed_windows=12, FPGA_records=2200, FPGA_windows=22,
    NPU_calls=23, arena_bytes=2097152, main_used_bytes=923648,
    known_allocation_bytes=65536, quota_bytes=1048576,
    trace_capacity=8192, trace_policy='bounded_RAM_then_flush',
    reuse=False, reset=False, retry=False, control='axi_registers', Cast='preserved',
    separate_physical_boots_per_arm=True, predecessor_chain_required=True,
    known_input_index=0, warmup_indices=[0]*10, timed_indices=list(range(1, 13)),
    primary_interval='submit_local entry through return',
    performance_runner_implemented=True)

_HASH = re.compile(r'[0-9a-f]{64}')
_EXTRA = {'src/finedge_host/udw_readback_bench.py', 'scripts/udw_readback_bench_check.py',
          'bench/inputs.f32', 'bench/expected.json', 'bench/plan.json', 'bench/anchor.json'}


def packet_fingerprint(review):
    return hashlib.sha256(json.dumps(review['packet_hashes'], sort_keys=True,
        separators=(',', ':')).encode('utf-8')).hexdigest()


def packet_files(root, review):
    if not _EXTRA <= set(review.get('packet_hashes', {})):
        raise ValueError('Incomplete paired benchmark packet')
    readback_packet_files(root, review)


def load_inputs(root):
    """All file reads, reference checks and synthetic WDM building precede init."""
    raw = (root/'bench/inputs.f32').read_bytes()
    if len(raw) != 32*16000 or not np.isfinite(np.frombuffer(raw, dtype='<f4')).all():
        raise ValueError('Exactly 32 finite frozen float32 windows required')
    tensors = tuple(raw[i*16000:(i+1)*16000] for i in range(32))
    hashes = [hashlib.sha256(t).hexdigest() for t in tensors]
    expected = read_json(root/'bench/expected.json')['logits']
    values = np.asarray(expected, dtype=np.float64)
    if values.shape != (32, 3) or not np.isfinite(values).all():
        raise ValueError('32 finite same-compiled-model Host references required')
    plan = read_json(root/'bench/plan.json')
    validate_measurement_plan(plan)
    if plan != read_json(root/'probe/measurement_plan.json'):
        raise ValueError('Frozen non-executable measurement plan must be preserved unchanged')
    models = {'json': sha256(root/'artifacts/icraft/tcn_ed4_ZG.json'),
              'raw': sha256(root/'artifacts/icraft/tcn_ed4_ZG.raw')}
    if plan['window_hashes'] != hashes or plan['model_hashes'] != models:
        raise ValueError('Benchmark input/model differs from frozen plan')
    if ((root/'probe/golden.f32').read_bytes() != tensors[0]
            or read_json(root/'probe/expected.json')['logits'] != expected[0]):
        raise ValueError('Known input/reference differs from accepted golden0')
    anchor = read_json(root/'bench/anchor.json')
    if (anchor.get('RB01_independently_accepted') is not True
            or anchor.get('label') != 'RB01_20261025'
            or anchor.get('arm') != 'host_readback'
            or not isinstance(anchor.get('boot_id'), str) or not anchor['boot_id']
            or not _HASH.fullmatch(anchor.get('result_sha256', ''))
            or not _HASH.fullmatch(anchor.get('expected_BOOT_sha256', ''))
            or anchor.get('expected_device_versions') != DEVICE_VERSIONS
            or anchor.get('model_hashes') != models):
        raise ValueError('Accepted RB01 identity/model anchor required')
    built = tuple(golden_records(t) for t in tensors)
    return dict(tensors=tensors, payloads=tuple(v[0] for v in built),
        endpoints=tuple((v[1], v[2]) for v in built), window_hashes=hashes,
        expected=values.tolist(), expected_sha256=sha256(root/'bench/expected.json'),
        inputs_sha256=hashlib.sha256(raw).hexdigest(), plan=plan, anchor=anchor,
        model_hashes=models)


def _validate_predecessor(review, predecessor, inputs):
    if (not isinstance(predecessor, dict) or not isinstance(predecessor.get('result'), dict)
            or not _HASH.fullmatch(predecessor.get('sha256', ''))):
        raise ValueError('Actual saved predecessor result and byte hash required')
    result, identity = predecessor['result'], review.get('predecessor', {})
    if (set(identity) != {'label', 'boot_id', 'result_sha256', 'arm'}
            or identity.get('result_sha256') != predecessor['sha256']
            or identity.get('label') != result.get('label')
            or identity.get('boot_id') != result.get('boot_id')):
        raise ValueError('Predecessor file identity differs from independent release')
    anchor = inputs['anchor']
    if review['arm'] == 'device_copy':
        if (identity != {k: anchor[k] for k in ('label','boot_id','result_sha256','arm')}
                or result.get('actual_NPU_calls') != 2
                or result.get('actual_ARM_buffer_constructor_verified') is not True
                or result.get('actual_NPU_readback_path_verified') is not True):
            raise ValueError('A arm must follow independently accepted actual RB01')
    else:
        if (identity.get('arm') != 'device_copy' or result.get('arm') != 'device_copy'
                or result.get('schema') != SCOPE['schema']
                or result.get('status') != 'bounded_paired_benchmark_verified'
                or result.get('audit_passed') is not True or result.get('trace_flushed') is not True
                or result.get('NPU_calls') != 23 or result.get('FPGA_batches') != 22
                or result.get('main_used_bytes') != 923648
                or len(result.get('warmups', [])) != 10 or len(result.get('samples', [])) != 12
                or result.get('packet_fingerprint') != packet_fingerprint(review)
                or result.get('input_hashes') != inputs['window_hashes']
                or result.get('expected_sha256') != inputs['expected_sha256']
                or result.get('model_hashes') != inputs['model_hashes']
                or result.get('expected_BOOT_sha256') != anchor['expected_BOOT_sha256']
                or result.get('device_versions') != DEVICE_VERSIONS
                or result.get('boot_id') == anchor['boot_id']
                or [v.get('golden_index') for v in result['samples']] != list(range(1,13))
                or [v.get('golden_index') for v in result['warmups']] != [0]*10):
            raise ValueError('B arm requires the complete successful A from the same frozen packet')
        _validate_prior_rows(result, inputs)
    return dict(identity)


def validate_request(root, review, confirmation, label, review_sha256, predecessor):
    if not re.fullmatch(r'[A-Za-z0-9_]{1,40}', label):
        raise ValueError('Safe unique benchmark label required')
    if (review.get('device_probe_approved') is not True
            or review.get('independent_review_passed') is not True
            or not FINDINGS <= set(review.get('closed_findings', []))
            or review.get('scope') != SCOPE
            or review.get('arm') not in ('device_copy','host_readback')
            or review.get('expected_device_versions') != DEVICE_VERSIONS):
        raise ValueError('Independent paired benchmark review/scope is not approved')
    packet_files(root, review)
    inputs = load_inputs(root)
    if review.get('expected_BOOT_sha256') != inputs['anchor']['expected_BOOT_sha256']:
        raise ValueError('Benchmark BOOT differs from accepted anchor')
    previous = _validate_predecessor(review, predecessor, inputs)
    if (confirmation.get('source') != 'human_user_current_session'
            or confirmation.get('physical_power_removed') is not True
            or confirmation.get('linux_boot_complete') is not True
            or confirmation.get('label') != label or label == previous['label']
            or not confirmation.get('verbatim_reply','').strip()
            or not _HASH.fullmatch(review_sha256)
            or confirmation.get('review_sha256') != review_sha256
            or confirmation.get('previous_result_sha256') != previous['result_sha256']):
        raise ValueError('Fresh human power-off confirmation bound to this arm/release required')
    return previous


def _verify_prediction(value, inputs, index, mode, call_number):
    logits = np.asarray(value.get('logits'), dtype=np.float64)
    expected = np.asarray(inputs['expected'][index], dtype=np.float64)
    transfer = value.get('input_transfer', {})
    if (logits.shape != (3,) or not np.isfinite(logits).all()
            or not np.allclose(logits, expected, rtol=1e-3, atol=1e-4)
            or int(logits.argmax()) != int(expected.argmax())
            or value.get('class_index') != int(logits.argmax())
            or value.get('tensor_sha256') != inputs['window_hashes'][index]
            or value.get('input_mode') != mode or value.get('npu_call') != call_number
            or transfer.get('explicit_device_Tensor_to_calls') != int(mode == 'device_copy')
            or transfer.get('explicit_device_Tensor_to_logical_bytes') != (16000 if mode == 'device_copy' else 0)
            or transfer.get('documented_Host_buffer_copy_bytes') != (16000 if mode == 'host_readback' else 0)
            or transfer.get('Host_verify_read_API_bytes') != 16000):
        raise ValueError('Benchmark input/numerical/mode/call verification differs')
    return float(np.max(np.abs(logits-expected)))


def _identity(ledger, request, inputs, index, offset):
    keys = [k for k in ledger.views if k.request_id == request]
    if len(keys) != 1:
        raise ValueError('One retained lease per independent FPGA batch required')
    key = keys[0]; view = ledger.views[key]
    first, last = inputs['endpoints'][index]
    if (key.window_index != 0 or key.boot_id != ledger.boot_id
            or key.allocation_id != ledger.allocation_id or key.run_id != ledger.run_id
            or key.generation != ledger.generation or view['state'] != 'completed_retained'
            or tuple(view['first']) != first or tuple(view['last']) != last
            or view['slot_offset'] != offset or view['slot_size'] != 41984
            or view['tensor_offset'] != offset+25920
            or view['tensor_sha256'] != inputs['window_hashes'][index]):
        raise ValueError('Retained WDO/slot/allocation identity differs')
    return dict(lease=asdict(key), first=list(first), last=list(last), slot_offset=offset,
        slot_size=view['slot_size'], tensor_offset=view['tensor_offset'],
        allocation_base=ledger.base, allocation_bytes=ledger.size)


def _validate_prior_rows(result, inputs):
    """An approved saved A must still contain all auditable, valid raw rows."""
    known = result.get('known')
    if not isinstance(known, dict):
        raise ValueError('A predecessor has no complete known-input comparison')
    error = _verify_prediction(known.get('prediction', {}), inputs, 0, 'device_copy', 1)
    if (known.get('phase') != 'known' or known.get('ordinal') != 0
            or known.get('golden_index') != 0
            or known.get('expected_tensor_sha256') != inputs['window_hashes'][0]
            or known.get('slot_offset') != 0 or known.get('slot_size') != 41984
            or known.get('tensor_offset') != 25920 or known.get('allocation_bytes') != 65536
            or known.get('max_abs_error') != error
            or known.get('first') != list(inputs['endpoints'][0][0])
            or known.get('last') != list(inputs['endpoints'][0][1])):
        raise ValueError('A known-input row differs from fixed reference/slot')
    prior_lease = None; main_base = None
    rows = result['warmups']+result['samples']
    for batch, row in enumerate(rows, 1):
        phase, ordinal, index = ('warmup', batch, 0) if batch <= 10 else ('timed', batch-10, batch-10)
        offset=(batch-1)*41984
        raw=row.get('result', {}); pred=raw.get('predictions', [])
        if len(pred) != 1:
            raise ValueError('A row lacks its one original prediction')
        error=_verify_prediction(pred[0], inputs, index, 'device_copy', batch+1)
        first,last=inputs['endpoints'][index]
        lease=row.get('lease', {}); desc=raw.get('descriptor', {})
        primary=row.get('primary_elapsed_ns')
        if (row.get('phase') != phase or row.get('ordinal') != ordinal
                or row.get('golden_index') != index or row.get('request_id') != batch
                or row.get('primary_performance_sample') is not (phase == 'timed')
                or type(primary) is not int or primary < 0
                or row.get('expected_tensor_sha256') != inputs['window_hashes'][index]
                or row.get('max_abs_error') != error
                or row.get('first') != list(first) or row.get('last') != list(last)
                or pred[0].get('first') != list(first) or pred[0].get('last') != list(last)
                or row.get('slot_offset') != offset or row.get('slot_size') != 41984
                or row.get('tensor_offset') != offset+25920 or row.get('allocation_bytes') != 2097152
                or raw.get('records') != 100 or raw.get('request_id') != batch
                or raw.get('used') != batch*41984
                or raw.get('input_sha256') != hashlib.sha256(inputs['payloads'][index]).hexdigest()
                or lease.get('boot_id') != result['boot_id']
                or lease.get('request_id') != batch or lease.get('window_index') != 0
                or not isinstance(lease.get('allocation_id'),str) or not lease['allocation_id']
                or type(lease.get('run_id')) is not int or type(lease.get('generation')) is not int
                or type(row.get('allocation_base')) is not int
                or desc.get('run_id') != lease['run_id'] or desc.get('generation') != lease['generation']
                or desc.get('request_id') != batch
                or desc.get('input_base') != row['allocation_base']+offset+64
                or desc.get('output_base') != row['allocation_base']+offset+25792
                or desc.get('record_count') != 100 or desc.get('max_windows') != 1
                or desc.get('input_bytes') != 25600 or desc.get('output_capacity') != 16128):
            raise ValueError('A predecessor row identity/count/timing/reference differs')
        identity={k:lease[k] for k in ('boot_id','allocation_id','run_id','generation')}
        if prior_lease is not None and (identity != prior_lease or row['allocation_base'] != main_base):
            raise ValueError('A predecessor changed allocation within its arm')
        prior_lease,main_base=identity,row['allocation_base']
    klease=known.get('lease', {})
    if (klease.get('boot_id') != result['boot_id'] or klease.get('request_id') != 1
            or klease.get('window_index') != 0 or klease.get('run_id') != prior_lease['run_id']
            or klease.get('generation') != prior_lease['generation']
            or klease.get('allocation_id') == prior_lease['allocation_id']
            or type(known.get('allocation_base')) is not int
            or not (known['allocation_base']+65536 <= main_base
                    or main_base+2097152 <= known['allocation_base'])):
        raise ValueError('A predecessor known and main allocation identities differ/overlap')


def run_benchmark(root, review, confirmation, label, *, review_sha256, predecessor,
                  execute, owner_factory, record, stop=None, clock_ns=time.perf_counter_ns):
    """Return calls-completed state. Final audit, flush and file saving remain gated.

    ``record`` must append to RAM only. Reference checking and this callback are
    outside every primary interval. Callers keep ProbeStop and its owner parked
    on any failure, including failures while finalizing or saving the result.
    """
    stop = stop if stop is not None else ProbeStop()
    progress = dict(schema=SCOPE['schema'], label=label, arm=review.get('arm'),
        status='preparing', known=None, warmups=[], samples=[], timings_ns={},
        owner_retained=True, audit_passed=False, trace_flushed=False)
    stop.benchmark_progress = progress
    def emit(stage, value):
        stop.checkpoint(stage+'.before_record')
        record(stage, value)
        stop.checkpoint(stage+'.recorded')
    try:
        stop.checkpoint('benchmark.before')
        begin = clock_ns()
        previous = validate_request(root, review, confirmation, label, review_sha256, predecessor)
        progress['timings_ns']['release_gate_and_file_validation'] = clock_ns()-begin
        begin = clock_ns(); inputs = load_inputs(root)
        progress['timings_ns']['file_input_and_payload_preparation'] = clock_ns()-begin
        observation = capture_boot(lambda text: stop.call('preflight.shell', lambda: execute(text)),
            label='bench_'+label, previous_boot_ids=[previous['boot_id'], inputs['anchor']['boot_id']],
            expected_BOOT_sha256=review['expected_BOOT_sha256'], expected_SDK='3.36.1')
        progress.update(boot_id=observation['boot_id'], predecessor=previous,
            packet_fingerprint=packet_fingerprint(review), input_hashes=inputs['window_hashes'],
            inputs_sha256=inputs['inputs_sha256'], expected_sha256=inputs['expected_sha256'],
            model_hashes=inputs['model_hashes'], expected_BOOT_sha256=review['expected_BOOT_sha256'],
            device_versions=DEVICE_VERSIONS, SDK_version='3.36.1',
            physical_boot_confirmation=confirmation, release_sha256=review_sha256)
        emit('preflight', observation)
        packet_files(root, review)
        owner = stop.call('owner.create', owner_factory, stop.bind_owner)
        owner.known_probe_chunks=[]; owner.known_probe_ledgers=[]; owner.known_probe_images=[]
        if (owner.input_mode != review['arm'] or not isinstance(owner.trace, BufferedTrace)
                or owner.trace.max_events != SCOPE['trace_capacity']):
            raise ValueError('Fixed input arm and identical bounded RAM diagnostics required')
        begin = clock_ns()
        info = stop.call('owner.initialize', lambda: owner.initialize(root/'artifacts/icraft/tcn_ed4_ZG.json',
            observation['boot_id'], int(datetime.now().timestamp()*1000000), control='axi_registers'))
        progress['timings_ns']['model_session_initialize'] = clock_ns()-begin
        stop.bind_ledger(owner.ledger)
        if (info.get('versions') != DEVICE_VERSIONS or info.get('input_mode') != review['arm']
                or owner.used != 0 or owner.call_count != 0 or owner.request_id != 0
                or owner.ledger.size != SCOPE['arena_bytes'] or owner.ledger.quota != SCOPE['quota_bytes']
                or owner.ledger.boot_id != observation['boot_id']
                or owner.ledger.run_id != owner.run_id or owner.ledger.generation != owner.generation):
            raise ValueError('Fresh bounded owner/device identity required')
        emit('initialized', info)
        packet_files(root, review)  # last file access before known/warmup/timed calls
        progress['current_attempt'] = dict(phase='known', ordinal=0, golden_index=0,
            expected_tensor_sha256=inputs['window_hashes'][0], raw_return=None,
            performance_sample=False)
        begin = clock_ns()
        known = stop.call('benchmark.known', lambda: known_sdk_input(owner, inputs['tensors'][0]),
            lambda value: progress['current_attempt'].update(raw_return=value))
        known_elapsed = clock_ns()-begin
        if known_elapsed < 0:
            raise ValueError('Monotonic timer reversed during known input')
        progress['current_attempt']['elapsed_ns'] = known_elapsed
        progress['timings_ns']['known_input'] = known_elapsed
        error = _verify_prediction(known, inputs, 0, review['arm'], 1)
        if len(owner.known_probe_ledgers) != 1:
            raise ValueError('Exactly one separately retained known allocation required')
        known_identity = _identity(owner.known_probe_ledgers[0], 1, inputs, 0, 0)
        progress['known'] = dict(phase='known', ordinal=0, golden_index=0,
            expected_tensor_sha256=inputs['window_hashes'][0], max_abs_error=error,
            prediction=known, **known_identity)
        progress['current_attempt'] = None
        emit('known_SDK_passed', progress['known'])
        warm_begin = clock_ns()
        schedule = [('warmup', i+1, 0) for i in range(10)] + [('timed', i+1, i+1) for i in range(12)]
        for batch, (phase, ordinal, index) in enumerate(schedule, 1):
            stop.checkpoint('benchmark.'+phase+'.before_submit')
            owner.trace.context = dict(phase=phase, ordinal=ordinal, golden_index=index,
                expected_tensor_sha256=inputs['window_hashes'][index], request_id=batch)
            offset = (batch-1)*41984
            progress['current_attempt'] = dict(phase=phase, ordinal=ordinal, golden_index=index,
                request_id=batch, expected_tensor_sha256=inputs['window_hashes'][index],
                raw_return=None, performance_sample=False)
            begin = clock_ns()
            value = stop.call('benchmark.submit', lambda: owner.submit_local(inputs['payloads'][index], False),
                lambda value: progress['current_attempt'].update(raw_return=value))
            elapsed = clock_ns()-begin
            progress['current_attempt']['elapsed_ns'] = elapsed
            if elapsed < 0:
                raise ValueError('Monotonic timer reversed during primary interval')
            # Deliberately outside primary interval; no application file I/O.
            if (value.get('records') != 100 or len(value.get('predictions', [])) != 1
                    or value.get('request_id') != batch or value.get('used') != batch*41984
                    or value.get('input_sha256') != hashlib.sha256(inputs['payloads'][index]).hexdigest()):
                raise ValueError('Independent 100-to-1 request count/input differs')
            desc = value.get('descriptor', {})
            if (desc.get('request_id') != batch or desc.get('run_id') != owner.run_id
                    or desc.get('generation') != owner.generation
                    or desc.get('input_base') != owner.base+offset+64
                    or desc.get('output_base') != owner.base+offset+25792
                    or desc.get('record_count') != 100 or desc.get('max_windows') != 1
                    or desc.get('input_bytes') != 25600 or desc.get('output_capacity') != 16128):
                raise ValueError('Returned descriptor/slot identity differs')
            prediction = value['predictions'][0]
            error = _verify_prediction(prediction, inputs, index, review['arm'], batch+1)
            identity = _identity(owner.ledger, batch, inputs, index, offset)
            if (prediction.get('first') != identity['first'] or prediction.get('last') != identity['last']
                    or owner.call_count != batch+1 or owner.used != batch*41984):
                raise ValueError('WDO endpoints/cumulative completions differ')
            row = dict(phase=phase, ordinal=ordinal, golden_index=index, request_id=batch,
                expected_tensor_sha256=inputs['window_hashes'][index], max_abs_error=error,
                primary_elapsed_ns=elapsed, primary_performance_sample=(phase == 'timed'),
                result=value, **identity)
            progress['warmups' if phase == 'warmup' else 'samples'].append(row)
            progress['current_attempt'] = None
            emit(phase+'_passed', row)
            if batch == 10:
                progress['timings_ns']['warmups_with_verification_and_RAM_record'] = clock_ns()-warm_begin
        if (owner.call_count != 23 or owner.used != 923648 or owner.request_id != 22
                or owner.ledger.used != 923648 or len(owner.retained) != 22
                or len(owner.trace.events) > 8192):
            raise ValueError('Final finite benchmark budget differs')
        times = [v['primary_elapsed_ns'] for v in progress['samples']]
        progress.update(status='bounded_paired_benchmark_calls_completed', NPU_calls=23,
            FPGA_batches=22, main_used_bytes=923648, known_used_bytes=41984,
            diagnostic_records_before_audit=len(owner.trace.events),
            primary_summary_ns=dict(min=min(times), max=max(times), median=float(np.median(times))),
            explicit_device_Tensor_to_calls=23 if review['arm']=='device_copy' else 0,
            documented_Host_buffer_copy_bytes=0 if review['arm']=='device_copy' else 23*16000,
            strict_visibility_proven=False, zero_copy=False, DMA_proven=False,
            steady_p99_proven=False, production_throughput_proven=False,
            L02_improvement_proven=False, SDK_internal_transfer_bytes=None)
        emit('calls_completed', progress)
        return progress
    except BaseException as error:
        progress.update(status='isolated', error=type(error).__name__+': '+str(error), owner_retained=True)
        stop.isolate(error)
        raise
