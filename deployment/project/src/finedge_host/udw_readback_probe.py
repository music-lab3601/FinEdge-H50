"""Closed two-input readback candidate and a separate, non-executable timing plan."""
from datetime import datetime
import re

import numpy as np

from .delivery import read_json, sha256
from .udw_copy_probe import packet_files as copy_packet_files, known_sdk_input as copy_known_sdk_input, DEVICE_VERSIONS, FINDINGS
from .udw_copy_probe import SCOPE as COPY_SCOPE
from .udw_shared_probe import golden_records
from .udw_e_preflight import capture_boot
from .udw_probe_stop import ProbeStop

SCOPE = dict(COPY_SCOPE, schema='protected-ARM-readback-probe-v1', input_mode='host_readback',
    copy_api='Tensor(readonly_float32_buffer, software_type)',
    input_API='Tensor(readonly_float32_buffer, software_type)',
    explicit_device_Tensor_to_calls=0, trace_policy='bounded_RAM_then_flush',
    performance_runner_implemented=False)


def known_sdk_input(owner, tensor):
    result = copy_known_sdk_input(owner, tensor)
    # The inherited helper's slot IO is identical, but its Tensor.to counter
    # describes CP01. Replace that counter with this owner's actual policy.
    result['known_SDK_io']['Tensor_to_logical_bytes'] = result['input_transfer']['explicit_device_Tensor_to_logical_bytes']
    result['known_SDK_io']['documented_Host_buffer_copy_bytes'] = result['input_transfer']['documented_Host_buffer_copy_bytes']
    return result


def packet_files(root, review):
    extra = {'src/finedge_host/udw_readback_owner.py', 'src/finedge_host/udw_readback_probe.py',
             'scripts/udw_readback_check.py', 'probe/readback_interface.json', 'probe/measurement_plan.json'}
    if not extra <= set(review.get('packet_hashes', {})):
        raise ValueError('Incomplete protected readback packet')
    copy_packet_files(root, review)


def validate_request(root, review, confirmation, label, review_sha256):
    if not re.fullmatch(r'[A-Za-z0-9_]{1,40}', label):
        raise ValueError('Safe unique readback probe label required')
    if (review.get('device_probe_approved') is not True or not FINDINGS <= set(review.get('closed_findings', []))
            or review.get('scope') != SCOPE or review.get('expected_device_versions') != DEVICE_VERSIONS):
        raise ValueError('Protected readback review/scope is not approved')
    packet_files(root, review)
    if (confirmation.get('source') != 'human_user_current_session'
            or confirmation.get('physical_power_removed') is not True
            or confirmation.get('linux_boot_complete') is not True
            or confirmation.get('label') != label or not confirmation.get('verbatim_reply', '').strip()
            or not re.fullmatch(r'[0-9a-f]{64}', review_sha256)
            or confirmation.get('review_sha256') != review_sha256):
        raise ValueError('Fresh human confirmation bound to readback probe required')
    previous = read_json(root/'probe/previous.json')
    if confirmation.get('previous_result_sha256') != previous['original_result_sha256']:
        raise ValueError('Readback predecessor differs')
    return previous


def measurement_plan(model_hashes, window_hashes):
    if len(window_hashes) != 32:
        raise ValueError('Frozen 32 golden windows required for the plan')
    plan = dict(schema='readback-paired-measurement-plan-v1', executable=False,
        device_probe_approved=False, performance_runner_implemented=False,
        arms=['device_copy', 'host_readback'], one_changed_variable='source.to(Host) vs Tensor(checked_ARM_buffer,type)',
        model_hashes=model_hashes, window_hashes=window_hashes,
        known_input_index=0, warmup_indices=[0]*10, timed_indices=list(range(1, 13)),
        planned_NPU_calls_per_arm=23, planned_FPGA_batches_per_arm=22,
        main_used_bytes=22*41984, quota_bytes=1048576, main_arena_bytes=2097152,
        known_allocation_bytes=65536, reuse=False, retry=False, reset=False,
        separate_physical_boots_per_arm=True, predecessor_chain_required=True,
        trace_policy='same bounded RAM trace in both arms; flush only after timing and final audit',
        primary_interval='submit_local entry through return; includes required preparation, control, readback, checks and inference',
        secondary_intervals=['input transfer API','Host validation','Cast','forward incl callbacks','output wait/read'],
        excluded_but_reported=['model/session initialization','file input','10 warmups','final memory audit','diagnostic flush'],
        timer='same board monotonic clock', tolerances=dict(rtol=1e-3,atol=1e-4),
        policy='abort on first input/identity/protection/numerical/cancellation error; retain owner',
        reports=['all 12 raw request times and input identities','median/min/max; no steady p99 claim',
                 'API counts and explicit bytes, unknown SDK internals separately','wall time and flush time separately'],
        no_claims=['zero-copy','DMA','strict PL visibility','market accuracy','full-day throughput/H50'],
        L02_comparison='L02 already uses ARM Host data/write. Do not claim savings over L02 from removing CP01 Tensor.to alone')
    validate_measurement_plan(plan)
    return plan


def validate_measurement_plan(plan):
    if (plan.get('schema') != 'readback-paired-measurement-plan-v1' or plan.get('executable') is not False
            or plan.get('device_probe_approved') is not False or plan.get('performance_runner_implemented') is not False
            or plan.get('arms') != ['device_copy', 'host_readback'] or plan.get('warmup_indices') != [0]*10
            or plan.get('known_input_index') != 0 or plan.get('known_allocation_bytes') != 65536
            or plan.get('timed_indices') != list(range(1, 13)) or plan.get('planned_NPU_calls_per_arm') != 23
            or plan.get('planned_FPGA_batches_per_arm') != 22 or plan.get('main_used_bytes') != 22*41984
            or plan.get('quota_bytes') != 1048576 or plan.get('main_arena_bytes') != 2097152
            or plan.get('main_used_bytes')+4096 > plan.get('quota_bytes')
            or plan.get('separate_physical_boots_per_arm') is not True or plan.get('predecessor_chain_required') is not True
            or any(plan.get(k) is not False for k in ('reuse','retry','reset'))
            or plan.get('tolerances') != dict(rtol=1e-3,atol=1e-4)
            or set(plan.get('model_hashes', {})) != {'json','raw'}
            or any(not re.fullmatch(r'[0-9a-f]{64}',h) for h in plan.get('model_hashes', {}).values())
            or len(plan.get('window_hashes', [])) != 32
            or any(not re.fullmatch(r'[0-9a-f]{64}',h) for h in plan['window_hashes'])):
        raise ValueError('Fair measurement plan changed or tries to become device authorization')


def run_probe(root, review, confirmation, label, *, review_sha256, execute, owner_factory, record, stop=None):
    stop = stop if stop is not None else ProbeStop()
    stop.checkpoint('readback.before')
    previous = validate_request(root, review, confirmation, label, review_sha256)
    preflight = capture_boot(lambda text: stop.call('preflight.shell', lambda: execute(text)),
        label='readback_'+label, previous_boot_ids=[previous['boot_id']],
        expected_BOOT_sha256=review['expected_BOOT_sha256'], expected_SDK='3.36.1')
    record('preflight', preflight)
    stop.checkpoint('preflight.recorded')
    packet_files(root, review)
    owner = stop.call('owner.create', owner_factory, stop.bind_owner)
    owner.known_probe_chunks = []; owner.known_probe_ledgers = []; owner.known_probe_images = []
    try:
        if owner.input_mode != 'host_readback':
            raise ValueError('Only host_readback is released by this bounded entry')
        info = stop.call('owner.initialize', lambda: owner.initialize(root/'artifacts/icraft/tcn_ed4_ZG.json',
            preflight['boot_id'], int(datetime.now().timestamp()*1000000), control='axi_registers'))
        stop.bind_ledger(owner.ledger)
        if info.get('versions') != DEVICE_VERSIONS:
            raise ValueError('Device/core differs')
        record('initialized', info)
        stop.checkpoint('initialized.recorded')
        packet_files(root, review)
        tensor = (root/'probe/golden.f32').read_bytes()
        expected = read_json(root/'probe/expected.json')['logits']

        def verify(value):
            logits = np.asarray(value['logits'])
            if (value['tensor_sha256'] != sha256(root/'probe/golden.f32')
                    or not np.allclose(logits, expected, rtol=1e-3,atol=1e-4)
                    or logits.argmax() != np.asarray(expected).argmax()
                    or value['input_mode'] != 'host_readback'
                    or value['input_transfer']['explicit_device_Tensor_to_calls'] != 0):
                raise ValueError('Readback input/numerical/transfer verification differs')

        known = known_sdk_input(owner, tensor); verify(known)
        record('known_SDK_passed', known)
        stop.checkpoint('known.recorded'); packet_files(root, review)
        owner.trace.context = {'phase': 'FPGA'}
        payload, _, _ = golden_records(tensor)
        fpga = stop.call('PL.submit', lambda: owner.submit_local(payload, False))
        if len(fpga['predictions']) != 1:
            raise ValueError('FPGA 100-to-1 count differs')
        verify(fpga['predictions'][0]); record('FPGA_passed', fpga)
        stop.checkpoint('FPGA.recorded'); packet_files(root, review)
        if owner.call_count != 2:
            raise ValueError('Unexpected NPU call count')
        return dict(status='bounded_protected_readback_verified', known=known, FPGA=fpga,
            NPU_calls=2, explicit_device_Tensor_to_calls=0, owner_retained=True,
            strict_visibility_proven=False, zero_copy=False, diagnostic_only=True, performance_sample=False)
    except BaseException as error:
        stop.isolate(error)
        raise
