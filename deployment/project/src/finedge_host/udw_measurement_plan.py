"""Offline E-stage planning contracts. This module cannot access a device.

The plan is review input, not an authorization or a faster UdwLab driver.
"""
from copy import deepcopy

from .ddr_transport_contract import OUTPUT_STRIDE
from .udw_runtime import CapacityBudget, capacity_plan
from .udw_e_frontend import Continuity


REQUIRED_CHECKS = frozenset({
    'abi_health_sticky', 'command_prestate_sequence_terminal',
    'descriptor_and_terminal_counts', 'run_request_generation_window_identity',
    'segment_continuity', 'current_input_output_guards_poison',
    'SDK_NPU_completion', 'finite_logits_and_NPU_identity',
    'capacity_before_reserve', 'monotonic_nonoverlap',
    'quarantine_and_retain_owner',
})


def bounded_schedule(batches, *, quota_bytes=512*1024):
    """Plan fresh slots; stop before reserve, allowing at most one lookahead batch.

    ``batches`` contains frozen row metadata, never tensors or live SDK objects.
    Selection is verified against the frozen single-segment emit=16 contract.
    """
    p = capacity_plan(record_limit=512, batch_size=64, emit_stride=16,
                      quota_bytes=quota_bytes)
    budget = CapacityBudget(p['arena_bytes'], quota_bytes)
    used = p['preflight_bytes']; continuity=Continuity(allow_segments=False)
    rows = []; submitted = []; stopped = None
    for batch_index, batch in enumerate(batches, 1):
        if len(batch) != 64:
            raise ValueError('Initial E plan requires complete 64-record batches')
        identities = []
        for row in batch:
            info=continuity.accept(row['metadata'],row['identity'])
            if row['opportunity'] != info['opportunity'] or row['selected'] != info['selected']:
                raise ValueError('Frozen opportunity/emit schedule mismatch')
            if info['selected']: identities.append(list(info['identity']))
        size = 64*256 + max(64, len(identities)*OUTPUT_STRIDE) + 256
        decision = budget.decision(used, size)
        item = dict(batch_index=batch_index, records=64, selected=len(identities),
                    selected_identities=identities, capacity=decision,
                    submitted=decision['fits'],record_rows=deepcopy(batch))
        rows.append(item)
        if not decision['fits']:
            stopped = batch_index
            break
        item['offset'] = used
        used += size; submitted.extend(identities)
    return dict(capacity_plan=p, batches=rows, used_bytes=used,
                submitted_records=sum(r['records'] for r in rows if r['submitted']),
                submitted_windows=len(submitted), selected_identities=submitted,
                stopped_before_batch=stopped, tail_reserved_bytes=p['reserved_tail'])


def measurement_plan(schedule, input_digest, compiled_hashes, onnx_hash):
    shared = dict(input_schedule_sha256=input_digest, batch_records=64, emit_stride=16,
                  warmup_predictions=10, selected_identities=schedule['selected_identities'])
    arms = []
    for name, backend in (('software_layout_NPU', 'ZG_socket'),
                          ('FPGA_layout_NPU', 'ZG_socket'), ('original_ONNX_CPU', 'ONNX_CPU')):
        arms.append(dict(name=name, backend=backend, **deepcopy(shared),
                         model_hashes=deepcopy(compiled_hashes) if backend == 'ZG_socket'
                         else {'onnx': onnx_hash}))
    return dict(schema='udw-e-measurement-plan-v2', status='proposal_pending_review',
                board_execution_enabled=False, faster_driver_implemented=False,
                strict_visibility_proven=False, default_runtime_changed=False,
                schedule=schedule, arms=arms,
                retained_hot_checks=sorted(REQUIRED_CHECKS),
                timed_boundary='raw_book_record_ready_to_prediction_available',
                timed_operations=['validation_normalization', 'ring_or_WDM_pack',
                    'batch_release_and_queue_wait', 'software_layout_or_FPGA_bridge',
                    'RPC_SDK_upload', 'descriptor_command_completion',
                    'necessary_current_readback_and_checks', 'NPU_or_ONNX_predict',
                    'result_identity_validation'],
                outside_hot_path=['file_read_and_ITCH_prefix_book_reconstruction',
                    'model_load_arena_allocation_poison_warmup',
                    'full_history_byte_audit_before_and_after',
                    'software_reference_tensor_construction_for_FPGA_arm',
                    'Torch_ONNX_Host_comparisons_before_or_after', 'report_UI_export'],
                setup_and_audit_time_reported_separately=True,
                setup_amortized_total_reported=True,
                per_window_Host=False, per_window_dual_model=False,
                automatic_reset=False, buffer_reuse=False, automatic_owner_termination=False,
                failure_policy='stop_all_submissions_and_predictions_retain_owner',
                session_rule='review_then_human_physical_stop_cold_boot_new_allocation_each_round',
                current_output_tensor_sha256_required=True,
                claims=dict(speedup_proven=False, realtime_peak_proven=False,
                            production_accuracy_accepted=False, P99_supported=False))


def validate_plan(plan):
    """Fail closed on common unfair/unsafe edits to a reviewable experiment plan."""
    if plan['schema'] != 'udw-e-measurement-plan-v2': raise ValueError('Schema')
    for field in ('board_execution_enabled', 'faster_driver_implemented',
                  'strict_visibility_proven', 'default_runtime_changed',
                  'per_window_Host', 'per_window_dual_model', 'automatic_reset',
                  'buffer_reuse', 'automatic_owner_termination'):
        if plan[field] is not False: raise ValueError(field)
    if not REQUIRED_CHECKS <= set(plan['retained_hot_checks']):
        raise ValueError('Required hot-path safety check removed')
    if plan['timed_boundary'] != 'raw_book_record_ready_to_prediction_available':
        raise ValueError('Timing scope changed')
    if plan['failure_policy'] != 'stop_all_submissions_and_predictions_retain_owner':
        raise ValueError('Failure lifecycle changed')
    arms = plan['arms']
    if [a['name'] for a in arms] != ['software_layout_NPU', 'FPGA_layout_NPU', 'original_ONNX_CPU']:
        raise ValueError('Missing comparison arm')
    if arms[0]['backend'] != 'ZG_socket' or arms[1]['backend'] != 'ZG_socket' or arms[2]['backend'] != 'ONNX_CPU':
        raise ValueError('Backend changed')
    for field in ('input_schedule_sha256', 'batch_records', 'emit_stride',
                  'warmup_predictions', 'selected_identities'):
        if any(a[field] != arms[0][field] for a in arms):
            raise ValueError('Asymmetric arms: '+field)
    if (arms[0]['batch_records'], arms[0]['emit_stride'], arms[0]['warmup_predictions']) != (64,16,10):
        raise ValueError('Unreviewed run dimensions')
    if arms[0]['model_hashes'] != arms[1]['model_hashes']:
        raise ValueError('NPU models differ')
    s = plan['schedule']; used = s['capacity_plan']['preflight_bytes']; ids=[]
    budget = CapacityBudget(s['capacity_plan']['arena_bytes'], s['capacity_plan']['quota_bytes'])
    stopped = False; count=0
    for b in s['batches']:
        if stopped: raise ValueError('Submission after capacity stop')
        if b['records'] != 64 or b['selected'] != len(b['selected_identities']):
            raise ValueError('Invalid batch counts')
        size = b['records']*256 + max(64, b['selected']*OUTPUT_STRIDE) + 256
        d=budget.decision(used,size)
        if b['capacity'] != d or b['submitted'] != d['fits']:
            raise ValueError('Capacity accounting differs')
        if b['submitted']:
            if b['offset'] != used: raise ValueError('Overlapping/nonmonotonic slot')
            used += size; ids.extend(b['selected_identities']); count += b['records']
        else: stopped=True
    if (used != s['used_bytes'] or ids != s['selected_identities']
            or count != s['submitted_records'] or len(ids) != s['submitted_windows']
            or ids != arms[0]['selected_identities']):
        raise ValueError('Schedule totals/identities differ')
    rebuilt=bounded_schedule([b['record_rows'] for b in s['batches']],quota_bytes=s['capacity_plan']['quota_bytes'])
    if rebuilt!=s:raise ValueError('Full metadata/continuity schedule differs')
    return True
