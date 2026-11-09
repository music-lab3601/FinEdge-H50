"""ED7 versioned finite TorchScript compatibility contract. Import has no ML, data or filesystem effects."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'artifacts/tcn_existing_data_optimization_2026-10-17'
BASE_STAGE = CAMPAIGN / 'step07_delivery'
STAGE = BASE_STAGE / 'torchscript_compatibility_v3'
PACKAGE = STAGE / 'package'
REGISTRATION = STAGE / 'registration.json'
TRAIN = ['2025-12-08', '2025-12-09', '2025-12-10', '2025-12-11', '2025-12-12']
FAMILY = 'ED4_feature42_lr0.001_seed42'
CP_SHA = 'c480db50073dbb099c5374dc5df9f942044e2ab09f8d6bcff396b7bc4cb6dc60'
NORM_SHA = '709114d09b2dd581bd9481e180973fdf1553452b644d24dcb211da7ea394caaf'
PROTOCOL_SHA = '09654e6dfdb10b69feb6ee950c6668e71640bcea1d56fa9691a9564c8641fcf3'
OPERATION_KEYS = ('eager_model_constructions', 'checkpoint_deserializations',
    'native_model_state_loads', 'folded_model_state_loads', 'weight_norm_folds',
    'script_compilations', 'torchscript_saves', 'torchscript_loads', 'forward_attempts',
    'forward_calls', 'forward_samples', 'source_cache_window_reads',
    'source_cache_payload_bytes_read', 'source_cache_full_hashes',
    'optimizer_constructions', 'optimizer_state_loads', 'rng_restorations',
    'backward_calls', 'optimizer_steps', 'training_runs', 'E_record_reads',
    'label_reads', 'DEV_cache_reads', 'sealed_stream_reads', 'saved_golden_read_bytes')
PRIMARY_BUDGET = dict(zip(OPERATION_KEYS, (2, 1, 1, 1, 10, 1, 1, 1, 6, 6, 99, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 512000)))
RELOAD_BUDGET = {key: (1 if key in ('torchscript_loads', 'forward_attempts', 'forward_calls')
                       else 32 if key == 'forward_samples' else 0) for key in OPERATION_KEYS}
FAILED_PRIMARY_OPERATIONS = dict(zip(OPERATION_KEYS, (2, 1, 1, 1, 10, 1, 0, 0, 0, 0, 0, 32, 512000, 0,
                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
CUMULATIVE_BUDGET = {key: FAILED_PRIMARY_OPERATIONS[key] + PRIMARY_BUDGET[key] + RELOAD_BUDGET[key]
                     for key in OPERATION_KEYS}
TOLERANCE = {'rtol': 1e-4, 'atol': 1e-5}


def need(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def safe(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    need('..' not in p.parts and p.is_relative_to(ROOT), 'Outside ED7 workspace')
    for q in (p, *p.parents):
        if q.exists():
            need(not q.is_symlink() and not getattr(q.lstat(), 'st_file_attributes', 0) & 0x400,
                 'Reparse paths are forbidden')
        if q == ROOT:
            break
    need(p.resolve().is_relative_to(ROOT.resolve()), 'Resolved path escaped workspace')
    return p


def sha(path):
    h = hashlib.sha256()
    with safe(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    p = safe(path)
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def equivalent(left, right):
    return (safe(left['path']).resolve() == safe(right['path']).resolve()
            and left['sha256'] == right['sha256'] and left['bytes'] == right['bytes'])


def verify(ref):
    need(isinstance(ref, dict) and {'path', 'sha256', 'bytes'} <= set(ref), 'Incomplete identity')
    p = safe(ref['path'])
    need(type(ref['bytes']) is int and p.is_file() and p.stat().st_size == ref['bytes']
         and sha(p) == ref['sha256'], 'Changed bound file: ' + str(p))
    return p


def read(path):
    p = safe(path)
    need(p.stat().st_size <= 32 * 1024**2, 'Oversized metadata')
    return json.loads(p.read_text('utf-8-sig'),
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonfinite JSON ' + s)))


def write_new(path, value):
    p = safe(path)
    need(p.is_relative_to(STAGE), 'ED7 write outside its stage')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def expected_window_plan(cache):
    """Metadata only. No cache, E source or labels are opened."""
    result = []
    for day in TRAIN:
        for example in [e for e in cache['boundary_examples'] if e['day'] == day]:
            r = next(r for r in cache['ranges'] if r['date'] == day
                     and r['endpoint_start'] <= example['endpoint'] < r['endpoint_stop_exclusive'])
            result.append(_window(cache, r, example['endpoint'], example['normalized_input_float32_sha256']))
    r = next(r for r in cache['ranges'] if r['date'] == TRAIN[0])
    for offset in range(2, 19):
        result.append(_window(cache, r, r['endpoint_start'] + offset, None))
    return result


def _window(cache, r, endpoint, expected):
    begin = r['cache_offset'] + endpoint - 99 - r['start_row']
    return {'date': r['date'], 'role': 'train', 'endpoint': endpoint,
        'research_segment_id': r['research_segment_id'], 'source_cache': cache['cache_files'][r['date']],
        'cache_row_start': begin, 'cache_row_stop_exclusive': begin + 100,
        'start_row': r['start_row'], 'stop_row': r['stop_row'],
        'cache_range_start': r['cache_offset'],
        'cache_range_stop_exclusive': r['cache_offset'] + r['stop_row'] - r['start_row'],
        'endpoint_start': r['endpoint_start'], 'endpoint_stop_exclusive': r['endpoint_stop_exclusive'],
        'expected_input_sha256': expected}


def validate_window_plan(plan, cache):
    need(plan == expected_window_plan(cache) and len(plan) == 32,
         'Frozen 32-window TRAIN plan changed')
    need(len({(w['date'], w['endpoint']) for w in plan}) == 32, 'Repeated TRAIN windows')
    for w in plan:
        ref = w['source_cache']; path = safe(ref['path'])
        need(w['date'] in TRAIN and w['role'] == ref['role'] == 'train'
             and path == CAMPAIGN / 'step04_features/prepared' / w['date'] / 'normalized_rows.f32',
             'Only registered TRAIN normalized caches admitted')
        need(ref['dtype'] == '<f4' and ref['shape'][1] == 40
             and ref['bytes'] == ref['shape'][0] * 160, 'Cache geometry changed')
        need(w['cache_row_stop_exclusive'] - w['cache_row_start'] == 100
             and w['cache_range_start'] <= w['cache_row_start']
             and w['cache_row_stop_exclusive'] <= w['cache_range_stop_exclusive']
             and w['endpoint'] - 99 >= w['start_row']
             and w['endpoint'] + 50 < w['stop_row']
             and w['endpoint_start'] <= w['endpoint'] < w['endpoint_stop_exclusive'],
             'Window crossed an accepted range or its H50 footprint')
    return True


def verify_protected(reg):
    inventory = read(verify(reg['protected_inventory_binding']))
    for ref in inventory['files']:
        verify(ref)  # opaque bytes only, including any historical binary backup
    snapshot = verify(reg['budget_snapshot_binding'])
    need(sha(CAMPAIGN / 'budget.json') == sha(snapshot), 'Closed training budget changed')
    return len(inventory['files'])


def checked_receipt(path, phase):
    receipt = read(path)
    need(receipt['phase'] == phase and receipt['exit_code'] == 0 and receipt['entry_unchanged'] is True,
         'Required actual CLI did not succeed: ' + phase)
    for name in ('entry_before', 'entry_after', 'input', 'capture_source', 'stdout', 'stderr'):
        verify(receipt[name])
    need(receipt['entry_before'] == receipt['entry_after']
         and receipt['sources_before'] == receipt['sources_after'] and receipt['sources_unchanged'] is True,
         'Actual implementation changed')
    for source in receipt['sources_before']:
        verify(source)
    for output in receipt['outputs'].values():
        verify(output)
    expected_capture = binding(ROOT / 'scripts/run_tcn_existing_data_ed7_compat_capture.py')
    if phase == 'tests':
        current = read(REGISTRATION)
        if 'test_capture_binding' in current:
            verify(current['test_capture_binding'])
            expected_capture = current['test_capture_binding']
    need(equivalent(receipt['capture_source'], expected_capture),
         'Wrong actual CLI capture implementation')
    expected_entry = {'tests': 'test_tcn_existing_data_ed7_compat.py', 'primary': 'tcn_existing_data_ed7_compat_export.py',
        'reload': 'tcn_existing_data_ed7_compat_export.py', 'audit': 'audit_tcn_existing_data_ed7_compat.py',
        'finalize': 'finalize_tcn_existing_data_ed7_compat.py'}[phase]
    need(equivalent(receipt['entry_before'], binding(ROOT / 'scripts' / expected_entry)),
         'Wrong actual phase entry')
    if phase == 'tests':
        need(equivalent(receipt['input'], receipt['entry_before']), 'Wrong synthetic test input')
    else:
        need(equivalent(receipt['input'], binding(REGISTRATION)), 'Actual CLI used another registration')
        expected_acceptance = STAGE / ('completion_acceptance.json' if phase == 'finalize'
                                      else phase + '_acceptance.json')
        need(equivalent(receipt['outputs']['acceptance'], binding(expected_acceptance)),
             'Wrong actual phase acceptance output')
    return receipt


def verify_failed_attempt(reg):
    """Bind the consumed original failure without calling it a successful phase."""
    refs = {'initial_failed_primary_binding': BASE_STAGE / 'primary_failure.json',
            'initial_primary_receipt_binding': BASE_STAGE / 'primary_execution_receipt.json',
            'original_registration_binding': BASE_STAGE / 'registration.json',
            'saved_golden_input_binding': BASE_STAGE / 'package/golden/inputs.f32',
            'saved_golden_selection_binding': BASE_STAGE / 'package/golden/selection.json'}
    for key, expected in refs.items():
        need(safe(reg[key]['path']) == expected, 'Wrong original failure/reuse reference')
        verify(reg[key])
    failure = read(reg['initial_failed_primary_binding']['path'])
    receipt = read(reg['initial_primary_receipt_binding']['path'])
    observed = dict(failure['operations_observed'], saved_golden_read_bytes=0)
    need(set(failure['operations_observed']) == set(OPERATION_KEYS) - {'saved_golden_read_bytes'}
         and observed == FAILED_PRIMARY_OPERATIONS and failure['phase'] == 'primary'
         and failure['error_type'] == 'RuntimeError'
         and failure['error'] == 'requires_grad_ is not supported on ScriptModules'
         and failure['reservation_remains_consumed'] is True,
         'Original actual failure/cost changed')
    need(receipt['phase'] == 'primary' and receipt['exit_code'] == 1
         and receipt['entry_unchanged'] is True and receipt['sources_unchanged'] is True
         and receipt['entry_before'] == receipt['entry_after']
         and receipt['sources_before'] == receipt['sources_after']
         and equivalent(receipt['input'], reg['original_registration_binding'])
         and equivalent(receipt['outputs']['failure'], reg['initial_failed_primary_binding']),
         'Original CLI failure provenance changed')
    for name in ('entry_before', 'entry_after', 'input', 'capture_source', 'stdout', 'stderr'):
        verify(receipt[name])
    for ref in receipt['sources_before']:
        verify(ref)
    need(equivalent(receipt['entry_before'], binding(ROOT / 'scripts/tcn_existing_data_ed7_export.py'))
         and equivalent(receipt['capture_source'], binding(ROOT / 'scripts/run_tcn_existing_data_ed7_capture.py'))
         and equivalent(failure['reservation']['registration'], reg['original_registration_binding'])
         and failure['worker_pid'] == failure['reservation']['worker_pid']
         == failure['process_provenance']['pid'], 'Original worker/source provenance changed')
    need(reg['saved_golden_input_binding']['bytes'] == 512000,
         'Saved TRAIN golden geometry changed')
    return observed


def admit(registration_sha, phase):
    need(phase in ('primary', 'reload', 'audit', 'finalize'), 'Unsupported ED7 phase')
    need(sha(REGISTRATION) == registration_sha, 'Explicit registration differs')
    reg = read(REGISTRATION)
    need(reg['version'] == 'TCN-ED7-delivery-registration-1.0'
         and reg['selected_family'] == FAMILY and reg['representative_seed'] == 42
         and reg['primary_budget'] == PRIMARY_BUDGET and reg['reload_budget'] == RELOAD_BUDGET
         and reg['tolerance'] == TOLERANCE
         and reg['compatibility_revision'] == 'torchscript_compatibility_v3'
         and reg['failed_primary_operations'] == FAILED_PRIMARY_OPERATIONS
         and reg['cumulative_budget'] == CUMULATIVE_BUDGET, 'Delivery identity or exact finite budget changed')
    need(reg['training_runs'] == 0 and reg['new_independent_test'] is False
         and reg['deployment_authorized'] is False and reg['historical17_19_reads'] is False,
         'ED7 scope expanded')
    verify_failed_attempt(reg)
    for ref in reg['source_bindings'].values():
        verify(ref)
    for ref in reg['evidence_bindings'].values():
        verify(ref)
    review = read(verify(reg['source_review_binding']))
    tests = read(verify(reg['test_evidence_binding']))
    need(review['passed'] is True and tests['passed'] is True and tests['actual_exit_code'] == 0,
         'Independent source review and actual synthetic tests required')
    actual_tests = checked_receipt(verify(tests['actual_receipt']), 'tests')
    need(equivalent(actual_tests['entry_before'], reg['source_bindings']['tests']), 'Wrong tested suite')
    need(set(review['source_bindings']) == set(reg['source_bindings'])
         and all(equivalent(review['source_bindings'][name], ref)
                 for name, ref in reg['source_bindings'].items()), 'Reviewed source set differs')
    test_sources = list(reg['source_bindings'].values())
    if 'test_capture_binding' in reg:
        verify(reg['test_capture_binding'])
        need(equivalent(review['test_capture_binding'], reg['test_capture_binding']),
             'Revised test capture was not independently reviewed')
        test_sources.append(reg['test_capture_binding'])
    tested = {safe(ref['path']).resolve(): ref for ref in actual_tests['sources_before']}
    need(set(tested) == {safe(ref['path']).resolve() for ref in test_sources}
         and all(equivalent(tested[safe(ref['path']).resolve()], ref)
                 for ref in test_sources), 'Actual tested source set differs')
    completion = read(verify(reg['evidence_bindings']['ED6_completion']))
    need(completion['passed'] is True and completion['stable_development_adoption'] is True
         and completion['representative_seed'] == 42 and completion['retained_family'] == FAMILY
         and completion['historical_final_quality_changed'] is False,
         'Frozen ED6 representative changed')
    need(equivalent(reg['checkpoint_binding'], completion['retained_checkpoint'])
         and reg['checkpoint_binding']['sha256'] == CP_SHA
         and equivalent(reg['normalization_binding'], completion['retained_normalization'])
         and reg['normalization_binding']['sha256'] == NORM_SHA, 'Retained checkpoint/normalization replaced')
    verify(reg['normalization_binding']); verify(reg['config_binding'])
    need(reg['protocol_binding']['sha256'] == PROTOCOL_SHA, 'Protocol identity changed')
    verify(reg['protocol_binding'])
    cache = read(verify(reg['cache_manifest_binding']))
    validate_window_plan(reg['window_plan'], cache)
    if phase == 'primary':
        verify(reg['checkpoint_binding'])
        need(not PACKAGE.exists(), 'Existing package cannot be overwritten')
        for copy in reg['package_copies']:
            target = Path(copy['target'])
            need(not target.is_absolute() and '..' not in target.parts, 'Unsafe package copy target')
            verify(copy['source'])
    else:
        checked_receipt(STAGE / 'primary_execution_receipt.json', 'primary')
    if phase in ('audit', 'finalize'):
        checked_receipt(STAGE / 'reload_execution_receipt.json', 'reload')
    verify_protected(reg)
    reg['_registration_binding'] = binding(REGISTRATION)
    return reg


def reserve(reg, phase):
    path = STAGE / (phase + '_phase_reservation.json')
    need(not path.exists() and not (STAGE / (phase + '_acceptance.json')).exists(),
         'Permanent ED7 phase already consumed; no automatic retry')
    value = {'version': 'TCN-ED7-permanent-phase-reservation-1.0', 'phase': phase,
        'started_utc': utc(), 'worker_pid': os.getpid(), 'registration': reg['_registration_binding'],
        'budget': reg.get(phase + '_budget', {'ML_operations': 0}),
        'permanent_single_use': True, 'automatic_retry': False}
    write_new(path, value)
    return value
