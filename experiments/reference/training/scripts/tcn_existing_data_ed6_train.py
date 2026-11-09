"""One root-selected cold-start ED6 paired replication slot; no family search.

Admission/import are stdlib only. Only the root operator launches the formal
command, after its independent source review and cache preparation. No resume,
automatic replacement, old campaign writer, historical checkpoint load, or
historical 17--19 stream is supported. Every optimizer attempt is permanently
charged by an atomic budget write BEFORE optimizer.step; failure stops the run.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'artifacts/tcn_existing_data_optimization_2026-10-17'
ROOT_STAGE = CAMPAIGN / 'step06_replication'
SLOT = 'control43'
STAGE = ROOT_STAGE / 'trials' / SLOT
BUDGET = CAMPAIGN / 'budget.json'
PROTOCOL_SHA = '09654e6dfdb10b69feb6ee950c6668e71640bcea1d56fa9691a9564c8641fcf3'
TRAIN = [f'2025-12-{i:02d}' for i in range(8, 13)]
DEV = ['2025-12-15', '2025-12-16']
EXCLUDED = ['2025-12-17', '2025-12-18', '2025-12-19']
VERSION = 'TCN-ED6-training-entry-1.0'
CP_VERSION = 'TCN-ED6-complete-epoch-checkpoint-1.0'
SETTINGS = {
    'version': 'TCN-ED6-paired-config-1.0', 'slot_id': 'control43',
    'seed': 43, 'device': 'cuda', 'precision': 'float32',
    'train_batch_size': 256, 'evaluation_batch_size': 256,
    'max_epochs': 10, 'early_stop_patience': 3, 'optimizer': 'AdamW',
    'learning_rate': .001, 'weight_decay': .0001,
    'loss': 'unweighted_cross_entropy', 'window_length': 100, 'horizon': 50,
    'dropout': 0, 'weight_norm': True, 'decision': 'original_argmax',
    'feature_family': 'original_physical_prices_raw_quantities',
}
SOURCE_NAMES = {
    'scripts/tcn_existing_data_ed6_train.py',
    'scripts/tcn_existing_data_ed6_data.py',
    'scripts/tcn_existing_data_ed4_features.py',
    'scripts/tcn_research_data_p2.py', 'scripts/tcn_research_evaluate_p2.py',
    'scripts/tcn_research_p5_model.py', 'scripts/tcn_candidate.py',
}


def configure(slot):
    global SLOT, STAGE
    need(slot in ('control43', 'candidate43'), 'Only predefined ED6 slots are supported')
    SLOT = slot; STAGE = ROOT_STAGE / 'trials' / slot
    SETTINGS['slot_id'] = slot
    SETTINGS['feature_family'] = ('original_physical_prices_raw_quantities' if slot == 'control43' else
        'relative_midprice_replaces_absolute_prices_odd_raw_quantities_unchanged')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def checked_path(value, base=ROOT):
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    need('..' not in p.parts and p.is_relative_to(base), 'Path outside approved root')
    for q in (p, *p.parents):
        if q.exists():
            need(not q.is_symlink() and not getattr(q.lstat(), 'st_file_attributes', 0) & 0x400,
                 'Reparse path forbidden')
        if q == base:
            break
    need(p.resolve().is_relative_to(base.resolve()), 'Resolved path outside approved root')
    return p


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    p = Path(path)
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def read_json(path):
    p = checked_path(path)
    need(p.is_file() and p.stat().st_size <= 16 * 1024**2, 'Missing/oversized metadata')
    return json.loads(p.read_text(encoding='utf-8-sig'),
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + s)))


def read_binding(ref):
    need(isinstance(ref, dict) and set(ref) >= {'path', 'sha256'}, 'Missing file binding')
    p = checked_path(ref['path'])
    need(p.is_file() and sha(p) == ref['sha256'], 'Bound identity changed: ' + str(p))
    need('bytes' not in ref or p.stat().st_size == ref['bytes'], 'Bound size changed')
    return read_json(p)


def verify_cache_payloads(data, prepared):
    """Seven D row caches plus labels; no E record payload is opened."""
    need(set(prepared.get('cache_files', {})) == set(TRAIN + DEV), 'Wrong cache payload dates')
    result = {}
    for day in TRAIN + DEV:
        ref = prepared['cache_files'][day]
        cache_stage = (ROOT_STAGE / 'prepared_control' if SLOT == 'control43' else
                       CAMPAIGN / 'step04_features/prepared')
        p = checked_path(ref['path'], cache_stage)
        need(p.is_file() and p.stat().st_size == ref['bytes'] and sha(p) == ref['sha256'],
             'Normalized cache payload identity changed: ' + day)
        label = data['day_label_reports'][day]
        lp = checked_path(label['label_path'])
        need(lp.stat().st_size == label['label_bytes'] and sha(lp) == label['label_sha256'],
             'Original P2 label stream changed: ' + day)
        result[day] = {'cache_sha256': ref['sha256'], 'label_sha256': label['label_sha256']}
    return result


def atomic_json(path, value):
    p = checked_path(path)
    need(p == BUDGET or p.is_relative_to(STAGE), 'No writes outside ED6 or current new budget')
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = checked_path(p.with_name(p.name + '.tmp'))
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(p)


def write_new(path, value):
    p = checked_path(path, STAGE)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def emit(value):
    print(json.dumps({'utc': utc(), **value}, ensure_ascii=True, allow_nan=False), flush=True)


def settings_for(slot):
    need(slot in ('control43', 'candidate43'), 'Unregistered paired slot')
    result = dict(SETTINGS); result['slot_id'] = slot
    result['feature_family'] = ('original_physical_prices_raw_quantities' if slot == 'control43' else
        'relative_midprice_replaces_absolute_prices_odd_raw_quantities_unchanged')
    return result


def validate_config(c, slot=None):
    expected = settings_for(SLOT if slot is None else slot)
    need(isinstance(c, dict) and all(c.get(k) == v and type(c.get(k)) is type(v)
                                  for k, v in expected.items()), 'ED6 default settings/factor differ')
    need(not any(k in c for k in ('bias', 'stationary_bias', 'resume', 'checkpoint_path')),
         'ED6 cannot silently add bias or historical checkpoint initialization')
    return c


def exact_rank(matrix):
    need(isinstance(matrix, list) and len(matrix) == 3 and all(isinstance(r, list) and len(r) == 3
         and all(type(x) is int and x >= 0 for x in r) for r in matrix), 'Invalid integer confusion')
    support = [sum(r) for r in matrix]
    total = sum(support); need(total > 0, 'Empty confusion')
    predicted = [sum(matrix[i][j] for i in range(3)) for j in range(3)]
    accuracy = Fraction(sum(matrix[i][i] for i in range(3)), total)
    f1 = sum((Fraction(2 * matrix[i][i], support[i] + predicted[i])
              if support[i] + predicted[i] else Fraction(0) for i in range(3)), Fraction(0)) / 3
    return accuracy, f1


def rank_json(rank):
    return [[v.numerator, v.denominator] for v in rank]


def recompute_history(history):
    need(isinstance(history, list) and 1 <= len(history) <= 10, 'Invalid history length')
    best = None; chosen = None; wait = 0
    for epoch, row in enumerate(history, 1):
        need(row['epoch'] == epoch and row['optimizer_updates'] == 48362,
             'Incomplete/nonconsecutive epoch cannot be selected')
        rank = exact_rank(row['development_validation']['confusion_true_rows_predicted_columns'])
        improved = best is None or rank > best
        need(row['improved'] is improved, 'Exact selection history mismatch')
        wait = 0 if improved else wait + 1
        if improved:
            best, chosen = rank, epoch
        if epoch < len(history):
            need(wait < 3, 'History exceeds patience boundary')
        need(row['wait'] == wait, 'Patience history differs')
    return chosen, wait, best


def validate_budget(value, reservation_id):
    need(value.get('version') == 'TCN-existing-data-permanent-budget-1.0', 'Budget version differs')
    need(value.get('maximum_runs') == 4 and value.get('maximum_epochs_per_run') == 10
         and value.get('maximum_complete_epochs') == 40
         and value.get('maximum_optimizer_updates') == 1934480
         and value.get('maximum_optimizer_updates_per_run') == 483620, 'Budget caps differ')
    rows = value.get('reserved_runs')
    expected = ['feature42', 'strategy42', 'control43'] + (['candidate43'] if SLOT == 'candidate43' else [])
    need(isinstance(rows, list) and [r.get('slot_id', r.get('id')) for r in rows] == expected,
         'The permanent reservation prefix/order differs')
    for i, row in enumerate(rows):
        need(type(row.get('optimizer_updates')) is int and 0 <= row['optimizer_updates'] <= 483620
             and type(row.get('consumed_epochs')) is int and 0 <= row['consumed_epochs'] <= 10,
             'Invalid permanent slot counters')
        need(row.get('seed') == (42 if i < 2 else 43), 'Permanent slot seed changed')
        if i < len(rows) - 1:
            need(row.get('status') == 'completed' and row['consumed_epochs'] >= 1
                 and row['optimizer_updates'] == 48362 * row['consumed_epochs'],
                 'Every earlier slot must already be completely closed')
    item = rows[-1]
    need(item.get('reservation_id') == reservation_id, 'Current slot reservation differs')
    need(value.get('optimizer_updates') == sum(r['optimizer_updates'] for r in rows)
         and value['optimizer_updates'] <= 1934480, 'Global update count mismatch/overrun')
    need(value.get('consumed_epochs') == sum(r['consumed_epochs'] for r in rows)
         and value['consumed_epochs'] <= 40, 'Global complete epoch mismatch/overrun')
    return item


def validate_prefix_snapshot(value, snapshot):
    need(isinstance(snapshot.get('reserved_runs'), list)
         and value['reserved_runs'][:-1] == snapshot['reserved_runs'],
         'Completed permanent reservation prefix was modified')
    need(snapshot.get('optimizer_updates') == sum(r['optimizer_updates'] for r in snapshot['reserved_runs'])
         and snapshot.get('consumed_epochs') == sum(r['consumed_epochs'] for r in snapshot['reserved_runs']),
         'Previous immutable budget snapshot counters differ')
    return snapshot['reserved_runs']


class BudgetWriter:
    """Only current slot/global counters change; all preceding rows stay exact."""
    def __init__(self, reservation_id, initial_binding):
        self.reservation_id = reservation_id
        self.expected_sha = initial_binding['sha256']
        value = read_binding(initial_binding)
        item = validate_budget(value, reservation_id)
        need(item['optimizer_updates'] == item['consumed_epochs'] == 0,
             'A consumed slot cannot cold restart or silently resume')
        self.prefix_snapshot = value['reserved_runs'][:-1]
        self.attempts = 0; self.completed = 0; self.epochs = 0

    def change(self, *, debit=False, complete_epoch=False, status=None):
        need(sha(BUDGET) == self.expected_sha, 'Budget changed outside the sole current trainer')
        value = read_json(BUDGET); item = validate_budget(value, self.reservation_id)
        need(value['reserved_runs'][:-1] == self.prefix_snapshot,
             'An earlier completed permanent reservation was modified')
        need(item['optimizer_updates'] == self.attempts and item['consumed_epochs'] == self.epochs,
             'Current trainer/ledger counters differ')
        if debit:
            need(item['optimizer_updates'] < 483620 and value['optimizer_updates'] < 1934480,
                 'Permanent optimizer attempt cap exhausted')
            item['optimizer_updates'] += 1; value['optimizer_updates'] += 1
        if complete_epoch:
            need(item['consumed_epochs'] < 10 and value['consumed_epochs'] < 40,
                 'Permanent complete epoch cap exhausted')
            item['consumed_epochs'] += 1; value['consumed_epochs'] += 1
        if status is not None:
            item['status'] = status
        item['actual_optimizer_steps_completed_last_snapshot'] = self.completed
        item['last_update_utc'] = utc()
        value['status'] = ('ED6_' + SLOT + '_completed' if status == 'completed' else
                           'ED6_' + SLOT + '_failed_budget_preserved' if status == 'failed_no_automatic_retry' else
                           'ED6_' + SLOT + '_training_active')
        validate_budget(value, self.reservation_id)
        atomic_json(BUDGET, value); self.expected_sha = sha(BUDGET)
        if debit: self.attempts += 1
        if complete_epoch: self.epochs += 1


def same_binding(left, right):
    return checked_path(left['path']) == checked_path(right['path']) and left['sha256'] == right['sha256']


def successful_receipt(ref):
    value = read_binding(ref)
    need(value.get('exit_code') == 0 and value.get('entry_unchanged') is True
         and value['entry_before'] == value['entry_after'], 'Actual preceding CLI did not succeed unchanged')
    for key in ('entry_before', 'entry_after', 'stdout', 'stderr', 'capture_source'):
        current = value[key]; path = checked_path(current['path'])
        need(path.is_file() and sha(path) == current['sha256']
             and ('bytes' not in current or path.stat().st_size == current['bytes']), 'Preceding CLI evidence changed')
    return value


def candidate_order_gate(reg, lock_ref):
    refs = {
        'training': reg['control_training_receipt_binding'],
        'audit': reg['control_audit_receipt_binding'],
        'checkpoint': reg['control_checkpoint_receipt_binding']}
    actual = {key: successful_receipt(ref) for key, ref in refs.items()}
    accepted = read_binding(reg['control_acceptance_binding'])
    audited = read_binding(reg['control_audit_binding'])
    checked = read_binding(reg['control_checkpoint_validation_binding'])
    need(checked_path(reg['control_acceptance_binding']['path'])
         == ROOT_STAGE / 'trials/control43/run/training_acceptance.json', 'Only this ED6 control may precede candidate')
    need(same_binding(actual['training']['outputs']['training_acceptance'], reg['control_acceptance_binding'])
         and same_binding(actual['audit']['outputs']['audit'], reg['control_audit_binding'])
         and same_binding(actual['checkpoint']['outputs']['checkpoint_validation'],
                          reg['control_checkpoint_validation_binding']), 'Preceding successful CLI outputs differ')
    need(accepted['status'] == 'completed' and accepted['slot_id'] == 'control43'
         and accepted['identity']['selection_lock_binding']['sha256'] == lock_ref['sha256']
         and audited.get('passed') is True and checked.get('passed') is True
         and checked.get('actual_new_checkpoint_CPU_loads') == 2,
         'Completed control / independent audit / bounded CPU check missing')
    need(same_binding(audited['training_receipt_binding'], reg['control_training_receipt_binding'])
         and same_binding(audited['registration_binding'], accepted['identity']['registration_binding']),
         'Control audit belongs to another actual trial/registration')
    checkpoint_registration = read_binding(checked['registration_binding'])
    need(same_binding(checkpoint_registration['training_receipt_binding'], reg['control_training_receipt_binding'])
         and same_binding(checkpoint_registration['training_registration_binding'],
                          accepted['identity']['registration_binding']), 'Control CPU check belongs to another trial')
    return accepted


def admit(registration_path, registration_sha):
    p = checked_path(registration_path, STAGE)
    need(p.parent == STAGE and sha(p) == registration_sha, 'Wrong ED6 trial registration')
    reg = read_json(p)
    need(reg.get('version') == 'TCN-ED6-training-registration-1.0' and reg.get('step') == 'ED6'
         and reg.get('slot_id') == SLOT, 'Wrong registered paired slot')
    auth = read_binding(reg['authorization_binding'])
    need(auth.get('human_authorization') == {'source': 'direct_current_human_user_message',
         'quote': '帮我完成第6步', 'step': 'ED6'} and auth.get('training_authorized') is True,
         'Missing current human ED6 authorization')
    protocol = read_binding(reg['protocol_binding'])
    need(reg['protocol_binding']['sha256'] == PROTOCOL_SHA and protocol['campaign_id'] == 'TCN_EDO_20261017',
         'Protocol identity differs')
    need(protocol['data']['train_dates'] == TRAIN and protocol['data']['development_dates'] == DEV
         and protocol['data']['historical_dates_excluded_from_new_execution'] == EXCLUDED, 'Date scope differs')
    lock = read_binding(reg['selection_lock_binding'])
    need(lock.get('version') == 'TCN-ED6-selection-lock-1.0' and lock.get('campaign_id') == 'TCN_EDO_20261017'
         and lock.get('branch') == 'trained_candidate' and lock.get('candidate_family') == 'ED4_feature42_lr0.001'
         and lock.get('replication_seed') == 43 and lock.get('representative_seed') == 42
         and lock.get('decision') == 'original_argmax' and lock.get('stationary_logit_bias') == 0
         and lock.get('window_length') == 100 and lock.get('horizon') == 50
         and lock.get('common_DEV_samples') == 6612655, 'Frozen pre-seed43 family/branch/anchor settings differ')
    need(lock.get('source_bindings') == reg['source_bindings'] and set(reg['source_bindings']) == SOURCE_NAMES,
         'Frozen paired numerical source set differs')
    for name, digest in reg['source_bindings'].items():
        need(sha(checked_path(name)) == digest, 'Frozen numerical source changed: ' + name)
    need(set(lock.get('config_bindings', {})) == {'control43', 'candidate43'}, 'Both configs must be frozen before control')
    for slot in ('control43', 'candidate43'):
        validate_config(read_binding(lock['config_bindings'][slot]), slot)
    need(same_binding(reg['config_binding'], lock['config_bindings'][SLOT]), 'Current config differs from family lock')
    need(set(lock.get('feature_cache_bindings', {})) == {'control43', 'candidate43'}
         and set(lock.get('normalization_bindings', {})) == {'control43', 'candidate43'}
         and same_binding(reg['feature_cache_binding'], lock['feature_cache_bindings'][SLOT]),
         'Both prepared data identities must be locked before the first seed43 model')
    config = validate_config(read_binding(reg['config_binding']))
    parent = read_binding(lock['frozen_parent_completion_binding'])
    need(parent.get('version') == 'TCN-ED4-completion-1.0' and parent.get('passed') is True
         and parent.get('phase_id') == 'ED4' and parent.get('strict_accuracy_improvement_for_later_replication') is True,
         'Representative seed42 ED4 family evidence differs')
    need(reg['prepared_manifest_binding'] == lock['prepared_manifest_binding'] == protocol['data']['source_manifest'],
         'P2 common anchor parent binding differs')
    manifest = read_binding(reg['prepared_manifest_binding'])
    need(set(manifest['sources']) == set(TRAIN + DEV)
         and manifest['sample_counts'] == {'train': 12380446, 'tuning_validation': 6612655}
         and len(manifest['ranges']) == 7 and sorted(r['date'] for r in manifest['ranges']) == TRAIN + DEV,
         'Prepared common data date/sample scope differs')
    for r in manifest['ranges']:
        need(r['endpoint_start'] == r['start_row'] + 99 and r['endpoint_stop_exclusive'] == r['stop_row'] - 50
             and r['window_count'] == r['endpoint_stop_exclusive'] - r['endpoint_start'],
             'Complete causal input/label footprint differs')
    prepared = read_binding(reg['feature_cache_binding'])
    need(prepared.get('p2_manifest_binding', {}).get('sha256') == reg['prepared_manifest_binding']['sha256'],
         'Cache common P2 anchor origin differs')
    norm = read_binding(prepared['normalization_binding'])
    need(norm == prepared['normalization']
         and same_binding(prepared['normalization_binding'], lock['normalization_bindings'][SLOT]),
         'Actual cache normalization metadata differs from the pre-seed43 lock')
    if SLOT == 'control43':
        need(prepared.get('version') == 'TCN-ED6-original-row-cache-1.0'
             and prepared.get('feature_id') == 'original_physical_prices_raw_quantities_frozen_P2_norm_v1'
             and prepared['normalization_binding']['sha256'] == protocol['data']['baseline_normalization']['sha256'],
             'Control must use original physical prices and the original P2 normalization')
        need(prepared.get('code_binding', {}).get('sha256')
             == reg['source_bindings']['scripts/tcn_existing_data_ed6_data.py'], 'Control adapter source identity differs')
        expected_cache_stage = ROOT_STAGE / 'prepared_control'
    else:
        need(prepared.get('version') == 'TCN-ED4-relative-row-cache-1.0'
             and prepared.get('feature_id') == 'same_record_mid_relative_prices_raw_quantities_v1'
             and prepared['normalization_binding']['sha256'] == parent['normalization']['sha256'],
             'Candidate must retain the already frozen relative-price family')
        need(prepared.get('code_binding', {}).get('sha256')
             == reg['source_bindings']['scripts/tcn_existing_data_ed4_features.py'], 'Candidate adapter source differs')
        parent_reg = read_binding(parent['registration'])
        need(same_binding(parent_reg['feature_cache_binding'], reg['feature_cache_binding']),
             'Candidate cache differs from the selected seed42 family')
        expected_cache_stage = CAMPAIGN / 'step04_features/prepared'
    if 'statistics_independent_review_binding' in prepared:
        need(read_binding(prepared['statistics_independent_review_binding']) == prepared['statistics_independent_review']
             and prepared['statistics_independent_review'].get('passed') is True, 'Inherited normalization review differs')
    review = read_binding(reg['source_review_binding']); tests = read_binding(reg['test_evidence_binding'])
    need(review.get('passed') is True and tests.get('passed') is True and tests.get('actual_exit_code') == 0,
         'Independent source review / actual synthetic tests have not passed')
    prep = successful_receipt(reg['preparation_receipt_binding'])
    saved_cache = prep.get('outputs', {}).get('feature_cache_manifest', {})
    need(checked_path(saved_cache.get('path', ''), expected_cache_stage)
         == checked_path(reg['feature_cache_binding']['path'], expected_cache_stage)
         and saved_cache.get('sha256') == reg['feature_cache_binding']['sha256']
         and saved_cache.get('bytes') == reg['feature_cache_binding']['bytes'], 'Actual cache preparation output differs')
    need(checked_path(reg['budget_binding_at_reservation']['path']) == BUDGET, 'Wrong campaign budget path')
    budget = read_binding(reg['budget_binding_at_reservation']); item = validate_budget(budget, reg['reservation_id'])
    snapshot = read_binding(reg['budget_prefix_snapshot_binding'])
    validate_prefix_snapshot(budget, snapshot)
    need(item['optimizer_updates'] == item['consumed_epochs'] == 0
         and item.get('config_sha256') == reg['config_binding']['sha256'], 'This permanent paired slot was already used')
    control = candidate_order_gate(reg, reg['selection_lock_binding']) if SLOT == 'candidate43' else None
    control_init = None
    if SLOT == 'candidate43':
        control_reg = read_binding(control['identity']['registration_binding'])
        need(same_binding(control_reg['selection_lock_binding'], reg['selection_lock_binding']),
             'Family was changed after observing control43')
        control_init = read_binding(reg['control_initialization_binding'])
        need(checked_path(reg['control_initialization_binding']['path'])
             == ROOT_STAGE / 'trials/control43/run/initialization.json'
             and control_init.get('seed') == 43 and control_init.get('fresh_cold_start') is True
             and control_init.get('historical_checkpoint_loaded') is False, 'Control seed43 initialization metadata differs')
    need(not (STAGE / 'training_phase_reservation.json').exists() and not (STAGE / 'run').exists(),
         'Paired slot already launched; inspect evidence rather than repeating')
    return {'registration': reg, 'registration_binding': binding(p), 'protocol': protocol, 'lock': lock,
            'config': config, 'manifest': manifest, 'prepared': prepared, 'control_acceptance': control,
            'control_initialization': control_init}


class TemporaryACAwake:
    class PowerStatus(ctypes.Structure):
        _fields_ = [('ACLineStatus', ctypes.c_ubyte), ('BatteryFlag', ctypes.c_ubyte),
                    ('BatteryLifePercent', ctypes.c_ubyte), ('SystemStatusFlag', ctypes.c_ubyte),
                    ('BatteryLifeTime', ctypes.c_uint32), ('BatteryFullLifeTime', ctypes.c_uint32)]

    def __init__(self):
        self.active = False; self.last_check = -math.inf; self.events = []
        self.cleanup_succeeded = False; self.started = False

    def update(self):
        if time.monotonic() - self.last_check < 10:
            return
        self.last_check = time.monotonic()
        api = ctypes.windll.kernel32
        api.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
        api.SetThreadExecutionState.restype = ctypes.c_uint32
        status = self.PowerStatus(); known = bool(api.GetSystemPowerStatus(ctypes.byref(status)))
        want = known and status.ACLineStatus == 1; self.started = True
        if want != self.active or not self.events:
            requested = 0x80000001 if want else 0x80000000
            previous = int(api.SetThreadExecutionState(requested))
            need(previous != 0, 'Temporary awake API failed')
            self.active = want
            self.events.append({'utc': utc(), 'ACLineStatus': int(status.ACLineStatus) if known else None,
                                'battery_percent': int(status.BatteryLifePercent) if known else None,
                                'requested': requested, 'returned_previous': previous,
                                'temporary_system_idle_request': want})
            emit({'phase': 'temporary_power_request', **self.events[-1]})

    def close(self):
        if self.started:
            returned = int(ctypes.windll.kernel32.SetThreadExecutionState(0x80000000))
            self.cleanup_succeeded = returned != 0; self.active = False
            self.events.append({'utc': utc(), 'requested': 0x80000000,
                                'returned_previous': returned, 'final_cleanup': True})
            need(self.cleanup_succeeded, 'Final temporary awake cleanup failed')


def rng_state(np, torch):
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])],
            'torch_cpu': torch.get_rng_state().clone(),
            'torch_cuda': [x.cpu().clone() for x in torch.cuda.get_rng_state_all()]}


def save_checkpoint(path, payload, torch):
    p = checked_path(path, STAGE)
    temp = p.with_name(p.name + '.tmp')
    with temp.open('wb') as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    temp.replace(p)


def stable_ce(z, labels, np):
    a = z.astype(np.float64); largest = a.max(axis=1)
    return largest + np.log(np.exp(a - largest[:, None]).sum(axis=1)) - a[np.arange(len(a)), labels]


def run_epoch(model, ds, c, epoch, np, torch, counters, guard, *, optimizer=None, budget=None, cache_dir=None):
    from software.training.tcn_research_evaluate_p2 import metrics
    training = optimizer is not None
    need(training == (budget is not None) and training == (cache_dir is None), 'Training/cache/budget roles differ')
    model.train(training)
    batch_size = c['train_batch_size'] if training else c['evaluation_batch_size']
    order = np.random.default_rng(c['seed'] + epoch).permutation(len(ds)) if training else np.arange(len(ds))
    total = np.zeros((3, 3), dtype=np.int64)
    dates = sorted({r['date'] for r in ds.entries})
    need(dates == (TRAIN if training else DEV), 'Dataset date scope differs')
    by_day = {day: np.zeros((3, 3), dtype=np.int64) for day in dates}
    day_loss = {day: 0. for day in dates}; loss_sum = 0.; updates = 0; forwards = 0
    streams = {}
    if not training:
        cache_dir = checked_path(cache_dir, STAGE); cache_dir.mkdir(parents=True, exist_ok=False)
        for day in DEV:
            folder = cache_dir / day; folder.mkdir()
            streams[day] = {kind: (folder / filename).open('xb') for kind, filename in
                            [('logits', 'logits.f32'), ('labels', 'labels.u8'), ('anchors', 'anchors.u64')]}
    try:
        with torch.set_grad_enabled(training):
            for begin in range(0, len(order), batch_size):
                guard.update()
                ids = order[begin:begin + batch_size]
                details = ds.batch_details(ids); x, y = details['X'], details['y']
                need(x.shape == (len(ids), 40, 100) and y.shape == (len(ids),), 'Invalid tensor batch shape')
                tx = torch.from_numpy(x).to(c['device']); ty = torch.from_numpy(y).to(c['device'])
                logits = model(tx); counters['forward_calls'] += 1; forwards += 1
                counters['train_windows' if training else 'development_windows'] += len(ids)
                each_loss = torch.nn.functional.cross_entropy(logits, ty, reduction='none'); loss = each_loss.mean()
                need(torch.isfinite(logits).all().item() and torch.isfinite(loss).item(), 'Nonfinite logits/loss')
                if training:
                    need(updates < 48362, 'Per-epoch optimizer cap exhausted')
                    optimizer.zero_grad(set_to_none=True); loss.backward(); counters['backward_calls'] += 1
                    need(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
                         'Missing/nonfinite gradients')
                    budget.change(debit=True)
                    counters['optimizer_attempts_charged'] = budget.attempts
                    optimizer.step(); budget.completed += 1; counters['optimizer_steps_completed'] += 1; updates += 1
                z = logits.detach().cpu().numpy()
                prediction = z.argmax(1)
                loss_values = each_loss.detach().cpu().numpy() if training else stable_ce(z, y, np)
                total += np.bincount(3 * y + prediction, minlength=9).reshape(3, 3)
                # Exactly the original global-role batches; grouping only separates reporting/cache rows.
                for day in np.unique(details['date']):
                    need(day in dates, 'Forbidden batch date')
                    mask = details['date'] == day
                    by_day[str(day)] += np.bincount(3 * y[mask] + prediction[mask], minlength=9).reshape(3, 3)
                    day_loss[str(day)] += float(loss_values[mask].sum(dtype=np.float64))
                    if not training:
                        streams[str(day)]['logits'].write(np.asarray(z[mask], dtype='<f4').tobytes())
                        streams[str(day)]['labels'].write(np.asarray(y[mask], dtype='u1').tobytes())
                        streams[str(day)]['anchors'].write(np.asarray(details['endpoints'][mask], dtype='<u8').tobytes())
                loss_sum += float(loss.detach().cpu()) * len(ids) if training else float(loss_values.sum(dtype=np.float64))
                if begin == 0 or (begin // batch_size + 1) % 1000 == 0:
                    emit({'phase': 'TRAIN' if training else 'DEV', 'epoch': epoch,
                          'windows_processed': begin + len(ids), 'windows_total': len(ds),
                          'optimizer_attempts_charged': counters['optimizer_attempts_charged']})
    finally:
        for day_streams in streams.values():
            for stream in day_streams.values():
                stream.flush(); os.fsync(stream.fileno()); stream.close()
    result = metrics(total.tolist())
    need(result['samples'] == len(ds), 'A window was dropped/repeated')
    result.update(loss=loss_sum / len(ds),
                  per_day={day: {**metrics(m.tolist()), 'loss': day_loss[day] / int(m.sum())}
                           for day, m in by_day.items()},
                  scope='online_training_before_each_update' if training else 'fixed_checkpoint_development',
                  loss_arithmetic='float32_CE_batch_mean' if training else 'cached_float32_logits_float64_logsumexp')
    cache = None
    if not training:
        cache = {'version': 'TCN-ED6-epoch-DEV-cache-1.0', 'epoch': epoch, 'decision': 'original_argmax',
                 'class_order': ['down', 'stationary', 'up'], 'per_day': {}}
        for day in DEV:
            count = result['per_day'][day]['samples']; folder = cache_dir / day
            refs = {name: {**binding(folder / filename), 'dtype': dtype, 'shape': shape} for name, filename, dtype, shape in
                    [('logits', 'logits.f32', '<f4', [count, 3]), ('labels', 'labels.u8', 'u1', [count]),
                     ('anchors', 'anchors.u64', '<u8', [count])]}
            need(refs['logits']['bytes'] == count * 12 and refs['labels']['bytes'] == count
                 and refs['anchors']['bytes'] == count * 8, 'Incomplete DEV cache')
            expected = next(r for r in ds.entries if r['date'] == day)
            need(count == expected['window_count'], 'Cache anchor denominator differs')
            cache['per_day'][day] = {'samples': count, 'metrics': result['per_day'][day], 'files': refs}
        write_new(cache_dir / 'cache_manifest.json', cache)
    return result, updates, forwards, cache


def train(registration_path, registration_sha):
    started = time.perf_counter(); context = admit(registration_path, registration_sha)
    reg, c = context['registration'], context['config']
    write_new(STAGE / 'training_phase_reservation.json', {
        'version': 'TCN-ED6-permanent-training-launch-1.0', 'reserved_utc': utc(),
        'registration_binding': context['registration_binding'], 'reservation_id': reg['reservation_id'],
        'counts_even_if_failure_before_model': True, 'worker_pid': os.getpid(), 'no_automatic_retry': True})
    folder = STAGE / 'run'; folder.mkdir(exist_ok=False)
    counters = {'models_constructed': 0, 'historical_checkpoint_loads': 0, 'optimizers_constructed': 0,
                'forward_calls': 0, 'backward_calls': 0, 'optimizer_attempts_charged': 0,
                'optimizer_steps_completed': 0, 'train_windows': 0, 'development_windows': 0}
    guard = TemporaryACAwake(); budget = None; train_ds = dev_ds = None; error = None
    try:
        budget = BudgetWriter(reg['reservation_id'], reg['budget_binding_at_reservation'])
        budget.change(status='running')
        # Metadata/label admission before first numerical/model import. This helper reads no raw records here.
        from software.training.tcn_research_data_p2 import load_prepared
        data = load_prepared(reg['prepared_manifest_binding']['path'], reg['prepared_manifest_binding']['sha256'])
        from experiments.reference.training.scripts.tcn_existing_data_ed6_data import CachedWindows, validate_cache
        prepared = context['prepared']; validate_cache(data, prepared, SLOT)
        cache_before = verify_cache_payloads(data, prepared)
        write_new(folder / 'cache_payloads_before.json', {'passed': True, 'per_day': cache_before})
        from software.training.tcn_research_evaluate_p2 import runtime, state_digest
        np, torch = runtime()
        need(str(np.__version__) == '2.2.6' and str(torch.__version__) == '2.7.1+cu128',
             'Registered numerical runtime changed')
        random.seed(43); np.random.seed(43); torch.manual_seed(43); torch.cuda.manual_seed_all(43)
        from software.model.tcn_research_p5_model import build_model
        model = build_model(True).to(c['device']); counters['models_constructed'] += 1
        need(sum(p.numel() for p in model.parameters()) == 33539, 'WN architecture changed')
        initialization = {'seed': 43, 'model_state_sha256': state_digest(model),
                          'parameter_count': 33539, 'all_gradients_none': all(p.grad is None for p in model.parameters()),
                          'historical_checkpoint_loaded': False, 'fresh_cold_start': True}
        if SLOT == 'candidate43':
            need(initialization['model_state_sha256'] == context['control_initialization']['model_state_sha256'],
                 'Paired seed43 initialization differs from the control43 recipe')
        write_new(folder / 'initialization.json', initialization)
        identity = {'entry_version': VERSION, 'registration_binding': context['registration_binding'],
                    'config_binding': reg['config_binding'], 'manifest_binding': reg['prepared_manifest_binding'],
                    'feature_cache_binding': reg['feature_cache_binding'], 'source_bindings': reg['source_bindings'],
                    'python': platform.python_version(), 'numpy': str(np.__version__), 'torch': str(torch.__version__),
                    'cuda': str(torch.version.cuda), 'gpu': torch.cuda.get_device_name(0),
                    'determinism': {'threads': 4, 'tf32': False, 'deterministic_algorithms': True, 'precision': 'float32'},
                    'reservation_id': reg['reservation_id']}
        identity.update(slot_id=SLOT, selection_lock_binding=reg['selection_lock_binding'],
                        budget_prefix_snapshot_binding=reg['budget_prefix_snapshot_binding'],
                        normalization_binding=prepared['normalization_binding'],
                        frozen_seed42_parent_binding=context['lock']['frozen_parent_completion_binding'],
                        control_initialization_binding=reg.get('control_initialization_binding'),
                        feature_family=c['feature_family'], seed=43)
        write_new(folder / 'identity.json', identity); write_new(folder / 'config.json', c)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        counters['optimizers_constructed'] += 1
        train_ds = CachedWindows(data, prepared, 'train', SLOT)
        dev_ds = CachedWindows(data, prepared, 'tuning_validation', SLOT)
        need(len(train_ds) == 12380446 and len(dev_ds) == 6612655, 'Prepared window counts differ')
        history = []; best = None; selected = None; wait = 0
        for epoch in range(1, 11):
            begin = time.perf_counter()
            training, updates, train_calls, _ = run_epoch(model, train_ds, c, epoch, np, torch, counters, guard,
                                                        optimizer=optimizer, budget=budget)
            need(updates == train_calls == 48362, 'Incomplete TRAIN epoch')
            development, eval_updates, eval_calls, cache = run_epoch(model, dev_ds, c, epoch, np, torch, counters, guard,
                                                                   cache_dir=folder / f'epoch{epoch:02d}_DEV')
            need(eval_updates == 0 and eval_calls == 25831, 'DEV optimizer/forward boundary differs')
            candidate = exact_rank(development['confusion_true_rows_predicted_columns'])
            improved = best is None or candidate > best
            wait = 0 if improved else wait + 1
            if improved:
                best, selected = candidate, epoch
            history.append({'epoch': epoch, 'train': training, 'development_validation': development,
                            'improved': improved, 'wait': wait, 'optimizer_updates': updates,
                            'train_forward_calls': train_calls, 'development_forward_calls': eval_calls,
                            'cache_binding': binding(folder / f'epoch{epoch:02d}_DEV/cache_manifest.json'),
                            'seconds': time.perf_counter() - begin})
            check_selected, check_wait, check_best = recompute_history(history)
            need((check_selected, check_wait, check_best) == (selected, wait, best), 'Selection replay differs')
            budget.change(complete_epoch=True)
            cp = {'version': CP_VERSION, 'experiment': 'TCN_EDO_20261017_ED6_' + SLOT,
                  'identity': identity, 'config': c, 'epoch': epoch, 'history': history,
                  'selected_epoch': selected, 'wait': wait, 'best_rank': rank_json(best),
                  'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': rng_state(np, torch),
                  'optimizer_attempts_charged': budget.attempts, 'optimizer_steps_completed': budget.completed,
                  'cache_binding': history[-1]['cache_binding']}
            save_checkpoint(folder / f'epoch{epoch:02d}.pt', cp, torch)
            save_checkpoint(folder / 'latest.pt', cp, torch)
            if improved:
                save_checkpoint(folder / 'best.pt', cp, torch)
            atomic_json(folder / 'history.json', history)
            atomic_json(folder / 'status.json', {'status': 'running', 'last_completed_epoch': epoch,
                        'selected_epoch': selected, 'wait': wait, 'counters': counters,
                        'current_epoch_metrics': development, 'budget_binding': binding(BUDGET)})
            emit({'phase': 'complete_epoch', **history[-1], 'selected_epoch': selected})
            if wait >= 3:
                break
        need(budget.attempts == budget.completed == counters['optimizer_steps_completed'],
             'Completed run has uncertain optimizer attempts')
        validate_cache(data, prepared, SLOT)
        cache_after = verify_cache_payloads(data, prepared)
        need(cache_after == cache_before, 'Feature cache or original labels changed during training')
        write_new(folder / 'cache_payloads_after.json', {'passed': True, 'per_day': cache_after})
        for name, digest in reg['source_bindings'].items():
            need(sha(ROOT / name) == digest, 'Source changed during training')
        budget.change(status='completed')
        budget_snapshot_path = folder / 'budget_at_completion_snapshot.json'
        with budget_snapshot_path.open('xb') as stream:
            stream.write(BUDGET.read_bytes()); stream.flush(); os.fsync(stream.fileno())
        need(sha(budget_snapshot_path) == budget.expected_sha,
             'Immutable terminal budget snapshot differs from this writer final budget')
        acceptance = {'version': 'TCN-ED6-training-acceptance-1.0', 'status': 'completed',
                      'worker_pid': os.getpid(), 'actual_worker_pid': os.getpid(), 'identity': identity,
                      'epochs_completed': len(history), 'selected_epoch': selected,
                      'stop_reason': 'patience3' if wait >= 3 else 'max10',
                      'selected_development_metrics': history[selected - 1]['development_validation'],
                      'selected_cache_binding': history[selected - 1]['cache_binding'],
                      'counters': counters, 'optimizer_attempts_charged': budget.attempts,
                      'optimizer_steps_completed': budget.completed, 'budget_binding': binding(BUDGET),
                      'checkpoint_bindings': {name: binding(folder / name) for name in ('best.pt', 'latest.pt')},
                      'epoch_checkpoint_bindings': {str(i): binding(folder / f'epoch{i:02d}.pt') for i in range(1, len(history) + 1)},
                      'history_binding': binding(folder / 'history.json'),
                      'cache_payloads_before_binding': binding(folder / 'cache_payloads_before.json'),
                      'cache_payloads_after_binding': binding(folder / 'cache_payloads_after.json'),
                      'seconds': time.perf_counter() - started, 'development_validation_only': True,
                      'sealed_dates_opened': False, 'original_flag64_cleared': False,
                      'source_control_verified': False, 'coverage_unknown': True,
                      'result_wording': '基于当前可观察数据和既定筛选规则的开发验证表现',
                      'original_argmax_without_ED3_bias': True, 'automatic_next_action': False}
        acceptance.update(slot_id=SLOT, selection_lock_binding=reg['selection_lock_binding'],
                          budget_prefix_snapshot_binding=reg['budget_prefix_snapshot_binding'],
                          budget_completion_snapshot_binding=binding(budget_snapshot_path),
                          normalization_binding=prepared['normalization_binding'],
                          paired_seed43_initialization_matches_control=(True if SLOT == 'candidate43' else None))
        write_new(folder / 'training_acceptance.json', acceptance)
        atomic_json(folder / 'status.json', acceptance)
        return acceptance
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        if budget is not None:
            try:
                budget.change(status='failed_no_automatic_retry')
            except BaseException as budget_exc:
                error['budget_cleanup_error'] = str(budget_exc)
        atomic_json(folder / 'status.json', {'status': 'failed_no_automatic_retry', 'error': error,
                    'counters': counters, 'worker_pid': os.getpid(),
                    'permanently_charged_attempts': budget.attempts if budget is not None else None,
                    'actual_steps_completed': budget.completed if budget is not None else None})
        raise
    finally:
        try:
            if train_ds is not None: train_ds.close()
            if dev_ds is not None: dev_ds.close()
        finally:
            try:
                guard.close()
            finally:
                write_new(folder / 'resource_cleanup.json', {
                    'worker_pid': os.getpid(), 'utc': utc(), 'temporary_AC_only_request': True,
                    'started': guard.started, 'events': guard.events, 'cleanup_succeeded': guard.cleanup_succeeded,
                    'request_active_after_finally': guard.active, 'permanent_power_or_display_settings_changed': False,
                    'run_error': error, 'counters': counters})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slot', required=True, choices=['control43', 'candidate43'])
    parser.add_argument('--registration', required=True)
    parser.add_argument('--registration-sha', '--registration-sha256', dest='registration_sha', required=True)
    parser.add_argument('command', choices=['admit', 'train'], nargs='?', default='train')
    args = parser.parse_args(argv)
    try:
        configure(args.slot)
        if args.command == 'admit':
            context = admit(args.registration, args.registration_sha)
            result = {'passed': True, 'registration_binding': context['registration_binding'],
                      'no_ML_imported': True, 'formal_training_started': False}
        else:
            result = train(args.registration, args.registration_sha)
        emit(result); return 0
    except BaseException as exc:
        emit({'status': 'rejected_or_failed', 'type': type(exc).__name__, 'message': str(exc),
              'traceback': traceback.format_exc(), 'automatic_retry': False})
        return 2


if __name__ == '__main__':
    sys.exit(main())
