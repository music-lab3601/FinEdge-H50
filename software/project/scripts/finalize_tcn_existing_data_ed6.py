"""ED6 paired-development publication after both actual, independently audited runs.

Standard-library metadata, opaque SHA and exact integer confusion arithmetic
only. No model/checkpoint deserialization, E records, new evaluation, budget
mutation, candidate search, or ED7 execution is implemented here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'artifacts/tcn_existing_data_optimization_2026-10-17'
STAGE = CAMPAIGN / 'step06_replication'
BUDGET = CAMPAIGN / 'budget.json'
PROTOCOL_SHA = '09654e6dfdb10b69feb6ee950c6668e71640bcea1d56fa9691a9564c8641fcf3'
SLOTS = ('control43', 'candidate43')
DEV = ('2025-12-15', '2025-12-16')


def need(value, message):
    if not value:
        raise ValueError(message)


def local(value):
    p = Path(value)
    if not p.is_absolute(): p = ROOT / p
    need('..' not in p.parts and p.is_relative_to(ROOT), 'Outside D workspace')
    for q in (p, *p.parents):
        if q.exists():
            need(not q.is_symlink() and not getattr(q.lstat(), 'st_file_attributes', 0) & 0x400,
                 'Reparse metadata/output forbidden')
        if q == ROOT: break
    need(p.resolve().is_relative_to(ROOT.resolve()), 'Resolved path escaped workspace')
    return p


def sha(path):
    h = hashlib.sha256()
    with local(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def binding(path):
    p = local(path)
    return {'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def verify(ref):
    p = local(ref['path'])
    need(p.is_file() and sha(p) == ref['sha256']
         and ('bytes' not in ref or p.stat().st_size == ref['bytes']), 'Evidence changed: ' + str(p))
    return p


def same(a, b):
    return local(a['path']) == local(b['path']) and a['sha256'] == b['sha256']


def read(path):
    p = local(path); need(p.stat().st_size <= 32 * 1024**2, 'Oversized metadata')
    return json.loads(p.read_text('utf-8-sig'),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))


def bound(ref):
    return read(verify(ref))


def put(path, value):
    p = local(path)
    need(p.is_relative_to(STAGE) or p == ROOT / 'state/tcn_software_plan_v26.json', 'Wrong publication output')
    with p.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def exact_accuracy(matrix):
    need(isinstance(matrix, list) and len(matrix) == 3 and all(isinstance(r, list) and len(r) == 3
         and all(type(v) is int and v >= 0 for v in r) for r in matrix), 'Bad confusion matrix')
    total = sum(sum(r) for r in matrix); need(total > 0, 'Empty confusion')
    return Fraction(sum(matrix[i][i] for i in range(3)), total)


def pair_rule(control42, candidate42, control43, candidate43):
    # Macro-F1/Recall deliberately do not enter this preregistered adoption gate.
    delta42 = exact_accuracy(candidate42) - exact_accuracy(control42)
    delta43 = exact_accuracy(candidate43) - exact_accuracy(control43)
    stable = delta42 >= 0 and delta43 >= 0 and (delta42 > 0 or delta43 > 0)
    return stable, delta42, delta43


def choose_representative(stable, parent, protocol, baseline_metric):
    """Artifacts always come from the predefined seed42, never best seed43."""
    return {'seed': 42, 'family': 'ED4_feature42_lr0.001_seed42' if stable else 'original_WN42_seed42',
            'metrics': parent['candidate_original_argmax_DEV_metrics'] if stable else baseline_metric,
            'checkpoint': parent['selected_checkpoint'] if stable else protocol['baseline']['checkpoint'],
            'normalization': parent['normalization'] if stable else protocol['data']['baseline_normalization']}


def validate_budget_chain(control_before, control_terminal, candidate_before, candidate_terminal, current):
    """Immutable trial snapshots are authoritative; no old live SHA is read."""
    need(len(control_before['reserved_runs']) == 2 and len(control_terminal['reserved_runs']) == 3
         and len(candidate_terminal['reserved_runs']) == 4, 'Paired reservation count differs')
    need(control_terminal['reserved_runs'][:-1] == control_before['reserved_runs']
         and candidate_before == control_terminal
         and candidate_terminal['reserved_runs'][:-1] == control_terminal['reserved_runs']
         and current == candidate_terminal, 'Immutable completed prefix / terminal snapshots differ')
    for budget in (control_before, control_terminal, candidate_before, candidate_terminal, current):
        rows = budget['reserved_runs']
        need(budget['optimizer_updates'] == sum(r['optimizer_updates'] for r in rows) <= 1934480
             and budget['consumed_epochs'] == sum(r['consumed_epochs'] for r in rows) <= 40
             and all(r['status'] == 'completed' and 0 < r['consumed_epochs'] <= 10
                     and r['optimizer_updates'] == 48362 * r['consumed_epochs'] <= 483620 for r in rows),
             'Immutable completed campaign counters/caps differ')
    return True


def rank(matrix):
    accuracy = exact_accuracy(matrix)
    support = [sum(r) for r in matrix]
    predicted = [sum(matrix[i][j] for i in range(3)) for j in range(3)]
    f1 = sum((Fraction(2 * matrix[i][i], support[i] + predicted[i])
              if support[i] + predicted[i] else Fraction(0) for i in range(3)), Fraction(0)) / 3
    return accuracy, f1


def checked_receipt(path, phase):
    result = read(path)
    need(result['phase'] == phase and result['exit_code'] == 0 and result['entry_unchanged'] is True
         and result['entry_before'] == result['entry_after'], 'Actual CLI not successful: ' + phase)
    for key in ('entry_before', 'entry_after', 'capture_source', 'stdout', 'stderr'):
        verify(result[key])
    if 'sources_before' in result:
        need(result['sources_before'] == result.get('sources_after')
             and result.get('sources_unchanged', True) is True, 'Actual CLI dependency sources changed')
        for ref in result['sources_before']: verify(ref)
    if 'input' in result: verify(result['input'])
    for ref in result.get('outputs', {}).values(): verify(ref)
    return result


def protected():
    inv = read(STAGE / 'protected_inventory.json')
    need(inv['protected_files'] == len(inv['files']) == 287, 'Historical inventory count differs')
    for ref in inv['files']: verify(ref)
    baseline = ROOT / 'artifacts/tcn_frozen_baseline_wn42_2026-10-17'
    paths = [p for p in (baseline / 'bundle').rglob('*') if p.is_file()]
    paths.append(baseline / 'TCN_WN42_Frozen_Baseline_20261017.zip')
    need(len(paths) == 45 and all(getattr(p.stat(), 'st_file_attributes', 0) & 1 for p in paths),
         'Original frozen baseline read-only evidence differs')
    return inv


def admission_trial(slot, lock_path, lock):
    trial = STAGE / 'trials' / slot
    reg_path = trial / 'training_registration.json'; reg = read(reg_path)
    need(reg['version'] == 'TCN-ED6-training-registration-1.0' and reg['slot_id'] == slot
         and same(reg['selection_lock_binding'], binding(lock_path)), 'Trial family/branch lock differs')
    for key in ('authorization_binding', 'protocol_binding', 'config_binding', 'prepared_manifest_binding',
                'feature_cache_binding', 'selection_lock_binding', 'source_review_binding', 'test_evidence_binding',
                'preparation_receipt_binding', 'budget_prefix_snapshot_binding', 'capture_source_binding'):
        verify(reg[key])
    config = bound(reg['config_binding'])
    need(config['seed'] == 43 and config['learning_rate'] == .001 and config['decision'] == 'original_argmax'
         and same(reg['config_binding'], lock['config_bindings'][slot]), 'Paired default config changed')
    need(reg['source_bindings'] == lock['source_bindings'], 'Paired numerical sources changed')
    for name, digest in reg['source_bindings'].items(): need(sha(ROOT / name) == digest, 'Numerical source changed')
    actual = checked_receipt(trial / 'training_execution_receipt.json', 'training')
    need(same(actual['capture_source'], reg['capture_source_binding']), 'Actual primary capture differs from registration')
    cp_actual = checked_receipt(trial / 'checkpoint_execution_receipt.json', 'checkpoint_validation')
    audit_actual = checked_receipt(trial / 'audit_execution_receipt.json', 'audit')
    need(same(actual['input'], binding(reg_path)) and same(audit_actual['input'], binding(reg_path)),
         'Actual training/audit input differs')
    need(datetime.fromisoformat(actual['ended_utc']) <= datetime.fromisoformat(cp_actual['started_utc'])
         and datetime.fromisoformat(cp_actual['ended_utc']) <= datetime.fromisoformat(audit_actual['started_utc']),
         'Actual trial predecessor terminal order differs')
    acceptance = bound(actual['outputs']['training_acceptance']); history = bound(actual['outputs']['history'])
    audit = bound(audit_actual['outputs']['audit']); cp = bound(cp_actual['outputs']['checkpoint_validation'])
    need(acceptance['status'] == 'completed' and acceptance['slot_id'] == slot
         and acceptance['actual_worker_pid'] == acceptance['worker_pid'] == actual['actual_worker_pid']
         and same(acceptance['selection_lock_binding'], binding(lock_path)), 'Actual paired worker/lock differs')
    need(audit['passed'] is True and same(audit['registration_binding'], binding(reg_path))
         and same(audit['training_receipt_binding'], binding(trial / 'training_execution_receipt.json')),
         'Independent trial audit not tied to actual original training')
    need(audit['protected_evidence'] == {'passed': True, 'protected_files': 287, 'baseline_readonly_files': 45}
         and audit['comparison_same_complete_DEV_anchors'] is True and audit['no_ED3_bias_added'] is True,
         'Independent historical/common-anchor/decision checks failed')
    epochs = acceptance['epochs_completed']; selected = acceptance['selected_epoch']
    need(1 <= epochs <= 10 and epochs == len(history) == audit['epochs_recomputed']
         and audit['selected_epoch'] == selected and audit['DEV_windows_recomputed'] == 6612655 * epochs,
         'All complete trial epochs not independently reconstructed')
    best = None; chosen = None; wait = 0
    for i, row in enumerate(history, 1):
        need(row['epoch'] == i and row['optimizer_updates'] == 48362, 'Partial/nonconsecutive complete epoch')
        candidate = rank(row['development_validation']['confusion_true_rows_predicted_columns'])
        improved = best is None or candidate > best; wait = 0 if improved else wait + 1
        if improved: best, chosen = candidate, i
        need(row['improved'] is improved and row['wait'] == wait and (i == epochs or wait < 3), 'Best/patience replay differs')
        verify(row['cache_binding'])
    need(chosen == selected and ((wait >= 3 and acceptance['stop_reason'] == 'patience3')
         or (epochs == 10 and wait < 3 and acceptance['stop_reason'] == 'max10')), 'Stop/selection boundary differs')
    metric = acceptance['selected_development_metrics']
    need(metric == history[selected - 1]['development_validation']
         and metric['confusion_true_rows_predicted_columns']
         == audit['selected_original_argmax_DEV_metrics']['confusion_true_rows_predicted_columns'], 'Selected matrix differs')
    counts = {'models_constructed': 1, 'historical_checkpoint_loads': 0, 'optimizers_constructed': 1,
              'forward_calls': 74193 * epochs, 'backward_calls': 48362 * epochs,
              'optimizer_attempts_charged': 48362 * epochs, 'optimizer_steps_completed': 48362 * epochs,
              'train_windows': 12380446 * epochs, 'development_windows': 6612655 * epochs}
    need(acceptance['counters'] == counts and audit['counters'] == counts, 'Formal paired ML counters differ')
    before = bound(acceptance['cache_payloads_before_binding']); after = bound(acceptance['cache_payloads_after_binding'])
    need(before == after and before['passed'] is True, 'Read-only D cache/labels changed')
    terminal_budget = bound(acceptance['budget_completion_snapshot_binding'])
    previous_budget = bound(reg['budget_prefix_snapshot_binding'])
    # Never read the now legitimately stale control live-budget binding.
    need(terminal_budget['reserved_runs'][:-1] == previous_budget['reserved_runs'], 'Completed reservation prefix changed')
    current = terminal_budget['reserved_runs'][-1]
    need(current.get('slot_id', current.get('id')) == slot and current['status'] == 'completed'
         and current['optimizer_updates'] == 48362 * epochs and current['consumed_epochs'] == epochs
         and terminal_budget['optimizer_updates'] == previous_budget['optimizer_updates'] + 48362 * epochs
         and terminal_budget['consumed_epochs'] == previous_budget['consumed_epochs'] + epochs, 'Terminal budget counters differ')
    cleanup_path = trial / 'run/resource_cleanup.json'; cleanup = read(cleanup_path)
    need(cleanup['worker_pid'] == acceptance['worker_pid'] and cleanup['cleanup_succeeded'] is True
         and cleanup['request_active_after_finally'] is False and cleanup['run_error'] is None
         and cleanup['permanent_power_or_display_settings_changed'] is False
         and cleanup['events'][-1].get('final_cleanup') is True and cleanup['events'][-1]['returned_previous'] != 0,
         'Actual AC-only finally resource cleanup failed')
    need(cp['passed'] is True and cp['actual_new_checkpoint_CPU_loads'] == 2
         and cp['worker_pid'] == cp_actual['actual_worker_pid'], 'Actual two-new-CPU-container check failed')
    cp_reg = bound(cp['registration_binding'])
    need(same(cp_reg['training_registration_binding'], binding(reg_path))
         and same(cp_reg['training_receipt_binding'], binding(trial / 'training_execution_receipt.json')),
         'CPU check belongs to another trial')
    for key in ('models_constructed', 'optimizers_constructed', 'forwards', 'backwards', 'optimizer_updates',
                'historical_checkpoint_loads', 'E_record_reads'): need(cp[key] == 0, 'CPU helper exceeded scope')
    for key in ('extra_model_loads', 'extra_models_constructed', 'extra_forwards', 'extra_optimizers',
                'extra_training_runs_or_updates', 'E_record_reads', 'historical17_19_stream_reads'):
        need(audit[key] == 0, 'Independent D audit exceeded scope')
    for ref in acceptance['checkpoint_bindings'].values(): verify(ref)
    for ref in acceptance['epoch_checkpoint_bindings'].values(): verify(ref)
    return {'slot': slot, 'trial': trial, 'reg': reg, 'actual': actual, 'acceptance': acceptance, 'history': history,
            'audit': audit, 'metric': metric, 'cp': cp, 'cp_actual': cp_actual, 'audit_actual': audit_actual,
            'terminal_budget': terminal_budget, 'previous_budget': previous_budget, 'cleanup': cleanup}


def pct(value): return f'{100 * value:.10f}%'


def metric_text(name, m):
    loss = m.get('loss', m.get('cross_entropy'))
    need(isinstance(loss, (int, float)) and math.isfinite(loss), 'Missing finite reported CE')
    return '\n'.join([f'{name}：N={m["samples"]:,}，Accuracy={pct(m["accuracy"])}，Macro-F1={m["macro_f1"]:.12f}，CE={loss:.12f}',
        '  跌/平/涨 Precision：' + ' / '.join(pct(v) for v in m['precision']),
        '  跌/平/涨 Recall：' + ' / '.join(pct(v) for v in m['recall']),
        '  跌/平/涨 F1：' + ' / '.join(f'{v:.12f}' for v in m['f1']),
        '  混淆矩阵（真实行/预测列，跌平涨）：' + json.dumps(m['confusion_true_rows_predicted_columns'])])


def synthetic_evidence(trials):
    observations = {}
    for name, count in (('cache_tests', 27), ('training_tests', 39)):
        evidence = read(STAGE / (name + '_evidence.json'))
        need(evidence['passed'] is True and evidence['actual_exit_code'] == 0 and evidence['tests_run'] == count,
             'Actual synthetic suite incomplete')
        actual = checked_receipt(verify(evidence['actual_receipt']), name)
        verify(evidence['test_source_binding'])
        need(same(evidence['test_source_binding'], actual['entry_before']), 'Actual synthetic suite source differs')
        observations[name] = binding(STAGE / (name + '_evidence.json'))
    clarification = read(STAGE / 'test_scope_clarification.json')
    need(clarification['cache_tests_actual_count'] == 27 and clarification['training_tests_actual_count'] == 39
         and clarification['actual_tests_repeated'] is False, 'Synthetic scope clarification missing')
    final_test = read(STAGE / 'finalizer_tests_evidence.json')
    need(final_test['passed'] is True and final_test['actual_exit_code'] == 0 and final_test['tests_run'] == 17
         and final_test['implementation_before'] == final_test['implementation_after'],
         'Actual separate finalizer boundary tests incomplete')
    verify(final_test['test_source_binding']); verify(final_test['implementation_before'])
    need(same(final_test['implementation_before'], binding(Path(__file__))), 'Finalizer changed after its actual tests')
    final_actual = bound(final_test['actual_receipt'])
    need(final_actual['phase'] == 'finalizer_tests' and final_actual['exit_code'] == 0
         and final_actual['entry_unchanged'] is True and final_actual['entry_before'] == final_actual['entry_after']
         and same(final_test['test_source_binding'], final_actual['entry_before']), 'Actual finalizer test source differs')
    need(final_actual['implementation_before'] == final_actual['implementation_after']
         == final_test['implementation_before'], 'Actual finalizer test implementation differs from its evidence')
    for key in ('entry_before', 'entry_after', 'stdout', 'stderr'): verify(final_actual[key])
    verify(final_actual['capture_source'])
    # Only the actual recorded capture source is required; no independently
    # unobserved Python worker PID or wrapper identity is invented.
    observations['finalizer_tests'] = binding(STAGE / 'finalizer_tests_evidence.json')
    evidence = read(STAGE / 'postrun_helpers_tests_evidence.json')
    need(evidence['passed'] is True and evidence['actual_exit_code'] == 0
         and type(evidence['separate_synthetic_test_count']) is int
         and evidence['separate_synthetic_test_count'] == 32 and evidence['no_real_data_or_ML'] is True,
         'Actual independent-helper synthetic tests incomplete')
    actual = checked_receipt(verify(evidence['actual_receipt']), 'postrun_helpers_tests')
    need(evidence['sources_before'] == evidence['sources_after'] == actual['sources_before'] == actual['sources_after']
         and same(evidence['test_source'], actual['entry_before']), 'Tested helper source set changed')
    for ref in evidence['sources_before']: verify(ref)
    tested = {local(r['path']) for r in evidence['sources_before']}
    required = {local(evidence['test_source']['path'])}
    for trial in trials.values():
        required.update(local(trial[k]['entry_before']['path']) for k in ('cp_actual', 'audit_actual'))
        required.update(local(trial[k]['capture_source']['path']) for k in ('cp_actual', 'audit_actual'))
        required.add(local(trial['actual']['capture_source']['path']))
    need(required <= tested, 'Actual posterior helper was not covered by the frozen synthetic source set')
    observations.update(postrun_helpers=binding(STAGE / 'postrun_helpers_tests_evidence.json'),
                        scope_clarification=binding(STAGE / 'test_scope_clarification.json'))
    return observations, evidence['separate_synthetic_test_count']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-lock-sha256', required=True)
    args = parser.parse_args(argv)
    need(not (STAGE / 'completion_acceptance.json').exists()
         and not (ROOT / 'state/tcn_software_plan_v26.json').exists(), 'ED6 already complete; inspect saved evidence')
    lock_path = STAGE / 'selection_lock.json'
    need(sha(lock_path) == args.selection_lock_sha256, 'Explicit pre-seed43 selection lock differs')
    lock = read(lock_path)
    need(lock['version'] == 'TCN-ED6-selection-lock-1.0' and lock['branch'] == 'trained_candidate'
         and lock['candidate_family'] == 'ED4_feature42_lr0.001' and lock['replication_seed'] == 43
         and lock['representative_seed'] == 42 and lock['decision'] == 'original_argmax'
         and lock['stationary_logit_bias'] == 0 and lock['common_DEV_samples'] == 6612655,
         'Frozen candidate/branch/representative/common-anchor choice differs')
    control_reg = read(STAGE / 'trials/control43/training_registration.json')
    protocol = bound(control_reg['protocol_binding'])
    need(control_reg['protocol_binding']['sha256'] == PROTOCOL_SHA, 'Original seven-step protocol differs')
    parent = bound(lock['frozen_parent_completion_binding'])
    need(parent['passed'] is True and parent['phase_id'] == 'ED4', 'Frozen representative seed42 parent differs')
    inv = protected()
    trials = {slot: admission_trial(slot, lock_path, lock) for slot in SLOTS}
    tests, helper_count = synthetic_evidence(trials)
    control, candidate = trials['control43'], trials['candidate43']
    need(datetime.fromisoformat(control['audit_actual']['ended_utc'])
         <= datetime.fromisoformat(candidate['actual']['started_utc']), 'Candidate preceded actual successful control audit')
    budget = read(BUDGET)
    need(candidate['reg']['budget_prefix_snapshot_binding']['sha256']
         == control['acceptance']['budget_completion_snapshot_binding']['sha256']
         and sha(BUDGET) == candidate['acceptance']['budget_completion_snapshot_binding']['sha256'],
         'Candidate prefix/current budget are not the exact original terminal bytes')
    validate_budget_chain(control['previous_budget'], control['terminal_budget'],
                          candidate['previous_budget'], candidate['terminal_budget'], budget)
    initial = {slot: read(trials[slot]['trial'] / 'run/initialization.json') for slot in SLOTS}
    need(initial['control43']['model_state_sha256'] == initial['candidate43']['model_state_sha256']
         and all(r['seed'] == 43 and r['historical_checkpoint_loaded'] is False and r['fresh_cold_start'] is True
                 for r in initial.values())
         and candidate['acceptance']['paired_seed43_initialization_matches_control'] is True,
         'The two fresh seed43 model initialization recipes differ')
    # Original WN42 and already frozen ED4 seed42 only: no highest-seed choice.
    baseline_cache_path = CAMPAIGN / 'step02_diagnosis/evaluation/cache_manifest.json'
    need(sha(baseline_cache_path) == '1808492b6b2e118b09a88114bf7af4a837414e1e96a41427fef832e08407c106',
         'Frozen WN42 cache metadata changed')
    baseline_cache = read(baseline_cache_path)
    base42 = protocol['baseline']['historical_full_DEV_metrics'].copy()
    base42['loss'] = sum(baseline_cache['per_day'][d]['metrics']['loss']
                         * baseline_cache['per_day'][d]['metrics']['samples'] for d in DEV) / 6612655
    base42['per_day'] = {d: baseline_cache['per_day'][d]['metrics'] for d in DEV}
    trial42 = parent['candidate_original_argmax_DEV_metrics']
    for slot in SLOTS:
        metric = trials[slot]['metric']
        need(metric['samples'] == 6612655, 'Seed43 complete common DEV denominator differs')
        chosen_cache = bound(trials[slot]['acceptance']['selected_cache_binding'])
        for day in DEV:
            a = chosen_cache['per_day'][day]['files']; b = baseline_cache['per_day'][day]['files']
            need(a['anchors']['sha256'] == b['anchors']['sha256'] and a['labels']['sha256'] == b['labels']['sha256'],
                 'A selected seed43 candidate uses different ordered common anchors/labels')
        prepared = bound(lock['feature_cache_bindings'][slot])
        need(same(prepared['normalization_binding'], lock['normalization_bindings'][slot]), 'Locked normalization changed')
        verify(prepared['normalization_binding'])
        for ref in prepared['cache_files'].values(): verify(ref)
    stable, delta42, delta43 = pair_rule(base42['confusion_true_rows_predicted_columns'],
        trial42['confusion_true_rows_predicted_columns'], control['metric']['confusion_true_rows_predicted_columns'],
        candidate['metric']['confusion_true_rows_predicted_columns'])
    representative = choose_representative(stable, parent, protocol, base42)
    retained, retained_checkpoint = representative['metrics'], representative['checkpoint']
    retained_normalization, retained_family = representative['normalization'], representative['family']
    verify(retained_checkpoint); verify(retained_normalization)
    now = datetime.now(timezone.utc); shanghai = now.astimezone(timezone(timedelta(hours=8)))
    date = shanghai.date().isoformat()
    report = ROOT / ('docs/TCN_现有数据优化第6步配对稳定性复核报告_' + date + '.txt')
    verdict = ('两个预定种子的Accuracy差均非负、至少一个严格为正，满足既定开发验证稳定采用规则；代表固定seed42。'
               if stable else '至少一个预定种子的Accuracy差为负或没有严格改善，不满足稳定采用；默认回退原WN42，ED4仅另存为不稳定改进选项。')
    lines = ['TCN 现有数据优化第6步：配对稳定性复核', f'完成时间（上海）：{shanghai.isoformat()}', '',
             '结论：' + verdict, f'seed42 Accuracy差：{100*float(delta42):+.10f}个百分点（精确{delta42}）；'
             f'seed43差：{100*float(delta43):+.10f}个百分点（精确{delta43}）。',
             f'本轮保留代表：{retained_family}。不是挑选最高分的seed43，也没有按43结果换家族或调参。',
             '稳定采用规则只看Accuracy：两个种子的candidate-minus-control均≥0，且至少一个>0。'
             'seed43若Accuracy完全并列，即使Macro-F1/某类Recall更低也不增加否决；这些指标完整报告。', '',
             '冻结与实际流程：',
             '控制组是原始physical价格、原P2归一化；候选组是已冻结ED4相对中点价格、原始数量及其TRAIN归一化。'
             '两组均seed43全新冷启动，原WN架构、lr0.001、AdamW、dropout0、L100/H50、固定标签/筛选、'
             '全部共同端点、max10/patience3、float32/确定性/TF32关闭/原argmax保持；不叠ED3偏置。',
             '0模型的原始价D缓存准备完成后，一次锁定家族、两个配置/数据/归一化/源码/代表42和共同端点，'
             '才启动control43；实际训练、两新CP容器核验与独立缓存审计成功后才登记candidate43。',
             '每组可因独立patience而有不同实际epoch；未人为对齐停止点或追加训练，未加载历史权重或optimizer。', '',
             metric_text('seed42原WN42控制（继承固定模型证据）', base42),
             metric_text('seed42冻结ED4候选（继承固定模型证据）', trial42),
             metric_text('seed43新控制', control['metric']), metric_text('seed43新候选', candidate['metric']),
             metric_text('固定seed42保留代表', retained), '', '逐日配对对照：']
    for seed, a, b in ((42, base42, trial42), (43, control['metric'], candidate['metric'])):
        for day in DEV:
            lines += [metric_text(f'seed{seed}/{day}控制', a['per_day'][day]),
                      metric_text(f'seed{seed}/{day}候选', b['per_day'][day]),
                      f'  当日Accuracy差：{100*(b["per_day"][day]["accuracy"]-a["per_day"][day]["accuracy"]):+.10f}个百分点。']
        lines.append(f'seed{seed}最差日Accuracy：控制{pct(min(a["per_day"][d]["accuracy"] for d in DEV))}，'
                     f'候选{pct(min(b["per_day"][d]["accuracy"] for d in DEV))}。')
    for slot in SLOTS:
        t = trials[slot]; acceptance = t['acceptance']
        lines += ['', f'{slot}：完整{acceptance["epochs_completed"]}轮、best第{acceptance["selected_epoch"]}轮，'
                  f'结束原因{acceptance["stop_reason"]}，真实训练CLI耗时{t["actual"]["wall_seconds"]:.6f}秒；'
                  f'capture/launcher/worker={t["actual"]["capture_pid"]}/{t["actual"]["launcher_pid"]}/{acceptance["worker_pid"]}。']
        for row in t['history']:
            m, online = row['development_validation'], row['train']
            lines.append(f'  epoch{row["epoch"]}: TRAIN在线Acc={pct(online["accuracy"])},CE={online["loss"]:.12f}; '
                         f'DEV Acc={pct(m["accuracy"])},Macro-F1={m["macro_f1"]:.12f},CE={m["loss"]:.12f},'
                         f'平Recall={pct(m["recall"][1])},improved={row["improved"]},wait={row["wait"]},秒={row["seconds"]:.6f}')
    total_updates = sum(t['acceptance']['optimizer_steps_completed'] for t in trials.values())
    total_forwards = sum(t['acceptance']['counters']['forward_calls'] for t in trials.values())
    lines += ['', '核验和边界：',
              f'本步实际2个新训练模型/2个AdamW，前向{total_forwards:,}次，反向/optimizer step各{total_updates:,}次。'
              f'新四组预算全部4slot已关闭，累计epoch={budget["consumed_epochs"]}、累计更新={budget["optimizer_updates"]:,}。',
              '每一步optimizer.step前永久预扣；旧feature42/strategy42以及候选启动前的control43完整条目保持。'
              '每组终态预算原字节另存，不将追加候选后合法改变的live budget当作旧控制证据。',
              '两个真实CPU核验CLI共仅载入4份新best/latest容器，0额外模型/optimizer构造、0前向，'
              '检查权重/AdamW累计step/RNG保存身份，不冒充GPU续训。独立D缓存程序逐组全epoch重建矩阵、CE、P/R/F1和精确epoch选择。',
              '训练前后D缓存/原标签SHA保持，控制缓存21个原P2黄金输入一致，候选继承ED4缓存和归一化；'
              '287历史保护及45基线只读保持。两组临时AC-only请求实际finally成功清除，不改永久电源/显示，不据心跳证明Windows从未睡眠。',
              f'27项NumPy合成缓存测试、39项stdlib/mock训练测试、17项独立stdlib最终器边界测试、'
              f'{helper_count}项后验辅助测试分别真实exit0。'
              '原缓存测试scope写stdlib过窄已用独立clarification纠正且保留旧证据，没有重跑或合并声称66项stdlib。',
              'TRAIN为每次更新前在线指标，不能当固定代表模型TRAIN成绩。当前比较仅已有15/16开发集，'
              '窗口相互重叠，两个种子不构成统计显著性或新独立跨日泛化证明。',
              '17—19记录/标签/预测流未新读或评价；历史最终45.3243568848%/quality=false不改。'
              '源control/覆盖未知，原flag64保持；CPU工程检查不代表板端/NPU/能耗/实盘验收。',
              '本步不导出、不部署，不开新训练。只有用户说“帮我完成第7步”才进行ED7开发结论与另目录交付。', '',
              '证据目录：' + str(STAGE)]
    outputs = [report, STAGE / 'completion_acceptance.json', STAGE / 'pair_decision.json',
               ROOT / 'state/tcn_software_plan_v26.json', STAGE / 'state_before_finalization.json',
               STAGE / 'before_finalization_state.bytes', STAGE / 'before_final_AGENTS.md',
               STAGE / 'before_final_PROGRESS.md', STAGE / 'finalization_receipt.json']
    need(all(not p.exists() for p in outputs), 'ED6 publication already exists; inspect evidence rather than repeat')
    state_path = ROOT / 'state/tcn_existing_data_optimization_20261017.json'
    snapshots = {'before_finalization_state.bytes': state_path.read_bytes(),
                 'before_final_AGENTS.md': (ROOT / 'AGENTS.md').read_bytes(),
                 'before_final_PROGRESS.md': (ROOT / 'PROGRESS.md').read_bytes()}
    put(STAGE / 'state_before_finalization.json', read(state_path))
    for name, data in snapshots.items():
        with (STAGE / name).open('xb') as stream: stream.write(data)
    with report.open('x', encoding='utf-8', newline='\n') as stream: stream.write('\n'.join(lines) + '\n')
    decision = {'version': 'TCN-ED6-paired-development-decision-1.0', 'completed_utc': now.isoformat(),
                'selection_lock': binding(lock_path), 'stable_development_adoption': stable,
                'accuracy_delta42_exact': str(delta42), 'accuracy_delta43_exact': str(delta43),
                'accuracy_delta42_percentage_points': 100*float(delta42),
                'accuracy_delta43_percentage_points': 100*float(delta43), 'Macro_F1_or_Recall_veto': False,
                'representative_seed': 42, 'retained_family': retained_family, 'retained_DEV_metrics': retained,
                'retained_checkpoint': retained_checkpoint, 'retained_normalization': retained_normalization,
                'unstable_ED4_option_preserved': not stable, 'candidate_family_switched_after43': False,
                'new_independent_test': False, 'ED7_started': False}
    put(STAGE / 'pair_decision.json', decision)
    completion = {'version': 'TCN-ED6-completion-1.0', 'passed': True, 'step': 6, 'phase_id': 'ED6',
        'campaign_id': 'TCN_EDO_20261017', 'status': 'ED6_completed_ED7_pending', 'worker_pid': os.getpid(),
        'completed_utc': now.isoformat(), 'completed_Asia_Shanghai': shanghai.isoformat(),
        'selection_lock': binding(lock_path), 'pair_decision': binding(STAGE / 'pair_decision.json'),
        'report': binding(report), 'stable_development_adoption': stable, 'representative_seed': 42,
        'retained_family': retained_family, 'retained_DEV_metrics': retained,
        'retained_checkpoint': retained_checkpoint, 'retained_normalization': retained_normalization,
        'seed42_control_DEV_metrics': base42, 'seed42_candidate_DEV_metrics': trial42,
        'seed43_control_DEV_metrics': control['metric'], 'seed43_candidate_DEV_metrics': candidate['metric'],
        'trial_evidence': {slot: {'training_acceptance': binding(trials[slot]['trial']/'run/training_acceptance.json'),
            'actual_training': binding(trials[slot]['trial']/'training_execution_receipt.json'),
            'actual_checkpoint': binding(trials[slot]['trial']/'checkpoint_execution_receipt.json'),
            'actual_audit': binding(trials[slot]['trial']/'audit_execution_receipt.json'),
            'immutable_budget_snapshot': trials[slot]['acceptance']['budget_completion_snapshot_binding']}
            for slot in SLOTS}, 'synthetic_evidence': tests, 'budget_final': binding(BUDGET),
        'new_training_runs_this_step': 2, 'total_campaign_training_reservations': 4,
        'this_step_forwards': total_forwards, 'this_step_optimizer_updates': total_updates,
        'postcheck_new_checkpoint_CPU_loads': 4, 'postcheck_forwards': 0,
        'protected_historical_files': 287, 'baseline_readonly_files': 45,
        'new_independent_test': False, 'historical_final_quality_changed': False,
        'source_control_verified': False, 'coverage_unknown': True, 'original_flag64_cleared': False,
        'next_step': 7, 'next_phase_id': 'ED7', 'requires_user_trigger': True, 'automatic_next_action': False,
        'finalizer_source': binding(Path(__file__))}
    put(STAGE / 'completion_acceptance.json', completion)
    plan = read(ROOT / 'state/tcn_software_plan_v25.json')
    plan.update(version='TCN-software-plan-26.0', status=completion['status'],
        previous_plan='state/tcn_software_plan_v25.json', current_step=6, current_phase_id='ED6',
        completion=binding(STAGE/'completion_acceptance.json'), guide=binding(report), completed_utc=now.isoformat(),
        next_step='ED7', next_trigger='帮我完成第7步', training_authorized_now=False,
        ED6_pair_decision=binding(STAGE/'pair_decision.json'), ED6_retained_family=retained_family)
    for step in plan['steps']: step['status'] = 'completed' if step['step'] <= 6 else 'awaiting_user_trigger'
    put(ROOT/'state/tcn_software_plan_v26.json', plan)
    state = {'version':'TCN-existing-data-optimization-state-1.0','status':completion['status'],
        'current_step':6,'current_phase_id':'ED6','current_plan':binding(ROOT/'state/tcn_software_plan_v26.json'),
        'completion':binding(STAGE/'completion_acceptance.json'),'stable_development_adoption':stable,
        'retained_family':retained_family,'representative_seed':42,'retained_DEV_accuracy':retained['accuracy'],
        'retained_DEV_macro_f1':retained['macro_f1'],'retained_DEV_stationary_recall':retained['recall'][1],
        'retained_checkpoint':retained_checkpoint,'retained_normalization':retained_normalization,
        'training_reservations':4,'training_runs':4,'optimizer_updates':budget['optimizer_updates'],
        'new_independent_test':False,'training_authorized_now':False,'next_step':7,'next_phase_id':'ED7',
        'requires_user_trigger':True,'automatic_next_action':False}
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    header = (f'## {date} 新七步ED6配对稳定性复核已完成，ED7待用户指令\n\n'
        '用户“帮我完成第6步”已执行，锁定ED4relative lr.001后原始价control43→relativecandidate43串行冷启动。'
        f'Acc差seed42={100*float(delta42):+.10f}pp、seed43={100*float(delta43):+.10f}pp，stableDEV={stable}，'
        f'代表固定42，保留{retained_family}。规则仅两个Accuracy差非负且至少一个正，不加F1/Recall否决，不择最高seed。\n'
        '两trial原训练/各2新CP CPU检查/独立全轮D缓存审计均真实exit0；4新slot全部关闭，旧prefix及每组终态原字节快照保持。'
        '287历史保护/45基线只读、seed43同初始化、全共同anchor和资源实际finally清理通过；27NumPy/39stdlibmock/后验suite分列。\n'
        '仅开发验证，无17—19新流或新独立测试；历史quality=false、源control/覆盖未知、flag64保持。'
        f'完成artifacts/tcn_existing_data_optimization_2026-10-17/step06_replication/completion_acceptance.json；'
        f'报告docs/{report.name}；v26 ED6_completed_ED7_pending，stage登记10-17保持。\n'
        '下一步仅用户“帮我完成第7步”后另目录交付，不自动训练/评价/导出部署；重复第6步先核现存完成证据，不重开预算。\n\n')
    (ROOT/'AGENTS.md').write_bytes(header.encode('utf-8')+snapshots['before_final_AGENTS.md'])
    (ROOT/'PROGRESS.md').write_bytes(header.encode('utf-8')+snapshots['before_final_PROGRESS.md'])
    protected(); need(sha(BUDGET)==completion['budget_final']['sha256'],'Finalizer changed budget')
    put(STAGE/'finalization_receipt.json',{'passed':True,'worker_pid':os.getpid(),
        'completed_utc':datetime.now(timezone.utc).isoformat(),'completion':binding(STAGE/'completion_acceptance.json'),
        'pair_decision':binding(STAGE/'pair_decision.json'),'report':binding(report),
        'plan':binding(ROOT/'state/tcn_software_plan_v26.json'),'state':binding(state_path),
        'budget_unchanged':binding(BUDGET),'protected_historical_files':287,'baseline_readonly_files':45,
        'new_ML_operations':0,'next_step_started':False,'source':binding(Path(__file__))})
    print(json.dumps({'passed':True,'completion':binding(STAGE/'completion_acceptance.json'),
        'stable_development_adoption':stable,'representative_seed':42,'retained_family':retained_family},ensure_ascii=True),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
