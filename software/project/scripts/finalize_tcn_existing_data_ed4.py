"""Publish ED4 only after original training, D-only audit and CPU-container checks.

This finalizer imports no numerical or ML libraries and performs no inference,
training, optimizer/budget mutation, E payload read, or next-step execution.
The root operator runs it only after its separate source review. All admission
gates finish before publication. Historical sources, checkpoints and plans stay
unchanged; current state/AGENTS/PROGRESS old bytes are separately snapshotted.
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
import stat

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'artifacts/tcn_existing_data_optimization_2026-10-17'
STAGE = CAMPAIGN / 'step04_features'
BUDGET = CAMPAIGN / 'budget.json'
PROTOCOL_SHA = '09654e6dfdb10b69feb6ee950c6668e71640bcea1d56fa9691a9564c8641fcf3'
TRAIN = [f'2025-12-{d:02}' for d in range(8, 13)]
DEV = ['2025-12-15', '2025-12-16']


def need(value, message):
    if not value:
        raise ValueError(message)


def local(value):
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    need('..' not in p.parts and p.is_relative_to(ROOT), 'Outside D workspace')
    for q in (p, *p.parents):
        if q.exists():
            need(not q.is_symlink() and not getattr(q.lstat(), 'st_file_attributes', 0) & 0x400,
                 'Reparse metadata/output forbidden')
        if q == ROOT:
            break
    need(p.resolve().is_relative_to(ROOT.resolve()), 'Resolved path escaped workspace')
    return p


def sha(path):
    h = hashlib.sha256()
    with local(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    p = local(path)
    return {'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def verify(ref):
    need(isinstance(ref, dict) and {'path', 'sha256'} <= set(ref), 'Missing evidence binding')
    p = local(ref['path'])
    need(p.is_file() and sha(p) == ref['sha256'], 'Evidence SHA changed: ' + str(p))
    need('bytes' not in ref or p.stat().st_size == ref['bytes'], 'Evidence size changed')
    return p


def same(ref, path):
    return local(ref['path']) == local(path) and ref['sha256'] == sha(path)


def read(path):
    p = local(path)
    need(p.stat().st_size <= 32 * 1024**2, 'Oversized metadata')
    return json.loads(p.read_text('utf-8-sig'),
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + s)))


def bound(ref):
    return read(verify(ref))


def put(path, value):
    p = local(path)
    need(p.is_relative_to(STAGE) or p == ROOT / 'state/tcn_software_plan_v24.json',
         'New JSON output outside this step')
    with p.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def exact_rank(matrix):
    need(isinstance(matrix, list) and len(matrix) == 3 and all(isinstance(r, list) and len(r) == 3
         and all(type(v) is int and v >= 0 for v in r) for r in matrix), 'Invalid confusion matrix')
    support = [sum(r) for r in matrix]; total = sum(support); need(total > 0, 'Empty confusion')
    predicted = [sum(matrix[i][j] for i in range(3)) for j in range(3)]
    return (Fraction(sum(matrix[i][i] for i in range(3)), total),
            sum((Fraction(2 * matrix[i][i], support[i] + predicted[i])
                 if support[i] + predicted[i] else Fraction(0) for i in range(3)), Fraction(0)) / 3)


def metrics_from_days(by_day):
    matrix = [[sum(by_day[d]['confusion_true_rows_predicted_columns'][i][j] for d in DEV)
               for j in range(3)] for i in range(3)]
    rank = exact_rank(matrix)
    support = [sum(r) for r in matrix]
    predicted = [sum(matrix[i][j] for i in range(3)) for j in range(3)]
    n = sum(support)
    return {'samples': n, 'accuracy': float(rank[0]), 'macro_f1': float(rank[1]),
            'precision': [matrix[i][i] / predicted[i] if predicted[i] else 0. for i in range(3)],
            'recall': [matrix[i][i] / support[i] if support[i] else 0. for i in range(3)],
            'f1': [2 * matrix[i][i] / (support[i] + predicted[i]) if support[i] + predicted[i] else 0.
                   for i in range(3)], 'support': support,
            'confusion_true_rows_predicted_columns': matrix,
            'loss': sum(by_day[d]['loss'] * by_day[d]['samples'] for d in DEV) / n,
            'per_day': by_day}


def pct(value):
    return f'{100 * value:.10f}%'


def ce(metric):
    return metric.get('loss', metric.get('cross_entropy'))


def metric_text(label, metric):
    loss = ce(metric)
    need(isinstance(loss, (int, float)) and math.isfinite(loss), 'Missing/nonfinite reported CE')
    return '\n'.join([
        f'{label}：N={metric["samples"]:,}，Accuracy={pct(metric["accuracy"])}，'
        f'Macro-F1={metric["macro_f1"]:.12f}，CE={loss:.12f}',
        '  跌/平/涨 Precision：' + ' / '.join(pct(v) for v in metric['precision']),
        '  跌/平/涨 Recall：' + ' / '.join(pct(v) for v in metric['recall']),
        '  跌/平/涨 F1：' + ' / '.join(f'{v:.12f}' for v in metric['f1']),
        '  混淆矩阵（真实为行，预测为列，顺序跌/平/涨）：'
        + json.dumps(metric['confusion_true_rows_predicted_columns']),
    ])


def receipt(phase):
    path = STAGE / (phase + '_execution_receipt.json')
    result = read(path)
    need(result.get('phase') == phase and result.get('exit_code') == 0
         and result.get('entry_unchanged') is True, 'Missing real CLI success: ' + phase)
    need(result.get('entry_before') == result.get('entry_after'), 'CLI source changed: ' + phase)
    for name in ('entry_before', 'entry_after', 'stdout', 'stderr', 'capture_source'):
        verify(result[name])
    if 'input' in result:
        verify(result['input'])
    for ref in result.get('outputs', {}).values():
        verify(ref)
    return result


def protected_files(inventory):
    need(inventory['protected_files'] == len(inventory['files']) == 85,
         'Historical protected inventory changed')
    for ref in inventory['files']:
        verify(ref)
    # The original pre-ED4 budget identity is deliberately historical. Only this
    # new budget is authorized to change; old P3/P5 ledgers remain protected above.
    need(local(inventory['excluded_mutable_new_campaign_budget']['path']) == BUDGET,
         'Wrong mutable budget exclusion')
    baseline = ROOT / 'artifacts/tcn_frozen_baseline_wn42_2026-10-17'
    paths = [p for p in (baseline / 'bundle').rglob('*') if p.is_file()]
    paths.append(baseline / 'TCN_WN42_Frozen_Baseline_20261017.zip')
    need(len(paths) == 45 and all(getattr(p.stat(), 'st_file_attributes', 0) & 1 for p in paths),
         'Frozen baseline read-only evidence changed')


def postrun_test_evidence():
    evidence = read(STAGE / 'postrun_helpers_tests_evidence.json')
    need(evidence.get('passed') is True and evidence.get('actual_exit_code') == 0
         and evidence.get('separate_synthetic_test_count') == 13
         and evidence.get('no_real_data_or_ML') is True, 'Actual separate helper tests did not pass')
    actual_path = verify(evidence['actual_receipt'])
    need(actual_path == STAGE / 'postrun_helpers_tests_execution_receipt.json', 'Wrong helper-test actual receipt')
    actual = read(actual_path)
    need(actual.get('phase') == 'postrun_helpers_tests' and actual.get('exit_code') == 0
         and actual.get('entry_unchanged') is True and actual['entry_before'] == actual['entry_after']
         and actual.get('actual_model_or_optimizer_operations') == 0
         and actual.get('real_records_read') is False, 'Separate helper-test actual execution differs')
    need(evidence['sources_before'] == evidence['sources_after']
         == actual['sources_before'] == actual['sources_after'], 'Tested helper sources changed')
    expected = {STAGE / 'independent_audit.py', STAGE / 'checkpoint_validation.py',
                ROOT / 'scripts/run_tcn_existing_data_ed4_checkpoint_capture.py', STAGE / 'test_postrun_helpers.py'}
    need(len(evidence['sources_before']) == 4
         and {local(ref['path']) for ref in evidence['sources_before']} == expected,
         'Wrong separately tested helper source set')
    for ref in evidence['sources_before']:
        verify(ref)
    need(evidence['test_source'] == actual['entry_before'], 'Actual helper suite did not use stated test source')
    for name in ('test_source',):
        verify(evidence[name])
    for name in ('entry_before', 'entry_after', 'stdout', 'stderr'):
        verify(actual[name])
    return evidence


def admit(registration_sha):
    need(not (STAGE / 'completion_acceptance.json').exists()
         and not (ROOT / 'state/tcn_software_plan_v24.json').exists(), 'ED4 already completed; inspect saved evidence')
    reg_path = STAGE / 'training_registration.json'
    need(sha(reg_path) == registration_sha, 'Explicit ED4 registration differs')
    reg = read(reg_path)
    need(reg['version'] == 'TCN-ED4-training-registration-1.0' and reg['step'] == 'ED4', 'Wrong phase')
    protocol = bound(reg['protocol_binding'])
    need(reg['protocol_binding']['sha256'] == PROTOCOL_SHA and protocol['campaign_id'] == 'TCN_EDO_20261017',
         'Registered protocol differs')
    need(protocol['data']['train_dates'] == TRAIN and protocol['data']['development_dates'] == DEV
         and protocol['data']['historical_dates_excluded_from_new_execution']
         == ['2025-12-17', '2025-12-18', '2025-12-19'], 'Date scope changed')
    for key in ('authorization_binding', 'config_binding', 'prepared_manifest_binding', 'feature_cache_binding',
                'reservation_binding', 'source_review_binding', 'test_evidence_binding',
                'preparation_receipt_binding', 'capture_source_binding', 'protected_inventory_binding'):
        verify(reg[key])
    for name, digest in reg['source_bindings'].items():
        need(sha(ROOT / name) == digest, 'Frozen numerical source changed: ' + name)
    auth = bound(reg['authorization_binding'])
    need(auth['human_authorization'] == {'source': 'direct_current_human_user_message',
         'quote': '帮我完成第4步', 'step': 'ED4'} and auth['training_authorized'] is True,
         'Current human authorization differs')
    need(bound(reg['source_review_binding'])['passed'] is True, 'Source review not passed')
    test_evidence = bound(reg['test_evidence_binding'])
    need(test_evidence['passed'] is True and test_evidence['actual_exit_code'] == 0, 'Actual tests not passed')
    helper_tests = postrun_test_evidence()
    config = bound(reg['config_binding'])
    need(config['seed'] == 42 and config['decision'] == 'original_argmax' and config['weight_norm'] is True
         and config['learning_rate'] == .001 and config['window_length'] == 100
         and config['max_epochs'] == 10 and config['early_stop_patience'] == 3,
         'Registered single-factor/default training settings differ')
    receipts = {p: receipt(p) for p in ('feature_tests', 'training_tests', 'preparation', 'training', 'audit')}
    for item in receipts.values():
        need(item['capture_source']['sha256'] == reg['capture_source_binding']['sha256'], 'Unregistered CLI capture')
    need(same(receipts['training']['input'], reg_path) and same(receipts['audit']['input'], reg_path),
         'Training/audit did not use exact registration')
    need(datetime.fromisoformat(receipts['training']['ended_utc'])
         <= datetime.fromisoformat(receipts['audit']['started_utc']), 'Audit preceded training terminal')
    actual = receipts['training']
    acceptance = bound(actual['outputs']['training_acceptance'])
    history = bound(actual['outputs']['history'])
    need(acceptance['status'] == 'completed' and acceptance['epochs_completed'] == len(history)
         and 1 <= len(history) <= 10, 'Incomplete formal training')
    need(acceptance['actual_worker_pid'] == acceptance['worker_pid'] == actual['actual_worker_pid']
         and acceptance['identity']['registration_binding']['sha256'] == registration_sha,
         'Actual worker/registration roles differ')
    need(acceptance['original_argmax_without_ED3_bias'] is True
         and acceptance['sealed_dates_opened'] is False and acceptance['original_flag64_cleared'] is False,
         'Forbidden bias/date/flag change')
    audit = bound(receipts['audit']['outputs']['audit'])
    need(audit['passed'] is True and same(audit['registration_binding'], reg_path)
         and same(audit['training_receipt_binding'], STAGE / 'training_execution_receipt.json'),
         'Independent audit does not bind actual training')
    need(audit['comparison_same_complete_DEV_anchors'] is True and audit['no_ED3_bias_added'] is True,
         'No complete common-anchor comparison')
    need(audit['protected_evidence'] == {'passed': True, 'protected_files': 85, 'baseline_readonly_files': 45},
         'Independent historical protection gate differs')
    for key in ('extra_model_loads', 'extra_models_constructed', 'extra_forwards', 'extra_optimizers',
                'extra_training_runs_or_updates', 'E_record_reads', 'historical17_19_stream_reads'):
        need(audit[key] == 0, 'Independent D audit exceeded scope: ' + key)
    need(audit['new_independent_test'] is False and audit['next_step_started'] is False, 'Audit scope differs')
    selected = acceptance['selected_epoch']; epochs = len(history)
    need(audit['selected_epoch'] == selected and audit['epochs_recomputed'] == epochs
         and audit['DEV_windows_recomputed'] == 6612655 * epochs
         and audit['DEV_payload_files_verified_before_after'] == 6 * epochs,
         'Independent all-epoch cache reconstruction incomplete')
    best = None; chosen = None; wait = 0
    for i, row in enumerate(history, 1):
        need(row['epoch'] == i and row['optimizer_updates'] == 48362, 'Partial/nonconsecutive epoch')
        rank = exact_rank(row['development_validation']['confusion_true_rows_predicted_columns'])
        improved = best is None or rank > best; wait = 0 if improved else wait + 1
        if improved:
            best, chosen = rank, i
        need(row['improved'] is improved and row['wait'] == wait, 'Exact selection replay differs')
        need(i == epochs or wait < 3, 'Patience overrun')
        verify(row['cache_binding'])
    need(chosen == selected and (wait >= 3 and acceptance['stop_reason'] == 'patience3'
         or epochs == 10 and wait < 3 and acceptance['stop_reason'] == 'max10'), 'Selection/stop mismatch')
    metric = acceptance['selected_development_metrics']
    need(metric == history[selected - 1]['development_validation'], 'Selected history differs')
    need(metric['confusion_true_rows_predicted_columns']
         == audit['selected_original_argmax_DEV_metrics']['confusion_true_rows_predicted_columns'],
         'Selected independently reconstructed matrix differs')
    updates = 48362 * epochs
    counts = {'models_constructed': 1, 'historical_checkpoint_loads': 0, 'optimizers_constructed': 1,
              'forward_calls': 74193 * epochs, 'backward_calls': updates,
              'optimizer_attempts_charged': updates, 'optimizer_steps_completed': updates,
              'train_windows': 12380446 * epochs, 'development_windows': 6612655 * epochs}
    need(acceptance['counters'] == audit['counters'] == counts, 'Actual formal ML counters differ')
    need(acceptance['optimizer_attempts_charged'] == acceptance['optimizer_steps_completed'] == updates,
         'Optimizer attempts/completions differ')
    budget = bound(acceptance['budget_binding'])
    need(same(acceptance['budget_binding'], BUDGET) and same(audit['budget_binding'], BUDGET),
         'Final budget identity differs')
    need(len(budget['reserved_runs']) == 1 and budget['optimizer_updates'] == updates
         and budget['consumed_epochs'] == epochs and updates <= 483620, 'New budget aggregate overrun/mismatch')
    slot = budget['reserved_runs'][0]
    need(slot.get('slot_id', slot.get('id')) == 'feature42' and slot['reservation_id'] == reg['reservation_id']
         and slot['status'] == 'completed' and slot['seed'] == 42 and slot['optimizer_updates'] == updates
         and slot['consumed_epochs'] == epochs and slot['actual_optimizer_steps_completed_last_snapshot'] == updates,
         'Permanent feature42 terminal counters differ')
    need(budget['maximum_runs'] == 4 and budget['maximum_optimizer_updates'] == 1934480,
         'Campaign cap changed')
    before = bound(acceptance['cache_payloads_before_binding'])
    after = bound(acceptance['cache_payloads_after_binding'])
    need(before == after and before['passed'] is True, 'Formal normalized-cache/label identity changed')
    prepared = bound(reg['feature_cache_binding'])
    need(prepared['source_records_unchanged'] is True and prepared['label_streams_unchanged'] is True,
         'Source preservation evidence missing')
    need(bound(prepared['normalization_binding']) == prepared['normalization']
         and bound(prepared['statistics_independent_review_binding']) == prepared['statistics_independent_review']
         and prepared['statistics_independent_review']['passed'] is True, 'Normalization identity differs')
    for ref in prepared['cache_files'].values():
        verify(ref)  # Opaque SHA of new D caches only; no new model or E data access.
    for ref in acceptance['checkpoint_bindings'].values():
        verify(ref)
    for ref in acceptance['epoch_checkpoint_bindings'].values():
        verify(ref)  # Opaque new checkpoints, never deserialized here.
    cleanup_path = STAGE / 'run/resource_cleanup.json'; cleanup = read(cleanup_path)
    need(same(audit['resource_cleanup_binding'], cleanup_path) and cleanup['worker_pid'] == acceptance['worker_pid']
         and cleanup['cleanup_succeeded'] is True and cleanup['request_active_after_finally'] is False
         and cleanup['permanent_power_or_display_settings_changed'] is False and cleanup['run_error'] is None,
         'Actual finally resource cleanup failed')
    need(cleanup['events'] and cleanup['events'][-1].get('final_cleanup') is True
         and cleanup['events'][-1]['returned_previous'] != 0, 'Actual request-clear API evidence missing')
    cp_receipt = receipt('checkpoint_validation')
    cp_reg_path = STAGE / 'checkpoint_validation_registration.json'
    need(same(cp_receipt['input'], cp_reg_path), 'CPU validation registration differs')
    cp_reg = read(cp_reg_path)
    need(same(cp_reg['training_receipt_binding'], STAGE / 'training_execution_receipt.json')
         and same(cp_reg['training_registration_binding'], reg_path), 'CPU validation parent differs')
    need(cp_reg['limits'] == {'new_checkpoint_CPU_loads': 2, 'models': 0, 'optimizers': 0,
                             'forwards': 0, 'backwards': 0, 'optimizer_updates': 0},
         'CPU validation finite scope differs')
    verify(cp_reg['source_binding'])
    need(cp_reg['source_binding']['sha256'] == cp_receipt['entry_before']['sha256'],
         'CPU CLI did not execute registered helper')
    cp_ref = cp_receipt['outputs']['checkpoint_validation']
    cp = bound(cp_ref)
    need(cp['passed'] is True and cp['actual_new_checkpoint_CPU_loads'] == 2
         and same(cp['registration_binding'], cp_reg_path), 'Two new CPU checkpoint checks incomplete')
    need(cp['source_binding']['sha256'] == cp_reg['source_binding']['sha256']
         and cp['worker_pid'] == cp_receipt['actual_worker_pid'], 'CPU helper source/worker receipt differs')
    for key in ('models_constructed', 'optimizers_constructed', 'forwards', 'backwards',
                'optimizer_updates', 'historical_checkpoint_loads', 'E_record_reads'):
        need(cp[key] == 0, 'CPU-container check exceeded scope: ' + key)
    need(cp['RNG_restored'] is False and cp['new_market_evaluation'] is False, 'CPU check scope differs')
    observations = cp['observations']
    need(len(observations) == 2 and {r['checkpoint_role'] for r in observations} == {'best.pt', 'latest.pt'}
         and all(r['passed'] is True for r in observations), 'Missing successful CPU checkpoint observations')
    for item in observations:
        need(item['epoch'] == (selected if item['checkpoint_role'] == 'best.pt' else epochs)
             and item['all_optimizer_steps'] == item['epoch'] * 48362, 'CPU checkpoint epoch/update mismatch')
    need(datetime.fromisoformat(actual['ended_utc']) <= datetime.fromisoformat(cp_receipt['started_utc']),
         'CPU check began before training terminal')
    inventory = bound(reg['protected_inventory_binding']); protected_files(inventory)
    old_plan = read(ROOT / 'state/tcn_software_plan_v23.json')
    base_cache = bound(old_plan['ED2_cache_manifest'])
    need(old_plan['ED2_cache_manifest']['sha256']
         == '1808492b6b2e118b09a88114bf7af4a837414e1e96a41427fef832e08407c106', 'Baseline cache metadata changed')
    baseline = metrics_from_days({d: base_cache['per_day'][d]['metrics'] for d in DEV})
    need(baseline['confusion_true_rows_predicted_columns']
         == audit['original_WN42_original_argmax_DEV_metrics']['confusion_true_rows_predicted_columns'],
         'Frozen-system comparison matrix differs')
    ed3 = bound(old_plan['completion'])
    need(ed3['phase_id'] == 'ED3' and ed3['passed'] is True, 'Previous decision evidence missing')
    need(metric['samples'] == baseline['samples'] == ed3['retained_DEV_metrics']['samples'] == 6612655,
         'Comparison denominator differs')
    promising = exact_rank(metric['confusion_true_rows_predicted_columns'])[0]
    promising = promising > exact_rank(baseline['confusion_true_rows_predicted_columns'])[0]
    need(audit['strict_accuracy_improvement_for_later_replication'] is promising,
         'Later-replication screen differs from exact count arithmetic')
    return {'reg': reg, 'reg_path': reg_path, 'protocol': protocol, 'config': config,
            'receipts': receipts, 'acceptance': acceptance, 'history': history, 'audit': audit,
            'metric': metric, 'baseline': baseline, 'ed3': ed3, 'promising': promising,
            'budget': budget, 'prepared': prepared, 'inventory': inventory, 'cleanup': cleanup,
            'cp_receipt': cp_receipt, 'cp': cp, 'old_plan': old_plan, 'helper_tests': helper_tests}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args(argv)
    c = admit(args.registration_sha256)
    actual, history, metric, baseline = c['acceptance'], c['history'], c['metric'], c['baseline']
    selected = actual['selected_epoch']; epochs = len(history); promising = c['promising']
    delta = 100 * float(exact_rank(metric['confusion_true_rows_predicted_columns'])[0]
                        - exact_rank(baseline['confusion_true_rows_predicted_columns'])[0])
    d3_metric = c['ed3']['retained_DEV_metrics']
    now = datetime.now(timezone.utc); shanghai = now.astimezone(timezone(timedelta(hours=8)))
    date = shanghai.date().isoformat(); completed_utc = now.isoformat()
    report = ROOT / ('docs/TCN_现有数据优化第4步相对价格特征训练报告_' + date + '.txt')
    verdict = ('本组相对价格特征在本轮共同开发集上有严格Accuracy改善，可作为后续有限复核的候选。'
               if promising else '本组相对价格特征未严格提高共同开发集Accuracy，保留原WN42作为原argmax训练家族的默认参考。')
    by_day = metric['per_day']; worst_day = min(DEV, key=lambda d: by_day[d]['accuracy'])
    lines = [
        'TCN 现有数据优化第4步：相对价格特征单因素训练',
        f'完成时间（上海）：{shanghai.isoformat()}', f'完成时间 UTC：{completed_utc}', '',
        f'结论：{verdict}',
        f'本组seed42；完整训练{epochs}轮，按精确Accuracy、精确Macro-F1选择第{selected}轮，'
        f'结束原因{actual["stop_reason"]}。全并列保留较早完整轮，patience3/max10，平稳类20%召回率仅观察、不作选点硬门。',
        f'同一DEV15/16全部6,612,655个端点，相对原WN42原argmax Accuracy变化{delta:+.10f}个百分点。',
        '该候选仍未完成seed43配对复核，不能称为稳定提升或新的最终模型。ED3偏置方案单独保存，本步不叠加偏置。', '',
        '唯一改变的输入因素：',
        '每条记录用其当前最佳买卖价的中点m=(raw_bid1+raw_ask1)/2，将20个偶数价格通道替换为(raw_price-m)/m；'
        '20个奇数数量通道保留原始数量，不做log1p，不加新通道。变换与归一化使用float64，最终缓存float32。',
        '新归一化只用原五日TRAIN完整有效研究区间的唯一记录；不重复窗口、不使用DEV拟合，不覆盖原归一化。'
        f'实际唯一TRAIN记录数{c["prepared"]["normalization"]["record_count"]:,}。',
        '本方法保留当前盘口相对形状，但逐记录中心化会去除窗口内中间价路径；这是登记时已明确的取舍，结果不能归因为单一已证明的机制。',
        '原L100/H50端点、标签、筛选规则、flag64、40输入通道、32隐藏通道、5残差块、WN、dropout0、'
        'AdamW lr0.001/weight_decay0.0001、unweighted CE、batch256、float32、TF32关闭、确定性算法及原全局shuffle规则保持。',
        '单组全新seed42冷启动；未载入旧权重/optimizer；未增加数据、改标签或开展其他方向搜索。', '',
        '两日合并开发验证对照：', metric_text('原冻结WN42、原argmax', baseline),
        metric_text(f'ED4相对价格特征、原argmax、第{selected}轮', metric),
        metric_text('ED3原权重加已冻结偏置0.1（仅并列描述，本组未使用）', d3_metric),
        f'ED4相对ED3 Accuracy描述性差值：{100 * (metric["accuracy"] - d3_metric["accuracy"]):+.10f}个百分点。'
        '不据此跳过ED6或声称新的独立测试提升。', '', '逐日共同端点对照：',
    ]
    for day in DEV:
        lines += [metric_text(day + ' 原WN42', baseline['per_day'][day]),
                  metric_text(day + f' ED4第{selected}轮', by_day[day]),
                  f'  Accuracy变化：{100 * (by_day[day]["accuracy"] - baseline["per_day"][day]["accuracy"]):+.10f}个百分点。']
    lines += [f'ED4最差日：{worst_day}，Accuracy={pct(by_day[worst_day]["accuracy"])}；'
              f'原WN42最差日Accuracy={pct(min(baseline["per_day"][d]["accuracy"] for d in DEV))}。', '',
              '完整训练历史（TRAIN为更新前在线分数，DEV为该轮固定权重原argmax）：']
    for row in history:
        train, dev = row['train'], row['development_validation']
        lines.append(f'  epoch{row["epoch"]}: TRAIN Accuracy={pct(train["accuracy"])}, CE={train["loss"]:.12f}; '
                     f'DEV Accuracy={pct(dev["accuracy"])}, Macro-F1={dev["macro_f1"]:.12f}, CE={dev["loss"]:.12f}, '
                     f'跌/平/涨Recall={" / ".join(pct(v) for v in dev["recall"])}; '
                     f'improved={row["improved"]}, wait={row["wait"]}, '
                     f'updates={row["optimizer_updates"]:,}, train+DEV秒={row["seconds"]:.6f}')
    lines += ['', '执行和核验：',
              f'实际训练CLI exit0，耗时{c["receipts"]["training"]["wall_seconds"]:.6f}秒；'
              f'capture PID={c["receipts"]["training"]["capture_pid"]}，launcher PID={c["receipts"]["training"]["launcher_pid"]}，'
              f'实际worker PID={actual["actual_worker_pid"]}，各角色分别保留。',
              f'正式阶段1模型、1AdamW，前向{actual["counters"]["forward_calls"]:,}次、'
              f'反向{actual["counters"]["backward_calls"]:,}次、实际optimizer完成{actual["optimizer_steps_completed"]:,}次；'
              f'预算预扣attempt同为{actual["optimizer_attempts_charged"]:,}，完整epoch={epochs}。',
              '新四组预算仅feature42一组已消耗；每步在optimizer.step前永久计数。失败/部分轮不退回预算，未重开旧P3/P5。',
              '每轮完整检查点、best/latest和两日DEV logits/labels/anchors均保存。独立D缓存程序逐轮重建矩阵、CE、'
              '每类P/R/F1、精确排名、完整轮及patience；与原WN42标签和有序端点SHA相同。',
              '独立缓存审计不载入模型、不增加前向。另有真实exit0的CPU容器检查，仅载入本组best/latest两份新检查点各一次，'
              '核权重、AdamW累计step、身份和RNG保存；0模型/optimizer构造、0前向，不代表实际GPU续训验收。',
              '7份新D特征缓存、7份原标签前后全SHA通过；21个边界输入及唯一TRAIN缓存归一化统计经独立核验。'
              '85份历史/基线保护文件及45份基线备份只读属性保持；原模型、原归一化、导出与归档完整保留。',
              '临时AC-only防系统空闲请求已在实际finally成功清除，不修改永久电源或显示设置；不以心跳证明Windows从未睡眠。',
              '合成测试25项训练边界、34项特征边界、13项后验辅助程序边界及源码审阅分别记录；'
              '各套件真实exit0，13项套件的4份辅助源码前后身份保持，不合并冒称单一套件，不把合成测试当成真实市场指标。', '',
              '结果边界和后续：',
              '以上仅为基于当前可观察数据和既定筛选规则的开发验证表现。15/16曾用于历史调参，没有新增独立测试日期，'
              '不能声称新的跨日泛化提升、统计显著性或实盘有效。',
              '17—19记录/标签/预测流未新读或评价；不拿本DEV结果与历史最终Accuracy45.3243568848%直接比较宣称提升。'
              '历史quality=false保持，源control/覆盖仍未知，原flag64保持。',
              '原WN42可直接回退；ED3固定偏置和本ED4候选各自保存。ED5须用户另行说“帮我完成第5步”，'
              '启动前再冻结唯一父方案及一个主要优化因素；当前不训练ED5、不启动ED6/7、不导出部署。', '',
              '证据目录：' + str(STAGE)]
    # All future output names are checked before any publication or mutable edit.
    outputs = [report, STAGE / 'completion_acceptance.json', ROOT / 'state/tcn_software_plan_v24.json',
               STAGE / 'state_before_finalization.json', STAGE / 'finalization_receipt.json',
               STAGE / 'before_finalization_state.bytes', STAGE / 'before_final_AGENTS.md', STAGE / 'before_final_PROGRESS.md']
    need(all(not p.exists() for p in outputs), 'Finalization output already exists; inspect rather than repeat')
    state_path = ROOT / 'state/tcn_existing_data_optimization_20261017.json'
    state_bytes = state_path.read_bytes(); agents_bytes = (ROOT / 'AGENTS.md').read_bytes()
    progress_bytes = (ROOT / 'PROGRESS.md').read_bytes()
    put(STAGE / 'state_before_finalization.json', read(state_path))
    for name, contents in [('before_finalization_state.bytes', state_bytes), ('before_final_AGENTS.md', agents_bytes),
                           ('before_final_PROGRESS.md', progress_bytes)]:
        with (STAGE / name).open('xb') as stream:
            stream.write(contents)
    with report.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines) + '\n')
    completion = {
        'version': 'TCN-ED4-completion-1.0', 'passed': True, 'step': 4, 'phase_id': 'ED4',
        'campaign_id': 'TCN_EDO_20261017', 'status': 'ED4_completed_ED5_pending',
        'completed_utc': completed_utc, 'completed_Asia_Shanghai': shanghai.isoformat(),
        'registration': binding(c['reg_path']), 'report': binding(report),
        'actual_receipts': {p: binding(STAGE / (p + '_execution_receipt.json')) for p in c['receipts']},
        'checkpoint_validation_receipt': binding(STAGE / 'checkpoint_validation_execution_receipt.json'),
        'postrun_helpers_tests_evidence': binding(STAGE / 'postrun_helpers_tests_evidence.json'),
        'postrun_helpers_tests_actual_receipt': c['helper_tests']['actual_receipt'],
        'separate_synthetic_test_counts': {'feature': 34, 'training': 25, 'postrun_helpers': 13},
        'training_acceptance': binding(STAGE / 'run/training_acceptance.json'),
        'independent_audit': binding(STAGE / 'independent_audit.json'),
        'checkpoint_validation': binding(STAGE / 'checkpoint_validation.json'),
        'resource_cleanup': binding(STAGE / 'run/resource_cleanup.json'),
        'history': binding(STAGE / 'run/history.json'), 'normalization': c['prepared']['normalization_binding'],
        'selected_checkpoint': actual['checkpoint_bindings']['best.pt'],
        'selected_cache': actual['selected_cache_binding'], 'selected_epoch': selected, 'epochs_completed': epochs,
        'baseline_original_argmax_DEV_metrics': baseline, 'candidate_original_argmax_DEV_metrics': metric,
        'ED3_bias_DEV_metrics_description_only': d3_metric,
        'candidate_accuracy_minus_original_WN42_percentage_points': delta,
        'strict_accuracy_improvement_for_later_replication': promising,
        'stable_development_adoption_not_yet_established': True,
        'one_common_complete_DEV_anchor_set': True, 'ED3_bias_added_to_training': False,
        'budget_final': binding(BUDGET), 'new_training_reservations': 1, 'new_training_runs': 1,
        'execution_counters': actual['counters'], 'postcheck_CPU_new_checkpoint_loads': 2,
        'postcheck_model_constructions': 0, 'postcheck_forwards': 0,
        'protected_historical_files': 85, 'baseline_readonly_files': 45,
        'new_independent_test': False, 'historical_final_quality_changed': False,
        'source_control_verified': False, 'coverage_unknown': True, 'original_flag64_cleared': False,
        'next_step': 5, 'next_phase_id': 'ED5', 'requires_user_trigger': True, 'automatic_next_action': False,
        'finalizer_source': binding(Path(__file__)),
    }
    put(STAGE / 'completion_acceptance.json', completion)
    plan = c['old_plan']
    plan.update(version='TCN-software-plan-24.0', status=completion['status'],
                previous_plan='state/tcn_software_plan_v23.json', current_step=4, current_phase_id='ED4',
                completed_utc=completed_utc, completion=binding(STAGE / 'completion_acceptance.json'),
                guide=binding(report), next_step='ED5', next_trigger='帮我完成第5步',
                training_authorized_now=False, ED4_selected_checkpoint=actual['checkpoint_bindings']['best.pt'],
                ED4_normalization=c['prepared']['normalization_binding'],
                ED4_promising_for_later_replication=promising)
    for row in plan['steps']:
        row['status'] = 'completed' if row['step'] <= 4 else 'awaiting_user_trigger'
    put(ROOT / 'state/tcn_software_plan_v24.json', plan)
    state = {'version': 'TCN-existing-data-optimization-state-1.0', 'status': completion['status'],
             'current_step': 4, 'current_phase_id': 'ED4',
             'current_plan': binding(ROOT / 'state/tcn_software_plan_v24.json'),
             'completion': binding(STAGE / 'completion_acceptance.json'),
             'selected_epoch': selected, 'epochs_completed': epochs, 'ED4_DEV_accuracy': metric['accuracy'],
             'ED4_DEV_macro_f1': metric['macro_f1'], 'ED4_DEV_stationary_recall': metric['recall'][1],
             'ED4_strict_accuracy_improvement': promising, 'ED4_accuracy_delta_percentage_points': delta,
             'ED4_selected_checkpoint': actual['checkpoint_bindings']['best.pt'],
             'ED4_normalization': c['prepared']['normalization_binding'],
             'training_reservations': 1, 'training_runs': 1, 'optimizer_updates': actual['optimizer_steps_completed'],
             'new_model_forwards': actual['counters']['forward_calls'], 'postcheck_CPU_new_checkpoint_loads': 2,
             'new_independent_test': False, 'training_authorized_now': False,
             'ED3_retained_bias': c['ed3']['retained_bias'], 'ED3_standalone_decision_preserved': True,
             'next_step': 5, 'next_phase_id': 'ED5', 'requires_user_trigger': True, 'automatic_next_action': False}
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    header = (f'## {date} 新七步ED4相对价格特征单因素训练已完成，ED5待用户指令\n\n'
              f'用户“帮我完成第4步”已执行ED4，唯一feature42冷启动WN，完整{epochs}轮、'
              f'{actual["optimizer_steps_completed"]}次AdamW更新，按新精确Acc/F1无Recall硬门选择第{selected}轮。'
              f'DEV Accuracy{pct(metric["accuracy"])}/Macro-F1 {metric["macro_f1"]:.12f}/平Recall{pct(metric["recall"][1])}；'
              f'原argmax WN42同anchor Accuracy变化{delta:+.10f}个百分点，promising={promising}。本步不加ED3偏置，未称seed43稳定采用。\n'
              f'实际training/audit/两新CP容器验证CLI均exit0；正式forward{actual["counters"]["forward_calls"]}，'
              f'backward/update各{actual["optimizer_steps_completed"]}，后验2CPUload/0模型构造/0额外前向。'
              '全epochDEV缓存独立重算、85历史保护、45基线只读、7新Dcache及原标签前后SHA通过；临时AC防idle实际清除。\n'
              '仅新相对价格输入和TRAIN归一化，odd数量不变，L100/H50端点/标签和旧原件保持；逐记录中心化移除mid路径取舍已报告。'
              '无17—19新流读取或新独立测试；历史quality=false、源control/覆盖未知、原flag64保持；原WN42和ED3另存可回退。\n'
              f'完成artifacts/tcn_existing_data_optimization_2026-10-17/step04_features/completion_acceptance.json；'
              f'报告docs/{report.name}；计划state/tcn_software_plan_v24.json，状态ED4_completed_ED5_pending。'
              'stage登记10-17保持，报告按实际上海完成日期；旧v23/running交接为历史。\n'
              '下一步仅用户“帮我完成第5步”后执行ED5；不自动优化/复评/导出部署。重复第4步先核完成证据，不重开feature42预约。\n\n')
    (ROOT / 'AGENTS.md').write_bytes(header.encode('utf-8') + agents_bytes)
    (ROOT / 'PROGRESS.md').write_bytes(header.encode('utf-8') + progress_bytes)
    protected_files(c['inventory'])
    need(same(actual['budget_binding'], BUDGET), 'Finalizer changed current budget')
    put(STAGE / 'finalization_receipt.json', {
        'passed': True, 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'completion': binding(STAGE / 'completion_acceptance.json'), 'report': binding(report),
        'plan': binding(ROOT / 'state/tcn_software_plan_v24.json'), 'state': binding(state_path),
        'protected_historical_files_unchanged': 85, 'baseline_readonly_files': 45,
        'budget_unchanged_by_finalizer': binding(BUDGET), 'new_model_forwards': 0,
        'training_or_optimizer_updates': 0, 'next_step_started': False,
        'source': binding(Path(__file__))})
    print(json.dumps({'passed': True, 'completion': binding(STAGE / 'completion_acceptance.json'),
                      'selected_epoch': selected, 'DEV_accuracy': metric['accuracy'],
                      'strict_accuracy_improvement_for_later_replication': promising}, ensure_ascii=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
