"""Publish the completed development delivery; no numerical/model imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import zipfile
import zlib

from experiments.reference.training.scripts import tcn_existing_data_ed7_contract as c


def put(path, value):
    with c.safe(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def verify_archive(path, entries):
    """Independently read every compressed stream; check SHA, bytes and CRC."""
    actual = []
    with zipfile.ZipFile(path, 'r') as archive:
        names = archive.namelist()
        c.need(len(names) == len(set(names)) == len(entries)
               and set(names) == set(entries), 'Archive names differ or repeat')
        for name in sorted(names):
            c.need(not Path(name).is_absolute() and '..' not in Path(name).parts,
                   'Unsafe archive entry')
            h = hashlib.sha256(); count = 0; crc = 0
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block); count += len(block); crc = zlib.crc32(block, crc)
            expected = entries[name]; info = archive.getinfo(name)
            c.need(count == expected['bytes'] == info.file_size
                   and h.hexdigest() == expected['sha256'] and (crc & 0xffffffff) == info.CRC,
                   'Archive stream identity failed: ' + name)
            c.verify(expected)
            actual.append({'name': name, 'bytes': count, 'sha256': h.hexdigest(),
                           'CRC32': crc & 0xffffffff})
    return actual


def finalize(reg_sha):
    reg = c.admit(reg_sha, 'finalize')
    receipts = {phase: c.checked_receipt(c.STAGE / (phase + '_execution_receipt.json'), phase)
                for phase in ('primary', 'reload', 'audit')}
    primary, reload, audit = [c.read(receipts[p]['outputs']['acceptance']['path'])
                              for p in ('primary', 'reload', 'audit')]
    c.need(primary['passed'] is True and reload['passed'] is True and audit['passed'] is True
           and primary['operations'] == c.PRIMARY_BUDGET and reload['operations'] == c.RELOAD_BUDGET
           and audit['operations'] == dict.fromkeys(c.OPERATION_KEYS, 0)
           and audit['all_class_agreement'] is True
           and audit['verified_total_forward_calls'] == 7
           and audit['verified_total_forward_samples'] == 131,
           'Actual finite engineering checks have not passed')
    c.need(audit['old_TRAIN_boundary_matches'] == 15 and audit['golden_input_windows'] == 32
           and audit['new_independent_test'] is False and audit['new_market_accuracy'] is None,
           'Local consistency scope changed')
    c.reserve(reg, 'finalize')
    now = datetime.now(timezone.utc); local = now.astimezone(timezone(timedelta(hours=8)))
    date = local.date().isoformat()
    before_count = c.verify_protected(reg)
    manifest = c.read(c.PACKAGE / 'package_manifest.json')
    for name, ref in manifest['files'].items():
        p = Path(name)
        c.need(not p.is_absolute() and '..' not in p.parts, 'Package manifest path escaped')
        actual = c.binding(c.PACKAGE / p)
        c.need(actual['sha256'] == ref['sha256'] and actual['bytes'] == ref['bytes'],
               'Package file changed before archive: ' + name)
    entries = {}
    for p in sorted(c.PACKAGE.rglob('*')):
        if p.is_file():
            entries['package/' + p.relative_to(c.PACKAGE).as_posix()] = c.binding(p)
    evidence = ['registration.json', 'authorization.json', 'protected_inventory.json',
                'source_review.json', 'tests_evidence.json', 'tests_execution_receipt.json',
                'tests.stdout.log', 'tests.stderr.log']
    if 'test_capture_binding' in reg:
        evidence += ['source_review_v2.json', 'tests_v2_execution_receipt.json',
                     'tests_v2_capture_reservation.json', 'tests_v2_launch.json',
                     'tests_v2.stdout.log', 'tests_v2.stderr.log', 'synthetic_import_failure_diagnosis.json']
    for phase in ('primary', 'reload', 'audit'):
        evidence += [phase + suffix for suffix in ('_acceptance.json', '_execution_receipt.json',
                    '_phase_reservation.json', '_capture_reservation.json', '_launch.json',
                    '.stdout.log', '.stderr.log')]
    evidence += ['reload_logits.f32', 'prepared/window_plan.json', 'before/budget.json']
    for name in evidence:
        entries['execution_evidence/' + name] = c.binding(c.STAGE / name)
    for name, ref in reg['source_bindings'].items():
        entries['source/' + Path(ref['path']).name] = c.binding(c.verify(ref))
    if 'test_capture_binding' in reg:
        ref = reg['test_capture_binding']
        entries['source/' + Path(ref['path']).name] = c.binding(c.verify(ref))
        for p in sorted((c.STAGE / 'initial_failed_synthetic_snapshot').rglob('*')):
            if p.is_file():
                entries['execution_evidence/initial_failed_synthetic_snapshot/'
                        + p.relative_to(c.STAGE / 'initial_failed_synthetic_snapshot').as_posix()] = c.binding(p)
    archive = c.STAGE / ('TCN_ED4_Seed42_Development_Delivery_' + date.replace('-', '') + '.zip')
    c.write_new(c.STAGE / 'delivery_zip_manifest.json', {'version': 'TCN-ED7-archive-inputs-1.0',
        'files': entries, 'scope': 'Portable package plus frozen source and actual checks up to audit; final completion/receipt remain adjacent to avoid cyclic identities'})
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        for name, ref in sorted(entries.items()):
            out.write(c.verify(ref), arcname=name)
    checked = verify_archive(archive, entries)
    c.write_new(c.STAGE / 'delivery_zip_verification.json', {'version': 'TCN-ED7-archive-verification-1.0',
        'passed': True, 'archive': c.binding(archive), 'files': checked, 'file_count': len(checked),
        'all_uncompressed_streams_SHA_bytes_CRC_verified': True, 'new_model_operations': 0})
    c.need(c.verify_protected(reg) == before_count, 'Historical identity count changed')
    old = c.read(c.verify(reg['evidence_bindings']['ED6_completion']))
    metrics = old['retained_DEV_metrics']
    strategy = c.read(c.verify(reg['evidence_bindings']['ED5_completion']))
    decision = c.read(c.verify(reg['evidence_bindings']['ED3_decision_config']))
    report = c.ROOT / ('docs/TCN_现有数据优化第7步交付与本轮结论报告_' + date + '.txt')
    plan_path = c.ROOT / 'state/tcn_software_plan_v27.json'
    state_path = c.ROOT / 'state/tcn_existing_data_optimization_20261017.json'
    completion_path = c.STAGE / 'completion_acceptance.json'
    c.need(not report.exists() and not plan_path.exists() and not completion_path.exists(),
           'Completed ED7 outputs already exist; do not overwrite')
    text = [
        'TCN 现有数据优化第7步：开发验证结论与独立软件交付',
        '实际完成时间（上海）：' + local.isoformat(), '',
        '本轮结论：按ED6冻结规则保留ED4相对价格方案，代表固定seed42。',
        f"开发验证 N={metrics['samples']}；Accuracy={metrics['accuracy']*100:.10f}%，Macro-F1={metrics['macro_f1']:.12f}，CE={metrics['loss']:.12f}。",
        '跌/平/涨 Recall：' + ' / '.join(f'{v*100:.10f}%' for v in metrics['recall']),
        '跌/平/涨 Precision：' + ' / '.join(f'{v*100:.10f}%' for v in metrics['precision']),
        '跌/平/涨 F1：' + ' / '.join(f'{v:.12f}' for v in metrics['f1']),
        '混淆矩阵（真实行、预测列，跌平涨）：' + json.dumps(metrics['confusion_true_rows_predicted_columns']),
        '以上完全继承ED6冻结代表在2025-12-15/16开发集上的既有结果；本步没有新测市场准确率。',
        '两seed同anchor Accuracy差分别+1.0928439485pp/+1.3590607706pp；不择较高seed43，不作统计显著性或新独立跨日泛化承诺。', '',
        '本轮优化取舍（继承已完成步骤的证据，不新搜索）：',
        'ED1固定现有五日TRAIN、15/16开发集与有限四组预算；ED2完成固定基线诊断。',
        f"ED3单独决策实验记录平稳logit偏置b={decision['stationary_logit_bias']}，没有把它叠加到本次特征模型；交付保持原argmax。",
        f"ED4相对价格seed42 DEV Accuracy={metrics['accuracy']*100:.10f}%；同开发集原WN42={old['seed42_control_DEV_metrics']['accuracy']*100:.10f}%。",
        f"同开发集平稳类Recall：原WN42 {old['seed42_control_DEV_metrics']['recall'][1]*100:.10f}% → 保留代表 {metrics['recall'][1]*100:.10f}%；不能与旧最终测试集平Recall直接比较。",
        f"ED5仅lr降为.0003后Accuracy={strategy['candidate_original_argmax_DEV_metrics']['accuracy']*100:.10f}%，未胜父方案；因此保留lr.001。",
        'ED6按原两个seed的Accuracy配对规则确认开发采用；ED7只交付该固定版本与工程证据。', '',
        '工程交付与实际验收：',
        '初次49项合成检查为48通过/1失败：导入静态规则误判普通预算字典update。原失败源码、预约、真实收据与日志原字节保留；只将预算改为同值纯声明，不改49项测试规则，另用独立v2预约完成49项实际检查。此修复在任何真实模型载入前完成，不增加模型或训练预算，不将两次合为98项通过。',
        '独立目录：' + str(c.PACKAGE), '压缩交付包：' + str(archive),
        'TorchScript CPU模型、原生完整checkpoint原字节备份、专属归一化/配置/协议、独立预处理与推理入口、环境/使用/回退说明、32黄金窗、输出和证据均已保存。',
        '输入为每档 ask_price,ask_size,bid_price,bid_size，共40列；每条当前mid相对价格，原数量，float64变换/归一化后float32。不得套用旧绝对价格归一化；原argmax、不加ED3平稳类偏置。',
        '原生WN33539参数，10个有效WN卷积权重复制到同架构33219参数导出副本；没有重新训练或改变模型有效权重。',
        '实际primary：1次源CP CPU载入、2个eager模型、1次script编译、1次导出载入，6次前向/99个样本；独立reload：1次TorchScript载入、1次前向/32个样本。合计7次前向、131个样本，32个独立TRAIN窗口。',
        f"所有预登记输出容差rtol=1e-4/atol=1e-5和类别核对通过；最大绝对logit差={audit['maximum_absolute_difference']:.12g}，检查内类别一致率100%。这不是100%市场准确率。",
        '只读取既有D TRAIN cache的32个100行窗，累计512000字节；其中15窗与ED4原边界SHA一致，其余17窗执行前固定。没有重验整份cache SHA，不将局部窗检查声称为全量原始行情链路验收。',
        '真实raw原始行情全链路未在本步重放；raw预处理/输入边界用独立纯NumPy/mock合成测试核验，真实工程窗口来自既有归一化TRAIN cache。',
        f'交付ZIP共{len(checked)}个文件，逐压缩流SHA256、字节数和CRC与原件独立核对通过。历史保护{before_count}项及关闭预算保持。', '',
        '边界与历史：',
        '0新增训练、0optimizer构造/恢复/更新、0反向、0RNG恢复、0归一化重拟合；未下载数据、未读E行情或标签、未读DEV cache或17—19记录/标签/预测流。',
        '原最终测试Accuracy45.3243568848%和quality=false原结论保持，不与本轮DEV成绩作新最终测试提升比较。',
        '源control/覆盖未知、原flag64保持。结果限“基于当前可观察数据和既定筛选规则的开发验证表现”。',
        'ED6外部电池守护真实失败exit2，未取得handle/未发请求；安全失败闭环不等于守护生效或清理API成功。失败证据与附页随包保存；主训练AC-only清理单列。',
        '本步没有板端/NPU/能耗/实盘/长时稳定性验收或部署。回退原WN42只读包保持；不得把相对价格模型与旧归一化/旧导出混用。',
        '本轮七步计划至此完成，无自动后续动作。后续训练/新测试/部署需另行具体任务和预算，不重开旧四组预算或重复选用17—19。', '',
        '工程审计：' + str(c.STAGE / 'audit_acceptance.json'),
        'package manifest SHA256：' + c.sha(c.PACKAGE / 'package_manifest.json'),
        '本报告不将工程一致性替代预测质量结论。',
    ]
    with report.open('x', encoding='utf-8', newline='\n') as out:
        out.write('\n'.join(text) + '\n')
    completion = {'version': 'TCN-ED7-completion-1.0', 'passed': True,
        'status': 'completed_existing_data_development_cycle_delivered', 'step': 7, 'phase_id': 'ED7',
        'worker_pid': os.getpid(), 'completed_utc': now.isoformat(), 'completed_Asia_Shanghai': local.isoformat(),
        'registration': c.binding(c.REGISTRATION), 'selected_family': c.FAMILY, 'representative_seed': 42,
        'retained_DEV_metrics': metrics, 'new_market_accuracy': None, 'new_independent_test': False,
        'historical_final_quality_changed': False, 'source_control_verified': False, 'coverage_unknown': True,
        'original_flag64_cleared': False, 'engineering_delivery_passed': True,
        'model_quality_newly_accepted': False, 'raw_market_end_to_end_verified': False,
        'primary_receipt': c.binding(c.STAGE / 'primary_execution_receipt.json'),
        'reload_receipt': c.binding(c.STAGE / 'reload_execution_receipt.json'),
        'audit_receipt': c.binding(c.STAGE / 'audit_execution_receipt.json'),
        'package_manifest': c.binding(c.PACKAGE / 'package_manifest.json'), 'package_directory': str(c.PACKAGE),
        'delivery_zip': c.binding(archive),
        'delivery_zip_verification': c.binding(c.STAGE / 'delivery_zip_verification.json'),
        'primary_operations': primary['operations'], 'reload_operations': reload['operations'],
        'forwards_this_step': 7, 'forward_samples_this_step': 131, 'unique_TRAIN_windows': 32,
        'historical_files_protected': before_count, 'report': c.binding(report),
        'source_checkpoint_unchanged': reg['checkpoint_binding'], 'normalization_unchanged': reg['normalization_binding'],
        'resource_support_failed_historically': True, 'resource_guard_passed_historically': False,
        'closed_budget_unchanged': c.binding(c.CAMPAIGN / 'budget.json'),
        'training_started': False, 'deployment_started': False, 'automatic_next_action': False,
        'plan': {'path': str(plan_path)}, 'state': {'path': str(state_path)}}
    # The plan/state point to completion by path only, avoiding a hash cycle.
    previous = c.read(c.ROOT / 'state/tcn_software_plan_v26.json')
    previous.update(version='TCN-software-plan-27.0', status=completion['status'],
        previous_plan='state/tcn_software_plan_v26.json', previous_plan_preserved=True,
        current_step=7, current_phase_id='ED7', completed_utc=now.isoformat(),
        completion={'path': str(completion_path)}, guide=c.binding(report), next_step=None,
        next_trigger=None, automatic_next_action=False, training_authorized_now=False,
        ED7_package_manifest=completion['package_manifest'], ED7_delivery_zip=completion['delivery_zip'])
    for step in previous['steps']:
        if step['id'] == 'ED7': step['status'] = 'completed'
    put(plan_path, previous)
    old_state = state_path.read_bytes()
    (c.STAGE / 'before_final_state.json').write_bytes(old_state)
    state = {'version': 'TCN-existing-data-development-state-7.0', 'status': completion['status'],
        'current_step': 7, 'current_phase_id': 'ED7', 'current_plan': str(plan_path),
        'completion': {'path': str(completion_path)}, 'guide': c.binding(report),
        'representative_seed': 42, 'retained_family': c.FAMILY,
        'retained_DEV_accuracy': metrics['accuracy'], 'retained_DEV_macro_f1': metrics['macro_f1'],
        'retained_DEV_stationary_recall': metrics['recall'][1],
        'package_directory': str(c.PACKAGE), 'package_manifest': completion['package_manifest'],
        'delivery_zip': completion['delivery_zip'], 'training_authorized_now': False,
        'new_independent_test': False, 'automatic_next_action': False, 'next_step': None}
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    completion['plan'] = c.binding(plan_path); completion['state'] = c.binding(state_path)
    c.write_new(completion_path, completion)
    c.need(c.verify_protected(reg) == before_count, 'Historical files changed during publication')
    print(json.dumps({'passed': True, 'completion': c.binding(completion_path),
                      'delivery_zip': completion['delivery_zip'], 'worker_pid': os.getpid()}, ensure_ascii=True), flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registration-sha256', required=True)
    return finalize(p.parse_args(argv).registration_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
