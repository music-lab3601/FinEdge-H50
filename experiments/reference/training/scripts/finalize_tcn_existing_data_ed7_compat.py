"""Publish the separately admitted ED7 compatibility delivery; no ML imports.

Original tests, failed primary, sources, registration and partial package remain
untouched. The compatibility run has its own finite registration and receipts.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import traceback
import zipfile
import zlib

from deployment.export import tcn_existing_data_ed7_compat_contract as c


def same(left, right):
    return (c.safe(left['path']).resolve() == c.safe(right['path']).resolve()
            and left['sha256'] == right['sha256'] and left['bytes'] == right['bytes'])


def new_json(path, value):
    path = c.safe(path)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def new_bytes(path, value):
    with c.safe(path).open('xb') as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())


def atomic_bytes(path, value):
    path = c.safe(path)
    temporary = path.with_name(path.name + '.ED7_compat_pending')
    new_bytes(temporary, value)
    os.replace(temporary, path)


def archive_name(name):
    c.need(isinstance(name, str) and name and '\\' not in name and ':' not in name,
           'Archive name must be relative POSIX')
    part = PurePosixPath(name)
    c.need(not part.is_absolute() and '..' not in part.parts and part.as_posix() == name,
           'Unsafe archive entry')
    return name


def add_entry(entries, name, ref):
    name = archive_name(name)
    c.need(name not in entries, 'Repeated archive entry: ' + name)
    c.verify(ref); entries[name] = ref


def verify_archive(path, entries):
    """Read every new ZIP stream independently; opaque SHA/bytes/CRC only."""
    result = []
    with zipfile.ZipFile(path, 'r') as archive:
        names = archive.namelist()
        c.need(len(names) == len(set(names)) == len(entries) and set(names) == set(entries),
               'Archive names differ or repeat')
        for name in sorted(names):
            archive_name(name)
            digest = hashlib.sha256(); count = 0; crc = 0
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block); count += len(block); crc = zlib.crc32(block, crc)
            expected = entries[name]; info = archive.getinfo(name)
            c.need(count == expected['bytes'] == info.file_size
                   and digest.hexdigest() == expected['sha256'] and (crc & 0xffffffff) == info.CRC,
                   'Archive stream identity failed: ' + name)
            c.verify(expected)
            result.append({'name': name, 'bytes': count, 'sha256': digest.hexdigest(),
                           'CRC32': crc & 0xffffffff})
    return result


def prior_failure():
    """Use the preserved failed terminal, never restart its consumed phase."""
    receipt = c.read(c.BASE_STAGE / 'primary_execution_receipt.json')
    failure = c.read(c.BASE_STAGE / 'primary_failure.json')
    c.need(receipt['phase'] == 'primary' and receipt['exit_code'] == 1
           and receipt['entry_unchanged'] is True and receipt['sources_unchanged'] is True
           and receipt['entry_before'] == receipt['entry_after']
           and receipt['sources_before'] == receipt['sources_after'],
           'Original failed primary evidence changed')
    for key in ('input', 'entry_before', 'entry_after', 'capture_source', 'stdout', 'stderr'):
        c.verify(receipt[key])
    for source in receipt['sources_before']:
        c.verify(source)
    c.need(same(receipt['outputs']['failure'], c.binding(c.BASE_STAGE / 'primary_failure.json'))
           and same(receipt['input'], c.binding(c.BASE_STAGE / 'registration.json'))
           and failure['phase'] == 'primary' and failure['error_type'] == 'RuntimeError'
           and failure['error'] == 'requires_grad_ is not supported on ScriptModules'
           and failure['reservation_remains_consumed'] is True and failure['automatic_retry'] is False,
           'Compatibility repair no longer matches the recorded failure')
    observed = dict(failure['operations_observed'])
    observed.setdefault('saved_golden_read_bytes', 0)
    c.need(observed == c.FAILED_PRIMARY_OPERATIONS
           and observed['forward_attempts'] == observed['forward_calls'] == observed['forward_samples'] == 0
           and observed['torchscript_saves'] == observed['torchscript_loads'] == 0,
           'Original actual failed-operation accounting differs')
    return failure, receipt


def test_count(reg):
    evidence = c.read(c.verify(reg['test_evidence_binding']))
    actual = c.read(c.verify(evidence['actual_receipt']))
    count = evidence['synthetic_tests_run']
    c.need(evidence['passed'] is True and evidence['actual_exit_code'] == actual['exit_code'] == 0
           and type(count) is int and count > 0, 'Compatibility synthetic evidence did not pass')
    stderr = c.verify(actual['stderr']).read_text('utf-8')
    match = re.search(r'\bRan (\d+) tests? in ', stderr)
    c.need(match is not None and int(match.group(1)) == count
           and '\nOK\n' in stderr.replace('\r\n', '\n') and 'FAILED' not in stderr,
           'Actual compatibility test count/terminal differs')
    return count, reg['test_evidence_binding'], evidence['actual_receipt']


def history_entries(reg, entries):
    frozen = reg['history_archive_bindings']
    c.need(isinstance(frozen, list) and len(frozen) == 78,
           'Exactly78 pre-frozen original stage history files required')
    seen = set()
    for ref in frozen:
        path = c.safe(ref['path'])
        c.need(path.is_relative_to(c.BASE_STAGE) and not path.is_relative_to(c.STAGE)
               and path.name != 'active_execution.json', 'Recursive/dynamic compatibility history forbidden')
        c.need(path not in seen, 'Duplicate original history input')
        seen.add(path)
        add_entry(entries, 'history/original_step07_delivery/'
                  + path.relative_to(c.BASE_STAGE).as_posix(), ref)
    required = ('registration.json', 'primary_failure.json', 'primary_execution_receipt.json',
                'primary_phase_reservation.json', 'tests_execution_receipt.json',
                'tests_v2_execution_receipt.json', 'source_review.json', 'source_review_v2.json',
                'synthetic_import_failure_diagnosis.json', 'initial_failed_synthetic_snapshot/preservation.json')
    c.need(all(c.BASE_STAGE / name in seen for name in required), 'Failed/test history omitted from archive')
    original = c.read(c.verify(reg['original_registration_binding']))
    sources = list(original['source_bindings'].values())
    if 'test_capture_binding' in original:
        sources.append(original['test_capture_binding'])
    source_paths = set()
    for ref in sources:
        path = c.safe(ref['path'])
        c.need(path not in source_paths, 'Repeated original source identity')
        source_paths.add(path)
        add_entry(entries, 'history/original_source/' + path.name, ref)
    c.need(len(original['source_bindings']) == 10 and len(source_paths) == 11,
           'Original registration core10 plus separate test capture history required')
    return {'original_stage_files': len(seen), 'original_source_files': len(source_paths),
            'total_original_history_files': len(seen) + len(source_paths)}


def finalize(registration_sha):
    reg = c.admit(registration_sha, 'finalize')
    receipts = {phase: c.checked_receipt(c.STAGE / (phase + '_execution_receipt.json'), phase)
                for phase in ('primary', 'reload', 'audit')}
    primary, reload, audit = [c.read(receipts[phase]['outputs']['acceptance']['path'])
                              for phase in ('primary', 'reload', 'audit')]
    failed, failed_receipt = prior_failure()
    cumulative = {key: c.FAILED_PRIMARY_OPERATIONS[key] + primary['operations'][key]
                  + reload['operations'][key] for key in c.OPERATION_KEYS}
    c.need(primary['passed'] is True and reload['passed'] is True and audit['passed'] is True
           and primary['operations'] == c.PRIMARY_BUDGET and reload['operations'] == c.RELOAD_BUDGET
           and cumulative == c.CUMULATIVE_BUDGET == audit['verified_cumulative_operations']
           and audit['verified_initial_failed_operations'] == c.FAILED_PRIMARY_OPERATIONS
           and audit['cumulative_operation_accounting_passed'] is True
           and audit['saved_golden_input_reused'] is True and audit['source_cache_reopened'] is False
           and audit['operations'] == dict.fromkeys(c.OPERATION_KEYS, 0)
           and audit['all_class_agreement'] is True and audit['verified_total_forward_calls'] == 7
           and audit['verified_total_forward_samples'] == 131
           and audit['old_TRAIN_boundary_matches'] == 15 and audit['golden_input_windows'] == 32
           and audit['new_independent_test'] is False and audit['new_market_accuracy'] is None,
           'Actual separate compatibility checks/cumulative accounting have not passed')
    c.need(cumulative['eager_model_constructions'] == 4 and cumulative['checkpoint_deserializations'] == 2
           and cumulative['weight_norm_folds'] == 20 and cumulative['script_compilations'] == 2
           and cumulative['forward_calls'] == 7 and cumulative['forward_samples'] == 131
           and cumulative['source_cache_window_reads'] == 32
           and cumulative['source_cache_payload_bytes_read'] == 512000
           and cumulative['saved_golden_read_bytes'] == 512000,
           'Original failed work was hidden or repeated source reads occurred')
    count, test_evidence, test_receipt = test_count(reg)
    bridge_path = c.BASE_STAGE / 'completion_acceptance.json'
    completion_path = c.STAGE / 'completion_acceptance.json'
    plan_path = c.ROOT / 'state/tcn_software_plan_v27.json'
    state_path = c.ROOT / 'state/tcn_existing_data_optimization_20261017.json'
    now = datetime.now(timezone.utc); local = now.astimezone(timezone(timedelta(hours=8)))
    date = local.date().isoformat()
    report = c.ROOT / ('docs/TCN_现有数据优化第7步交付与本轮结论报告_' + date + '.txt')
    c.need(not any(path.exists() for path in (bridge_path, completion_path, plan_path, report)),
           'Completed ED7 output already exists; do not overwrite')
    c.reserve(reg, 'finalize')
    protected_count = c.verify_protected(reg)
    manifest = c.read(c.PACKAGE / 'package_manifest.json')
    entries = {}
    for name, ref in manifest['files'].items():
        archive_name(name); actual = c.binding(c.PACKAGE / name)
        c.need(actual['sha256'] == ref['sha256'] and actual['bytes'] == ref['bytes'],
               'New package file changed before archive')
    for path in sorted(c.PACKAGE.rglob('*')):
        if path.is_file():
            add_entry(entries, 'package/' + path.relative_to(c.PACKAGE).as_posix(), c.binding(path))
    for phase in ('primary', 'reload', 'audit'):
        for suffix in ('_acceptance.json', '_execution_receipt.json', '_phase_reservation.json',
                       '_capture_reservation.json', '_launch.json', '.stdout.log', '.stderr.log'):
            name = phase + suffix
            add_entry(entries, 'execution_evidence/' + name, c.binding(c.STAGE / name))
    for name, ref in (('registration.json', c.binding(c.REGISTRATION)),
                      ('source_review.json', reg['source_review_binding']),
                      ('tests_evidence.json', test_evidence), ('tests_execution_receipt.json', test_receipt),
                      ('reload_logits.f32', c.binding(c.STAGE / 'reload_logits.f32'))):
        add_entry(entries, 'execution_evidence/' + name, ref)
    for name in ('authorization.json', 'protected_inventory.json', 'prepared/window_plan.json',
                 'before/budget.json', 'tests_capture_reservation.json', 'tests_launch.json',
                 'worker_finalizer_independent_source_review.json'):
        add_entry(entries, 'execution_evidence/' + name, c.binding(c.STAGE / name))
    actual_tests = c.read(c.verify(test_receipt))
    for name in ('stdout', 'stderr'):
        add_entry(entries, 'execution_evidence/tests.' + name + '.log', actual_tests[name])
    for name, ref in reg['source_bindings'].items():
        add_entry(entries, 'source/' + Path(ref['path']).name, ref)
    if 'test_capture_binding' in reg:
        add_entry(entries, 'source/' + Path(reg['test_capture_binding']['path']).name, reg['test_capture_binding'])
    history_counts = history_entries(reg, entries)
    archived_history = history_counts['total_original_history_files']
    archive = c.STAGE / ('TCN_ED4_Seed42_Development_Delivery_' + date.replace('-', '') + '.zip')
    c.write_new(c.STAGE / 'delivery_zip_manifest.json', {'version': 'TCN-ED7-compat-archive-inputs-1.0',
        'files': entries, 'frozen_original_history_files': archived_history,
        'original_history_counts': history_counts,
        'scope': 'New package/sources/actual evidence through audit plus frozen original failed history; completion/bridge/final receipt adjacent to avoid cycles'})
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for name, ref in sorted(entries.items()):
            output.write(c.verify(ref), arcname=name)
    verified = verify_archive(archive, entries)
    c.write_new(c.STAGE / 'delivery_zip_verification.json', {'version': 'TCN-ED7-compat-archive-verification-1.0',
        'passed': True, 'archive': c.binding(archive), 'files': verified, 'file_count': len(verified),
        'all_uncompressed_streams_SHA_bytes_CRC_verified': True, 'new_model_operations': 0})
    c.need(c.verify_protected(reg) == protected_count, 'Historical identities changed during archive')
    old = c.read(c.verify(reg['evidence_bindings']['ED6_completion']))
    strategy = c.read(c.verify(reg['evidence_bindings']['ED5_completion']))
    decision = c.read(c.verify(reg['evidence_bindings']['ED3_decision_config']))
    metrics = old['retained_DEV_metrics']
    text = [
        'TCN 现有数据优化第7步：开发验证结论与独立软件交付',
        '实际完成时间（上海）：' + local.isoformat(), '',
        '按ED6冻结规则保留ED4相对价格、原数量、lr.001、原argmax；代表固定seed42/epoch3。',
        f"开发验证N={metrics['samples']}；Accuracy={metrics['accuracy']*100:.10f}%，Macro-F1={metrics['macro_f1']:.12f}，CE={metrics['loss']:.12f}。",
        '跌/平/涨Recall：' + ' / '.join(f'{v*100:.10f}%' for v in metrics['recall']),
        '跌/平/涨Precision：' + ' / '.join(f'{v*100:.10f}%' for v in metrics['precision']),
        '跌/平/涨F1：' + ' / '.join(f'{v:.12f}' for v in metrics['f1']),
        '混淆矩阵（真实行、预测列，跌平涨）：' + json.dumps(metrics['confusion_true_rows_predicted_columns']),
        '以上继承15/16重复使用开发集的既有结果；本步没有新测市场准确率或独立测试。',
        f"同DEV原WN42 Accuracy={old['seed42_control_DEV_metrics']['accuracy']*100:.10f}%，平Recall={old['seed42_control_DEV_metrics']['recall'][1]*100:.10f}%；当前平Recall={metrics['recall'][1]*100:.10f}%。",
        '两seed共同anchor Accuracy差+1.0928439485pp/+1.3590607706pp；不择较高seed43，不作统计显著性或新跨日泛化承诺。',
        f"ED3单独决策实验b={decision['stationary_logit_bias']}未叠加；ED5仅lr.0003成绩{strategy['candidate_original_argmax_DEV_metrics']['accuracy']*100:.10f}%未胜父方案，保留lr.001。", '',
        '工程兼容修复与真实消耗：',
        '原49套件48通过/1静态误报失败保持；预算声明同值修复后原49源码通过独立v2实际检查，未将两次合为98项通过。',
        f'本次兼容版本实际合成套件{count}项通过，按独立真实收据计数；原49-v2作为历史分列。',
        '原唯一primary真实exit1：ScriptModule不支持模块级requires_grad_；原预约、源码、注册、partial package及日志原字节保留。',
        '随后显式登记一次独立兼容工程阶段，只修ScriptModule梯度接口；编译模型eval并断言继承无梯度，导出载入后逐参数设置Tensor标志。未改权重、归一化、特征、32窗或选型规则。',
        '原失败实际2eager/1源CP载入/10fold/1compile，0save/load/forward；补充primary2eager/1源CP/10fold/1compile/1save/1scriptload/6forward99样本，独立reload1scriptload/1forward32样本。',
        '本步累计4eager、2次源CP CPU载入、20fold、2compile，合计7次前向/131个样本、32个独立TRAIN窗口；不冒称原2eager预算一次完成。',
        '原失败已读取32个D TRAIN缓存窗/512000应用层数值字节；补充阶段复用保存黄金输入，一次数值载入512000字节，源cache新读取0。SHA/复制/ZIP的不透明身份读取另计，不把数值读取量称为所有文件I/O总量。',
        '15窗与ED4原TRAIN边界SHA一致，17窗预选；未重验整份cache SHA，也未真实raw全链路回放。raw预处理仅合成边界证据。',
        f"rtol=1e-4/atol=1e-5及类别核对通过；最大绝对logit差={audit['maximum_absolute_difference']:.12g}，检查内类别一致率100%，不是市场准确率100%。",
        '原生WN33539参数、10有效卷积权重复制到33219参数同结构导出模型；CPU TorchScript默认推理。',
        '交付目录：' + str(c.PACKAGE), 'ZIP：' + str(archive),
        f"ZIP共{len(verified)}文件，逐压缩流SHA/字节/CRC通过；原stage历史{history_counts['original_stage_files']}文件、原登记源码{history_counts['original_source_files']}文件，共{archived_history}文件分列；保护{protected_count}项和关闭预算保持。", '',
        '边界：0新增训练/optimizer构造、恢复或更新/反向/RNG恢复/归一化重拟合；不读E/标签/DEV cache/17—19流，不下载新数据。',
        '原最终测试Accuracy45.3243568848%和quality=false保持，不与本轮DEV作新最终提升比较；源control/覆盖未知和原flag64保持。',
        '结果限“基于当前可观察数据和既定筛选规则的开发验证表现”。',
        'ED6外部电池守护失败exit2/无handle/无请求的安全失败闭环不等于守护生效或清理API成功；主训练AC-only清理单列。附页随包。',
        '无板端/NPU/能耗/实盘/长时稳定性或部署验收；原WN42只读回退整包保持，不与本相对价模型/归一化混用。',
        '七步计划至此完成，无自动后续动作；后续需独立任务及有限预算，不重开旧四组训练预算或重复选用17—19。',
        '工程审计：' + str(c.STAGE / 'audit_acceptance.json'),
        'package manifest SHA256：' + c.sha(c.PACKAGE / 'package_manifest.json'),
    ]
    new_bytes(report, ('\n'.join(text) + '\n').encode('utf-8'))
    status = 'completed_existing_data_development_cycle_delivered'
    manifest_ref = c.binding(c.PACKAGE / 'package_manifest.json'); zip_ref = c.binding(archive)
    previous = c.read(c.ROOT / 'state/tcn_software_plan_v26.json')
    previous.update(version='TCN-software-plan-27.0', status=status,
        previous_plan='state/tcn_software_plan_v26.json', previous_plan_preserved=True,
        current_step=7, current_phase_id='ED7', completed_utc=now.isoformat(),
        completion={'path': str(bridge_path)}, detailed_completion={'path': str(completion_path)},
        guide=c.binding(report), next_step=None, next_trigger=None, automatic_next_action=False,
        training_authorized_now=False, ED7_package_manifest=manifest_ref, ED7_delivery_zip=zip_ref)
    for step in previous['steps']:
        if step['id'] == 'ED7': step['status'] = 'completed'
    new_json(plan_path, previous)
    new_bytes(c.STAGE / 'before_final_state.json', state_path.read_bytes())
    state = {'version': 'TCN-existing-data-development-state-7.0', 'status': status,
        'current_step': 7, 'current_phase_id': 'ED7', 'current_plan': str(plan_path),
        'completion': {'path': str(bridge_path)}, 'detailed_completion': {'path': str(completion_path)},
        'guide': c.binding(report), 'representative_seed': 42, 'retained_family': c.FAMILY,
        'retained_DEV_accuracy': metrics['accuracy'], 'retained_DEV_macro_f1': metrics['macro_f1'],
        'retained_DEV_stationary_recall': metrics['recall'][1], 'package_directory': str(c.PACKAGE),
        'package_manifest': manifest_ref, 'delivery_zip': zip_ref, 'training_authorized_now': False,
        'new_independent_test': False, 'automatic_next_action': False, 'next_step': None}
    atomic_bytes(state_path, (json.dumps(state, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
    handoff = c.STAGE / 'handoff_before'; handoff.mkdir(exist_ok=False)
    header = ('## ' + date + ' 新七步ED7交付完成，本轮结束\n\n'
        '固定ED4 seed42/epoch3，相对价格、原数量、原归一化、原argmax。DEV Accuracy46.8784020942%、'
        'Macro-F1 .463728341443、平Recall34.1787780038%；仅开发验证，无新独立测试。\n'
        '原ScriptModule接口失败及预约保留，另注册有限兼容阶段已实际通过；累计4eager/2CP载入/20fold/2compile、'
        '7forward/131样本，0训练/optimizer。源cache总32窗512000数值字节，新阶段源cache读取0；'
        '另读取保存黄金512000数值字节，opaque身份/复制I/O单列。\n'
        'ED6辅助guard仍failed_no_handle_no_request/guard_passed=false，安全失败闭环与主训练AC-only清理分列。'
        '原最终quality=false、源control/覆盖未知及flag64保持；17—19不重评。\n'
        '完成桥接：' + str(bridge_path) + '；详情：' + str(completion_path) + '\n'
        '报告：' + str(report) + '；当前计划v27。无自动后续训练、测试、下载或部署。\n\n')
    handoff_rows = []
    for name in ('AGENTS.md', 'PROGRESS.md'):
        path = c.ROOT / name; before = path.read_bytes(); before_ref = c.binding(path)
        new_bytes(handoff / name, before)
        atomic_bytes(path, header.encode('utf-8') + before)
        handoff_rows.append({'path': str(path), 'before': before_ref,
                             'saved_before': c.binding(handoff / name), 'after': c.binding(path)})
    c.write_new(c.STAGE / 'handoff_update_receipt.json', {'version': 'TCN-ED7-compat-handoff-update-1.0',
        'files': handoff_rows, 'completed_utc': now.isoformat(), 'new_model_operations': 0})
    completion = {'version': 'TCN-ED7-compat-completion-1.0', 'passed': True, 'status': status,
        'step': 7, 'phase_id': 'ED7', 'worker_pid': os.getpid(), 'completed_utc': now.isoformat(),
        'completed_Asia_Shanghai': local.isoformat(), 'registration': c.binding(c.REGISTRATION),
        'selected_family': c.FAMILY, 'representative_seed': 42, 'retained_DEV_metrics': metrics,
        'new_market_accuracy': None, 'new_independent_test': False, 'historical_final_quality_changed': False,
        'source_control_verified': False, 'coverage_unknown': True, 'original_flag64_cleared': False,
        'engineering_delivery_passed': True, 'model_quality_newly_accepted': False,
        'raw_market_end_to_end_verified': False, 'original_failed_operations': c.FAILED_PRIMARY_OPERATIONS,
        'primary_operations': primary['operations'], 'reload_operations': reload['operations'],
        'cumulative_operations': cumulative, 'forwards_this_step': 7, 'forward_samples_this_step': 131,
        'unique_TRAIN_windows': 32, 'compatibility_synthetic_test_count': count,
        'compatibility_test_evidence': test_evidence, 'compatibility_test_receipt': test_receipt,
        'original_failed_primary_receipt': c.binding(c.BASE_STAGE / 'primary_execution_receipt.json'),
        'original_failed_primary_failure': c.binding(c.BASE_STAGE / 'primary_failure.json'),
        'primary_receipt': c.binding(c.STAGE / 'primary_execution_receipt.json'),
        'reload_receipt': c.binding(c.STAGE / 'reload_execution_receipt.json'),
        'audit_receipt': c.binding(c.STAGE / 'audit_execution_receipt.json'),
        'package_manifest': manifest_ref, 'package_directory': str(c.PACKAGE), 'delivery_zip': zip_ref,
        'delivery_zip_verification': c.binding(c.STAGE / 'delivery_zip_verification.json'),
        'archived_original_history_files': archived_history, 'historical_files_protected': protected_count,
        'original_history_counts': history_counts,
        'report': c.binding(report), 'plan': c.binding(plan_path), 'state': c.binding(state_path),
        'source_checkpoint_unchanged': reg['checkpoint_binding'], 'normalization_unchanged': reg['normalization_binding'],
        'resource_support_failed_historically': True, 'resource_guard_passed_historically': False,
        'closed_budget_unchanged': c.binding(c.CAMPAIGN / 'budget.json'),
        'handoff_update_receipt': c.binding(c.STAGE / 'handoff_update_receipt.json'),
        'bridge_path': str(bridge_path), 'training_started': False, 'deployment_started': False,
        'automatic_next_action': False}
    c.need(c.verify_protected(reg) == protected_count, 'Original protected evidence changed before completion')
    c.write_new(completion_path, completion)
    bridge = dict(completion)
    bridge.update(version='TCN-ED7-compatibility-bridge-completion-1.0', bridge_only=True,
                  detailed_completion=c.binding(completion_path))
    new_json(bridge_path, bridge)
    print(json.dumps({'passed': True, 'completion': c.binding(completion_path),
        'completion_bridge': c.binding(bridge_path), 'delivery_zip': zip_ref,
        'worker_pid': os.getpid()}, ensure_ascii=True), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        return finalize(args.registration_sha256)
    except BaseException as error:
        failure = c.STAGE / 'finalize_failure.json'
        if c.STAGE.exists() and not failure.exists():
            c.write_new(failure, {'version': 'TCN-ED7-compat-finalize-failure-1.0',
                'worker_pid': os.getpid(), 'failed_utc': datetime.now(timezone.utc).isoformat(),
                'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
                'automatic_retry': False, 'new_model_operations': 0})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
