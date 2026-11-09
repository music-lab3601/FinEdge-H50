"""One bounded TRAIN-only raw-to-frozen-ED4 reference extraction authorized on 2026-10-20."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'artifacts/tcn_hardware_raw_handoff_2026-10-20'
MODEL = ROOT / 'artifacts/tcn_existing_data_optimization_2026-10-17/step07_delivery/torchscript_compatibility_v3/package'
P2 = ROOT / 'artifacts/tcn_research_p2_2026-10-14/data_20261014T020631_732415/manifest.json'
CACHE = ROOT / 'artifacts/tcn_existing_data_optimization_2026-10-17/step04_features/prepared/cache_manifest.json'
PROTOCOL = ROOT / 'configs/tcn_research_protocol_b_v1.json'
DATE, SYMBOL, COUNT = '2025-12-08', 'AAPL', 1024
STARTS = (0, 1, 2, 100, 500, 924)
PINS = {
    P2: '95b45a9faebf2dd5657901c919ac89aff82ef4a0b137c073b7865ec715493ba0',
    CACHE: '24d4989704db8b4eebf7692ebed9d2221bf6ff52f16e8944ee1b3a810760c08c',
    PROTOCOL: '7003d63f5b6b49f7d6f757ebaee1128c06091f1d8a0cbd0c1abb53b044a0c9aa',
    MODEL / 'package_manifest.json': 'bf8244237ac5fc4726b33fe2aee49a15b6f5ac2fdb46485bbd6289933fbdabee',
    MODEL / 'normalization.json': '709114d09b2dd581bd9481e180973fdf1553452b644d24dcb211da7ea394caaf',
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def readj(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def writej(path, obj):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def binding(path):
    p = Path(path)
    return {'bytes': p.stat().st_size, 'sha256': sha(p)}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def signature(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def read_span(path, offset, count):
    with Path(path).open('rb') as f:
        f.seek(offset)
        result = f.read(count)
    need(len(result) == count, 'Truncated bounded input span')
    return result


def main():
    STAGE.mkdir(exist_ok=False)
    package = STAGE / 'package'
    package.mkdir()
    writej(STAGE / 'execution_plan.json', {
        'authorized_request': 'Provide a consecutive screened raw top10 slice, preprocessing references, explicit windows, logits/classes and hashes for the delivered ED4 seed42 model.',
        'date': DATE, 'symbol': SYMBOL, 'role': 'train', 'record_count': COUNT,
        'selection': 'First 1024 original rows of the first previously accepted 2025-12-08 research range; chosen before reading model outputs.',
        'window_starts_0based': list(STARTS), 'window_length': 100,
        'maximum_torchscript_loads': 1, 'maximum_forward_calls': 2, 'maximum_forward_samples': 12,
        'source_record_numeric_bytes': COUNT * 352, 'ED4_cache_numeric_bytes': COUNT * 160,
        'screening_mask_numeric_bytes': COUNT * 4, 'training': False, 'optimizer': False,
        'labels_or_DEV_or_17_19_streams': False, 'old_campaign_budget_reopened': False,
    })
    for path, expected in PINS.items():
        need(sha(path) == expected, 'Frozen identity differs: ' + str(path))
    protected = {str(p): binding(p) for p in PINS}
    for relative in ('state/tcn_software_plan_v27.json', 'state/tcn_existing_data_optimization_20261017.json',
                     'artifacts/tcn_existing_data_optimization_2026-10-17/budget.json'):
        path = ROOT / relative
        protected[str(path)] = binding(path)
    p2, cache, protocol, model_manifest = readj(P2), readj(CACHE), readj(PROTOCOL), readj(MODEL / 'package_manifest.json')
    need(p2['status'] == 'prepared_real_data_passed' and cache['complete'], 'Preparation not accepted')
    need(protocol['symbol'] == SYMBOL and protocol['date_roles'][DATE] == 'train', 'Instrument/role differs')
    p1ref = p2['p1_manifest_binding']
    p1path = Path(p1ref['path'])
    need(sha(p1path) == p1ref['sha256'], 'P1 metadata identity differs')
    p1 = readj(p1path)
    need(p1['engineering_screening_passed'] and p1['research_data_index_ready'], 'P1 not accepted')
    selected = next(r for r in p2['ranges'] if r['date'] == DATE and r['role'] == 'train')
    p1range = next(r for r in p1['ranges'] if r['date'] == DATE)
    need(all(selected[k] == v for k, v in p1range.items()), 'P1/P2 ranges differ')
    source = p2['sources'][DATE]
    need(source['source_kind'] == 'native' and source['source_control_verified'] and not source['coverage_unknown'], 'Unexpected source profile')
    start, stop = selected['start_row'], selected['start_row'] + COUNT
    need(stop <= selected['stop_row'] and start == 120855, 'Fixed snippet outside selected range')
    src = Path(source['records_path'])
    need(src.drive.upper() == 'E:' and src.name == 'records.bin', 'Wrong source path')
    for ref in source['metadata_bindings']:
        need(binding(ref['path']) == {k: ref[k] for k in ('bytes', 'sha256')}, 'Source metadata differs')
    report = readj(src.parent / 'report.json')
    need(report['symbol'] == SYMBOL and report['trading_date'] == DATE, 'Source symbol/date mismatch')
    schema = readj(src.parent / 'schema.json')
    need(schema['row_bytes'] == 352 and schema['timestamp_unit'] == 'ns since exchange midnight', 'Source schema differs')
    screen = readj(p1path.parent / DATE / 'screening.json')
    need(screen['research_window_acceptance_passed'] and screen['screening_passed'] and not screen['day_problem_codes'], 'Day screening failed')
    mask_path = Path(screen['mask_path'])
    need(sha(mask_path) == screen['mask_sha256'], 'Historical rejection mask differs')
    cache_ref = cache['cache_files'][DATE]
    cr = next(r for r in cache['ranges'] if r['date'] == DATE)
    need(all(cr[k] == v for k, v in selected.items()), 'ED4 cache range differs')
    cache_path = Path(cache_ref['path'])
    before = {str(p): signature(p) for p in (src, cache_path, mask_path)}
    need(before[str(src)]['bytes'] == source['record_bytes'] == source['rows'] * 352, 'Source file size differs')
    need(before[str(cache_path)]['bytes'] == cache_ref['bytes'], 'ED4 cache size differs')
    original_bytes = read_span(src, start * 352, COUNT * 352)
    mask_bytes = read_span(mask_path, start * 4, COUNT * 4)
    cache_start = cr['cache_offset'] + start - cr['start_row']
    cached_bytes = read_span(cache_path, cache_start * 160, COUNT * 160)
    print('Bounded TRAIN source/cache/mask extraction completed.', flush=True)

    import numpy as np
    dtype = np.dtype([('sequence', '<u8'), ('timestamp_ns', '<u8'), ('segment_id', '<u4'),
                      ('flags', '<u4'), ('bid_levels', '<u4'), ('ask_levels', '<u4'), ('features', '<u8', (40,))])
    need(dtype.itemsize == 352, 'Record dtype geometry differs')
    records = np.frombuffer(original_bytes, dtype=dtype)
    raw = np.ascontiguousarray(records['features'], dtype='<u8')
    mask = np.frombuffer(mask_bytes, dtype='<u4')
    native_segments = readj(source['segments_path'])
    native = next(s for s in native_segments if s['start_row'] <= start and s['end_row'] + 1 >= stop)
    need(np.all(mask == 0) and np.all(records['flags'] == 0), 'Rejected/native-invalid row in snippet')
    need(np.all(records['segment_id'] == native['segment_id']), 'Native segment crossed')
    need(np.all(records['bid_levels'] >= 10) and np.all(records['ask_levels'] >= 10), 'Insufficient depth')
    need(np.all(records['sequence'][1:] > records['sequence'][:-1]), 'Source order broken')
    need(np.all(records['timestamp_ns'][1:] >= records['timestamp_ns'][:-1]), 'Time regression')
    need(np.all(records['timestamp_ns'] >= 34500 * 10**9) and np.all(records['timestamp_ns'] < 57300 * 10**9), 'Outside frozen screening clock')
    need(np.all(raw > 0) and np.all(raw[:, 0::2] < 2**51), 'Bad raw prices/quantities')
    need(np.all(raw[:, 0] > raw[:, 2]) and np.all(raw[:, 4::4] > raw[:, 0:-4:4])
         and np.all(raw[:, 6::4] < raw[:, 2:-4:4]), 'Invalid top10 book geometry')
    for relative, ref in model_manifest['files'].items():
        need(binding(MODEL / relative) == ref, 'Delivered model payload changed: ' + relative)
    shutil.copytree(MODEL, package / 'ed4_model')
    copied_model = package / 'ed4_model'
    runtime = load_module('frozen_ed4_runtime_for_raw_handoff', copied_model / 'runtime.py')
    normalization = readj(copied_model / 'normalization.json')
    # Apply the exact delivered API to all 1024 rows via eleven overlapping 100-row windows.
    covering_starts = list(range(0, 1000, 100)) + [924]
    covering_raw = np.stack([raw[i:i + 100] for i in covering_starts])
    covering_normalized = runtime.preprocess_raw(covering_raw, normalization)
    normalized = np.concatenate([covering_normalized[:10].transpose(0, 2, 1).reshape(1000, 40),
                                 covering_normalized[10].T[-24:]], axis=0).astype('<f4', copy=False)
    # Separate historical training implementation; importing it has no model/data access.
    
    ed4 = load_module('frozen_ed4_feature_reference', MODEL / 'source/tcn_existing_data_ed4_features.py')
    independent_normalized = ed4.normalized_rows(raw, normalization)
    need(normalized.tobytes() == independent_normalized.tobytes() == cached_bytes,
         'Delivered raw preprocessing / training implementation / original ED4 cache differ')
    windows_raw = np.stack([raw[i:i + 100] for i in STARTS])
    window_inputs = np.stack([normalized[i:i + 100].T for i in STARTS]).astype('<f4')
    need(runtime.preprocess_raw(windows_raw, normalization).tobytes() == window_inputs.tobytes(), 'Window layout differs')
    print('All 1024 normalized rows match the original ED4 training cache byte-for-byte.', flush=True)
    predictor = runtime.Predictor(copied_model, PINS[MODEL / 'package_manifest.json'])
    via_raw = predictor.predict_raw(windows_raw)
    via_normalized = predictor.predict_normalized(window_inputs)
    need(via_raw['logits'].tobytes() == via_normalized['logits'].tobytes()
         and np.array_equal(via_raw['class_indices'], via_normalized['class_indices']), 'Raw/normalized prediction paths differ')
    need(predictor._runner.forward_calls == 2 and predictor._runner.forward_samples == 12, 'Inference scope exceeded')
    logits = np.asarray(via_raw['logits'], dtype='<f4')
    classes = np.asarray(via_raw['class_indices'], dtype='u1')
    old_selection = readj(MODEL / 'golden/selection.json')['per_input']
    old_logits = np.frombuffer((MODEL / 'golden/script_batch32.f32').read_bytes(), dtype='<f4').reshape(32, 3)
    old_matches = []
    for i, offset in enumerate(STARTS):
        old = next((w for w in old_selection if w['date'] == DATE and w['endpoint'] == start + offset + 99), None)
        if old is not None:
            need(digest(window_inputs[i].tobytes()) == old['input_sha256'], 'Existing ED7 golden input differs')
            reference = old_logits[old['index']]
            need(np.allclose(logits[i], reference, rtol=1e-4, atol=1e-5)
                 and int(classes[i]) == int(reference.argmax()), 'Existing ED7 golden output differs')
            old_matches.append({'new_window_index': i, 'ED7_golden_index': old['index'],
                                'input_byte_identical': True, 'class_identical': True,
                                'maximum_absolute_logit_difference': float(np.abs(logits[i] - reference).max())})
    need(len(old_matches) == 3, 'Expected three old golden overlaps')
    need(before == {str(p): signature(p) for p in (src, cache_path, mask_path)}, 'Source changed during extraction')
    need(all(binding(p) == ref for p, ref in protected.items()), 'Frozen software identity changed')

    (package / 'source_slice.records.bin').write_bytes(original_bytes)
    (package / 'raw_book.u64le.bin').write_bytes(raw.tobytes())
    (package / 'normalized_rows.f32le.bin').write_bytes(normalized.tobytes())
    (package / 'screening_reject_masks.u32le.bin').write_bytes(mask_bytes)
    (package / 'window_inputs.f32le.bin').write_bytes(window_inputs.tobytes())
    (package / 'window_logits.f32le.bin').write_bytes(logits.tobytes())
    (package / 'window_classes.u8.bin').write_bytes(classes.tobytes())
    epoch_base = datetime(2025, 12, 8, 5, tzinfo=timezone.utc)
    unix_midnight = (epoch_base - datetime(1970, 1, 1, tzinfo=timezone.utc)).days * 86400 * 10**9 + 5 * 3600 * 10**9
    # Integer UTC origin: New York midnight on this registered EST date is 05:00Z.
    need(unix_midnight == int(epoch_base.timestamp()) * 10**9, 'UTC origin conversion differs')

    def stamps(t):
        t = int(t)
        sec, nano = divmod(t, 10**9)
        local_time = f'{sec // 3600:02}:{sec // 60 % 60:02}:{sec % 60:02}.{nano:09}'
        utc_value = epoch_base + timedelta(seconds=sec)
        return {'timestamp_ns_since_exchange_midnight': t, 'exchange_datetime_est': DATE + 'T' + local_time + '-05:00',
                'derived_utc_datetime': utc_value.strftime('%Y-%m-%dT%H:%M:%S') + f'.{nano:09}Z',
                'derived_unix_timestamp_ns': unix_midnight + t}

    row_objects = []
    with (package / 'row_metadata.jsonl').open('x', encoding='utf-8', newline='\n') as f:
        for i, r in enumerate(records):
            row = {'sample_row_0based': i, 'sample_order_1based': i + 1, 'source_record_row_0based': start + i,
                   'date': DATE, 'symbol': SYMBOL, 'snippet_id': 'AAPL_20251208_R1_rows120855_121879',
                   'research_segment_id': selected['research_segment_id'], 'native_segment_id': int(r['segment_id']),
                   'source_message_position_1based': int(r['sequence']), 'source_flags': int(r['flags']),
                   'research_reject_mask': int(mask[i]), 'bid_levels': int(r['bid_levels']), 'ask_levels': int(r['ask_levels']),
                   **stamps(r['timestamp_ns'])}
            row_objects.append(row)
            f.write(json.dumps(row, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n')
    window_objects = []
    for i, offset in enumerate(STARTS):
        end = offset + 99
        window_objects.append({'window_id': f'W{i:02}', 'window_index': i,
            'sample_start_inclusive_0based': offset, 'sample_stop_exclusive_0based': offset + 100,
            'sample_endpoint_0based': end, 'source_start_row_0based': start + offset,
            'source_endpoint_row_0based': start + end, 'date': DATE, 'symbol': SYMBOL,
            'research_segment_id': selected['research_segment_id'],
            'first_source_message_position_1based': int(records[offset]['sequence']),
            'last_source_message_position_1based': int(records[end]['sequence']),
            'first_time': stamps(records[offset]['timestamp_ns']), 'last_time': stamps(records[end]['timestamp_ns']),
            'input_sha256': digest(window_inputs[i].tobytes()), 'input_binary_offset_bytes': i * 100 * 40 * 4,
            'logits_down_stationary_up': logits[i].tolist(), 'logits_sha256': digest(logits[i].tobytes()),
            'predicted_class_index': int(classes[i]), 'predicted_class_name': via_raw['class_names'][i],
            'is_market_truth_label': False})
    writej(package / 'windows.json', {'indexing': 'zero-based; start inclusive and stop exclusive',
           'input_shape': [len(STARTS), 40, 100], 'input_dtype': '<f4', 'logits_shape': [len(STARTS), 3],
           'class_order': ['down', 'stationary', 'up'], 'decision': 'original argmax; first index on tie; no stationary bias',
           'windows': window_objects})
    evidence = package / 'provenance'
    evidence.mkdir()
    for name, path in {'P1_manifest.json': p1path, 'P2_manifest.json': P2,
                       'ED4_cache_manifest.json': CACHE, 'screening_protocol.json': PROTOCOL,
                       'day_screening.json': p1path.parent / DATE / 'screening.json',
                       'source_schema.json': src.parent / 'schema.json',
                       'native_segments.json': Path(source['segments_path']),
                       'native_report.json': src.parent / 'report.json',
                       'execution_plan.json': STAGE / 'execution_plan.json'}.items():
        shutil.copyfile(path, evidence / name)
    shutil.copyfile(__file__, evidence / Path(__file__).name)
    shutil.copyfile(ROOT / 'scripts/verify_tcn_raw_handoff.py', package / 'verify_reference.py')
    acceptance = {'status': 'passed', 'created_shanghai': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'raw_records': COUNT, 'source_original_rows_contiguous': True, 'all_preexisting_reject_masks_zero': True,
        'same_research_and_native_segment': True, 'source_sequence_strictly_increasing': True,
        'source_sequence_is_exchange_transport_sequence': False, 'timestamp_non_decreasing': True,
        'delivered_runtime_vs_frozen_ED4_feature_code_byte_identical': True,
        'all_1024_rows_vs_ED4_training_cache_byte_identical': True, 'raw_and_normalized_model_outputs_byte_identical': True,
        'ED7_prior_golden_overlap_comparisons': old_matches, 'torchscript_loads': 1,
        'model_forward_calls': predictor._runner.forward_calls, 'model_forward_samples': predictor._runner.forward_samples,
        'eager_model_constructions': 0, 'native_checkpoint_deserializations': 0, 'training_runs': 0,
        'optimizer_constructions_or_updates': 0, 'normalization_refits': 0, 'market_accuracy_evaluations': 0,
        'DEV_or_17_19_record_label_prediction_streams_read': False, 'source_drive_writes': 0,
        'original_source_slice_numeric_bytes': len(original_bytes), 'ED4_cache_slice_numeric_bytes': len(cached_bytes),
        'mask_slice_numeric_bytes': len(mask_bytes), 'metadata_hash_copy_IO_counted_separately': True,
        'source_full_file_SHA_recomputed': False, 'ED4_cache_full_file_SHA_recomputed': False,
        'source_file_signatures_before_after': before, 'frozen_software_files_unchanged': protected,
        'NPU_or_FPGA_or_INT8_acceptance': False,
        'scope': 'Engineering references for this single preselected TRAIN slice; no new market test or model selection.'}
    writej(package / 'acceptance.json', acceptance)
    readme = f'''ED4 seed42硬件交接：连续原始十档价量与软件参考

日期：{DATE}；股票：{SYMBOL}；样本数：{COUNT}。
片段：AAPL_20251208_R1_rows120855_121879。原records行[{start},{stop})，0-based；研究段1，原生段{native['segment_id']}。
固定选取既有TRAIN第一accepted范围的开头，不按类别或分数挑片段；无删除坏行后的拼接、重排、补值或重采样。

主要文件（所有二进制都没有文件头或行padding，C顺序）：
raw_book.u64le.bin：little-endian uint64 [{COUNT},40]，每行320字节。
  依次为ask_price_1,ask_size_1,bid_price_1,bid_size_1,...,ask_price_10,ask_size_10,bid_price_10,bid_size_10。
  保留原整数价格单位和原数量；不要先除10000；源schema的price_divisor是显示单位说明。
normalized_rows.f32le.bin：little-endian float32 [{COUNT},40]，每行160字节。已完成相对价变换与固定归一化。
row_metadata.jsonl：逐行日期、股票、片段号、两种连续样本索引、原record行号、原消息位置、原flags、研究reject mask、深度、原始及派生时间。
  source_message_position_1based是原ITCH文件的消息位置，不是交易所网络序号。筛选同一股票后允许有间隔；连续性以原records行号相邻为准。
  timestamp_ns_since_exchange_midnight是纽约当日00:00起纳秒，不是Unix时间。此日期为EST(UTC-05:00)。
  derived_utc_datetime和derived_unix_timestamp_ns是按固定日期时区计算的派生值；没有捏造ts_recv或网络到达时间。
source_slice.records.bin：原352字节结构记录的原字节截取，字段偏移见manifest/provenance/source_schema.json；用于核对sidecar与原flags。
screening_reject_masks.u32le.bin：原筛选mask同位置切片，全部为0，未修改原flags。
window_inputs.f32le.bin：little-endian float32 [6,40,100]；channel-first，时间从旧到新。
window_logits.f32le.bin：little-endian float32 [6,3]，顺序跌/平/涨，未做softmax。
window_classes.u8.bin：uint8 [6]，0跌、1平、2涨。只是固定模型预测，不是市场真实标签。
windows.json：六个窗口的确切边界、起止消息位置/时间、logits、类别、输入和输出SHA256。
  窗口起点(0-based)={list(STARTS)}，每个100条；stop为exclusive。

预处理严格沿用交付runtime：
  每条记录mid=(ask1+bid1)/2，各价格列=(price-mid)/mid；数量列保持原值。
  float64变换后以ed4_model/normalization.json的40列mean/std做(x-mean)/std，最后转float32。
  先保存[N,40]；单个窗口取rows[start:start+100].T并连续存储，得到[40,100]。
  ed4_model为原ED7完整package的原字节复制，含CPU TorchScript、native/best.pt、源码、归一化及原黄金窗。
  model manifest SHA256={PINS[MODEL / 'package_manifest.json']}。
  ED4 seed42/epoch3，原argmax，无平稳类logit偏置，不修改模型或重新拟合归一化。

复核（使用Python和NumPy；从本目录运行）：
  python verify_reference.py
  python verify_reference.py --run-model
第一条仅文件hash/原始数据预处理/窗口映射复核；第二条额外加载CPU TorchScript并一次推理6窗，需安装匹配的PyTorch。
requirements及实际验证环境见ed4_model/environment.json和requirements_snapshot.txt。

真实验证：这1024行已与原ED4训练缓存逐字节一致；六窗raw入口与normalized入口输出逐字节一致；三个重叠旧ED7黄金窗也通过容差及类别核对。
所有文件SHA256见manifest.json与SHA256SUMS.txt；manifest自身hash列于SHA256SUMS.txt。
原完整records/cache的历史SHA保留用于溯源，本次仅按位置读片段并新算片段SHA，未声称重算两份完整大文件SHA；stat前后相同。

用途限制：本包是真实TRAIN片段的FPGA预处理/布局/NPU迁移工程对照，不是独立准确率测试、足量代表性INT8校准集或持续性能测试流。
原DEV准确率46.8784020942%及历史quality=false结论不变；17—19封存流未读。此旧原生日期有已验收控制段，不能外推其他日期控制覆盖已解决。
本次0训练/optimizer/归一化重拟合；仅1次CPU TorchScript载入、2次forward共12个窗口样本。无上板、量化或NPU验收。
'''
    (package / 'README_先读我.txt').write_text(readme, encoding='utf-8', newline='\n')
    files = {p.relative_to(package).as_posix(): binding(p) for p in sorted(package.rglob('*')) if p.is_file()}
    manifest = {'version': 'TCN-ED4-raw-hardware-handoff-1.0', 'date': DATE, 'symbol': SYMBOL,
        'snippet_id': 'AAPL_20251208_R1_rows120855_121879', 'role': 'train', 'record_count': COUNT,
        'research_segment_id': selected['research_segment_id'], 'native_segment_id': native['segment_id'],
        'source_rows': {'start_inclusive': start, 'stop_exclusive': stop, 'index_base': 0},
        'source_record_dtype': dtype.descr, 'source_record_bytes': 352,
        'feature_names': [f'{name}_{level}' for level in range(1, 11) for name in ('ask_price', 'ask_size', 'bid_price', 'bid_size')],
        'raw_dtype': '<u8', 'raw_shape': [COUNT, 40], 'raw_price_units': 'unchanged original integers; no divide by10000',
        'normalized_dtype': '<f4', 'normalized_shape': [COUNT, 40], 'window_input_shape': [6, 40, 100],
        'logits_shape': [6, 3], 'class_order': ['down', 'stationary', 'up'],
        'timestamp_semantics': 'integer ns since 2025-12-08T00:00:00-05:00; UTC values derived by integer arithmetic',
        'first_row': row_objects[0], 'last_row': row_objects[-1],
        'source_control_verified_for_this_native_day': True, 'coverage_unknown_for_this_native_day': False,
        'other_dates_coverage_limitations_unchanged': True, 'source_range': selected,
        'source_full_file_sha256_historical_not_recomputed': source['records_sha256'],
        'source_slice_original_352byte_records_sha256': digest(original_bytes),
        'ED4_cache_full_file_sha256_historical_not_recomputed': cache_ref['sha256'],
        'ED4_cache_slice_sha256_recomputed': digest(cached_bytes),
        'model_family': model_manifest['selected_family'], 'model_seed': 42, 'model_epoch': 3,
        'model_manifest_sha256': PINS[MODEL / 'package_manifest.json'],
        'normalization_sha256': PINS[MODEL / 'normalization.json'], 'files': files}
    writej(package / 'manifest.json', manifest)
    print(json.dumps({'status': 'references_created', 'records': COUNT, 'windows': len(STARTS),
                      'classes': classes.tolist(), 'model_forward_calls': 2, 'package': str(package)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if STAGE.is_dir() and not (STAGE / 'failure.json').exists():
            writej(STAGE / 'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'completed': False})
        raise
