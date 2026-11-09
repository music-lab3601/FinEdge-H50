import platform
import sys
from time import perf_counter_ns

import numpy as np

from software.runtime.backends import CpuBackend, WorkerBackend, warmup
from software.runtime.delivery import read_json, runtime_module, sha256, write_json
from shared.model_paths import ARTIFACTS, MANIFEST_SHA, PACKAGE, RAW
from software.runtime.stream import InputBuilder, PREPROCESS_ID, RawSource


def verify_cpu():
    import torch
    runtime = runtime_module()
    source = RawSource()
    norm = read_json(PACKAGE / 'normalization.json')
    reference_rows = np.fromfile(RAW / 'normalized_rows.f32le.bin', dtype='<f4').reshape(-1, 40)
    builder = InputBuilder(norm)
    incremental = np.stack([builder.normalize(record.values) for record in source])
    rows_identical = incremental.tobytes() == reference_rows.tobytes()
    if not rows_identical:
        raise AssertionError('Incremental preprocessing differs from delivered rows')
    cpu = CpuBackend()
    golden = np.fromfile(PACKAGE / 'golden/inputs.f32', dtype='<f4').reshape(32, 40, 100)
    expected = np.fromfile(PACKAGE / 'golden/script_batch32.f32', dtype='<f4').reshape(32, 3)
    actual = np.stack([cpu.predict(i + 1, x[None].copy())['logits'] for i, x in enumerate(golden)]).astype('<f4')
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
    specification = read_json(RAW / 'windows.json')['windows']
    expected_inputs = np.fromfile(RAW / 'window_inputs.f32le.bin', dtype='<f4').reshape(6, 40, 100)
    expected_logits = np.fromfile(RAW / 'window_logits.f32le.bin', dtype='<f4').reshape(6, 3)
    indices = {x['sample_endpoint_0based']: i for i, x in enumerate(specification)}
    windows, selected = 0, []
    for i, record in enumerate(source):
        window = builder.push(record)
        if window is None:
            continue
        windows += 1
        if i in indices:
            j = indices[i]
            np.testing.assert_array_equal(window.tensor[0], expected_inputs[j])
            raw_reference = runtime.preprocess_raw(source.rows[i-99:i+1][None].copy(), norm)
            np.testing.assert_array_equal(window.tensor, raw_reference)
            result = cpu.predict(windows, window.tensor)
            np.testing.assert_allclose(result['logits'], expected_logits[j], rtol=1e-4, atol=1e-5)
            selected.append(result)
    assert windows == len(source) - 99 == 925 and len(selected) == 6
    reference = ARTIFACTS / 'fpga_reference'
    reference.mkdir(parents=True, exist_ok=True)
    np.save(reference / 'raw_book.npy', source.rows, allow_pickle=False)
    np.save(reference / 'normalized_rows.npy', incremental, allow_pickle=False)
    np.save(reference / 'window_inputs.npy', expected_inputs, allow_pickle=False)
    np.save(reference / 'window_logits.npy', expected_logits, allow_pickle=False)
    write_json(reference / 'windows.json', specification)
    write_json(reference / 'manifest.json', {
        'model_manifest_sha256': MANIFEST_SHA, 'raw_manifest_sha256': sha256(RAW / 'manifest.json'),
        'preprocess_id': PREPROCESS_ID, 'feature_order_per_level': ['ask_price', 'ask_size', 'bid_price', 'bid_size'],
        'input_shape': [1, 40, 100], 'time_order': 'oldest_to_newest',
        'window_starts': [w['sample_endpoint_0based']-99 for w in specification],
        'files': {p.name: sha256(p) for p in (reference / n for n in
                  ('raw_book.npy', 'normalized_rows.npy', 'window_inputs.npy', 'window_logits.npy', 'windows.json'))}})
    report = {'status': 'passed', 'python': platform.python_version(), 'torch': torch.__version__,
              'numpy': np.__version__, 'model_manifest_sha256': MANIFEST_SHA,
              'rows': len(source), 'preprocess_byte_identical': rows_identical,
              'stream_windows': windows, 'golden_windows': 32, 'golden_class_matches': int(np.sum(actual.argmax(1) == expected.argmax(1))),
              'golden_max_abs_difference': float(np.max(np.abs(actual-expected))), 'raw_reference_windows': selected,
              'rtol': 1e-4, 'atol': 1e-5, 'market_accuracy_evaluated': False}
    write_json(ARTIFACTS / 'reports/cpu_verification.json', report)
    return report


def verify_icraft(socket=True):
    golden = np.fromfile(PACKAGE / 'golden/inputs.f32', dtype='<f4').reshape(32, 40, 100)
    raw = np.fromfile(RAW / 'window_inputs.f32le.bin', dtype='<f4').reshape(6, 40, 100)
    inputs = np.concatenate((golden, raw))
    cpu = CpuBackend()
    host = WorkerBackend('host', timeout=45)
    device = None
    report = {'status': 'running', 'backend_info': {'host': host.info}, 'comparisons': []}
    report['compiled_json_sha256'] = sha256(ARTIFACTS / 'icraft/tcn_ed4_ZG.json')
    report['compiled_raw_sha256'] = sha256(ARTIFACTS / 'icraft/tcn_ed4_ZG.raw')
    report['model_manifest_sha256'] = MANIFEST_SHA
    try:
        if socket:
            device = WorkerBackend('socket', timeout=45)
            report['backend_info']['socket'] = device.info
        report['warmup'] = {'cpu': warmup(cpu, inputs[:1].copy()), 'host': warmup(host, inputs[:1].copy())}
        if device:
            report['warmup']['socket'] = warmup(device, inputs[:1].copy())
        for i, x in enumerate(inputs):
            tensor = x[None].copy()
            baseline = cpu.predict(i + 1, tensor)
            simulated = host.predict(i + 1, tensor)
            actual = device.predict(i + 1, tensor) if device else None
            a, b = np.array(baseline['logits']), np.array(simulated['logits'])
            row = {'input_index': i, 'cpu': baseline, 'host': simulated,
                   'conversion_max_abs_error': float(np.max(np.abs(a-b))),
                   'conversion_class_match': baseline['class_index'] == simulated['class_index']}
            if actual:
                c = np.array(actual['logits'])
                row.update(socket=actual, npu_max_abs_error=float(np.max(np.abs(c-b))),
                           npu_within_tolerance=bool(np.allclose(c, b, rtol=1e-3, atol=1e-4)),
                           npu_class_match=actual['class_index'] == simulated['class_index'])
            report['comparisons'].append(row)
        report['status'] = 'passed' if all(r.get('npu_within_tolerance', True) and r.get('npu_class_match', True) for r in report['comparisons']) else 'failed'
        report['summary'] = {'windows': len(inputs), 'npu_execution_verified': bool(device),
                             'conversion_class_matches': sum(r['conversion_class_match'] for r in report['comparisons']),
                             'npu_class_matches': sum(r.get('npu_class_match', False) for r in report['comparisons']),
                             'conversion_max_abs_error': max(r['conversion_max_abs_error'] for r in report['comparisons']),
                             'npu_max_abs_error': max((r.get('npu_max_abs_error', 0) for r in report['comparisons']), default=0),
                             'rtol': 1e-3, 'atol': 1e-4, 'market_accuracy_evaluated': False}
        if device and report['status'] == 'passed':
            reference = ARTIFACTS / 'fpga_reference'
            reference.mkdir(parents=True, exist_ok=True)
            np.save(reference / 'window_tf32_host_logits.npy',
                    np.asarray([r['host']['logits'] for r in report['comparisons'][32:]], dtype='<f4'), allow_pickle=False)
            np.save(reference / 'window_npu_logits.npy',
                    np.asarray([r['socket']['logits'] for r in report['comparisons'][32:]], dtype='<f4'), allow_pickle=False)
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        host.close()
        if device:
            device.close()
        write_json(ARTIFACTS / ('reports/npu_verification.json' if socket else 'reports/host_verification.json'), report)
        if socket:
            compilation_path = ARTIFACTS / 'icraft/compilation.json'
            compilation = read_json(compilation_path)
            matches = (compilation.get('generated_json_sha256') == report['compiled_json_sha256']
                       and compilation.get('generated_raw_sha256') == report['compiled_raw_sha256'])
            if matches:
                compilation['npu_execution_verified'] = report['status'] == 'passed'
                compilation['npu_verification_report'] = 'artifacts/reports/npu_verification.json'
                write_json(compilation_path, compilation)
    if report['status'] != 'passed':
        raise AssertionError('NPU/Host comparison failed; all differences retained')
    return report

