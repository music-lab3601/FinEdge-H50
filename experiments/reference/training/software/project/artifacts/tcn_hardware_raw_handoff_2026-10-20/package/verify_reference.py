"""Verify the portable raw-book handoff; model inference is explicit and optional."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--run-model', action='store_true', help='One CPU TorchScript load and one six-window forward')
    args = parser.parse_args()
    root = args.package.resolve()
    m = json.loads((root / 'manifest.json').read_text('utf-8'))
    for relative, ref in m['files'].items():
        path = (root / relative).resolve()
        assert path.is_relative_to(root) and path.is_file(), relative
        assert path.stat().st_size == ref['bytes'] and sha(path) == ref['sha256'], relative

    import numpy as np
    n = m['record_count']
    raw = np.fromfile(root / 'raw_book.u64le.bin', dtype='<u8').reshape(n, 40)
    expected = np.fromfile(root / 'normalized_rows.f32le.bin', dtype='<f4').reshape(n, 40)
    norm = json.loads((root / 'ed4_model/normalization.json').read_text('utf-8'))
    # Independent row-wise formula; preserve the frozen F64 subtract/divide order.
    values = raw.astype(np.float64)
    midpoint = (values[:, 0] + values[:, 2]) / 2.0
    values[:, 0::2] = (values[:, 0::2] - midpoint[:, None]) / midpoint[:, None]
    actual = np.ascontiguousarray((values - np.asarray(norm['mean'], dtype=np.float64))
                                  / np.asarray(norm['std'], dtype=np.float64), dtype='<f4')
    assert actual.tobytes() == expected.tobytes(), 'Preprocessing bytes differ'
    rows = [json.loads(s) for s in (root / 'row_metadata.jsonl').read_text('utf-8').splitlines()]
    assert len(rows) == n
    for i, row in enumerate(rows):
        assert row['sample_row_0based'] == i and row['sample_order_1based'] == i + 1
        assert row['source_record_row_0based'] == m['source_rows']['start_inclusive'] + i
        assert row['date'] == m['date'] and row['symbol'] == m['symbol']
        assert row['research_segment_id'] == m['research_segment_id']
        if i:
            assert row['source_message_position_1based'] > rows[i - 1]['source_message_position_1based']
            assert row['timestamp_ns_since_exchange_midnight'] >= rows[i - 1]['timestamp_ns_since_exchange_midnight']
    windows = json.loads((root / 'windows.json').read_text('utf-8'))['windows']
    inputs = np.fromfile(root / 'window_inputs.f32le.bin', dtype='<f4').reshape(len(windows), 40, 100)
    logits = np.fromfile(root / 'window_logits.f32le.bin', dtype='<f4').reshape(len(windows), 3)
    classes = np.fromfile(root / 'window_classes.u8.bin', dtype='u1')
    for i, w in enumerate(windows):
        start, stop = w['sample_start_inclusive_0based'], w['sample_stop_exclusive_0based']
        assert stop - start == 100 and 0 <= start < stop <= n
        assert inputs[i].tobytes() == np.ascontiguousarray(expected[start:stop].T).tobytes()
        assert hashlib.sha256(inputs[i].tobytes()).hexdigest() == w['input_sha256']
        assert w['logits_down_stationary_up'] == logits[i].tolist()
        assert int(classes[i]) == int(logits[i].argmax()) == w['predicted_class_index']
    result = {'file_hashes_verified': len(m['files']), 'raw_records': n,
              'preprocessing_byte_identical': True, 'window_count': len(windows),
              'model_forward_calls': 0, 'training': False}
    if args.run_model:
        spec = importlib.util.spec_from_file_location('handoff_ed4_runtime', root / 'ed4_model/runtime.py')
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        predictor = runtime.Predictor(root / 'ed4_model', m['model_manifest_sha256'])
        output = predictor.predict_normalized(inputs)
        assert np.allclose(output['logits'], logits, rtol=1e-4, atol=1e-5)
        assert np.array_equal(output['class_indices'], classes)
        result.update(model_forward_calls=1, model_forward_samples=len(windows),
                      maximum_absolute_logit_difference=float(np.abs(output['logits'] - logits).max()))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
