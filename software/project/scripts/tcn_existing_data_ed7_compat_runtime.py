"""Portable ED7 compatibility-v3 CPU inference for the fixed ED4 seed42 model.

Copy this file as runtime.py with the complete package. No project path,
checkpoint deserializer, optimizer, label reader or market-data reader is used.
NumPy validation is separate from the lazy TorchScript runner so boundary tests
can replace _new_runner without importing or constructing a real model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

TRAIN_DATES = ['2025-12-08', '2025-12-09', '2025-12-10', '2025-12-11', '2025-12-12']
FEATURE_ID = 'same_record_mid_relative_prices_raw_quantities_v1'
FROZEN_NORMALIZATION_SHA = '709114d09b2dd581bd9481e180973fdf1553452b644d24dcb211da7ea394caaf'
FROZEN_CHECKPOINT_SHA = 'c480db50073dbb099c5374dc5df9f942044e2ab09f8d6bcff396b7bc4cb6dc60'
CLASS_ORDER = ['down', 'stationary', 'up']
SEMANTIC_CONTRACT = {
    'version': 'TCN-ED7-relative-price-inference-1.0', 'feature_id': FEATURE_ID,
    'raw_shape': ['B', 100, 40], 'normalized_shape': ['B', 40, 100],
    'raw_dtype': 'uint64', 'normalized_dtype': 'float32', 'maximum_batch_size': 32,
    'raw_price_units': 'original_uint64_price_units_no_divide_10000',
    'price_formula': '(raw_price - (raw_ask1 + raw_bid1)/2) / ((raw_ask1 + raw_bid1)/2)',
    'quantity_formula': 'raw_quantity_unchanged_no_log',
    'transform_precision': 'float64_then_frozen_float64_normalization_then_float32',
    'feature_order': 'ask_price,ask_size,bid_price,bid_size for each level1..10',
    'class_order': CLASS_ORDER, 'decision': 'original_argmax_first_index_tie',
    'stationary_logit_bias': 0, 'device': 'cpu', 'training': False,
}


def need(value, message):
    if not value:
        raise ValueError(message)


def _strict_equal(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    path = Path(path); need(path.stat().st_size <= 8 * 1024**2, 'Oversized package metadata')
    return json.loads(path.read_text('utf-8-sig'),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))


def _file(root, relative):
    need(isinstance(relative, str) and relative and '\\' not in relative and ':' not in relative,
         'Package filenames must be relative POSIX paths')
    part = PurePosixPath(relative)
    need(not part.is_absolute() and '..' not in part.parts and '.' not in part.parts
         and part.as_posix() == relative, 'Unsafe package filename')
    path = root.joinpath(*part.parts)
    for ancestor in (path, *path.parents):
        if ancestor.exists():
            need(not ancestor.is_symlink() and not getattr(ancestor.lstat(), 'st_file_attributes', 0) & 0x400,
                 'Reparse package file forbidden')
        if ancestor == root:
            break
    need(path.resolve().is_relative_to(root.resolve()), 'Package file escaped root')
    return path


def validate_normalization(normalization):
    need(isinstance(normalization, dict) and normalization.get('feature_id') == FEATURE_ID
         and normalization.get('fit_dates') == TRAIN_DATES
         and normalization.get('TUNE_fitted') is False and normalization.get('sealed_fitted') is False,
         'Frozen TRAIN-only relative normalization required')
    need(normalization.get('price_formula') == '(raw_price - (raw_bid1+raw_ask1)/2) / ((raw_bid1+raw_ask1)/2)'
         and normalization.get('quantity_formula') == 'raw_quantity_unchanged_no_log'
         and normalization.get('transform_and_normalization_precision') == 'float64; final stored rows float32',
         'Normalization feature semantics differ')
    for key in ('mean', 'std'):
        need(isinstance(normalization.get(key), list) and len(normalization[key]) == 40
             and all(type(v) in (int, float) and np.isfinite(v) for v in normalization[key]),
             'Finite forty-channel normalization required')
    mean = np.asarray(normalization['mean'], dtype=np.float64)
    std = np.asarray(normalization['std'], dtype=np.float64)
    need(np.all(std > 0), 'Normalization standard deviations must be positive')
    return mean, std


def validate_normalized(inputs):
    need(isinstance(inputs, np.ndarray) and inputs.dtype == np.dtype('<f4')
         and inputs.ndim == 3 and 1 <= inputs.shape[0] <= 32 and inputs.shape[1:] == (40, 100),
         'Expected float32 [B,40,100], 1<=B<=32')
    need(np.isfinite(inputs).all(), 'Nonfinite normalized input')
    return np.ascontiguousarray(inputs, dtype='<f4')


def preprocess_raw(raw, normalization):
    need(isinstance(raw, np.ndarray) and raw.dtype == np.dtype('<u8')
         and raw.ndim == 3 and 1 <= raw.shape[0] <= 32 and raw.shape[1:] == (100, 40),
         'Expected raw uint64 [B,100,40], 1<=B<=32')
    prices = raw[..., 0::2]
    need(np.all(prices > 0) and np.all(prices < 2**51), 'Prices must be positive and below 2**51')
    physical = np.array(raw, dtype=np.float64, copy=True)
    midpoint = (physical[..., 0] + physical[..., 2]) / 2.0
    physical[..., 0::2] = (physical[..., 0::2] - midpoint[..., None]) / midpoint[..., None]
    mean, std = validate_normalization(normalization)
    normalized = (physical - mean[None, None, :]) / std[None, None, :]
    # Preserve the exact ED4 subtract-then-divide order and final F32 storage.
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.ascontiguousarray(normalized.transpose(0, 2, 1), dtype='<f4')
    return validate_normalized(result)


def validate_package(package, expected_manifest_sha256):
    root = Path(package).absolute()
    need(root.is_dir() and not root.is_symlink()
         and not getattr(root.lstat(), 'st_file_attributes', 0) & 0x400, 'Package root must be a regular directory')
    need(isinstance(expected_manifest_sha256, str) and len(expected_manifest_sha256) == 64
         and all(c in '0123456789abcdef' for c in expected_manifest_sha256), 'External manifest SHA256 required')
    manifest_path = _file(root, 'package_manifest.json')
    need(_sha(manifest_path) == expected_manifest_sha256, 'Package manifest identity differs')
    manifest = _json(manifest_path)
    need(manifest.get('version') == 'TCN-ED7-package-manifest-1.0'
         and manifest.get('selected_family') == 'ED4_feature42_lr0.001_seed42'
         and manifest.get('epoch') == 3 and manifest.get('seed') == 42
         and manifest.get('model_format') == 'TorchScript' and manifest.get('execution_device') == 'cpu'
         and _strict_equal(manifest.get('semantic_contract'), SEMANTIC_CONTRACT),
         'Package semantic/model identity differs')
    files = manifest.get('files')
    required = {'runtime.py', 'normalization.json', 'model_cpu.ts', 'native/best.pt',
                'golden/inputs.f32', 'golden/selection.json',
                *(f'golden/{kind}_batch{batch}.f32' for kind in ('native', 'effective', 'script') for batch in (32, 1))}
    need(isinstance(files, dict) and required <= set(files) and 'package_manifest.json' not in files,
         'Incomplete or circular package file manifest')
    need(files['normalization.json']['sha256'] == manifest.get('normalization_sha256') == FROZEN_NORMALIZATION_SHA
         and files['native/best.pt']['sha256'] == manifest.get('checkpoint_sha256') == FROZEN_CHECKPOINT_SHA,
         'The fixed ED4 family normalization/native checkpoint was replaced')
    for relative, ref in files.items():
        path = _file(root, relative)
        need(isinstance(ref, dict) and set(ref) == {'sha256', 'bytes'}
             and type(ref['bytes']) is int and ref['bytes'] >= 0 and path.is_file()
             and path.stat().st_size == ref['bytes'] and _sha(path) == ref['sha256'],
             'Package payload identity differs: ' + relative)
    need(_sha(Path(__file__)) == files['runtime.py']['sha256'], 'Executing runtime differs from the portable package source')
    normalization = _json(_file(root, 'normalization.json')); validate_normalization(normalization)
    return manifest, normalization


class _TorchRunner:
    def __init__(self, script_path):
        import torch
        self._torch = torch
        self.model = torch.jit.load(str(script_path), map_location='cpu')
        self.model.eval()
        # ScriptModule blocks Module.requires_grad_; the Tensor API is supported.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        need(sum(parameter.numel() for parameter in self.model.parameters()) == 33219,
             'Folded CPU architecture differs')
        need(all(parameter.device.type == 'cpu' and parameter.dtype == torch.float32
                 and not parameter.requires_grad and parameter.grad is None for parameter in self.model.parameters()),
             'CPU eval/no-gradient parameters required')
        self.forward_attempts = self.forward_calls = self.forward_samples = 0

    def run(self, inputs):
        values = validate_normalized(inputs)
        self.forward_attempts += 1
        with self._torch.inference_mode():
            output = self.model(self._torch.from_numpy(values))
        self.forward_calls += 1; self.forward_samples += len(values)
        need(output.device.type == 'cpu' and output.dtype == self._torch.float32
             and tuple(output.shape) == (len(values), 3) and self._torch.isfinite(output).all().item(),
             'Invalid CPU float32 logits')
        return np.array(output.detach().cpu().numpy(), dtype='<f4', copy=True)


def _new_runner(script_path):
    return _TorchRunner(script_path)


class Predictor:
    def __init__(self, package, expected_manifest_sha256):
        self.package = Path(package).absolute()
        self.manifest, self.normalization = validate_package(self.package, expected_manifest_sha256)
        self._runner = _new_runner(_file(self.package, 'model_cpu.ts'))

    def predict_normalized(self, inputs):
        values = validate_normalized(inputs)
        logits = self._runner.run(values)
        need(isinstance(logits, np.ndarray) and logits.dtype == np.dtype('<f4')
             and logits.shape == (len(values), 3) and np.isfinite(logits).all(), 'Runner logits contract differs')
        classes = logits.argmax(axis=1).astype(np.int64)
        return {'logits': logits, 'class_indices': classes,
                'class_names': [CLASS_ORDER[int(index)] for index in classes]}

    def predict_raw(self, raw):
        return self.predict_normalized(preprocess_raw(raw, self.normalization))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--mode', choices=('raw', 'normalized'), required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    data = np.load(args.input, allow_pickle=False)
    predictor = Predictor(args.package, args.manifest_sha256)
    result = predictor.predict_raw(data) if args.mode == 'raw' else predictor.predict_normalized(data)
    output = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items()}
    with Path(args.output).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(output, stream, ensure_ascii=False, allow_nan=False); stream.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
