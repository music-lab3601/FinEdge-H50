"""Model export and Icraft stages with explicit calibration and frozen scales."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys

from shared.io import atomic_json
from shared.paths import ConfigurationError, ROOT, environment, load_settings, resolve_path, tool_executable, work_path


def prepare_calibration(path):
    import numpy as np
    samples = np.load(path, allow_pickle=False)
    if samples.dtype != np.dtype('<f4') or samples.ndim != 3 or samples.shape[1:] != (40, 100):
        raise ValueError('Calibration requires float32[N,40,100] normalized by the frozen H50 parameters')
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Calibration samples must be finite and nonempty')
    directory = work_path('model_build/base/data')
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / 'calibration_bin'
    target.mkdir()
    names = []
    for index, tensor in enumerate(samples):
        name = f'{index:06d}.bin'
        (target / name).write_bytes(tensor.tobytes())
        names.append(name)
    shutil.copyfile(path, directory / 'inputs.npy')
    (directory / 'forward_list.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
    atomic_json(directory / 'manifest.json', dict(count=len(samples), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                                normalized=True, synthetic=False, normalization_ref='repo://models/normalization.json'))
    # Native scale folding consumes the same explicitly provided tensors.
    mixed = work_path('model_build/mixed/data')
    mixed.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, mixed / 'inputs.npy')
    return dict(status='prepared', windows=len(samples), directory=str(directory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    export = commands.add_parser('export')
    export.add_argument('--raw-reference', action='store_true')
    polyphase = commands.add_parser('polyphase')
    polyphase.add_argument('--label', required=True)
    polyphase.add_argument('--preserve-taps', action='store_true')
    calibration = commands.add_parser('calibration')
    calibration.add_argument('--input', type=str)
    compile_model = commands.add_parser('compile')
    compile_model.add_argument('--kind', choices=['tf32', 'int8', 'mixed', 'native'], required=True)
    helpers = commands.add_parser('tools')
    helpers.add_argument('--stage', choices=['base', 'mixed', 'native'], required=True)
    args = parser.parse_args()
    if args.action == 'tools':
        from tools.build_sdk import build_stage
        build_stage(args.stage)
        return 0
    if args.action == 'export':
        from deployment.export.onnx import export_onnx
        print(json.dumps(export_onnx(args.raw_reference), indent=2))
        return 0
    if args.action == 'polyphase':
        options = ['--label', args.label] + (['--preserve-taps'] if args.preserve_taps else [])
        return subprocess.call([sys.executable, '-B', '-m', 'deployment.export.polyphase', *options], env=environment())
    if args.action == 'calibration':
        path = resolve_path(args.input or load_settings()['inputs']['calibration'])
        if not path.is_file():
            raise ConfigurationError('Supply normalized calibration tensors with model calibration --input')
        print(json.dumps(prepare_calibration(path), indent=2))
        return 0
    tool_executable('icraft')
    base = work_path('model_build/base')
    (base/'.tmp').mkdir(parents=True,exist_ok=True)
    model = ROOT / 'models/onnx/tcn_polyphase.onnx'
    (base / 'model').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model, base / 'model/tcn_polyphase.onnx')
    if args.kind != 'tf32' and not (base / 'data/manifest.json').is_file():
        raise ConfigurationError('INT8 compilation requires model calibration --input first')
    modules = dict(tf32=('deployment.icraft.base.compile_models', ['--kind', 'tf32', '--shared']),
                   int8=('deployment.icraft.base.compile_models', ['--kind', 'int8', '--shared']),
                   mixed=('deployment.icraft.mixed.compile_mixed', ['--tf32-compat']),
                   native=('deployment.icraft.native.rebuild_native_models', []))
    required = {'mixed': work_path('model_build/mixed/artifacts/layer_diagnostics/report.json'),
                'native': work_path('model_build/mixed/artifacts/mixed_tf32_compat/tcn_d_quantized.json')}
    if args.kind in required and not required[args.kind].is_file():
        raise ConfigurationError('Stage input missing. ' + str(required[args.kind]) +
                                 '. See docs/MODEL_BUILD.md for the source stage and required receipts')
    module, options = modules[args.kind]
    return subprocess.call([sys.executable, '-B', '-m', module, *options], env=environment())


if __name__ == '__main__':
    raise SystemExit(main())
