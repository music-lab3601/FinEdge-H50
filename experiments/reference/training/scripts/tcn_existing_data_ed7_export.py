"""One finite ED7 CPU export or independent portable-package reload.

Import is standard-library/contract only. Admission and permanent reservation
precede numerical imports or payload reads. No optimizer, RNG restoration,
labels, raw E records, DEV cache or historical 17--19 stream is used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys

from experiments.reference.training.scripts import tcn_existing_data_ed7_contract as contract

ROOT, STAGE, PACKAGE = contract.ROOT, contract.STAGE, contract.PACKAGE
CP_VERSION = 'TCN-ED4-complete-epoch-checkpoint-1.0'


def operations():
    return dict.fromkeys(contract.OPERATION_KEYS, 0)


def provenance(started):
    return {'pid': os.getpid(), 'entry_path': str(Path(__file__).resolve()),
            'entry_sha256': contract.sha(__file__), 'argv': list(sys.argv), 'started_utc': started}


def numeric_runtime():
    import numpy as np
    import torch
    contract.need(str(np.__version__) == '2.2.6' and str(torch.__version__) == '2.7.1+cu128',
                  'Frozen numerical runtime differs')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return np, torch


def state_digest(model):
    return tensors_digest(model.state_dict())


def tensors_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_ready(model, torch):
    contract.need(model.training is False and all(parameter.device.type == 'cpu'
        and parameter.dtype == torch.float32 and not parameter.requires_grad and parameter.grad is None
        for parameter in model.parameters()), 'CPU eval/no-gradient model required')


def target(relative):
    contract.need(isinstance(relative, str) and '\\' not in relative and ':' not in relative,
                  'Package targets must use relative POSIX paths')
    part = PurePosixPath(relative)
    contract.need(not part.is_absolute() and '..' not in part.parts and part.as_posix() == relative,
                  'Unsafe package target')
    path = contract.safe(PACKAGE.joinpath(*part.parts))
    contract.need(path.is_relative_to(PACKAGE) and not path.exists(), 'Existing package target preserved')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_binary(path, values, np):
    path = contract.safe(path)
    contract.need(path.is_relative_to(STAGE), 'Binary output outside ED7')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(np.ascontiguousarray(values, dtype='<f4').tobytes())
        stream.flush(); os.fsync(stream.fileno())
    return contract.binding(path)


def cache_stat(path):
    path = contract.safe(path); info = path.stat()
    contract.need(path.is_file(), 'Registered cache is not a regular file')
    return {'path': str(path), 'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns,
            'inode': info.st_ino, 'device': info.st_dev}


def prepare_golden(reg, counters, np):
    """Exactly 32 seek/read windows; no whole source-cache payload hash or map."""
    plan = reg['window_plan']
    contract.need(len(plan) == 32 and len({(w['date'], w['endpoint']) for w in plan}) == 32,
                  'Exactly32 distinct registered TRAIN windows required')
    paths = {w['date']: contract.safe(w['source_cache']['path']) for w in plan}
    before = {day: cache_stat(path) for day, path in paths.items()}
    windows = []; entries = []
    for index, window in enumerate(plan):
        ref = window['source_cache']; begin = window['cache_row_start']; stop = window['cache_row_stop_exclusive']
        contract.need(window['date'] in contract.TRAIN and window['role'] == ref['role'] == 'train'
            and ref['dtype'] == '<f4' and ref['shape'][1] == 40 and before[window['date']]['bytes'] == ref['bytes']
            and ref['bytes'] == ref['shape'][0] * 160, 'Registered TRAIN cache stat/geometry differs')
        contract.need(type(begin) is int and type(stop) is int and stop - begin == 100
            and window['cache_range_start'] <= begin < stop <= window['cache_range_stop_exclusive']
            and begin == window['cache_range_start'] + window['endpoint'] - 99 - window['start_row']
            and window['endpoint_start'] <= window['endpoint'] < window['endpoint_stop_exclusive']
            and window['endpoint'] - 99 >= window['start_row'] and window['endpoint'] + 50 < window['stop_row'],
            'Window crosses its registered accepted range/footprint')
        with paths[window['date']].open('rb', buffering=0) as stream:
            stream.seek(begin * 160)
            counters['source_cache_window_reads'] += 1
            raw = stream.read(16000)
            counters['source_cache_payload_bytes_read'] += len(raw)
        contract.need(len(raw) == 16000, 'Incomplete registered cache window')
        values = np.frombuffer(raw, dtype='<f4').reshape(100, 40)
        inputs = np.ascontiguousarray(values.T, dtype='<f4')
        contract.need(np.isfinite(inputs).all(), 'Nonfinite normalized golden input')
        digest = hashlib.sha256(inputs.tobytes()).hexdigest()
        expected = window['expected_input_sha256']
        contract.need(expected is None or digest == expected, 'Original ED4 boundary input SHA differs')
        entries.append({'index': index, **window, 'input_sha256': digest,
                        'boundary_match': digest == expected if expected is not None else None})
        windows.append(inputs)
    contract.need(sum(w['expected_input_sha256'] is not None for w in plan) == 15,
                  'Fifteen original TRAIN boundary identities required')
    after = {day: cache_stat(path) for day, path in paths.items()}
    contract.need(before == after, 'Source cache stat identity changed while reading the32 windows')
    return np.ascontiguousarray(np.stack(windows), dtype='<f4'), entries, before


def load_native(reg, counters, torch):
    from software.model.tcn_research_p5_model import build_model
    path = contract.verify(reg['checkpoint_binding']); before = contract.sha(path)
    config = contract.read(contract.verify(reg['config_binding']))
    counters['checkpoint_deserializations'] += 1
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    contract.need(type(checkpoint) is dict and checkpoint.get('version') == CP_VERSION
        and checkpoint.get('experiment') == 'TCN_EDO_20261017_ED4_feature42'
        and checkpoint.get('epoch') == checkpoint.get('selected_epoch') == 3 and checkpoint.get('config') == config
        and config.get('slot_id') == 'feature42' and config.get('seed') == 42
        and config.get('weight_norm') is True and config.get('learning_rate') == .001
        and config.get('window_length') == 100 and config.get('horizon') == 50
        and config.get('decision') == 'original_argmax', 'Frozen ED4 complete-epoch identity differs')
    identity = checkpoint.get('identity')
    contract.need(isinstance(identity, dict)
        and contract.equivalent(identity['config_binding'], reg['config_binding'])
        and contract.equivalent(identity['feature_cache_binding'], reg['cache_manifest_binding'])
        and identity['numpy'] == '2.2.6' and identity['torch'] == '2.7.1+cu128', 'Frozen CP identity differs')
    state = checkpoint.get('model')
    contract.need(isinstance(state, dict) and state and all(isinstance(value, torch.Tensor)
        and value.device.type == 'cpu' and value.dtype == torch.float32 and torch.isfinite(value).all().item()
        for value in state.values()), 'Invalid frozen CPU model tensors')
    source_state_sha = tensors_digest(state)
    counters['eager_model_constructions'] += 1; native = build_model(weight_norm=True)
    contract.need(sum(p.numel() for p in native.parameters()) == 33539, 'WN architecture differs')
    counters['native_model_state_loads'] += 1; native.load_state_dict(state, strict=True)
    contract.need(all(torch.equal(state[name], tensor) for name, tensor in native.state_dict().items()),
                  'Native model-state copy differs')
    native.eval().requires_grad_(False); model_ready(native, torch)
    contract.need(state_digest(native) == source_state_sha, 'Loaded native state fingerprint differs')
    # Historical optimizer/RNG objects are neither inspected nor restored.
    del state, checkpoint
    return native, before, source_state_sha


def fold_model(native, counters, torch):
    from software.model.tcn_candidate import TemporalConvCandidate, CausalConv1d
    counters['eager_model_constructions'] += 1; effective = TemporalConvCandidate()
    native_state = native.state_dict(); modules = dict(native.named_modules()); folded = {}; folds = []
    for key, initial in effective.state_dict().items():
        if key.endswith('.conv.weight'):
            module = modules[key[:-len('.weight')]]
            contract.need(hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'),
                          'Expected WN temporal convolution')
            counters['weight_norm_folds'] += 1
            value = module.weight.detach().cpu().clone()
            contract.need(value.shape == initial.shape and torch.isfinite(value).all().item(), 'Invalid effective WN weight')
            folded[key] = value
            folds.append({'state_key': key, 'shape': list(value.shape),
                          'effective_weight_sha256': hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()})
        else:
            contract.need(key in native_state and native_state[key].shape == initial.shape, 'Unexpected non-temporal state')
            folded[key] = native_state[key].detach().cpu().clone()
    contract.need(len(folds) == sum(isinstance(m, CausalConv1d) for m in effective.modules()) == 10,
                  'Exactly ten WN temporal folds required')
    counters['folded_model_state_loads'] += 1; effective.load_state_dict(folded, strict=True)
    effective.eval().requires_grad_(False); model_ready(effective, torch)
    contract.need(sum(p.numel() for p in effective.parameters()) == 33219
        and all(torch.equal(value, folded[key]) for key, value in effective.state_dict().items()),
        'Effective plain architecture/state differs')
    return effective, folds


def checked_eager_forward(model, values, counters, torch, np):
    counters['forward_attempts'] += 1
    with torch.inference_mode():
        result = model(torch.from_numpy(np.ascontiguousarray(values)))
    counters['forward_calls'] += 1; counters['forward_samples'] += len(values)
    contract.need(result.device.type == 'cpu' and result.dtype == torch.float32
        and tuple(result.shape) == (len(values), 3) and torch.isfinite(result).all().item(), 'Invalid eager logits')
    return np.array(result.detach().cpu().numpy(), dtype='<f4', copy=True)


def checked_script_forward(runner, values, counters):
    before = (runner.forward_attempts, runner.forward_calls, runner.forward_samples)
    try:
        return runner.run(values)
    finally:
        for key, old, new in zip(('forward_attempts', 'forward_calls', 'forward_samples'), before,
                                (runner.forward_attempts, runner.forward_calls, runner.forward_samples)):
            counters[key] += new - old


def compare(reference, observed, tolerance, np):
    contract.need(reference.dtype == observed.dtype == np.dtype('<f4') and reference.shape == observed.shape
        and reference.ndim == 2 and reference.shape[1] == 3 and np.isfinite(reference).all()
        and np.isfinite(observed).all(), 'Invalid comparison logits')
    expected = reference.astype(np.float64); actual = observed.astype(np.float64)
    difference = np.abs(expected - actual)
    return {'passed': bool(np.all(difference <= tolerance['atol'] + tolerance['rtol'] * np.abs(expected))),
            'max_absolute_difference': float(np.max(difference)),
            'class_agreement': float(np.mean(reference.argmax(1) == observed.argmax(1)))}


def copy_package_assets(reg):
    receipts = []; names = set()
    for item in reg['package_copies']:
        contract.need(item['target'] not in names, 'Duplicate package copy target')
        source = contract.verify(item['source']); destination = target(item['target'])
        with source.open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024); dst.flush(); os.fsync(dst.fileno())
        copied = contract.binding(destination)
        contract.need(copied['sha256'] == item['source']['sha256'] and copied['bytes'] == item['source']['bytes'],
                      'Opaque package copy identity differs')
        receipts.append({'source': item['source'], 'target': copied, 'relative': item['target']}); names.add(item['target'])
    contract.need({'runtime.py', 'normalization.json', 'native/best.pt'} <= names, 'Required portable copies absent')
    copied = {item['relative']: item for item in receipts}
    contract.need(copied['native/best.pt']['source']['sha256'] == reg['checkpoint_binding']['sha256']
        and copied['normalization.json']['source']['sha256'] == reg['normalization_binding']['sha256']
        and copied['runtime.py']['source']['sha256'] == contract.sha(Path(__file__).with_name('tcn_existing_data_ed7_runtime.py')),
        'Portable runtime/native checkpoint/normalization copy differs')
    return receipts


def common_acceptance(phase, reg, reservation, counters, started, process, np, torch):
    return {'version': 'TCN-ED7-' + phase + '-acceptance-1.0', 'status': 'completed', 'passed': True,
        'phase': phase, 'worker_pid': os.getpid(), 'started_utc': started, 'completed_utc': contract.utc(),
        'process_provenance': process, 'reservation': reservation, 'registration_binding': reg['_registration_binding'],
        'operations': counters, 'tolerance': reg['tolerance'], 'training_started': False,
        'new_independent_test': False, 'labels_read': False, 'E_records_read': False,
        'DEV_cache_read': False, 'historical17_19_stream_read': False, 'normalization_refitted': False,
        'model_quality_changed': False, 'source_control_verified': False, 'coverage_unknown': True,
        'real_raw_input_replay': False, 'real_golden_path': 'normalized_float32_only',
        'raw_preprocessing_evidence': 'Pure synthetic boundary tests only; no real raw payload or additional model forward',
        'original_flag64_cleared': False, 'NPU_verified': False, 'deployment_started': False,
        'runtime': {'numpy': str(np.__version__), 'torch': str(torch.__version__), 'device': 'cpu',
                    'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                    'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}}


def run_primary(reg, reservation, counters, started, process):
    np, torch = numeric_runtime()
    from experiments.reference.training.scripts import tcn_existing_data_ed7_runtime as portable
    PACKAGE.mkdir(exist_ok=False)
    copies = copy_package_assets(reg)
    portable.validate_normalization(contract.read(PACKAGE / 'normalization.json'))
    inputs, entries, stat_before = prepare_golden(reg, counters, np)
    inputs_path = PACKAGE / 'golden/inputs.f32'; write_binary(inputs_path, inputs, np)
    contract.write_new(PACKAGE / 'golden/selection.json', {'version': 'TCN-ED7-golden-selection-1.0',
        'selection': '15 fixed ED4 TRAIN boundary windows plus17 consecutive predefined12-08 windows',
        'windows': 32, 'per_input': entries, 'input_sha256': contract.sha(inputs_path),
        'source_payload_identity_scope': 'Only32 local normalized windows and15 original boundary SHAs; no whole source cache payload hash'})
    native, checkpoint_before, source_state_sha = load_native(reg, counters, torch)
    native_before = state_digest(native)
    effective, folds = fold_model(native, counters, torch); effective_before = state_digest(effective)
    counters['script_compilations'] += 1; scripted = torch.jit.script(effective)
    scripted.eval().requires_grad_(False); model_ready(scripted, torch); script_before = state_digest(scripted)
    contract.need(script_before == effective_before, 'Compiled script state differs from effective eager state')
    script_path = target('model_cpu.ts')
    counters['torchscript_saves'] += 1; torch.jit.save(scripted, str(script_path)); script_sha = contract.sha(script_path)
    contract.need(contract.sha(script_path) == script_sha, 'Registered direct script-load artifact changed')
    counters['torchscript_loads'] += 1; runner = portable._new_runner(script_path)
    loaded_before = state_digest(runner.model)
    contract.need(loaded_before == script_before, 'Sole primary TorchScript load state differs')
    outputs = {}; checks = {}
    for batch in (32, 1):
        values = portable.validate_normalized(inputs[:batch])
        native_logits = checked_eager_forward(native, values, counters, torch, np)
        effective_logits = checked_eager_forward(effective, values, counters, torch, np)
        script_logits = checked_script_forward(runner, values, counters)
        outputs[batch] = {'native': native_logits, 'effective': effective_logits, 'script': script_logits}
        checks['batch' + str(batch)] = {
            'native_effective': compare(native_logits, effective_logits, reg['tolerance'], np),
            'native_script': compare(native_logits, script_logits, reg['tolerance'], np),
            'effective_script': compare(effective_logits, script_logits, reg['tolerance'], np)}
        contract.need(all(check['passed'] and check['class_agreement'] == 1.0
                          for check in checks['batch' + str(batch)].values()), 'Three-way output equivalence failed')
        for kind, logits in outputs[batch].items():
            write_binary(PACKAGE / f'golden/{kind}_batch{batch}.f32', logits, np)
    native_after, effective_after = state_digest(native), state_digest(effective)
    script_after, loaded_after = state_digest(scripted), state_digest(runner.model)
    contract.need(source_state_sha == native_before == native_after
        and effective_before == effective_after == script_before == script_after == loaded_before == loaded_after,
        'Model state changed during finite verification')
    for model in (native, effective, scripted, runner.model):
        model_ready(model, torch)
    stat_after = {day: cache_stat(path['path']) for day, path in stat_before.items()}
    contract.need(stat_after == stat_before and contract.sha(script_path) == script_sha
        and contract.sha(reg['checkpoint_binding']['path']) == checkpoint_before,
        'Source cache stat/checkpoint/script identity changed')
    contract.need(counters == reg['primary_budget'] == contract.PRIMARY_BUDGET, 'Primary exact finite operations differ')
    manifest = {'version': 'TCN-ED7-package-manifest-1.0', 'created_utc': contract.utc(),
        'selected_family': reg['selected_family'], 'epoch': 3, 'seed': 42, 'model_format': 'TorchScript',
        'execution_device': 'cpu', 'semantic_contract': portable.SEMANTIC_CONTRACT,
        'native_parameter_count': 33539, 'folded_parameter_count': 33219,
        'checkpoint_sha256': reg['checkpoint_binding']['sha256'],
        'normalization_sha256': reg['normalization_binding']['sha256'],
        'training_started': False, 'new_independent_test': False, 'historical_final_quality_changed': False,
        'files': {path.relative_to(PACKAGE).as_posix(): {'sha256': contract.sha(path), 'bytes': path.stat().st_size}
                  for path in sorted(PACKAGE.rglob('*')) if path.is_file()}}
    contract.write_new(PACKAGE / 'package_manifest.json', manifest)
    manifest_binding = contract.binding(PACKAGE / 'package_manifest.json')
    portable.validate_package(PACKAGE, manifest_binding['sha256'])  # opaque hashes/metadata; no model/load/forward
    value = common_acceptance('primary', reg, reservation, counters, started, process, np, torch)
    value.update(package_manifest_binding=manifest_binding, package_copy_receipts=copies,
        comparisons=checks,
        golden={'windows': 32, 'input_sha256': contract.sha(inputs_path), 'per_input': entries, **checks},
        folded_weights=folds, source_checkpoint_model_state_sha256=source_state_sha,
        native_state_sha256_before=native_before, native_state_sha256_after=native_after,
        effective_state_sha256_before=effective_before, effective_state_sha256_after=effective_after,
        script_state_sha256_before=script_before, script_state_sha256_after=script_after,
        loaded_script_state_sha256_before=loaded_before, loaded_script_state_sha256_after=loaded_after,
        parameter_counts={'native': 33539, 'effective': 33219, 'script': 33219},
        state_tensor_counts={'native': len(native.state_dict()), 'effective': len(effective.state_dict()),
            'script': len(scripted.state_dict()), 'loaded_script': len(runner.model.state_dict())},
        primary_script_sha256=script_sha,
        checkpoint_sha256_before=checkpoint_before, checkpoint_sha256_after=contract.sha(reg['checkpoint_binding']['path']),
        source_cache_stat_before=stat_before, source_cache_stat_after=stat_after,
        source_cache_access={'windows': 32, 'bytes': 512000, 'complete_files_hashed': 0,
            'identity_scope': '32 unbuffered application-level local normalized window reads and15 exact original boundary SHAs; stat identities before/after only for complete5TRAIN cache files; no physical disk/page-cache byte-count claim'},
        all_eval_CPU_no_gradients=True,
        primary_script_entry='Direct sole TorchScript runner under registered script SHA/semantic checks; complete package Predictor is exercised only by reload')
    contract.write_new(STAGE / 'primary_acceptance.json', value)
    return value


def load_portable_module():
    spec = importlib.util.spec_from_file_location('ed7_portable_package_runtime', PACKAGE / 'runtime.py')
    contract.need(spec is not None and spec.loader is not None, 'Portable runtime loader unavailable')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def run_reload(reg, reservation, counters, started, process):
    np, torch = numeric_runtime(); primary = contract.read(STAGE / 'primary_acceptance.json')
    manifest = primary['package_manifest_binding']; contract.verify(manifest)
    portable = load_portable_module()
    counters['torchscript_loads'] += 1
    predictor = portable.Predictor(PACKAGE, manifest['sha256']); runner = predictor._runner
    before = state_digest(runner.model)
    raw = (PACKAGE / 'golden/inputs.f32').read_bytes()
    contract.need(len(raw) == 512000, 'Golden byte count differs')
    inputs = np.frombuffer(raw, dtype='<f4').reshape(32, 40, 100).copy()
    reference = np.frombuffer((PACKAGE / 'golden/native_batch32.f32').read_bytes(), dtype='<f4').reshape(32, 3)
    previous = np.frombuffer((PACKAGE / 'golden/script_batch32.f32').read_bytes(), dtype='<f4').reshape(32, 3)
    # Predictor verifies the complete portable package and uses this exact input validator.
    observed_before = (runner.forward_attempts, runner.forward_calls, runner.forward_samples)
    try:
        output = predictor.predict_normalized(inputs)['logits']
    finally:
        for key, old, new in zip(('forward_attempts', 'forward_calls', 'forward_samples'), observed_before,
                                (runner.forward_attempts, runner.forward_calls, runner.forward_samples)):
            counters[key] += new - old
    native_reloaded = compare(reference, output, reg['tolerance'], np)
    previous_reloaded = compare(previous, output, reg['tolerance'], np)
    contract.need(all(check['passed'] and check['class_agreement'] == 1.0
        for check in (native_reloaded, previous_reloaded)), 'Independent portable reload logits differ')
    after = state_digest(runner.model); model_ready(runner.model, torch)
    contract.need(before == after == primary['loaded_script_state_sha256_after'], 'Independent script state differs')
    contract.need(counters == reg['reload_budget'] == contract.RELOAD_BUDGET, 'Reload exact finite operations differ')
    portable.validate_package(PACKAGE, manifest['sha256'])  # post-forward identity check, no additional load
    output_ref = write_binary(STAGE / 'reload_logits.f32', output, np)
    value = common_acceptance('reload', reg, reservation, counters, started, process, np, torch)
    value.update(primary_acceptance_binding=contract.binding(STAGE / 'primary_acceptance.json'),
        package_manifest_binding=manifest, golden_windows=32,
        native_reloaded=native_reloaded, previous_script_reloaded=previous_reloaded,
        comparisons={'native_reloaded': native_reloaded, 'previous_script_reloaded': previous_reloaded},
        loaded_script_state_sha256_before=before, loaded_script_state_sha256_after=after,
        output_binding=output_ref, all_eval_CPU_no_gradients=True,
        portable_module_source_binding=contract.binding(PACKAGE / 'runtime.py'))
    contract.write_new(STAGE / 'reload_acceptance.json', value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('primary', 'reload'))
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args(argv)
    started = contract.utc(); process = provenance(started)
    reg = contract.admit(args.registration_sha256, args.phase)
    reservation = contract.reserve(reg, args.phase)
    counters = operations()
    try:
        value = (run_primary if args.phase == 'primary' else run_reload)(reg, reservation, counters, started, process)
    except BaseException as error:
        contract.write_new(STAGE / (args.phase + '_failure.json'), {
            'version': 'TCN-ED7-phase-failure-1.0', 'phase': args.phase, 'worker_pid': os.getpid(),
            'failed_utc': contract.utc(), 'process_provenance': process, 'reservation': reservation,
            'operations_observed': counters, 'error_type': type(error).__name__, 'error': str(error),
            'reservation_remains_consumed': True, 'automatic_retry': False})
        raise
    print(json.dumps({'phase': args.phase, 'passed': value['passed'], 'worker_pid': os.getpid(),
                      'operations': value['operations'], 'completed_utc': value['completed_utc']}, ensure_ascii=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
