from shared.paths import resolve_path
from datetime import datetime, timezone
import json
import platform
import subprocess

import numpy as np

from software.runtime.delivery import read_json, runtime_module, sha256, write_json
from shared.model_paths import ARTIFACTS, MANIFEST_SHA, PACKAGE, RAW, ROOT
from software.runtime.stream import InputBuilder, RawSource


def export_onnx(raw_reference=False):
    import onnx
    import onnxruntime as ort
    import torch
    runtime_module()
    torch.set_num_threads(1)
    output = ARTIFACTS / 'export'
    output.mkdir(parents=True, exist_ok=True)
    model = torch.jit.load(str(PACKAGE / 'model_cpu.ts'), map_location='cpu').eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    golden = np.fromfile(PACKAGE / 'golden' / 'inputs.f32', dtype='<f4').reshape(32, 40, 100)
    dummy = torch.from_numpy(golden[:1].copy())
    target = output / 'tcn_ed4.onnx'
    torch.onnx.export(model, (dummy,), str(target), export_params=True, opset_version=11,
                      input_names=['lob'], output_names=['logits'], do_constant_folding=True,
                      dynamo=False)
    onnx.checker.check_model(onnx.load(str(target)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(target), options, providers=['CPUExecutionProvider'])
    all_inputs = [row[None].copy() for row in golden]
    if raw_reference:
        source = RawSource()
        builder = InputBuilder(read_json(PACKAGE.parent / 'normalization.json'))
        for record in source:
            window = builder.push(record)
            if window is not None:
                all_inputs.append(window.tensor)
    max_difference = 0.0
    mismatches = []
    for i, x in enumerate(all_inputs):
        with torch.inference_mode():
            expected = model(torch.from_numpy(x)).numpy()
        actual = session.run(None, {'lob': x})[0]
        difference = float(np.max(np.abs(actual - expected)))
        max_difference = max(max_difference, difference)
        if not np.allclose(actual, expected, rtol=1e-4, atol=1e-5) or not np.array_equal(actual.argmax(1), expected.argmax(1)):
            mismatches.append({'input_index': i, 'max_abs_difference': difference,
                               'cpu': expected.tolist(), 'onnx': actual.tolist()})
    result = {'status': 'passed' if not mismatches else 'failed', 'opset': 11,
              'input_shape': [1, 40, 100], 'output_shape': [1, 3],
              'golden_windows': 32, 'raw_windows': len(all_inputs) - 32,
              'max_abs_difference': max_difference, 'rtol': 1e-4, 'atol': 1e-5,
              'mismatches': mismatches, 'model_manifest_sha256': MANIFEST_SHA,
              'onnx_sha256': sha256(target), 'torch': torch.__version__, 'onnx': onnx.__version__,
              'onnxruntime': ort.__version__, 'python': platform.python_version(),
              'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(output / 'verification.json', result)
    if mismatches:
        raise RuntimeError('ONNX equivalence failed; see export/verification.json')
    return result


def compile_icraft():
    verified = read_json(ARTIFACTS / 'export' / 'verification.json')
    model = ARTIFACTS / 'export' / 'tcn_ed4.onnx'
    if verified['status'] != 'passed' or verified['onnx_sha256'] != sha256(model):
        raise ValueError('ONNX has not passed verification')
    output = ARTIFACTS / 'icraft'
    output.mkdir(parents=True, exist_ok=True)
    f = lambda p: str(p).replace('\\', '/')
    config = f'''[parse]
net_name = "tcn_ed4"
framework = "onnx"
target = "zhuge"
network = "{f(model)}"
jr_path = "{f(output)}"
inputs = "1,40,100"
inputs_layout = "FD"
inputs_dtype = "FP32"
pre_method = "nop"
pre_mean = "nop"
pre_scale = "nop"
channel_swap = "nop"

[optimize]
target = "zhuge"
json = "{f(output / 'tcn_ed4_parsed.json')}"
raw = "{f(output / 'tcn_ed4_parsed.raw')}"
jr_path = "{f(output)}"

[quantize]
target = "zhuge"
qdtype = "tf32"
no_transinput = true
json = "{f(output / 'tcn_ed4_optimized.json')}"
raw = "{f(output / 'tcn_ed4_optimized.raw')}"
jr_path = "{f(output)}"

[adapt]
target = "zhuge"
json = "{f(output / 'tcn_ed4_quantized.json')}"
raw = "{f(output / 'tcn_ed4_quantized.raw')}"
jr_path = "{f(output)}"

[generate]
json = "{f(output / 'tcn_ed4_adapted.json')}"
raw = "{f(output / 'tcn_ed4_adapted.raw')}"
jr_path = "{f(output)}"
'''
    config_file = output / 'tcn_ed4_tf32.toml'
    config_file.parent.mkdir(parents=True,exist_ok=True)
    config_file.write_text(config, encoding='utf-8')
    commands = []
    for stage in ('parse', 'optimize', 'quantize', 'adapt', 'generate'):
        with (output / (stage + '.log')).open('wb') as log:
            result = subprocess.run([str(resolve_path('tool://icraft/bin/icraft.exe')), stage, str(config_file)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        commands.append({'stage': stage, 'exit_code': result.returncode})
        write_json(output / 'compilation.json', {'status': 'running', 'stages': commands})
        if result.returncode:
            write_json(output / 'compilation.json', {'status': 'failed', 'stages': commands,
                       'failed_stage': stage, 'log': str(output / (stage + '.log'))})
            raise RuntimeError(f'Icraft {stage} failed; see {output / (stage + ".log")}')
    generated = output / 'tcn_ed4_ZG.json'
    if not generated.is_file() or not generated.with_suffix('.raw').is_file():
        raise RuntimeError('Compiler returned without the expected generated JSON/RAW')
    report = {'status': 'passed', 'stages': commands, 'precision': 'tf32',
              'generated_json_sha256': sha256(generated),
              'generated_raw_sha256': sha256(generated.with_suffix('.raw')),
              'onnx_sha256': verified['onnx_sha256'], 'npu_execution_verified': False}
    write_json(output / 'compilation.json', report)
    return report
