"""Exercise the public frozen model using the included synthetic fixtures."""
from pathlib import Path
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    package = ROOT / 'models/cpu'
    identity = json.loads((ROOT / 'models/identity.json').read_text(encoding='utf-8'))
    spec = importlib.util.spec_from_file_location('public_cpu_runtime', package / 'runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    predictor = runtime.Predictor(package, identity['CPU_manifest_sha256'])
    inputs = np.fromfile(package / 'golden/inputs.f32', dtype='<f4').reshape(32, 40, 100)
    expected = np.fromfile(package / 'golden/script_batch32.f32', dtype='<f4').reshape(32, 3)
    result = predictor.predict_normalized(inputs)
    error = float(np.abs(result['logits'] - expected).max())
    passed = np.allclose(result['logits'], expected, rtol=1e-4, atol=1e-5)
    passed = passed and np.array_equal(result['class_indices'], expected.argmax(axis=1))
    print(json.dumps(dict(status='passed' if passed else 'failed', synthetic_windows=32,
                          max_abs_error=error, model_parameters=33219, board_access=False)))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
