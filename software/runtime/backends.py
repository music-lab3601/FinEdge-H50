from shared.paths import resolve_path
import json
import os
from pathlib import Path
import queue
import struct
import subprocess
import threading
from time import perf_counter_ns

import numpy as np

from software.runtime.delivery import runtime_module
from shared.model_paths import ARTIFACTS, CLASS_NAMES, MANIFEST_SHA, PACKAGE, ROOT


def validate_tensor(tensor):
    if not isinstance(tensor, np.ndarray) or tensor.dtype != np.dtype('<f4') or tensor.shape != (1, 40, 100) or not np.isfinite(tensor).all():
        raise ValueError('Expected finite float32[1,40,100]')
    return np.ascontiguousarray(tensor)


def prediction(window_id, logits, backend, elapsed_ns, **extra):
    values = np.asarray(logits, dtype='<f4').reshape(-1)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError('Invalid three-class logits')
    index = int(values.argmax())
    return {'window_id': int(window_id), 'logits': values.tolist(), 'class_index': index,
            'class_name': CLASS_NAMES[index], 'backend': backend, 'elapsed_ns': int(elapsed_ns), **extra}


class CpuBackend:
    name = 'torch_cpu'

    def __init__(self):
        import torch
        torch.set_num_threads(1)
        self.runtime = runtime_module()
        self.predictor = self.runtime.Predictor(PACKAGE, MANIFEST_SHA)

    def predict(self, window_id, tensor):
        tensor = validate_tensor(tensor)
        start = perf_counter_ns()
        output = self.predictor.predict_normalized(tensor)['logits'][0]
        return prediction(window_id, output, self.name, perf_counter_ns() - start)

    def close(self):
        pass


class OnnxBackend:
    name = 'onnx_cpu'

    def __init__(self, path=None):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path or ARTIFACTS / 'export' / 'tcn_ed4.onnx'),
                                           sess_options=options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, window_id, tensor):
        tensor = validate_tensor(tensor)
        start = perf_counter_ns()
        values = self.session.run(None, {self.input_name: tensor})[0][0]
        return prediction(window_id, values, self.name, perf_counter_ns() - start)

    def close(self):
        pass


class WorkerBackend:
    """One persistent C++ Session. No shell invocation and no CPU fallback."""
    def __init__(self, mode, url='socket://zg330aiu@192.168.126.49:9981', timeout=15,
                 executable=None, network=None, log_path=None):
        if mode not in ('host', 'socket', 'local'):
            raise ValueError('Worker mode must be host, socket or board local')
        if mode == 'local' and os.name == 'nt':
            raise ValueError('Board local mode requires ARM Linux')
        self.name = 'icraft_' + mode
        self.timeout = timeout
        self.messages = queue.Queue()
        self.closed = False
        self.calls = 0
        executable = Path(executable or ROOT / 'build' / 'Release' / 'finedge_worker.exe')
        if not executable.is_file():
            raise FileNotFoundError('C++ worker not built; run scripts/build_worker.ps1')
        network = Path(network or ARTIFACTS / 'icraft' / 'tcn_ed4_ZG.json')
        args = [str(executable), '--mode', mode, '--json', str(network), '--raw', str(network.with_suffix('.raw')),
                '--url', url]
        env = os.environ.copy()
        if os.name == 'nt':
            env['PATH'] = str(Path(str(resolve_path('tool://icraft/bin')))) + os.pathsep + env.get('PATH', '')
        env['OMP_NUM_THREADS'] = '1'
        log_path = Path(log_path or ARTIFACTS / 'worker_logs' / (self.name + '.log'))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr = log_path.open('ab', buffering=0)
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, env=env, creationflags=flags)

        def read_stdout():
            try:
                for line in iter(self.process.stdout.readline, b''):
                    try:
                        item = json.loads(line.decode('utf-8'))
                        if isinstance(item, dict) and 'type' in item:
                            self.messages.put(item)
                        else:
                            self.stderr.write(line)
                    except (ValueError, UnicodeError):
                        self.stderr.write(line)
            finally:
                self.messages.put({'type': 'eof'})
        self.reader = threading.Thread(target=read_stdout, daemon=True)
        self.reader.start()
        try:
            self.info = self._receive()
            if self.info.get('type') != 'ready' or self.info.get('backend') != self.name:
                raise RuntimeError(f'Worker initialization failed: {self.info}')
            if mode != 'host' and not self.info.get('npu_bound', False):
                raise RuntimeError('No NPU hard operation bound; refusing NPU claim')
        except BaseException:
            self.close()
            raise

    def _receive(self):
        try:
            value = self.messages.get(timeout=self.timeout)
        except queue.Empty as e:
            self.close()
            raise TimeoutError(f'{self.name} timed out; its own worker was terminated') from e
        if value.get('type') in ('error', 'eof'):
            raise RuntimeError(f'{self.name}: {value}')
        return value

    def predict(self, window_id, tensor):
        values = validate_tensor(tensor)
        if self.closed:
            raise RuntimeError('Worker is closed')
        start = perf_counter_ns()
        try:
            self.process.stdin.write(struct.pack('<Q', window_id) + values.tobytes(order='C'))
            self.process.stdin.flush()
            result = self._receive()
            if result.get('type') != 'prediction' or result.get('window_id') != window_id or result.get('backend') != self.name:
                raise RuntimeError('Worker response identity mismatch')
            self.calls += 1
            return prediction(window_id, result['logits'], self.name, perf_counter_ns() - start,
                              device_call_ns=result['device_call_ns'],
                              stages={key: result[key] for key in ('input_cast_ns', 'forward_submit_ns',
                                      'completion_wait_ns', 'output_read_ns') if key in result})
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)
        if threading.current_thread() is not self.reader:
            self.reader.join(timeout=2)
        self.stderr.close()


def warmup(backend, first_tensor, count=10):
    start = perf_counter_ns()
    for _ in range(count):
        backend.predict(0, first_tensor)
    return {'count': count, 'elapsed_ns': perf_counter_ns() - start}

