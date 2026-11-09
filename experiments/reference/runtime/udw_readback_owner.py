"""Reuse protected ARM readback; no second explicit device Tensor.to request.

The buffer constructor is documented to COPY into Host memory. It is not a
zero-copy alias. The comparison owner and bounded RAM trace are independent
of the accepted CP01 owner and of the default service.
"""
from dataclasses import asdict
from datetime import timedelta
import importlib
import io
from pathlib import Path
import time

import numpy as np

from software.runtime.delivery import read_json
from experiments.reference.runtime.udw_copy_probe_owner import CopyProbeBridge, ProbeTrace, verify_sdk_binding
from experiments.reference.runtime.udw_shared_probe_owner import ProbeBridge
from experiments.reference.runtime.udw_probe_stop import ProbeCancelled
from experiments.reference.runtime.udw_lab import digest


class BufferedTrace(ProbeTrace):
    """Bounded RAM diagnostics; file output is an explicit outside-call action."""

    def __init__(self, stop, max_events=8192):
        if type(max_events) is not int or max_events < 2:
            raise ValueError('Finite trace capacity required')
        super().__init__(stop, lambda *args: None)
        self.max_events, self.depth = max_events, 0

    def event(self, name, value):
        if len(self.events) >= self.max_events:
            self.stop.request(None)
            self.stop.isolate('RAM diagnostic capacity exhausted')
            raise ProbeCancelled('RAM diagnostic capacity exhausted; no overwrite or retry')
        super().event(name, value)

    def call(self, stage, operation, retain=None):
        self.depth += 1
        try:
            return super().call(stage, operation, retain)
        finally:
            self.depth -= 1

    def flush(self, writer):
        if self.depth:
            raise RuntimeError('No diagnostic file output inside a guarded call')
        # Do not clear the in-memory evidence even when the writer fails.
        writer(self.snapshot())

    def elapsed(self, stage):
        for row in reversed(self.events):
            if row['event'] == 'call_end' and row['value']['stage'] == stage:
                if row['value']['outcome'] != 'returned':
                    raise ValueError('Failed call is not a performance sample')
                return row['value']['operation_elapsed_ns']
        raise ValueError('Call did not execute: ' + stage)


def verify_readback_interface(path):
    evidence = verify_sdk_binding(path)
    constructor = importlib.import_module('xrt').Tensor.__init__
    doc = constructor.__doc__ or ''
    if not callable(constructor) or 'buffer' not in doc or 'xir.TensorType' not in doc:
        raise ValueError('Reviewed Tensor(buffer, TensorType) overload is unavailable')
    return dict(evidence, buffer_constructor_doc=doc,
                buffer_constructor_semantics='copies the supplied buffer into a new Host Tensor',
                actual_ARM_buffer_constructor_verified=False)


class ReadbackCompareBridge(CopyProbeBridge):
    MODES = ('device_copy', 'host_readback')

    def __init__(self, stop, mode='host_readback', max_events=8192):
        if mode not in self.MODES:
            raise ValueError('One explicit input mode required')
        self._input_mode = mode
        super().__init__(stop, lambda *args: None)
        self.trace = BufferedTrace(stop, max_events)

    @property
    def input_mode(self):
        return self._input_mode

    def initialize(self, model, expected_boot, run_id, control='axi_registers'):
        import sys
        sys.path.insert(0, '/usr/lib/aarch64-linux-gnu')
        interface = self.sdk('init.readback_interface', lambda: verify_readback_interface(
            Path(model).parents[2]/'probe/sdk_interface.json'))
        self.trace.event('readback_interface', interface)
        contract = read_json(Path(model).parents[2]/'probe/cast_contract.json')
        self.expected_cast_dtype = contract['returned_dtype']
        info = ProbeBridge.initialize(self, model, expected_boot, run_id, control)
        if info['input_type'] != contract['model_input_dtype']:
            raise ValueError('Compiled input differs from frozen Cast contract')
        info.update(mode='protected_ARM_readback_candidate', input_mode=self.input_mode,
                    trace_policy='bounded_RAM_then_flush_outside_calls', performance_sample=False)
        return info

    def _predict_retained(self, tensor):
        if type(tensor) is not bytes or len(tensor) != 16000 or not np.isfinite(np.frombuffer(tensor, dtype='<f4')).all():
            raise ValueError('Immutable finite 16000-byte protected readback required')
        if self.input_mode != self.info.get('input_mode'):
            raise ValueError('Input mode changed during a session')
        if self.pending is None or self.pending_index >= len(self.pending):
            raise ValueError('Protected window required; no software fallback')
        begin = time.perf_counter_ns()
        key = self.pending[self.pending_index]
        self.pending_index += 1
        self.trace.context.update(lease=asdict(key), input_mode=self.input_mode)
        self.ledger.observe_protected_readback(key, tensor)
        device_region = self.sdk('input.device_region', self.device.defaultMemRegion)
        host_region = self.sdk('input.host_region', self.xrt.HostDevice.MemRegion)
        source = self.ledger.bind(key, tensor_factory=lambda *a: self.sdk('input.bind', lambda: self.xrt.Tensor(*a)),
            dtype=self.software_type, chunk=self.chunk, region=device_region, allow_experimental_readback=True)
        source_meta = self.tensor_metadata(source, 'source', device_region)
        self.validate_tensor(source_meta, self.software_type, ('ADDR',))
        if self.sdk('source.on_host', lambda: source.isOn(host_region)):
            raise ValueError('Source must identify the protected device slot')
        transfer_begin = time.perf_counter_ns()
        if self.input_mode == 'device_copy':
            api = 'input.to_Host'
            host_tensor = self.sdk(api, lambda: source.to(host_region), self.ledger.handles.append)
        else:
            # Keep the original immutable bytes and its view even if the SDK
            # constructor fails or returns immediately before cancellation.
            self.stop_control.hold(tensor)
            view = np.frombuffer(tensor, dtype='<f4').reshape(1, 40, 100)
            self.stop_control.hold(view)
            api = 'input.from_readback_buffer'
            host_tensor = self.sdk(api, lambda: self.xrt.Tensor(view, self.software_type), self.ledger.handles.append)
        transfer_end = time.perf_counter_ns()
        self.trace.event('input_transfer', dict(mode=self.input_mode, input_sha256=digest(tensor),
            logical_bytes=16000, explicit_device_Tensor_to_calls=int(self.input_mode == 'device_copy'),
            explicit_device_Tensor_to_logical_bytes=16000 if self.input_mode == 'device_copy' else 0,
            documented_Host_buffer_copy_bytes=16000 if self.input_mode == 'host_readback' else 0,
            all_protection_reads_retained=True, SDK_internal_transfer_bytes=None, zero_copy=False))
        host_meta = self.tensor_metadata(host_tensor, 'host_input', host_region)
        self.validate_tensor(host_meta, self.software_type, ('CPTR', 'BOTH'))
        if self.sdk('host_input.on_device', lambda: host_tensor.isOn(device_region)):
            raise ValueError('Host input still on the device region')
        copied = np.empty(16000, dtype=np.uint8)
        self.sdk('host_input.read', lambda: host_tensor.read(copied, 0, 16000))
        actual = copied.tobytes()
        self.trace.event('Host_bytes', dict(input_sha256=digest(tensor), Host_sha256=digest(actual),
            all_bytes_equal=actual == tensor, logical_bytes=16000, float32_values=4000))
        if actual != tensor:
            raise ValueError('Host input differs from protected ARM readback')
        ids = []
        adapted = self.sdk('input.Cast', lambda: self.host.Cast(host_tensor, self.software_type, self.input_type, self.converter),
            lambda value: ids.append(self.ledger.retain_cast(value, source='shared', key=key)))
        adapted_meta = self.tensor_metadata(adapted, 'adapted', host_region)
        self.validate_tensor(adapted_meta, self.expected_cast_dtype, ('CPTR', 'BOTH'))
        self.adapted_info = dict(chunk_bytes=adapted_meta['chunk_bytes'], offset=adapted_meta['offset'], dtype=adapted_meta['complete_dtype'])
        call_id = ids[0]
        self.stop_control.checkpoint('NPU.before_mark')
        self.ledger.npu_may_start(call_id)
        outputs = self.sdk('NPU.forward', lambda: self.session.forward([adapted]), self.ledger.forward_outputs.append)
        if len(outputs) != 1:
            raise ValueError('Wrong NPU output count')
        self.sdk('NPU.output_not_ready', lambda: outputs[0].setReady(False))
        if not self.sdk('NPU.output_wait', lambda: outputs[0].waitForReady(timedelta(seconds=10), timedelta(milliseconds=1))):
            raise TimeoutError('NPU output timeout')
        stream = io.BytesIO()
        self.sdk('NPU.output_dump', lambda: outputs[0].dump(stream, 'SFB'))
        logits = np.frombuffer(stream.getvalue(), dtype='<f4')
        end = time.perf_counter_ns()
        if logits.shape != (3,) or not np.isfinite(logits).all():
            raise ValueError('Invalid logits')
        self.ledger.npu_completed(call_id)
        self.call_count += 1
        self.stop_control.checkpoint('NPU.completed')
        elapsed = {name: self.trace.elapsed(name) for name in
            (api, 'input.Cast', 'NPU.forward', 'NPU.output_wait', 'NPU.output_dump')}
        return dict(logits=logits.tolist(), class_index=int(logits.argmax()), npu_call=self.call_count,
            tensor_sha256=digest(tensor), input_mode=self.input_mode, source=source_meta,
            host_input=host_meta, adapted=adapted_meta,
            input_transfer=dict(api=api, explicit_device_Tensor_to_calls=int(self.input_mode == 'device_copy'),
                explicit_device_Tensor_to_logical_bytes=16000 if self.input_mode == 'device_copy' else 0,
                documented_Host_buffer_copy_bytes=16000 if self.input_mode == 'host_readback' else 0,
                Host_verify_read_API_bytes=16000, SDK_internal_transfer_bytes=None),
            operation_elapsed_ns=elapsed,
            stages_ns=dict(input_transfer_wrapper=transfer_end-transfer_begin, guarded_predict_total=end-begin),
            same_buffer_proven_for_HardOp=False, strict_visibility_proven=False, zero_copy=False,
            trace_policy='bounded_RAM_no_hot_file_output', diagnostic_only=True, performance_sample=False)
