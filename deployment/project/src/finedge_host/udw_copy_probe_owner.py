"""Independent device-to-Host copy diagnostic; never used by the default service.

Only Tensor.to(HostDevice.MemRegion()) is used. Copies, Cast results and
outputs are retained before cancellation/metadata checks. No pointer coercion.
"""
from dataclasses import asdict
from datetime import timedelta
import importlib
import io
import json
from pathlib import Path
import time

import numpy as np

from .delivery import read_json, sha256
from .udw_probe_stop import ProbeCancelled
from .udw_shared_probe_owner import ProbeBridge
from .udw_lab import digest


class ProbeTrace:
    """Persist only plain snapshots; failure reporting never queries the device."""

    def __init__(self, stop, record, clock=time.perf_counter_ns):
        self.stop, self.record, self.clock = stop, record, clock
        self.events = []
        self.context = {'phase': 'initialize'}

    def event(self, name, value):
        row = dict(event=name, context=dict(self.context), value=value)
        # Reject SDK objects before they can enter the diagnostic writer.
        row = json.loads(json.dumps(row, allow_nan=False))
        self.events.append(row)
        self.record('diagnostic', row)

    def call(self, stage, operation, retain=None):
        self.stop.checkpoint(stage + ':trace_before')
        self.event('call_begin', dict(stage=stage))
        state = dict(stage=stage, operation_started=False, operation_returned=False,
                     operation_elapsed_ns=None)

        def invoke():
            state['operation_started'] = True
            begin = self.clock()
            try:
                result = operation()
                state['operation_returned'] = True
                return result
            finally:
                state['operation_elapsed_ns'] = self.clock() - begin

        try:
            result = self.stop.call(stage, invoke, retain)
        except BaseException as error:
            state.update(outcome='cancelled' if isinstance(error, ProbeCancelled) else 'failed',
                         error=type(error).__name__ + ': ' + str(error),
                         timing_is_failed_or_cancelled_call=True)
            self.event('call_end', state)
            raise
        self.event('call_end', dict(state, outcome='returned', timing_is_failed_or_cancelled_call=False))
        return result

    def snapshot(self):
        return json.loads(json.dumps(self.events, allow_nan=False))


def verify_sdk_binding(interface_path):
    """Check the exact local ARM extension and advertised API before Device.Open.

    Static package/stub evidence does not replace a successful ARM call.
    No alternative copying API is attempted if this contract is absent.
    """
    spec = read_json(interface_path)
    if spec['copy_api'] != 'Tensor.to(HostDevice.MemRegion())':
        raise ValueError('Single explicit copy API required')
    modules = {}
    for name, expected in spec['ARM_extensions'].items():
        module = importlib.import_module(name)
        path = Path(module.__file__)
        if path.parent != Path('/usr/lib/aarch64-linux-gnu') or sha256(path) != expected:
            raise ValueError('ARM Python extension identity differs: ' + name)
        modules[name] = module
    api = modules['xrt'].Tensor.to
    doc = api.__doc__ or ''
    if not callable(api) or 'xrt.Tensor' not in doc or 'xrt.MemRegion' not in doc:
        raise ValueError('Tensor.to Python signature is not the reviewed contract')
    return dict(copy_api=spec['copy_api'],Tensor_to_doc=doc,
                ARM_extensions=spec['ARM_extensions'],actual_copy_verified=False)


class CopyProbeBridge(ProbeBridge):
    def __init__(self, stop, record):
        self.trace = ProbeTrace(stop, record)
        super().__init__(stop)

    def sdk(self, stage, operation, retain=None):
        if stage == 'arena.write' and self.pending:
            key = self.pending[0]
            slot = self.ledger.views[key]
            self.trace.event('arena_write_plan', dict(offset=slot['slot_offset'],
                explicit_API_bytes=slot['slot_size'], logical_tensor_bytes=16000,
                SDK_internal_transfer_bytes=None, device_bus_bytes=None))
        return self.trace.call(stage, operation, retain)

    def initialize(self, model, expected_boot, run_id, control='axi_registers'):
        import sys
        sys.path.insert(0, '/usr/lib/aarch64-linux-gnu')
        interface = self.sdk('init.copy_interface', lambda: verify_sdk_binding(
            Path(model).parents[2] / 'probe/sdk_interface.json'))
        self.trace.event('copy_interface', interface)
        contract = read_json(Path(model).parents[2] / 'probe/cast_contract.json')
        self.expected_cast_dtype = contract['returned_dtype']
        info = super().initialize(model, expected_boot, run_id, control)
        if info['input_type'] != contract['model_input_dtype']:
            raise ValueError('Compiled input differs from frozen L02 Cast contract')
        info.update(mode='explicit_device_to_Host_copy_diagnostic', copy_api=interface['copy_api'])
        return info

    def tensor_metadata(self, tensor, name, expected_region):
        """Use ptype before legal address getters; never call Python cptr()."""
        get = lambda suffix, fn: self.sdk(name + '.' + suffix, fn)
        dtype = get('dtype', tensor.dtype)
        chunk = get('chunk', tensor.chunk)
        ptr = get('pointer', tensor.data)
        kind = get('ptype', ptr.ptype)
        types = self.xrt.PtrType
        labels = [(types.ADDR, 'ADDR'), (types.CPTR, 'CPTR'), (types.BOTH, 'BOTH')]
        label = next((text for value, text in labels if kind == value), None)
        if label is None:
            raise ValueError('Unknown SDK pointer type')
        address = get('address', lambda: int(ptr.addr())) if label in ('ADDR', 'BOTH') else None
        begin = get('chunk_begin', lambda: chunk.begin)
        begin_kind = get('chunk_ptype', begin.ptype)
        base = get('chunk_address', lambda: int(begin.addr())) if begin_kind in (types.ADDR, types.BOTH) else None
        region = get('region', tensor.memRegion)
        metadata = dict(
            tensor_handle_identity=id(tensor), chunk_handle_identity=id(chunk),
            handle_identity_is_memory_address=False,
            ptype=label, device_address=address, allocation_base=base,
            CPU_pointer_address=None,
            CPU_pointer_address_note='Python cptr() is declared to return str, not used as an address getter',
            offset=get('offset', lambda: int(tensor.offset())),
            chunk_bytes=get('chunk_bytes', lambda: int(chunk.byte_size)),
            storage_bytes=get('storage_bytes', lambda: int(dtype.bytes())),
            shape=get('shape', lambda: list(dtype.shape)),
            layout=get('layout', lambda: str(dtype.layout)),
            complete_dtype=get('complete_dtype', lambda: str(dtype)),
            region_type=get('region_type', region.typeKey),
            is_on_expected_region=get('is_on_region', lambda: bool(tensor.isOn(expected_region))))
        self.trace.event(name + '_metadata', metadata)
        return metadata

    def validate_tensor(self, meta, expected_type, expected_pointer):
        expected = self.sdk('metadata.expected_type', lambda: str(expected_type))
        if (meta['ptype'] not in expected_pointer or not meta['is_on_expected_region']
                or meta['shape'] != [1, 40, 100] or meta['storage_bytes'] != 16000
                or meta['offset'] < 0 or meta['offset'] + 16000 > meta['chunk_bytes']
                or meta['complete_dtype'] != expected):
            raise ValueError('Copy probe Tensor pointer/region/type/extent mismatch')

    def _predict_retained(self, tensor):
        if len(tensor) != 16000 or not np.isfinite(np.frombuffer(tensor, dtype='<f4')).all():
            raise ValueError('Invalid tensor')
        if self.pending is None or self.pending_index >= len(self.pending):
            raise ValueError('Copy diagnostic requires a protected shared input, no software fallback')
        begin = time.perf_counter_ns()
        key = self.pending[self.pending_index]
        self.pending_index += 1
        self.trace.context.update(lease=asdict(key))
        self.ledger.observe_protected_readback(key, tensor)
        device_region = self.sdk('input.device_region', self.device.defaultMemRegion)
        host_region = self.sdk('input.host_region', self.xrt.HostDevice.MemRegion)
        source = self.ledger.bind(key, tensor_factory=lambda *a: self.sdk('input.bind', lambda: self.xrt.Tensor(*a)),
            dtype=self.software_type, chunk=self.chunk, region=device_region, allow_experimental_readback=True)
        source_meta = self.tensor_metadata(source, 'source', device_region)
        self.validate_tensor(source_meta, self.software_type, ('ADDR',))
        if self.sdk('source.on_host', lambda: source.isOn(host_region)):
            raise ValueError('Expected a device source, not a Host alias')
        self.trace.event('copy_input', dict(source=source_meta, input_sha256=digest(tensor),
            logical_bytes=16000, float32_values=4000, copy_api='Tensor.to(HostDevice.MemRegion())',
            SDK_internal_transfer_bytes=None, device_bus_bytes=None))
        bind_end = time.perf_counter_ns()
        host_copy = self.sdk('input.to_Host', lambda: source.to(host_region), self.ledger.handles.append)
        copy_end = time.perf_counter_ns()
        target_meta = self.tensor_metadata(host_copy, 'host_copy', host_region)
        self.validate_tensor(target_meta, self.software_type, ('CPTR', 'BOTH'))
        if self.sdk('host_copy.on_device', lambda: host_copy.isOn(device_region)):
            raise ValueError('Copy still on the source device region')
        copied = np.empty(16000, dtype=np.uint8)
        self.sdk('host_copy.read', lambda: host_copy.read(copied, 0, 16000))
        actual = copied.tobytes()
        self.trace.event('copy_bytes', dict(input_sha256=digest(tensor), copied_sha256=digest(actual),
            logical_bytes=16000, float32_values=4000, all_bytes_equal=actual == tensor,
            host_read_API_bytes=16000, SDK_internal_transfer_bytes=None))
        if actual != tensor:
            raise ValueError('Copied 4000 float32 values differ from protected input')
        copy_verified = time.perf_counter_ns()
        ids = []
        adapted = self.sdk('input.Cast', lambda: self.host.Cast(host_copy, self.software_type, self.input_type, self.converter),
            lambda value: ids.append(self.ledger.retain_cast(value, source='shared', key=key)))
        cast_end = time.perf_counter_ns()
        adapted_meta = self.tensor_metadata(adapted, 'adapted', host_region)
        # The accepted Cast removes merged distributions from its return dtype.
        # Keep its original arguments; check the exact observed L02 return type.
        self.validate_tensor(adapted_meta, self.expected_cast_dtype, ('CPTR', 'BOTH'))
        self.adapted_info = dict(chunk_bytes=adapted_meta['chunk_bytes'], offset=adapted_meta['offset'], dtype=adapted_meta['complete_dtype'])
        call_id = ids[0]
        self.stop_control.checkpoint('NPU.before_mark')
        self.ledger.npu_may_start(call_id)
        outputs = self.sdk('NPU.forward', lambda: self.session.forward([adapted]), self.ledger.forward_outputs.append)
        forward_end = time.perf_counter_ns()
        if len(outputs) != 1:
            raise ValueError('Wrong NPU output count')
        self.sdk('NPU.output_not_ready', lambda: outputs[0].setReady(False))
        if not self.sdk('NPU.output_wait', lambda: outputs[0].waitForReady(timedelta(seconds=10), timedelta(milliseconds=1))):
            raise TimeoutError('NPU output timeout')
        ready_end = time.perf_counter_ns()
        stream = io.BytesIO()
        self.sdk('NPU.output_dump', lambda: outputs[0].dump(stream, 'SFB'))
        logits = np.frombuffer(stream.getvalue(), dtype='<f4')
        end = time.perf_counter_ns()
        if logits.shape != (3,) or not np.isfinite(logits).all():
            raise ValueError('Invalid logits')
        self.ledger.npu_completed(call_id)
        self.call_count += 1
        self.stop_control.checkpoint('NPU.completed')
        return dict(logits=logits.tolist(), class_index=int(logits.argmax()), npu_call=self.call_count,
            tensor_sha256=digest(tensor), shared_input=dict(path='explicit_device_to_Host_copy',
                allocation_id=key.allocation_id, source=source_meta, host_copy=target_meta, adapted=adapted_meta,
                retained_call_id=call_id, same_buffer_proven_for_HardOp=False, protective_readback_removed=False,
                explicit_host_input_write_bytes=0, copy_logical_bytes=16000, host_verify_read_bytes=16000,
                SDK_internal_transfer_bytes=None, zero_copy=False),
            stages_ns=dict(shared_bind=bind_end-begin, explicit_copy=copy_end-bind_end,
                copied_metadata_verify=copy_verified-copy_end, host_input_cast=cast_end-copy_verified,
                forward=forward_end-cast_end, output_wait=ready_end-forward_end,
                output_read=end-ready_end, diagnostic_total=end-begin),
            diagnostic_only=True, performance_sample=False)
