"""Bounded demo owner. Retain SDK returns and failed call frames until cold stop."""
from datetime import timedelta
import io
import time

import numpy as np

from experiments.reference.runtime.udw_board_bridge import BoardBridge, digest
from experiments.reference.runtime.udw_probe_stop import ProbeStop


class DemoOwner(BoardBridge):
    def __init__(self):
        super().__init__()
        self.stop = ProbeStop()
        self.stop.bind_owner(self)
        self.failures = []

    def quarantine(self, reason):
        self.blocked = True

    def guard(self, point):
        self.stop.checkpoint(point)
        return super().guard(point)

    def predict_local(self, tensor):
        if len(tensor) != 16000 or not np.isfinite(np.frombuffer(tensor, dtype='<f4')).all():
            raise ValueError('Invalid demo NPU input')
        call = self.stop.call
        begin = time.perf_counter_ns()
        call('input_write', lambda: self.input.write(0, np.frombuffer(tensor, dtype=np.uint8), len(tensor)))
        adapted = call('cast', lambda: self.host.Cast(self.input, self.software_type, self.input_type, self.converter))
        cast_end = time.perf_counter_ns()
        self.adapted_info = dict(chunk_bytes=int(adapted.chunk().byte_size),
                                 offset=int(adapted.offset()), dtype=str(adapted.dtype()))
        outputs = call('forward', lambda: self.session.forward([adapted]))
        forward_end = time.perf_counter_ns()
        if len(outputs) != 1:
            raise ValueError('Wrong demo output count')
        call('not_ready', lambda: outputs[0].setReady(False))
        if not call('output_wait', lambda: outputs[0].waitForReady(timedelta(seconds=10), timedelta(milliseconds=1))):
            raise TimeoutError('NPU output timeout')
        ready_end = time.perf_counter_ns()
        buffer = self.stop.hold(io.BytesIO())
        call('dump', lambda: outputs[0].dump(buffer, 'SFB'))
        logits = np.frombuffer(buffer.getvalue(), dtype='<f4')
        if logits.shape != (3,) or not np.isfinite(logits).all():
            raise ValueError('Invalid demo logits')
        end = time.perf_counter_ns()
        self.call_count += 1
        return dict(logits=logits.tolist(), class_index=int(logits.argmax()), npu_call=self.call_count,
                    tensor_sha256=digest(tensor), stages_ns=dict(host_input_cast=cast_end-begin,
                    forward=forward_end-cast_end, output_wait=ready_end-forward_end,
                    output_read=end-ready_end, npu_total=end-begin))
