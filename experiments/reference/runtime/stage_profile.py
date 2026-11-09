"""Opt-in exclusive frontend spans. Default execution never creates this object."""
import time
import numpy as np


class StageProfile:
    def __init__(self):
        self.spans = {}

    def start(self):
        return time.perf_counter_ns(), time.thread_time_ns()

    def end(self, name, token):
        wall = time.perf_counter_ns() - token[0]
        cpu = time.thread_time_ns() - token[1]
        self.spans.setdefault(name, []).append((wall, cpu))

    def report(self):
        result = {}
        for name, values in self.spans.items():
            data = np.asarray(values, dtype=np.float64)
            item = {'count': len(data)}
            for i, clock in enumerate(('wall', 'thread_cpu')):
                x = data[:, i] / 1000
                item[clock + '_us'] = {'sum': float(x.sum()), 'mean': float(x.mean()),
                    'p50': float(np.percentile(x, 50)), 'p95': float(np.percentile(x, 95)),
                    'p99': float(np.percentile(x, 99)), 'max': float(x.max())}
            result[name] = item
        return result
