"""Display pacing is distinct from the backend's achieved processing rate."""
from collections import deque
import math

MAX_TARGET_RATE = 1000


def validate_rate(value):
    rate = float(value)
    if not math.isfinite(rate) or rate < 0 or rate > MAX_TARGET_RATE:
        raise ValueError('目标速度范围0～1000；0表示不限速，不代表设备能达到1000窗/秒。')
    return rate


def remaining_period(rate, elapsed_ns):
    rate = validate_rate(rate)
    if elapsed_ns < 0: raise ValueError('Elapsed time must be nonnegative')
    return max(0.0, 1.0/rate-elapsed_ns/1e9) if rate else 0.0


class OutputRate:
    """Actual emitted-window intervals, including pacing/pause, not UI polling."""
    def __init__(self):
        self.times = deque(maxlen=32)

    def observe(self, now_ns):
        if self.times and now_ns <= self.times[-1]:
            raise ValueError('Output clock must advance')
        self.times.append(now_ns)
        if len(self.times) < 2: return None
        return (len(self.times)-1)*1e9/(self.times[-1]-self.times[0])
