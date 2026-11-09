"""Actual dispatch speed uses integer nanoseconds, independent of viewing speed."""
SUPPORTED_SPEEDS = (1, 2, 4)


def checked_speed(speed):
    if type(speed) is not int or speed not in SUPPORTED_SPEEDS:
        raise ValueError('实际派发倍速仅支持1、2、4。')
    return speed


def wall_duration_ns(market_duration_ns, speed):
    checked_speed(speed)
    if type(market_duration_ns) is not int or market_duration_ns < 0:
        raise ValueError('历史片段时长必须为非负整数纳秒。')
    return market_duration_ns // speed
