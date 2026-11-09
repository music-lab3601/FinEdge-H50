"""Display-only persistence: tolerate short Windows reader/rename conflicts."""
import time
from shared.io import atomic_json as _atomic_json


def atomic_json(path,value):
    deadline=time.monotonic()+2
    while True:
        try:return _atomic_json(path,value)
        except PermissionError:
            if time.monotonic()>=deadline:raise
            time.sleep(.01)
