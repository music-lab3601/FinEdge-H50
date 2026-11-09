"""Runtime raw DBN reconstruction -> frozen normalization -> shared slot.

No model/reference inference. No prebuilt book/tensor input. The bounded PC
history is used only for sampled transfer-integrity checks, never NPU input.
"""
from collections import deque
import hashlib

from .ddr_transport_contract import pack_record
from .udw_e_frontend import CandidateFrontend
from .shared_input_contract import crc32


def batches(source,normalization,*,sample_every=64,max_windows=None):
    if type(sample_every) is not int or sample_every<1:raise ValueError('Positive sample interval required')
    if max_windows is not None and (type(max_windows) is not int or max_windows<1):raise ValueError('Positive finite window limit required')
    frontend=CandidateFrontend(normalization,'H')
    history=deque(maxlen=100)
    pending=[];expected_windows=[];window=0
    for record in source.iter_records():
        info=frontend.push(record)
        if info['segment_changed']:history.clear()
        row=info['normalized'].astype('<f4',copy=False).tobytes()
        history.append(row)
        emit=info['opportunity']
        packet=pack_record(tuple(int(v) for v in info['normalized'].view('<u4')),info['identity'],emit)
        pending.append(packet)
        if emit:
            window+=1
            expected=dict(first=info['first'],last=info['last'])
            if window==1 or window%sample_every==0:
                tensor=b''.join(row[c*4:c*4+4] for c in range(40) for row in history)
                expected.update(crc32=crc32(tensor),sample_sha256=hashlib.sha256(tensor).hexdigest())
            expected_windows.append(expected)
        if len(pending)==32 or (max_windows is not None and window==max_windows):
            yield b''.join(pending),expected_windows
            pending=[];expected_windows=[]
            if max_windows is not None and window==max_windows:return
    if pending:yield b''.join(pending),expected_windows
