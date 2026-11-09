"""D v3 actual native INT8 input. No runtime ARM dequantization.

Feature scales are folded into first main/shortcut HWIO weights. Input dtype
has unit scale, so the SDK consumes signed codes, not normalized FP32 numbers.
"""
import json
from pathlib import Path
import numpy as np

from shared.paths import ROOT
SCALE=np.fromfile(ROOT/'models/npu/scales.f32',dtype='<f4')
if SCALE.shape!=(40,) or not np.isfinite(SCALE).all() or not (SCALE>0).all():raise ValueError('Invalid frozen scales')
SCALE.flags.writeable=False
LOGICAL_SHAPE=(1,40,100)
STORAGE_SHAPE=(1,3,1,100,16)
STORAGE_BYTES=4800

def validate(x,shape):
    if not isinstance(x,np.ndarray) or x.dtype!=np.dtype('<f4') or x.shape!=shape:
        raise ValueError('Expected little-endian float32 '+str(shape))
    if not np.isfinite(x).all():raise ValueError('Nonfinite input')
    return x

def codes(x):
    validate(x,LOGICAL_SHAPE)
    with np.errstate(over='ignore'):
        return np.clip(np.floor(x/SCALE[None,:,None]+np.float32(.5)),-128,127).astype(np.int8)

def pack(q):
    if q.dtype!=np.int8 or q.shape!=LOGICAL_SHAPE:raise ValueError('Expected int8 [1,40,100]')
    padded=np.zeros((1,48,100),dtype=np.int8);padded[:,:40]=q
    return padded.reshape(1,3,16,100).transpose(0,1,3,2).reshape(STORAGE_SHAPE).tobytes()

def unpack(blob):
    if not isinstance(blob,bytes) or len(blob)!=STORAGE_BYTES:raise ValueError('Expected 4800 bytes')
    p=np.frombuffer(blob,np.int8).reshape(1,3,100,16).transpose(0,1,3,2).reshape(1,48,100)
    if np.any(p[:,40:]!=0):raise ValueError('Nonzero feature padding')
    return p[:,:40].copy()

def encode(x):return pack(codes(x))

def decode(blob):return unpack(blob).astype('<f4')*SCALE[None,:,None]

def encode_record(x):
    """One record -> 40 codes; F scatters into channel-blocked window."""
    validate(x,(40,))
    with np.errstate(over='ignore'):
        return np.clip(np.floor(x/SCALE+np.float32(.5)),-128,127).astype(np.int8).tobytes()

def sdk_source(x):
    """FP32 integer-coordinate values for SDK codec verification only."""
    validate(x,LOGICAL_SHAPE)
    p=np.zeros((1,48,100),dtype='<f4');p[:,:40]=x/SCALE[None,:,None]
    return p.reshape(1,3,16,100).transpose(0,1,3,2).reshape(STORAGE_SHAPE).copy()
