"""Explicit external FP32 ABI and SDK-verified INTERNAL quantized-cast codec.

The internal 4000-byte encoding is NOT the full generated model's input ABI.
"""
import numpy as np

SHAPE=(1,40,100)
STEP_BITS=1025970844
STEP=np.array([STEP_BITS],dtype='<u4').view('<f4')[0]


def tensor(value):
    if not isinstance(value,np.ndarray) or value.dtype!=np.dtype('<f4') or value.shape!=SHAPE:
        raise ValueError('Expected float32 [1,40,100]')
    if not np.isfinite(value).all():raise ValueError('Nonfinite input')
    return np.ascontiguousarray(value)


def encode_external_fp32(value):
    return tensor(value).tobytes(order='C')


def decode_external_fp32(value):
    if not isinstance(value,bytes) or len(value)!=16000:raise ValueError('Expected 16000 external FP32 bytes')
    return tensor(np.frombuffer(value,dtype='<f4').reshape(SHAPE)).copy()


def tf32_rne(value):
    """Round the F32 significand to ten explicit bits, ties to even."""
    value=np.ascontiguousarray(value,dtype='<f4')
    bits=value.view('<u4')
    rounded=(bits+np.uint32(0xFFF)+((bits>>np.uint32(13))&np.uint32(1)))&np.uint32(0xFFFFE000)
    return rounded.view('<f4')


def encode_internal_int8(value,step=STEP):
    """Only quantized graph Value246: FP32->TF32 RNE->INT8 RNE/saturation."""
    value=tensor(value)
    step=np.float32(step)
    if not np.isfinite(step) or step<=0:raise ValueError('Positive finite F32 step required')
    with np.errstate(over='ignore',under='ignore',invalid='raise'):
        quantized=np.clip(np.rint(tf32_rne(value)/step),-128,127).astype(np.int8)
    return quantized.tobytes(order='C')


def decode_internal_int8(value,step=STEP):
    if not isinstance(value,bytes) or len(value)!=4000:raise ValueError('Expected 4000 logical internal bytes')
    step=np.float32(step)
    if not np.isfinite(step) or step<=0:raise ValueError('Positive finite F32 step required')
    return np.frombuffer(value,dtype=np.int8).astype('<f4').reshape(SHAPE)*step


def encode_for_full_graph(value,dtype='float32'):
    if dtype!='float32':
        raise ValueError('Compiled full graph Value0 requires FP32; direct INT8 binding is unverified')
    return encode_external_fp32(value)
