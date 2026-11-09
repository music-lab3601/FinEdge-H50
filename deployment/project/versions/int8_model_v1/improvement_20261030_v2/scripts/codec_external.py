"""External ABI only. The mixed graph still consumes 16KB FP32 windows."""
import numpy as np

def encode(value):
    if not isinstance(value,np.ndarray) or value.shape!=(1,40,100) or value.dtype!=np.dtype('<f4'):
        raise ValueError('Expected little-endian float32[1,40,100]')
    if not np.isfinite(value).all():raise ValueError('Nonfinite')
    return np.ascontiguousarray(value).tobytes(order='C')

def decode(value):
    if not isinstance(value,bytes) or len(value)!=16000:raise ValueError('16000 FP32 bytes required')
    result=np.frombuffer(value,dtype='<f4').reshape(1,40,100).copy()
    if not np.isfinite(result).all():raise ValueError('Nonfinite')
    return result

def encode_as_int8(value):
    raise ValueError('Mixed full graph does not accept the v1 4000B logical INT8 codec')
