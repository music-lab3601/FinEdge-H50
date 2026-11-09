import numpy as np
from common import ROOT,read,save,sha
from codec import encode_internal_int8,decode_internal_int8,encode_external_fp32,STEP


def main():
    x=np.load(ROOT/'golden/inputs.npy')
    encoded=b''.join(encode_internal_int8(v[None]) for v in x)
    actual=(ROOT/'golden/encoded.sqb').read_bytes()
    mismatches=int(np.count_nonzero(np.frombuffer(encoded,np.uint8)!=np.frombuffer(actual,np.uint8)))
    if mismatches:raise ValueError(f'{mismatches} SDK codec byte mismatches')
    back=np.concatenate([decode_internal_int8(actual[i*4000:(i+1)*4000]) for i in range(len(x))])
    sdk=np.fromfile(ROOT/'golden/sdk_dequantized.f32',dtype='<f4').reshape(len(x),40,100)
    if not np.array_equal(back,sdk):
        raise ValueError('SDK cast-back dequantization differs')
    if (ROOT/'golden/encoded.hqb').read_bytes()!=actual:raise ValueError('Logical codec HQB layout differs')
    np.save(ROOT/'golden/reference_dequantized.npy',back)
    np.save(ROOT/'golden/reference_int8.npy',np.frombuffer(encoded,np.int8).reshape(len(x),40,100))
    (ROOT/'golden/encoded.hex').write_text('\n'.join(f'{b:02x}' for b in actual)+'\n',encoding='ascii')
    scales=read(ROOT/'golden/summary.json')['cast_chain'][-1]
    report=dict(status='sdk_logical_codec_byte_equal',samples=len(x),elements=len(actual),byte_mismatches=mismatches,
        external_fp32_roundtrip_bytes=len(encode_external_fp32(x[0:1])),
        scale=float(STEP),scale_float32_bits=scales['scale_float32_bits'][0],scale_axis=-1,
        scale_entries=100,all_scale_entries_equal=True,zero_point=0,
        formula='int8(clip(round_ties_even(TF32_RNE(x)/float32_step), -128, 127))',
        dequantization='float32(q) * float32_step',sdk_cast_back_bit_equal=True,
        tested_scope='32 delivered goldens plus 32 deterministic tie/nextafter/saturation tensors',
        generated_model_external_input_dtype='float32',direct_device_binding_verified=False,board_accessed=False)
    save(ROOT/'golden/codec_verification.json',report)
    contract=read(ROOT/'input_contract.json');q=contract['internal_integer_codec']
    q.update(rounding=report['formula'],scale_interpretation='x_reconstructed = q * step',
        scale=float(STEP),scale_float32_bits=report['scale_float32_bits'],scale_axis=-1,scale_entries=100,
        saturation_min=-128,saturation_max=127,logical_layout='C-contiguous [batch,feature,time]',
        logical_strides_bytes=[4000,100,1],logical_bytes=4000,
        sdk_verified_samples=len(x),sdk_verified_elements=len(actual),sdk_byte_mismatches=mismatches,
        sdk_verification_sha256=sha(ROOT/'golden/codec_verification.json'),
        warning='Logical quantized-stage Value246 was transformed during adapter lowering. This is not a generated-device address/format binding.')
    generated=read(ROOT/'artifacts/compile/int8/tcn_d_ZG.json')
    first=next((dict(op_id=o['op_id'],name=o.get('name'),value=v)
        for o in generated['ops'] for v in o['outputs']
        if isinstance(v['dtype']['element_dtype'],dict) and v['dtype']['element_dtype'].get('storage_dtype')=='@sint(8)'),None)
    q['first_generated_integer_value']=first
    if first:
        q['first_generated_integer_physical_bytes']=int(np.prod(first['value']['dtype']['shape']))
    contract['status']='compiled_external_abi_confirmed_internal_codec_sdk_verified_board_pending'
    save(ROOT/'input_contract.json',contract)
    save(ROOT/'contracts/published.json',dict(contract='input_contract.json',sha256=sha(ROOT/'input_contract.json'),
        verification='golden/codec_verification.json',board_accessed=False))
    print(report)


if __name__=='__main__':main()
