"""Publish facts from actual compiled XIR; no guessed integer input ABI."""
from collections import Counter
import math
import re
from deployment.icraft.base.common import ROOT, read, save, sha


def type_size(dtype):
    element=dtype['element_dtype']
    storage=element['storage_dtype'] if isinstance(element,dict) else element
    bits=int(re.search(r'\((\d+)\)',storage).group(1))
    return math.prod(dtype['shape'])*bits//8


def main():
    quant=read(ROOT/'artifacts/compile/int8/tcn_d_quantized.json')
    generated=read(ROOT/'artifacts/compile/int8/tcn_d_ZG.json')
    external=generated['ops'][0]['outputs'][0]
    int_op=next(o for o in quant['ops'] if o['_type_key'].endswith('::Cast') and
                any(v['dtype'].get('element_dtype',{}).get('storage_dtype')=='@sint(8)'
                    for v in o['outputs'] if isinstance(v['dtype'].get('element_dtype'),dict)))
    internal=int_op['outputs'][0]
    preceding=next(o for o in quant['ops'] if any(v['v_id']==int_op['inputs'][0]['v_id'] for v in o['outputs']))
    device_integer=[dict(op_id=o['op_id'],name=o.get('name'),value=v)
        for o in generated['ops'] for v in o['outputs']
        if v['v_id']==internal['v_id']]
    save(ROOT/'input_contract.json',dict(
        schema='D_icraft_input_contract_v1',contract_version='D_input_r1',status='compiled_external_abi_confirmed_host_codec_pending',
        core_contract_version='finedge_realtime_r1',core_contract_sha256=sha(ROOT/'contracts/core_r1.json'),
        graph_sha256=sha(ROOT/'model/tcn_polyphase.onnx'),
        model_quantized_json_sha256=sha(ROOT/'artifacts/compile/int8/tcn_d_quantized.json'),
        model_generated_json_sha256=sha(ROOT/'artifacts/compile/int8/tcn_d_ZG.json'),
        model_generated_params_sha256=sha(ROOT/'artifacts/compile/int8/tcn_d_ZG.raw'),
        input=dict(value_id=external['v_id'],dtype='float32',signedness='IEEE754',shape=[1,40,100],
            byte_order='little',logical_axes=['batch','feature','time'],
            physical_shape=external['dtype']['shape'],layout=external['dtype']['layout'],
            strides_bytes=[16000,400,4],valid_bytes=16000,storage_bytes=type_size(external['dtype']),
            scale=None,zero_point=None,rounding='IEEE754 FP32 external storage',
            saturation='no added external clipping',merged_distributions=external['dtype']['merged_distrs'],
            compiled_dtype=external['dtype'],memory_type=external['mtype']),
        internal_integer_codec=dict(status='not_a_replacement_for_external_model_input',
            value_id=internal['v_id'],op_id=int_op['op_id'],shape=internal['dtype']['shape'],
            logical_bytes=type_size(internal['dtype']),compiled_dtype=internal['dtype'],
            preceding_cast=dict(op_id=preceding['op_id'],value_id=preceding['outputs'][0]['v_id'],
                                dtype=preceding['outputs'][0]['dtype']),
            generated_integer_value=device_integer,
            signedness='signed int8',zero_point=0,
            zero_point_evidence='compiler zero_points array is empty; symmetric quantized type',
            rounding='pending SDK encoded boundary-vector verification',
            scale_interpretation='pending SDK encoded/dequantized comparison'),
        direct_int8_input_supported_by_current_full_graph=False,
        direct_int8_device_binding_verified=False,board_accessed=False,
        instruction='Feed FP32 [1,40,100] to current full graph. Do not bind a 4000-byte int8 tensor to Value0.',
        fallback=dict(kind='paired_tf32',json='artifacts/compile/tf32/tcn_d_ZG.json',
                      raw='artifacts/compile/tf32/tcn_d_ZG.raw',input_dtype='float32',input_bytes=16000)))
    save(ROOT/'contracts/published.json',dict(contract='input_contract.json',sha256=sha(ROOT/'input_contract.json'),
                                             board_accessed=False))
    print('Published',ROOT/'input_contract.json')


if __name__=='__main__':main()
