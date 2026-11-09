"""Publish real ABI and freeze all numerical choices BEFORE fresh holdout."""
from datetime import datetime,timezone
import numpy as np
from common3 import ROOT,BASE,V2,PROJECT,read,save,sha
from codec3 import SCALE,STORAGE_SHAPE

def main():
    target=ROOT/'contracts/frozen_candidate.json'
    if target.exists():raise FileExistsError('Frozen candidate exists')
    if not read(ROOT/'golden_v2/verification.json')['sdk_bytes_equal']:raise ValueError('Codec not verified')
    models={}
    for key,path in [('mixed_int8_entry','artifacts/int8_physical/d3p_ZG.json'),('tf32_body_int8_entry','artifacts/int8_entry_tf32_body/d3tp_ZG.json')]:
        p=ROOT/path;n=read(p)
        models[key]=dict(json=path,raw=str(p.with_suffix('.raw').relative_to(ROOT)).replace('\\','/'),json_sha256=sha(p),raw_sha256=sha(p.with_suffix('.raw')),
            expected_layers_per_forward=sum(o.get('layer_count',0) for o in n['ops']),input=n['ops'][0]['outputs'][0]['dtype'])
    amendment=dict(date_utc=datetime.now(timezone.utc).isoformat(),before_fresh_holdout=True,
        original_plan_sha256=sha(ROOT/'contracts/study_plan.json'),
        change='Use symmetric fixed feature scales, zero point 0; fold scales into first main/shortcut weights; native NCHWc16 external SINT8 ABI.',
        reasons=['Affine per-feature dense input failed adapter scale-length validation.',
                 'Padded INT8 decoding prefixes adapted but generator terminated 0xC0000409.',
                 'Short paths did not solve generation. Native blocked entry succeeded without integer layout transforms.',
                 'Symmetric zero point preserves causal zero padding and permits algebraic weight scaling.',
                 'SDK physical codec is clip(floor(float32(float32(x/s)+0.5f)),-128,127), not RNE. Boundary failure retained.'],
        unchanged=['one accepted mixed body, no new protection search or training','original normalization','fixed input scaling fit from already observed v2 inputs','fresh 1024 endpoints and <=1pp quality budgets'],
        int16='Not used: actual Zhuge INT16 Cast support not established; no fabricated fallback.',fresh_data_observed=False)
    save(ROOT/'contracts/plan_amendment.json',amendment)
    contract=dict(schema='finedge_D_native_int8_input_v3',contract_version='D.int8.native.r3',
        base_contract_version='r1',base_contract_sha256=sha(PROJECT/'versions/realtime_integration_v1/contracts/r1/contract.json'),
        status='compiled_and_Host_verified_codec_pending_new_holdout_and_G_board_qualification',
        model_binding=models,logical_shape=[1,40,100],batch=1,feature_order=[f'{side}_{kind}{level}' for level in range(1,11) for side,kind in [('ask','p'),('ask','q'),('bid','p'),('bid','q')]],
        normalized_dtype='<f4',normalization_sha256=sha(PROJECT/'vendor/ed4_model/normalization.json'),
        external_dtype='signed_int8',external_value_id=0,external_sdk_quantized_dtype=models['mixed_int8_entry']['input']['element_dtype'],
        external_shape=list(STORAGE_SHAPE),external_axis_order=['batch','feature_block','height','time','feature_lane'],
        layout='NCHWc16',byte_order='one byte per signed code; two-complement',logical_valid_bytes=4000,storage_bytes=4800,
        physical_dense_strides_bytes=[4800,1600,1600,16,1],
        offset_bytes='(feature//16)*1600 + time*16 + feature%16; feature=0..39,time=0..99',
        padding=dict(feature_range=[40,48],code=0,bytes=800,required=True,time_padding=0),
        address_alignment=dict(required_by_D_codec=1,device_allocation='Use SDK allocator/alignment; D does not claim hardware DMA alignment from one-byte codec'),
        numerical_quantization=dict(bits=8,signed=True,min_code=-128,max_code=127,scale=SCALE.tolist(),scale_f32_bits=SCALE.view('<u4').tolist(),zero_point=[0]*40,
            scale_axis='logical feature, invariant across time/window/stream',
            rule='u=float32(x/scale[f]); v=float32(u+0.5f); q=int8(clip(floor(v),-128,127))',
            nonfinite='Reject before encoder, invalidate window; no silent NaN conversion',
            subnormals='IEEE float32 operations; C fixed-point implementation must match published golden or obtain a versioned tolerance/contract amendment',
            dequantization_reference='x_hat=float32(q)*scale[f]',
            calibration_inputs_sha256=sha(V2/'data/inputs.npy'),calibration_labels_used_for_scale_fit=False),
        model_transform=dict(original_body='accepted D v2 mixed, no layer-protection search',scale_fold_parameters=[6,20],
            scale_fold_formula='W_new[h,w,f,o]=TF32_RNE(float32(W_old[h,w,f,o]*scale[f]))',
            biases_changed=False,training=False,scale_decoded_on_ARM=False,
            hardware_prefix='one native SINT8 -> TF32 Cast; feature scale absorbed by first convolution and shortcut weights',
            host_input_adapter=False,remaining_layout_transforms='Original body/output transforms retained after first conv input; see adapted graph',
            potential_error_sources=['external input quantization/saturation','additional TF32 rounding of folded weights','unchanged mixed-body PTQ']),
        record_once=dict(supported=True,record_payload_bytes=40,encoder='scripts/codec3.py:encode_record',
            assembler='F sole RTL owner: store 40 codes per accepted row, scatter chronological window using offset_bytes; append eight zero padding features',
            segment_reset='Never join across invalid/segment boundary; first99 valid rows have no inference'),
        golden=dict(inputs='golden_v2/normalized.npy',encoded='golden_v2/encoded.bin',hex='golden_v2/encoded.hex',samples=64,
            encoded_sha256=sha(ROOT/'golden_v2/encoded.bin'),verification='golden_v2/verification.json'),
        fallback='Accepted v2 FP32 entry 16000B. Do not feed this encoding to the v2 graph.',
        board_tested=False,performance=dict(npu_us=None,ddr_us=None,end_to_end_us=None),production_default_changed=False)
    save(ROOT/'input_contract.json',contract)
    a=np.load(V2/'artifacts/fresh/original_fp32.npy');y=np.load(V2/'data/labels.npy')
    b=np.fromfile(ROOT/'artifacts/observed_screen/integer/logits.f32','<f4').reshape(-1,3)
    frozen=dict(schema='D_lowbit_frozen_v3',frozen_at_utc=datetime.now(timezone.utc).isoformat(),models=models,
        input_contract_sha256=sha(ROOT/'input_contract.json'),codec_sha256=sha(ROOT/'scripts/codec3.py'),
        plan_sha256=sha(ROOT/'contracts/study_plan.json'),amendment_sha256=sha(ROOT/'contracts/plan_amendment.json'),
        scale_source_sha256=sha(V2/'data/inputs.npy'),already_observed_screen=dict(samples=len(y),original_correct=int((a.argmax(1)==y).sum()),integer_correct=int((b.argmax(1)==y).sum())),
        fresh_evaluation_not_yet_generated=True,quality_budget=read(ROOT/'contracts/study_plan.json')['budget'],board_accessed=False)
    save(target,frozen);print('Candidate and true native INT8 contract frozen',sha(target))

if __name__=='__main__':main()
