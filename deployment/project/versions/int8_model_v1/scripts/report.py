"""Report the frozen evaluation, including failed quality budgets."""
from pathlib import Path
import hashlib
import math
import shutil
import numpy as np
from common import ROOT, PROJECT, PACKAGE, read, save, sha
from reference import summary
from codec import tf32_rne, STEP


def wilson(correct,count):
    z=1.959963984540054;p=correct/count;den=1+z*z/count
    center=(p+z*z/(2*count))/den
    half=z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
    return [center-half,center+half]


def model_info(kind):
    directory=ROOT/f'artifacts/compile/{kind}'; n=read(directory/'tcn_d_ZG.json')
    q=read(directory/'tcn_d_quantized.json')
    return dict(json=str((directory/'tcn_d_ZG.json').relative_to(ROOT)),
        raw=str((directory/'tcn_d_ZG.raw').relative_to(ROOT)),json_sha256=sha(directory/'tcn_d_ZG.json'),
        raw_sha256=sha(directory/'tcn_d_ZG.raw'),generated_raw_bytes=(directory/'tcn_d_ZG.raw').stat().st_size,
        quantized_raw_bytes=(directory/'tcn_d_quantized.raw').stat().st_size,
        generated_ops=len(n['ops']),hardware_ops=sum(o['_type_key'].endswith('::HardOp') for o in n['ops']),
        expected_layers_per_forward=sum(o.get('layer_count',0) for o in n['ops']),
        external_input_dtype='float32',external_input_bytes=16000,
        compiled_target_evidence='generated HardOp compile_target @zhuget(330); actual device execution untested',
        board_tested=False,npu_latency_us=None,end_to_end_latency_us=None)


def main():
    manifest=read(ROOT/'data/manifest.json')
    release=read(ROOT/'contracts/evaluation_release.json')
    if release['compiled_json_sha256']!=sha(ROOT/'artifacts/compile/int8/tcn_d_ZG.json'):
        raise ValueError('Frozen candidate was modified after release')
    evaluation=np.load(ROOT/'data/evaluation.inputs.npy')
    labels=np.load(ROOT/'data/evaluation.labels.npy')
    fp=np.load(ROOT/'artifacts/reference/evaluation.torch.npy')
    report=dict(schema='D_int8_offline_acceptance_v1',date='2026-10-30',board_accessed=False,
        hardware_calls=0,scope='same-day temporally isolated engineering screening',
        independent_training_data_available=False,original_validation_alignment=False,
        interface_contract_version='finedge_realtime_r1',interface_contract_sha256=sha(ROOT/'contracts/core_r1.json'),
        quant_input_contract_sha256=sha(ROOT/'input_contract.json'),
        calibration_samples=256,evaluation_samples=512,
        unused_records_between_footprints=manifest['unused_records_between_footprints'],
        candidate_count=1,mixed_candidate_count=0,
        candidate_selection=release,
        reference_original_fp32=summary(fp.argmax(1),labels),models={})
    refquality=report['reference_original_fp32']; count=len(labels)
    for kind in ('tf32','int8'):
        values=np.load(ROOT/f'artifacts/host/{kind}/ZG/evaluation/logits.npy')
        quality=summary(values.argmax(1),labels)
        correct=int((values.argmax(1)==labels).sum())
        diff=values-fp
        quality.update(correct=correct,accuracy_drop_percentage_points=(refquality['accuracy']-quality['accuracy'])*100,
            macro_f1_drop_percentage_points=(refquality['macro_f1']-quality['macro_f1'])*100,
            class_changes_vs_fp32=int((values.argmax(1)!=fp.argmax(1)).sum()),
            class_agreement_vs_fp32=float(np.mean(values.argmax(1)==fp.argmax(1))),
            accuracy_nominal_wilson_95_interval=wilson(correct,count),
            confidence_interval_limitation='nominal binomial interval; overlapping windows and temporal dependence reduce effective sample size',
            logits_max_abs=float(abs(diff).max()),logits_mean_abs=float(abs(diff).mean()),
            old_strict_logits_tolerance_passed=bool(np.allclose(values,fp,rtol=1e-4,atol=1e-5)))
        quality['predeclared_quality_budget_passed']=quality['accuracy_drop_percentage_points']<=1 and quality['macro_f1_drop_percentage_points']<=1
        report['models'][kind]=dict(compile=model_info(kind),evaluation=quality)
    q=report['models']['int8'];t=report['models']['tf32']
    with np.errstate(over='ignore'):
        unbounded=np.rint(tf32_rne(evaluation)/STEP)
    clipped=(unbounded<-128)|(unbounded>127)
    report['input_quantization_saturation']=dict(elements=int(clipped.size),clipped_elements=int(clipped.sum()),
        clipped_fraction=float(clipped.mean()),scope='logical quantized-stage Cast prefix',
        per_feature_clipped_fraction=clipped.mean(axis=(0,2)).tolist(),step=float(STEP))
    report['measured_storage_effects']=dict(
        generated_raw_bytes_tf32=t['compile']['generated_raw_bytes'],generated_raw_bytes_int8=q['compile']['generated_raw_bytes'],
        generated_raw_size_reduction_fraction=1-q['compile']['generated_raw_bytes']/t['compile']['generated_raw_bytes'],
        quantized_raw_size_reduction_fraction=1-q['compile']['quantized_raw_bytes']/t['compile']['quantized_raw_bytes'],
        external_input_bytes_tf32=16000,external_input_bytes_int8_candidate=16000,
        external_input_byte_reduction_fraction=0.,logical_internal_int8_bytes=4000,
        internal_generated_integer_physical_bytes=read(ROOT/'input_contract.json')['internal_integer_codec'].get('first_generated_integer_physical_bytes'),
        npu_speedup=None,transport_latency_reduction=None,
        explanation='Artifact size reduction measured. Runtime and DDR traffic savings require G board measurement.')
    report['int8_quality_passed']=q['evaluation']['predeclared_quality_budget_passed']
    report['status']='offline_implementation_complete_quality_failed_board_pending' if not report['int8_quality_passed'] else 'offline_quality_screen_passed_board_pending'
    report['production_recommendation']='retain_existing_default; INT8 candidate is for explicit research/qualification only'
    report['why_no_mixed_retry']='Calibration passed the predeclared budget; candidate was frozen. After seeing evaluation failure no new candidate was selected against that evaluation split.'
    report['limitations']=['512 windows are a finite engineering screen, not original validation reproduction or full-day acceptance',
        'Calibration/evaluation are both from development date 2025-12-15, with no shared input/label records',
        'Input ABI remains FP32 despite internal INT8',
        'Logical SDK codec is not a verified device binding',
        'No board, NPU latency, DDR bandwidth or end-to-end speedup has been measured by task D']
    save(ROOT/'quality_report.json',report)
    request=ROOT/'board_handoff';request.mkdir(exist_ok=True)
    golden=np.fromfile(PACKAGE/'golden/inputs.f32',dtype='<f4').reshape(32,40,100)
    golden.tofile(request/'inputs.f32')
    for kind in ('tf32','int8'):
        np.load(ROOT/f'artifacts/host/{kind}/ZG/golden/logits.npy').astype('<f4').tofile(request/f'{kind}.expected_host_logits.f32')
    save(request/'request.json',dict(schema='D_to_G_model_qualification_request_v1',
        priority='optional bounded research qualification after G mainline benchmark',
        authorization_boundary='G schedules all actual device access; task D made no device calls',
        int8_quality_passed=report['int8_quality_passed'],production_enable_allowed=False,
        request_type='same-input numerical qualification and service-time measurement, not a production promotion',
        inputs=dict(path='board_handoff/inputs.f32',shape_per_tensor=[1,40,100],dtype='<f4',bytes_per_tensor=16000,
                    count=32,sha256=sha(request/'inputs.f32')),
        models={kind:model_info(kind) for kind in ('tf32','int8')},
        expected={kind:dict(path=f'board_handoff/{kind}.expected_host_logits.f32',dtype='<f4',shape=[32,3],
                  sha256=sha(request/f'{kind}.expected_host_logits.f32')) for kind in ('tf32','int8')},
        execution=dict(known_input_calls_per_model=1,warmup_calls_per_model=10,measured_calls_per_model=32,
                       total_forward_calls_per_model=43,total_forward_calls_both_models=86,batch=1,
                       model_and_session_load_once=True,strict_ordering=True,
                       no_parallel_sdk_sessions=True,completion_layer_counts_from_model_not_hardcoded=True),
        checks=dict(preferred_same_graph_byte_equal=True,numeric_rtol=1e-3,numeric_atol=1e-4,
                    class_indices_all_equal_to_same_graph_host=True,preserve_all_failures=True),
        measure=['startup','NPU forward service','input handoff','completion/output','board total', 'actual input bytes'],
        do_not_claim=['INT8 deployment quality passed','4000-byte direct model input','Host simulator time is NPU time'],
        core_contract_sha256=sha(ROOT/'contracts/core_r1.json'),input_contract_sha256=sha(ROOT/'input_contract.json')))
    save(ROOT/'fallback_config.json',dict(status='explicit_fallback_not_applied_to_current_demo',
        model='paired_tf32',json='artifacts/compile/tf32/tcn_d_ZG.json',raw='artifacts/compile/tf32/tcn_d_ZG.raw',
        input_dtype='float32',input_shape=[1,40,100],input_bytes=16000,
        expected_layers_per_forward=t['compile']['expected_layers_per_forward'],
        note='Existing demo stays on its own already validated release; this file is an optional same-graph TF32 reference, not an automatic default change.'))
    print(report['status'])
    print('Accuracy pp drop',q['evaluation']['accuracy_drop_percentage_points'],'macroF1 pp drop',q['evaluation']['macro_f1_drop_percentage_points'])
    print(report['measured_storage_effects'])


if __name__=='__main__':main()
