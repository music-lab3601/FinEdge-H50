import math,subprocess
from collections import Counter
import numpy as np
from common2 import ROOT,BASE,PROJECT,read,save,sha,env


def quality(values,labels,reference):
    pred=values.argmax(1);ref=reference.argmax(1);cm=np.zeros((3,3),dtype=np.int64)
    np.add.at(cm,(labels,pred),1);tp=np.diag(cm).astype(float);support=cm.sum(1);selected=cm.sum(0)
    f1=np.divide(2*tp,support+selected,out=np.zeros(3),where=(support+selected)>0)
    recall=np.divide(tp,support,out=np.zeros(3),where=support>0)
    return dict(samples=len(labels),correct=int((pred==labels).sum()),accuracy=float(np.mean(pred==labels)),
        macro_f1=float(f1.mean()),class_recall=recall.tolist(),confusion=cm.tolist(),
        class_changes_vs_original=int((pred!=ref).sum()),class_agreement_vs_original=float(np.mean(pred==ref)),
        logits_max_abs=float(abs(values-reference).max()),logits_mean_abs=float(abs(values-reference).mean()))


def main():
    frozen=read(ROOT/'contracts/frozen_candidate.json');model=ROOT/frozen['model_path'];net=read(model)
    if sha(model)!=frozen['model_sha256']:raise ValueError('Frozen mixed model changed')
    labels=np.load(ROOT/'data/labels.npy');ref=np.load(ROOT/'artifacts/fresh/original_fp32.npy')
    rows={'original_fp32':quality(ref,labels,ref)}
    for kind in ('ordinary_int8','tf32','mixed'):
        rows[kind]=quality(np.load(ROOT/f'artifacts/fresh/{kind}/logits.npy'),labels,ref)
        rows[kind]['accuracy_drop_pp']=(rows['original_fp32']['accuracy']-rows[kind]['accuracy'])*100
        rows[kind]['macro_f1_drop_pp']=(rows['original_fp32']['macro_f1']-rows[kind]['macro_f1'])*100
        rows[kind]['quality_budget_passed']=rows[kind]['accuracy_drop_pp']<=1 and rows[kind]['macro_f1_drop_pp']<=1
    qnet=read(ROOT/'artifacts/mixed_tf32_compat/tcn_d_quantized.json')
    arithmetic=[]
    for op in qnet['ops']:
        if not(op['_type_key'].endswith('::Conv2d') or op['_type_key'].endswith('::Matmul')):continue
        dtype=op['outputs'][0]['dtype']['element_dtype']
        arithmetic.append(dict(op_id=op['op_id'],type=op['_type_key'],storage_dtype=dtype['storage_dtype']))
    counts=dict(Counter(o['storage_dtype'] for o in arithmetic))
    ext=net['ops'][0]['outputs'][0]
    contract=dict(schema='D_icraft_input_contract_v2',contract_version='D_mixed_input_r2',
        parent_contract='../input_contract.json',core_contract_sha256=sha(BASE/'contracts/core_r1.json'),
        model_json=frozen['model_path'],model_sha256=sha(model),params_sha256=sha(model.with_suffix('.raw')),
        input=dict(dtype='float32',shape=[1,40,100],signedness='IEEE754',byte_order='little',
            logical_axes=['batch','feature','time'],physical_shape=ext['dtype']['shape'],layout=ext['dtype']['layout'],
            strides_bytes=[16000,400,4],valid_bytes=16000,storage_bytes=16000,
            compiled_dtype=ext['dtype'],scale=None,zero_point=None,saturation='none on external input',
            alignment_bytes=None,allocation='SDK allocator and G/F verified device binding'),
        arithmetic_precision_counts=counts,arithmetic_ops=arithmetic,
        first_generated_integer_value=next((dict(op_id=o['op_id'],name=o.get('name'),value=v) for o in net['ops'] for v in o['outputs']
            if isinstance(v['dtype']['element_dtype'],dict) and v['dtype']['element_dtype'].get('storage_dtype')=='@sint(8)'),None),
        full_window_direct_int8_input_supported=False,direct_device_integer_binding_verified=False,
        old_internal_value246_codec_applicable=False,
        warning='Full-window INT8 codec from v1 is not an input for this mixed model. Its first integer tensor occurs after the protected first block.',
        expected_layers_per_forward=sum(o.get('layer_count',0) for o in net['ops']),board_accessed=False)
    save(ROOT/'input_contract.json',contract)
    manifest=read(ROOT/'data/manifest.json')
    actual_gap=int(np.diff(np.load(ROOT/'data/endpoints.npy')).min()-150)
    if manifest['min_gap_records_between_footprints']!=actual_gap:
        old_hash=sha(ROOT/'data/manifest.json');old_value=manifest['min_gap_records_between_footprints']
        manifest['min_gap_records_between_footprints']=actual_gap
        save(ROOT/'data/manifest.json',manifest)
        save(ROOT/'artifacts/gap_metadata_correction.json',dict(before_sha256=old_hash,
            after_sha256=sha(ROOT/'data/manifest.json'),field='min_gap_records_between_footprints',
            before=old_value,after=actual_gap,reason='Count unused records, not endpoint difference; inputs/labels/candidate unchanged.'))
    sizes={name:path.stat().st_size for name,path in (
        ('tf32',BASE/'artifacts/compile/tf32/tcn_d_ZG.raw'),
        ('ordinary_int8',BASE/'artifacts/compile/int8/tcn_d_ZG.raw'),('mixed',model.with_suffix('.raw')))}
    report=dict(schema='D_precision_improvement_result_v2',date='2026-10-30',
        status='fresh_offline_quality_passed_board_pending' if rows['mixed']['quality_budget_passed'] else 'fresh_offline_quality_failed',
        fresh_evaluation_count=len(labels),scope='previously unused later same-day engineering holdout; not independent date validation',
        candidate_frozen_before_fresh_data=True,full_footprints_nonoverlapping=True,
        dataset_manifest_sha256=sha(ROOT/'data/manifest.json'),candidate_freeze_sha256=sha(ROOT/'contracts/frozen_candidate.json'),
        rows=rows,calibration_comparison=frozen,
        improvement_vs_ordinary_int8=dict(
            accuracy_gain_pp=(rows['mixed']['accuracy']-rows['ordinary_int8']['accuracy'])*100,
            macro_f1_gain_pp=(rows['mixed']['macro_f1']-rows['ordinary_int8']['macro_f1'])*100,
            additional_correct=rows['mixed']['correct']-rows['ordinary_int8']['correct'],
            mean_absolute_logits_error_reduction_fraction=1-rows['mixed']['logits_mean_abs']/rows['ordinary_int8']['logits_mean_abs']),
        precision_design='First and last blocks and classifier TF32; middle three blocks INT8',
        arithmetic_precision_counts=counts,
        pure_int8=False,successful_generated_new_models=1,precision_encodings_attempted=2,
        fp16_attempt='Quantize/adapt and Host precheck worked, but generate crashed 0xC0000409; conservative codegen also failed. Preserved, not deployed.',
        artifact_sizes=sizes,mixed_size_reduction_vs_tf32=1-sizes['mixed']/sizes['tf32'],
        mixed_size_increase_vs_ordinary_int8=sizes['mixed']/sizes['ordinary_int8']-1,
        layers_per_forward=dict(tf32=26,ordinary_int8=29,mixed=contract['expected_layers_per_forward']),
        external_input_bytes_all_models=16000,
        input_contract_sha256=sha(ROOT/'input_contract.json'),npu_speedup=None,end_to_end_speedup=None,
        board_tested=False,board_calls=0,training=False,normalization_refit=False,
        limitations=['512 samples of one later region from the same development date; no full-day/generalization acceptance',
            'Classification still differs from original FP32 on some windows despite equal aggregate accuracy',
            'Larger than ordinary INT8 and more generated layers; hardware latency benefit unmeasured',
            'No 4KB integer external input has been produced for this model'])
    save(ROOT/'quality_report.json',report)
    out=ROOT/'board_handoff';out.mkdir(exist_ok=True)
    inputs=np.load(ROOT/'data/inputs.npy')[:32];inputs.astype('<f4').tofile(out/'inputs.f32')
    expected=np.load(ROOT/'artifacts/fresh/mixed/logits.npy')[:32];expected.astype('<f4').tofile(out/'expected_host_logits.f32')
    save(out/'request.json',dict(schema='D_to_G_mixed_model_qualification_v2',
        status='requested_not_run',offline_budget_passed=rows['mixed']['quality_budget_passed'],
        priority='G schedules after its active benchmark work',model_json=frozen['model_path'],
        model_raw=str(model.with_suffix('.raw').relative_to(ROOT)),model_sha256=sha(model),params_sha256=sha(model.with_suffix('.raw')),
        input_contract='input_contract.json',input_contract_sha256=sha(ROOT/'input_contract.json'),
        inputs='board_handoff/inputs.f32',inputs_sha256=sha(out/'inputs.f32'),count=32,dtype='<f4',shape=[1,40,100],bytes_per_input=16000,
        expected='board_handoff/expected_host_logits.f32',expected_sha256=sha(out/'expected_host_logits.f32'),
        expected_layers_per_forward=contract['expected_layers_per_forward'],
        run=dict(known=1,warmup=10,measured=32,total=43,batch=1),
        checks=dict(same_graph_host_rtol=1e-3,same_graph_host_atol=1e-4,classes_all_equal=True),
        performance_measured=False,production_default_changed=False,board_access_by_D=False,
        reminder='Do not assume original 26 or INT8 29 layer counts; no direct INT8 input; SDK owns allocation/completion.'))
    print(report['status'],rows,report['artifact_sizes'])

if __name__=='__main__':main()
