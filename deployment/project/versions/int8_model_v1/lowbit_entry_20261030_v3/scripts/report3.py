"""Recompute quality, saturated input counts, and bounded G board request."""
import shutil
import numpy as np
from common3 import ROOT,BASE,V2,PROJECT,read,save,sha
from codec3 import SCALE,encode
from prepare_holdout3 import frozen_check

def metrics(logits,y,ref):
    pred=logits.argmax(1);cm=np.zeros((3,3),dtype=np.int64);np.add.at(cm,(y,pred),1)
    f1=2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)
    return dict(correct=int((pred==y).sum()),samples=len(y),accuracy=float((pred==y).mean()),macro_f1=float(f1.mean()),
        class_f1=f1.tolist(),confusion_true_rows_pred_cols=cm.tolist(),class_changes_vs_original=int((pred!=ref.argmax(1)).sum()),
        logits_max_abs_vs_original=float(abs(logits-ref).max()),logits_mean_abs_vs_original=float(abs(logits-ref).mean()),
        strict_logits_allclose_original=bool(np.allclose(logits,ref,rtol=1e-4,atol=1e-5)))

def main():
    frozen=frozen_check();folder=ROOT/'artifacts/fresh';y=np.load(ROOT/'data/labels.npy');x=np.load(ROOT/'data/inputs.npy')
    a=np.load(folder/'original_fp32.npy');values={'original_fp32':a}
    for key in ['full_tf32','accepted_mixed','dequantized_mixed','mixed_int8_entry','tf32_body_int8_entry']:values[key]=np.load(folder/key/'logits.npy')
    results={k:metrics(v,y,a) for k,v in values.items()};ref=results['original_fp32']
    for k,row in results.items():
        row['accuracy_drop_pp']=(ref['accuracy']-row['accuracy'])*100
        row['macro_f1_drop_pp']=(ref['macro_f1']-row['macro_f1'])*100
        row['quality_budget_passed']=row['accuracy_drop_pp']<=1 and row['macro_f1_drop_pp']<=1
    u=x/SCALE[None,:,None];rounded=np.floor(u+np.float32(.5));sat=(rounded<-128)|(rounded>127)
    integer=values['mixed_int8_entry'];decoded=values['dequantized_mixed']
    q=dict(schema='D_lowbit_quality_v3',status='offline_budget_passed_board_pending' if results['mixed_int8_entry']['quality_budget_passed'] else 'offline_budget_failed',
        evaluation='fresh1024 after candidate freeze; same-day engineering evidence',candidate_sha256=sha(ROOT/'contracts/frozen_candidate.json'),
        rows=results,budget=frozen['quality_budget'],
        saturation=dict(denominator_elements=int(x.size),clipped_elements=int(sat.sum()),fraction=float(sat.mean()),
            windows_with_clipping=int(sat.any((1,2)).sum()),clipped_per_feature=sat.sum((0,2)).tolist(),
            input_quantization_max_abs=float(abs(x-np.array([__import__('codec3').decode(encode(v[None]))[0] for v in x])).max())),
        error_separation=dict(external_quantization='accepted_mixed vs dequantized_mixed',fold_and_native_entry='dequantized_mixed vs mixed_int8_entry',
            fold_logits_max_abs=float(abs(integer-decoded).max()),fold_logits_mean_abs=float(abs(integer-decoded).mean()),
            fold_class_changes=int((integer.argmax(1)!=decoded.argmax(1)).sum()),
            all_tf32_with_same_lowbit_entry='tf32_body_int8_entry; diagnostic paired reference, not a post-holdout candidate search'),
        input_bytes=dict(old_fp32=16000,logical_int8=4000,native_padded_int8=4800,padding=800,saved=11200,reduction_percent=70.0),
        board_tested=False,performance_measured=False,npu_us=None,end_to_end_us=None,
        limitations=['1024 is finite and same day, not a new out-of-sample trading day','9/1024 fewer correct than original; narrow remaining margin against 1pp budget',
            'H50 uses endpoint mid-price direction from current engineering reconstruction, not teammate original validation cache',
            'lossy quantization does not satisfy original FP32 strict-logits equivalence',
            'fixed per-feature input scales may saturate new market regimes; no adaptation was fitted on this holdout',
            '70% fewer input bytes does not establish DDR bandwidth or latency speedup'])
    save(ROOT/'quality_report.json',q)
    handoff=ROOT/'board_handoff';handoff.mkdir(exist_ok=False)
    payload=b''.join(encode(v[None]) for v in x[:32]);(handoff/'inputs.i8').write_bytes(payload)
    np.save(handoff/'normalized_reference.npy',x[:32]);np.save(handoff/'identities.npy',np.load(ROOT/'data/identities.npy')[:32])
    requests=[]
    for name,m in frozen['models'].items():
        expected=values[name][:32].astype('<f4');expected.tofile(handoff/(name+'.expected_logits.f32'))
        requests.append(dict(name=name,model_json=m['json'],model_raw=m['raw'],model_sha256=m['json_sha256'],params_sha256=m['raw_sha256'],
            expected='board_handoff/'+name+'.expected_logits.f32',expected_sha256=sha(handoff/(name+'.expected_logits.f32')),
            expected_layers_per_forward=m['expected_layers_per_forward'],known_input_index=0,warmup_input_index=0,
            run=dict(known=1,warmup=10,measured=32,total=43,batch=1)))
    save(handoff/'request.json',dict(schema='D_to_G_native_int8_qualification_v3',status='requested_not_run',agent='D',
        offline_budget_passed=q['rows']['mixed_int8_entry']['quality_budget_passed'],
        contract_version='D.int8.native.r3',core_contract_sha256=read(ROOT/'input_contract.json')['base_contract_sha256'],
        input_contract='input_contract.json',input_contract_sha256=sha(ROOT/'input_contract.json'),
        inputs='board_handoff/inputs.i8',inputs_sha256=sha(handoff/'inputs.i8'),samples=32,bytes_per_tensor=4800,
        direct_integer_write=True,runtime_fp32_adapter=False,physical_shape=[1,3,1,100,16],layout='NCHWc16',
        models=requests,total_forwards=86,checks=dict(same_graph_host_rtol=1e-3,same_graph_host_atol=1e-4,classes_all_equal=True),
        performance=dict(measured=False,input_prepare_us=None,input_transfer_us=None,npu_us=None,total_us=None),
        required_measurements=['same boundaries across arms','input preparation/transfer','submit/wait','NPU completion','total mean/P50/P95/P99/max and sample count',
            'confirm SDK/device graph actually accepts native INT8 and no Host fallback','read actual completion increment: use each model count, not old26/29/33'],
        board_access_by_D=False,production_default_changed=False,recovery='G manages device/session ownership and software-first recovery',
        integration='F owns encoder/layout RTL; C consumes scales; E selects graph/hash and completion count; only G may promote'))
    save(ROOT/'fallback_config.json',dict(status='read_only_no_switch_performed',directory=str(V2),input_contract=str(V2/'input_contract.json'),
        external_dtype='float32',bytes=16000,model=str(V2/'artifacts/mixed_tf32_compat/tcn_d_ZG.json'),model_sha256=sha(V2/'artifacts/mixed_tf32_compat/tcn_d_ZG.json')))
    print(q['status']);print({k:(r['correct'],r['accuracy_drop_pp'],r['macro_f1_drop_pp']) for k,r in results.items()})

if __name__=='__main__':main()
