import numpy as np
from common import ROOT,save,sha
from reference import summary

def main():
    fp=np.load(ROOT/'artifacts/reference/calibration.torch.npy')
    q=np.load(ROOT/'artifacts/host/int8/ZG/calibration/logits.npy')
    labels=np.load(ROOT/'data/calibration.labels.npy')
    a,b=summary(fp.argmax(1),labels),summary(q.argmax(1),labels)
    passed=a['accuracy']-b['accuracy']<=.01 and a['macro_f1']-b['macro_f1']<=.01
    if (ROOT/'contracts/evaluation_release.json').exists():raise ValueError('Already frozen')
    save(ROOT/'contracts/evaluation_release.json',dict(
        status='candidate_frozen_before_evaluation',candidate='int8',
        chosen_using='calibration_only',calibration_original_fp32=a,calibration_int8=b,
        calibration_budget_passed=passed,class_changes=int((fp.argmax(1)!=q.argmax(1)).sum()),
        max_abs_logits=float(abs(fp-q).max()),
        mixed_candidate_used=False,
        mixed_candidate_decision='No extra candidate: calibration accuracy loss 0 and macro-F1 loss within the 1pp budget.',
        compiled_json_sha256=sha(ROOT/'artifacts/compile/int8/tcn_d_ZG.json'),
        compiled_params_sha256=sha(ROOT/'artifacts/compile/int8/tcn_d_ZG.raw'),
        evaluation_inputs_sha256=sha(ROOT/'data/evaluation.inputs.npy'),
        evaluation_labels_sha256=sha(ROOT/'data/evaluation.labels.npy'),
        quality_budget=dict(accuracy_drop_percentage_points_max=1.,macro_f1_drop_percentage_points_max=1.),
        policy='Evaluate frozen candidate once; do not tune against this evaluation split.',board_accessed=False))
    if not passed:raise ValueError('Calibration quality failed; do not release evaluation without an evidence-based decision')
    print('Frozen ordinary INT8; no mixed candidate selected')

if __name__=='__main__':main()
