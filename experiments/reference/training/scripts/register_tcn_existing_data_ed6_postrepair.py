"""Root-only bounded metadata repair; never train, load CP, or alter old evidence."""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from experiments.reference.training.scripts import register_tcn_existing_data_ed6 as base

ROOT=base.ROOT;STAGE=base.STAGE;BUDGET=base.BUDGET;STATE=base.STATE
read=base.read;ref=base.ref;bound=base.bound;put=base.put;update=base.update;now=base.now
AUDIT=STAGE/'independent_audit_v2.py'
CAPTURE=ROOT/'scripts/run_tcn_existing_data_ed6_audit_v2_capture.py'
GLOBAL=STAGE/'audit_path_repair_v2_registration.json'
REVIEW=STAGE/'audit_path_repair_v2_source_review.json'
def verify(r):
    p=Path(r['path']);p=p if p.is_absolute()else ROOT/p
    assert ref(p)['sha256']==r['sha256'] and ref(p)['bytes']==r['bytes'],str(p)
    return p
def review_ok():
    v=read(REVIEW);assert v['passed']
    for r in v['source_bindings'].values():verify(r)
    assert base.tested_sources()==read(STAGE/'selection_lock.json')['source_bindings']
    return v
def test_evidence():
    review_ok();p=STAGE/'repair_tests_execution_receipt.json';r=read(p)
    assert r['phase']=='repair_tests' and r['exit_code']==0 and r['entry_unchanged'] and r['entry_before']==r['entry_after']
    assert r['sources_before']==r['sources_after']
    for x in r['sources_before']:verify(x)
    msg=verify(r['stderr']).read_text('utf-8');m=re.search(r'Ran (\d+) tests? in ',msg)
    assert m and int(m.group(1))==43 and re.search(r'\nOK\s*$',msg)
    put(STAGE/'repair_tests_evidence.json',{'passed':True,'actual_exit_code':0,'separate_synthetic_test_count':43,
        'no_real_data_or_ML':True,'actual_receipt':ref(p),'test_source':r['entry_before'],
        'sources_before':r['sources_before'],'sources_after':r['sources_after'],
        'scope':'43 separate v2 repair synthetic fixtures; no real streams, models, optimizer or CP loads'})
    print(json.dumps({'repair_synthetic_tests':43,'passed':True}))
def global_registration():
    review_ok();assert read(STAGE/'repair_tests_evidence.json')['passed']
    assert read(STAGE/'finalizer_v2_tests_evidence.json')['passed']
    old=STAGE/'trials/control43/audit_execution_receipt.json';failed=read(old)
    assert failed['exit_code']==1 and failed['entry_unchanged'] and failed['outputs']=={}
    assert 'ValueError: Locked norm differs' in verify(failed['stderr']).read_text('utf-8')
    prepared=read(STAGE/'prepared_control/cache_manifest.json')['normalization_binding']
    lock=read(STAGE/'selection_lock.json');locked=lock['normalization_bindings']['control43']
    assert prepared!=locked and verify(prepared).resolve()==verify(locked).resolve()
    assert prepared['sha256']==locked['sha256'] and prepared['bytes']==locked['bytes']==2289
    v={'version':'TCN-ED6-global-audit-path-repair-2.0','registered_utc':now(),
      'repair_scope':'normalized_path_SHA256_bytes_only','selection_lock_binding':ref(STAGE/'selection_lock.json'),
      'authorization_binding':ref(STAGE/'authorization.json'),
      'original_failed_audit_receipt_binding':ref(old),'original_failure_stderr_binding':failed['stderr'],
      'original_finalizer_binding':ref(ROOT/'scripts/finalize_tcn_existing_data_ed6.py'),
      'original_audit_source_binding':ref(STAGE/'independent_audit.py'),
      'audit_v2_source_binding':ref(AUDIT),'audit_v2_capture_source_binding':ref(CAPTURE),
      'repair_source_review_binding':ref(REVIEW),'repair_tests_evidence_binding':ref(STAGE/'repair_tests_evidence.json'),
      'finalizer_v2_tests_evidence_binding':ref(STAGE/'finalizer_v2_tests_evidence.json'),
      'numeric_source_bindings':lock['source_bindings'],
      'normalization_identity':{'prepared':prepared,'locked':locked},
      'limits':{'control43_audits':1,'candidate43_audits':1,'ML':0,'CPUloads':0},
      'full_original_D_only_audit_retained':True,'training_or_selection_lock_changes':False,
      'control_completed_budget_snapshot_binding':read(STAGE/'trials/control43/run/training_acceptance.json')['budget_completion_snapshot_binding'],
      'root_registry_source_binding':ref(Path(__file__))}
    put(GLOBAL,v);print(json.dumps({'global_repair_registration':ref(GLOBAL)},ensure_ascii=True))
def audit_registration(slot):
    review_ok();g=read(GLOBAL)
    assert g['limits']=={'control43_audits':1,'candidate43_audits':1,'ML':0,'CPUloads':0}
    trial=STAGE/'trials'/slot;tr=read(trial/'training_execution_receipt.json');cp=read(trial/'checkpoint_execution_receipt.json')
    assert tr['exit_code']==cp['exit_code']==0 and tr['entry_unchanged'] and cp['entry_unchanged']
    v={'version':'TCN-ED6-audit-v2-registration-1.0','slot_id':slot,'registered_utc':now(),
      'training_registration_binding':ref(trial/'training_registration.json'),
      'global_repair_registration_binding':ref(GLOBAL),'repair_scope':g['repair_scope'],
      'full_original_D_only_audit_retained':True,'additional_training_or_CP_loads':0,
      'v2_audit_source_binding':ref(AUDIT),'v2_capture_source_binding':ref(CAPTURE),
      'original_audit_source_binding':g['original_audit_source_binding'],
      'original_failure_receipt_binding':g['original_failed_audit_receipt_binding'],
      'original_failure_stderr_binding':g['original_failure_stderr_binding'],
      'synthetic_test_evidence_binding':g['repair_tests_evidence_binding'],'source_review_binding':g['repair_source_review_binding']}
    assert v['training_registration_binding']==tr['input']
    put(trial/'audit_v2_registration.json',v);print(json.dumps({'audit_v2_registration':ref(trial/'audit_v2_registration.json')},ensure_ascii=True))
def candidate_reserve():
    review_ok();g=read(GLOBAL);assert g['numeric_source_bindings']==base.tested_sources()
    selection=read(STAGE/'selection_lock.json');trial=STAGE/'trials/candidate43';ctrl=STAGE/'trials/control43'
    control_repair=read(ctrl/'audit_v2_registration.json')
    assert control_repair['global_repair_registration_binding']==ref(GLOBAL)
    assert g['limits']=={'control43_audits':1,'candidate43_audits':1,'ML':0,'CPUloads':0}
    assert not (trial/'training_registration.json').exists()
    paths={'control_training_receipt_binding':ctrl/'training_execution_receipt.json',
      'control_acceptance_binding':ctrl/'run/training_acceptance.json',
      'control_audit_receipt_binding':ctrl/'audit_v2_execution_receipt.json',
      'control_audit_binding':ctrl/'readonly_audit_v2.json',
      'control_checkpoint_receipt_binding':ctrl/'checkpoint_execution_receipt.json',
      'control_checkpoint_validation_binding':ctrl/'checkpoint_validation.json',
      'control_initialization_binding':ctrl/'run/initialization.json'}
    extra={k:ref(v)for k,v in paths.items()}
    from experiments.reference.training.scripts import tcn_existing_data_ed6_train as entry
    accepted=entry.candidate_order_gate(extra,ref(STAGE/'selection_lock.json'))
    assert ref(BUDGET)['sha256']==accepted['budget_completion_snapshot_binding']['sha256']
    budget=read(BUDGET);prior=read(STAGE/'before/budget_before_ED6.json')
    assert len(budget['reserved_runs'])==3 and budget['reserved_runs'][:2]==prior['reserved_runs']
    assert all(r['status']=='completed'for r in budget['reserved_runs'])
    assert read(paths['control_audit_binding'])['passed'] and read(paths['control_audit_receipt_binding'])['exit_code']==0
    snapshot=trial/'budget_prefix_snapshot.json'
    with snapshot.open('xb')as f:f.write(BUDGET.read_bytes());f.flush();os.fsync(f.fileno())
    cfg=ref(trial/'config.json');cache=selection['feature_cache_bindings']['candidate43'];assert cfg==selection['config_bindings']['candidate43']
    rid='TCN_EDO_20261017_ED6_candidate43';sources=selection['source_bindings']
    item={'reservation_id':rid,'slot_id':'candidate43','step':'ED6','seed':43,'reserved_utc':now(),
      'status':'reserved','optimizer_updates':0,'consumed_epochs':0,'config_sha256':cfg['sha256'],
      'source_bindings':sources,'feature_cache_binding':cache,'selection_lock_binding':ref(STAGE/'selection_lock.json'),
      'counts_even_before_first_checkpoint':True,'maximum_optimizer_updates':483620,'maximum_epochs':10,
      'automatic_retry':False,'counter_policy':'Atomic precharge1 before optimizer.step; failed attempts never refunded; no cold restart.'}
    budget['reserved_runs'].append(item);budget['status']='ED6_candidate43_reserved';budget['current_authorization_binding']=ref(STAGE/'authorization.json');update(BUDGET,budget)
    put(trial/'reservation.json',{'reservation_id':rid,'permanent_single_use':True,'item':item,'budget_at_reservation':ref(BUDGET),'budget_prefix_snapshot_binding':ref(snapshot)})
    reg={'version':'TCN-ED6-training-registration-1.0','step':'ED6','slot_id':'candidate43','registered_utc':now(),
      'selection_lock_binding':ref(STAGE/'selection_lock.json'),'authorization_binding':ref(STAGE/'authorization.json'),
      'protocol_binding':ref(ROOT/'configs/tcn_existing_data_optimization_v1.json'),'config_binding':cfg,
      'prepared_manifest_binding':selection['prepared_manifest_binding'],'feature_cache_binding':cache,'source_bindings':sources,
      'source_review_binding':ref(STAGE/'prelaunch_source_review.json'),'test_evidence_binding':ref(STAGE/'training_tests_evidence.json'),
      'preparation_receipt_binding':ref(base.ED4/'preparation_execution_receipt.json'),
      'capture_source_binding':ref(ROOT/'scripts/run_tcn_existing_data_ed6_capture.py'),
      'protected_inventory_binding':ref(STAGE/'protected_inventory.json'),'budget_binding_at_reservation':ref(BUDGET),
      'budget_prefix_snapshot_binding':ref(snapshot),'reservation_id':rid,'reservation_binding':ref(trial/'reservation.json'),
      'audit_path_repair_global_registration_binding':ref(GLOBAL),**extra}
    put(trial/'training_registration.json',reg)
    state=read(STATE);state.update(status='ED6_candidate43_reserved',current_trial='candidate43',current_registration=ref(trial/'training_registration.json'),candidate43_reserved=True);update(STATE,state)
    print(json.dumps({'candidate_registration':ref(trial/'training_registration.json'),'new_ML':0},ensure_ascii=True))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['tests_evidence','global','control_audit','candidate_audit','candidate_reserve']);v=a.parse_args()
    {'tests_evidence':test_evidence,'global':global_registration,'control_audit':lambda:audit_registration('control43'),
     'candidate_audit':lambda:audit_registration('candidate43'),'candidate_reserve':candidate_reserve}[v.phase]()
