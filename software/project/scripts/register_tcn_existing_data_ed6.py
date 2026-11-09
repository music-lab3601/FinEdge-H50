"""Root-owned ED6 frozen paired protocol and bounded metadata registration."""
from pathlib import Path
import argparse
import copy
import datetime as dt
from fractions import Fraction
import hashlib
import json
import os
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'artifacts/tcn_existing_data_optimization_2026-10-17'
STAGE=CAMPAIGN/'step06_replication'
ED4=CAMPAIGN/'step04_features'
ED5=CAMPAIGN/'step05_strategy'
BUDGET=CAMPAIGN/'budget.json'
STATE=ROOT/'state/tcn_existing_data_optimization_20261017.json'
SOURCE_NAMES=['scripts/tcn_existing_data_ed6_train.py','scripts/tcn_existing_data_ed6_data.py',
              'scripts/tcn_existing_data_ed4_features.py','scripts/tcn_research_data_p2.py',
              'scripts/tcn_research_evaluate_p2.py','scripts/tcn_research_p5_model.py','scripts/tcn_candidate.py']
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text('utf-8-sig'))
def ref(p):
    p=p if p.is_absolute()else ROOT/p
    with p.open('rb')as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':digest}
def bound(r):
    p=Path(r['path']);p=p if p.is_absolute()else ROOT/p
    actual=ref(p);assert actual['sha256']==r['sha256'] and ('bytes'not in r or actual['bytes']==r['bytes']),str(p)
    return read(p)
def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n')as f:
        json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def update(p,v):
    temp=p.with_name(p.name+'.ed6.tmp');put(temp,v);temp.replace(p)
def rank(m):
    n=sum(map(sum,m));pred=[sum(r[i]for r in m)for i in range(3)];support=list(map(sum,m))
    return Fraction(sum(m[i][i]for i in range(3)),n),sum((Fraction(2*m[i][i],support[i]+pred[i])if support[i]+pred[i]else Fraction(0)for i in range(3)),Fraction(0))/3

def init():
    assert read(STATE)['status']=='ED5_completed_ED6_pending'
    assert not (STAGE/'authorization.json').exists()
    d4=read(ED4/'completion_acceptance.json');d5=read(ED5/'completion_acceptance.json')
    assert d4['passed'] and d5['passed'] and d5['retained_original_argmax_family']=='ED4_feature42_lr0.001'
    assert rank(d4['candidate_original_argmax_DEV_metrics']['confusion_true_rows_predicted_columns'])>rank(d5['candidate_original_argmax_DEV_metrics']['confusion_true_rows_predicted_columns'])
    assert ref(BUDGET)['sha256']==d5['budget_final']['sha256']
    budget=read(BUDGET);assert len(budget['reserved_runs'])==2 and all(r['status']=='completed'for r in budget['reserved_runs'])
    assert budget['consumed_epochs']==11 and budget['optimizer_updates']==531982
    before=STAGE/'before';before.mkdir(parents=True,exist_ok=True)
    for p,name in [(ROOT/'AGENTS.md','AGENTS.md'),(ROOT/'PROGRESS.md','PROGRESS.md'),
                   (STATE,'state.json'),(ROOT/'state/tcn_software_plan_v25.json','plan_v25.json'),(BUDGET,'budget_before_ED6.json')]:
        with (before/name).open('xb')as f:f.write(p.read_bytes())
    a={'version':'TCN-ED6-authorization-1.0','registered_utc':now(),
       'human_authorization':{'source':'direct_current_human_user_message','quote':'帮我完成第6步','step':'ED6'},
       'training_authorized':True,'maximum_new_training_runs':2,'slots':['control43','candidate43'],
       'replication_seed':43,'representative_seed':42,'maximum_complete_epochs_per_run':10,
       'patience':3,'maximum_optimizer_updates_per_run':483620,'maximum_new_optimizer_updates':967240,
       'sequential_training_workers':1,'no_new_downloads':True,'next_step_authorized':False,
       'new_independent_test':False,'excluded_dates':['2025-12-17','2025-12-18','2025-12-19']}
    put(STAGE/'authorization.json',a)
    parent=read(ED4/'config.json')
    for slot in ('control43','candidate43'):
        cfg=copy.deepcopy(parent);cfg.update(version='TCN-ED6-paired-config-1.0',slot_id=slot,seed=43)
        cfg['feature_family']=('original_physical_prices_raw_quantities'if slot=='control43'else parent['feature_family'])
        cfg['description']=('Fresh seed43 paired replication, original argmax; raw physical prices/original P2 normalization'if slot=='control43'else 'Fresh seed43 replication of the locked ED4 relative-midprice family and ED4 TRAIN normalization')+'; same L100/H50/all original anchors/WN32/lr0.001/batch256/max10/pat3; no bias, old checkpoint initialization or new data.'
        put(STAGE/'trials'/slot/'config.json',cfg)
    old=read(ED5/'protected_inventory.json')
    refs={str((ROOT/Path(r['path'])).resolve()):r for r in old['files']}
    for p in [*ED5.rglob('*'),ROOT/'state/tcn_software_plan_v25.json',ROOT/d5['report']['path'],
              ROOT/'scripts/tcn_existing_data_ed5_train.py',ROOT/'scripts/run_tcn_existing_data_ed5_capture.py',
              ROOT/'scripts/run_tcn_existing_data_ed5_postrun_capture.py',ROOT/'scripts/finalize_tcn_existing_data_ed5.py']:
        if p.is_file() and p.stat().st_size<20*1024**2 and p.suffix in {'.json','.py','.txt','.pt','.log'}:
            refs[str(p.resolve())]=ref(p)
    for r in refs.values():
        p=Path(r['path']);actual=ref(p);assert actual['sha256']==r['sha256'] and actual['bytes']==r['bytes']
    put(STAGE/'protected_inventory.json',{'version':'TCN-ED6-protected-inventory-1.0','files':list(refs.values()),
        'protected_files':len(refs),'excluded_mutable_new_campaign_budget':ref(BUDGET),
        'mutable_budget_authorization':'Only current control43/candidate43 items and global counters; all completed prior full items immutable'})
    put(STAGE/'candidate_selection_evidence.json',{'version':'TCN-ED6-pre-seed43-candidate-choice-1.0',
        'registered_utc':now(),'parent_ED4_completion_binding':ref(ED4/'completion_acceptance.json'),
        'ED5_completion_binding':ref(ED5/'completion_acceptance.json'),'candidate_family':'ED4_feature42_lr0.001',
        'branch':'trained_candidate','decision':'original_argmax','stationary_logit_bias':0,
        'seed42_control_metrics':d4['baseline_original_argmax_DEV_metrics'],
        'seed42_candidate_metrics':d4['candidate_original_argmax_DEV_metrics'],
        'seed42_ED5_metrics_description_only':d5['candidate_original_argmax_DEV_metrics'],
        'seed42_candidate_checkpoint':d4['selected_checkpoint'],'seed42_candidate_normalization':d4['normalization'],
        'reason':'ED4 exceeds both original WN42 and ED5 under same complete DEV exact Accuracy/F1; freeze before any seed43 result.',
        'common_DEV_samples':6612655,'seed43_scores_observed':False,'new_ML':0})
    s=read(STATE);s.update(status='ED6_preparing',current_step=6,current_phase_id='ED6',training_authorized_now=True,
        previous_completion=s['completion'],current_stage=str(STAGE),authorization=ref(STAGE/'authorization.json'),
        next_step=7,next_phase_id='ED7',automatic_next_action=False)
    update(STATE,s)
    heading=('## 2026-10-18 新七步ED6已授权，固定候选与配对入口准备中\n\n'
      '用户“帮我完成第6步”授权control43→candidate43两组串行冷seed43复核。唯一候选由已完成ED5选择为ED4相对价格/lr.001/原argmax家族，代表固定seed42，不择高seed；新43任何分数前须冻结selection_lock。control使用原物理价/旧TRAINnorm，candidate用ED4相对价/cache/norm，其他WN/L100/H50/同全部anchors/max10/pat3相同，平Recall20%仅观察。\n'
      '预算当前旧两slot完整11轮/531982更新保持，未新预约或训练。将先准备原physical价D缓存不fitnorm，以原21黄金X字节核对；现有E7日只此准备有限读取，17—19封存。当前仅元信息/配置登记与旧证据保护，不代表模型或稳定性通过。旧P3/P5关闭，源control/覆盖未知、flag64与历史quality=false保持。ED7未授权。\n\n')
    for name in ('AGENTS.md','PROGRESS.md'):
        p=ROOT/name;p.write_text(heading+p.read_text('utf-8-sig'),'utf-8')
    print(json.dumps({'status':'ED6_preparing','protected_files':len(refs),'new_training_reservations':0,'new_ML':0},ensure_ascii=True))

def tested_sources():
    review=read(STAGE/'prelaunch_source_review.json');assert review['passed']
    sources={name:ref(ROOT/name)['sha256']for name in SOURCE_NAMES}
    assert review['source_bindings']==sources
    assert review['capture_source_binding']['sha256']==ref(ROOT/'scripts/run_tcn_existing_data_ed6_capture.py')['sha256']
    for phase in ('cache_tests','training_tests'):
        actual=read(STAGE/(phase+'_execution_receipt.json'));ev=read(STAGE/(phase+'_evidence.json'))
        assert actual['phase']==phase and actual['exit_code']==0 and actual['entry_unchanged']
        assert ev['passed'] and ev['actual_exit_code']==0 and ev['tests_run']>0
        assert actual['entry_before']==actual['entry_after']
        assert actual['entry_before']['sha256']==review['test_source_bindings'][phase]['sha256']
        bound(ev['actual_receipt'])
    return sources

def preparation():
    sources=tested_sources();old=read(ED4/'preparation_registration.json')
    assert not (STAGE/'prepared_control').exists()
    p=read(ROOT/'configs/tcn_existing_data_optimization_v1.json')
    reg={'version':'TCN-ED6-control-cache-registration-1.0','step':'ED6','registered_utc':now(),
         'authorization_binding':ref(STAGE/'authorization.json'),
         'protocol_binding':ref(ROOT/'configs/tcn_existing_data_optimization_v1.json'),
         'prepared_manifest_binding':old['prepared_manifest_binding'],
         'normalization_binding':p['data']['baseline_normalization'],
         'source_bindings':sources,'source_review_binding':ref(STAGE/'prelaunch_source_review.json'),
         'test_evidence_binding':ref(STAGE/'cache_tests_evidence.json'),
         'capture_source_binding':ref(ROOT/'scripts/run_tcn_existing_data_ed6_capture.py'),
         'protected_inventory_binding':ref(STAGE/'protected_inventory.json'),
         'candidate_choice_binding':ref(STAGE/'candidate_selection_evidence.json'),
         'output_directory':str(STAGE/'prepared_control'),'training_runs':0,'model_forwards':0,
         'normalization_refit':False,'new_labels':False,'raw_source_dates':['2025-12-08','2025-12-09','2025-12-10','2025-12-11','2025-12-12','2025-12-15','2025-12-16'],
         'excluded_dates':['2025-12-17','2025-12-18','2025-12-19']}
    put(STAGE/'preparation_registration.json',reg)
    print(json.dumps({'preparation_registration':ref(STAGE/'preparation_registration.json'),'formal_training_started':False},ensure_ascii=True))

def lock():
    sources=tested_sources();prep=read(STAGE/'preparation_execution_receipt.json')
    assert prep['exit_code']==0 and prep['entry_unchanged'] and prep['phase']=='preparation'
    cache=bound(prep['outputs']['feature_cache_manifest'])
    assert cache['complete'] and cache['model_forwards']==0 and cache['training_runs']==0
    assert read(STATE)['status']=='ED6_preparing'
    assert ref(BUDGET)['sha256']==ref(STAGE/'before/budget_before_ED6.json')['sha256']
    choices=read(STAGE/'candidate_selection_evidence.json')
    p2=bound(read(ROOT/'configs/tcn_existing_data_optimization_v1.json')['data']['source_manifest'])
    original_norm=read(ROOT/'configs/tcn_existing_data_optimization_v1.json')['data']['baseline_normalization']
    relative=read(ED4/'training_registration.json')['feature_cache_binding']
    control=prep['outputs']['feature_cache_manifest']
    bounds=[{'date':r['date'],'role':r['role'],'endpoint_start':r['endpoint_start'],
             'endpoint_stop_exclusive':r['endpoint_stop_exclusive'],'window_count':r['window_count'],
             'label_sha256':p2['day_label_reports'][r['date']]['label_sha256'],
             'anchor_stream_sha256':p2['day_label_reports'][r['date']]['endpoint_stream_sha256']}for r in p2['ranges']]
    reg={'version':'TCN-ED6-selection-lock-1.0','campaign_id':'TCN_EDO_20261017','locked_utc':now(),
         'branch':'trained_candidate','candidate_family':'ED4_feature42_lr0.001',
         'replication_seed':43,'representative_seed':42,'decision':'original_argmax','stationary_logit_bias':0,
         'config_bindings':{slot:ref(STAGE/'trials'/slot/'config.json')for slot in ('control43','candidate43')},
         'source_bindings':sources,'prepared_manifest_binding':read(ED4/'training_registration.json')['prepared_manifest_binding'],
         'common_DEV_samples':6612655,'common_TRAIN_samples':12380446,'window_length':100,'horizon':50,
         'feature_cache_bindings':{'control43':control,'candidate43':relative},
         'normalization_bindings':{'control43':original_norm,'candidate43':read(ED4/'completion_acceptance.json')['normalization']},
         'frozen_parent_completion_binding':ref(ED4/'completion_acceptance.json'),
         'candidate_choice_binding':ref(STAGE/'candidate_selection_evidence.json'),
         'prior_budget_snapshot_binding':ref(STAGE/'before/budget_before_ED6.json'),
         'seed42_control_metrics':choices['seed42_control_metrics'],'seed42_candidate_metrics':choices['seed42_candidate_metrics'],
         'seed42_representative_checkpoint':choices['seed42_candidate_checkpoint'],
         'seed42_baseline_checkpoint':read(ROOT/'configs/tcn_existing_data_optimization_v1.json')['baseline']['checkpoint'],
         'same_complete_anchor_bounds':bounds,'new_seed43_scores_observed':False,'new_seed43_models_constructed':0,
         'paired_adoption_rule':'candidate-minus-control exact Accuracy>=0 for both predeclared seeds and >0 for at least one; no extra F1/Recall/day veto',
         'checkpoint_selection':'exact Accuracy then exact Macro-F1; full tie earlier complete epoch',
         'no_candidate_branch_bias_or_settings_switch_after_any43_result':True,'no_new_independent_test':True,
         'next_step_authorized':False}
    put(STAGE/'selection_lock.json',reg)
    s=read(STATE);s.update(status='ED6_locked_ready_for_control',selection_lock=ref(STAGE/'selection_lock.json'),
                          frozen_candidate_family=reg['candidate_family'],representative_seed=42,replication_seed=43)
    update(STATE,s)
    print(json.dumps({'selection_lock':ref(STAGE/'selection_lock.json'),'new_seed43_models':0,'formal_training_started':False},ensure_ascii=True))

def test_evidence():
    review=read(STAGE/'prelaunch_source_review.json');assert review['passed']
    for phase in ('cache_tests','training_tests'):
        path=STAGE/(phase+'_execution_receipt.json');receipt=read(path)
        assert receipt['exit_code']==0 and receipt['entry_unchanged']
        assert receipt['entry_before']==receipt['entry_after']
        assert receipt['entry_before']['sha256']==review['test_source_bindings'][phase]['sha256']
        err=Path(receipt['stderr']['path']);assert ref(err)['sha256']==receipt['stderr']['sha256']
        result=err.read_text('utf-8');match=re.search(r'Ran (\d+) tests? in ',result)
        assert match and re.search(r'\nOK\s*$',result)
        put(STAGE/(phase+'_evidence.json'),{'passed':True,'actual_exit_code':0,
            'tests_run':int(match.group(1)),'actual_receipt':ref(path),
            'test_source_binding':receipt['entry_before'],'scope':'stdlib synthetic only; no actual ML or market payload'})
    tested_sources();print(json.dumps({'actual_test_evidence':'passed'}))

def reserve(slot):
    assert slot in ('control43','candidate43')
    sources=tested_sources();selection=read(STAGE/'selection_lock.json')
    assert selection['source_bindings']==sources
    trial=STAGE/'trials'/slot;assert not (trial/'training_registration.json').exists()
    budget=read(BUDGET);prior_count=2 if slot=='control43'else 3
    assert len(budget['reserved_runs'])==prior_count and all(r['status']=='completed'for r in budget['reserved_runs'])
    old=read(STAGE/'before/budget_before_ED6.json');assert budget['reserved_runs'][:2]==old['reserved_runs']
    extra={}
    if slot=='control43':
        assert read(STATE)['status']=='ED6_locked_ready_for_control'
        assert ref(BUDGET)['sha256']==ref(STAGE/'before/budget_before_ED6.json')['sha256']
    else:
        ctrl=STAGE/'trials/control43'
        paths={'control_training_receipt_binding':ctrl/'training_execution_receipt.json',
          'control_acceptance_binding':ctrl/'run/training_acceptance.json',
          'control_audit_receipt_binding':ctrl/'audit_execution_receipt.json',
          'control_audit_binding':ctrl/'readonly_audit.json',
          'control_checkpoint_receipt_binding':ctrl/'checkpoint_execution_receipt.json',
          'control_checkpoint_validation_binding':ctrl/'checkpoint_validation.json',
          'control_initialization_binding':ctrl/'run/initialization.json'}
        extra={k:ref(v)for k,v in paths.items()}
        sys.path.insert(0,str(ROOT/'scripts'))
        import tcn_existing_data_ed6_train as entry
        entry.candidate_order_gate(extra,ref(STAGE/'selection_lock.json'))
        ctrl_accept=read(paths['control_acceptance_binding'])
        assert ref(BUDGET)['sha256']==ctrl_accept['budget_completion_snapshot_binding']['sha256']
    snapshot=trial/'budget_prefix_snapshot.json'
    with snapshot.open('xb')as f:f.write(BUDGET.read_bytes());f.flush();os.fsync(f.fileno())
    cfg=ref(trial/'config.json');assert cfg==selection['config_bindings'][slot]
    cache=selection['feature_cache_bindings'][slot];bound(cache)
    rid='TCN_EDO_20261017_ED6_'+slot
    item={'reservation_id':rid,'slot_id':slot,'step':'ED6','seed':43,'reserved_utc':now(),
          'status':'reserved','optimizer_updates':0,'consumed_epochs':0,'config_sha256':cfg['sha256'],
          'source_bindings':sources,'feature_cache_binding':cache,'selection_lock_binding':ref(STAGE/'selection_lock.json'),
          'counts_even_before_first_checkpoint':True,'maximum_optimizer_updates':483620,'maximum_epochs':10,
          'automatic_retry':False,'counter_policy':'Atomic precharge1 before optimizer.step; failed attempts never refunded; no cold restart.'}
    budget['reserved_runs'].append(item);budget['status']='ED6_'+slot+'_reserved'
    budget['current_authorization_binding']=ref(STAGE/'authorization.json')
    update(BUDGET,budget)
    put(trial/'reservation.json',{'reservation_id':rid,'permanent_single_use':True,'item':item,
                                 'budget_at_reservation':ref(BUDGET),'budget_prefix_snapshot_binding':ref(snapshot)})
    reg={'version':'TCN-ED6-training-registration-1.0','step':'ED6','slot_id':slot,'registered_utc':now(),
         'selection_lock_binding':ref(STAGE/'selection_lock.json'),'authorization_binding':ref(STAGE/'authorization.json'),
         'protocol_binding':ref(ROOT/'configs/tcn_existing_data_optimization_v1.json'),
         'config_binding':cfg,'prepared_manifest_binding':selection['prepared_manifest_binding'],
         'feature_cache_binding':cache,'source_bindings':sources,'source_review_binding':ref(STAGE/'prelaunch_source_review.json'),
         'test_evidence_binding':ref(STAGE/'training_tests_evidence.json'),
         'preparation_receipt_binding':ref(STAGE/'preparation_execution_receipt.json'if slot=='control43'else ED4/'preparation_execution_receipt.json'),
         'capture_source_binding':ref(ROOT/'scripts/run_tcn_existing_data_ed6_capture.py'),
         'protected_inventory_binding':ref(STAGE/'protected_inventory.json'),'budget_binding_at_reservation':ref(BUDGET),
         'budget_prefix_snapshot_binding':ref(snapshot),'reservation_id':rid,'reservation_binding':ref(trial/'reservation.json'),**extra}
    put(trial/'training_registration.json',reg)
    s=read(STATE);s.update(status='ED6_'+slot+'_reserved',current_trial=slot,
                          current_registration=ref(trial/'training_registration.json'),selection_lock=ref(STAGE/'selection_lock.json'))
    update(STATE,s)
    print(json.dumps({'registration':ref(trial/'training_registration.json'),'slot':slot,'new_ML':0},ensure_ascii=True))

def handoff_running():
    a=read(STAGE/'active_execution.json');assert a['exec_session']==92822 and a['actual_worker_pid']==14400
    heading=('## 2026-10-18 新七步ED6原始种子43已实际启动，整步未完成\n\n'
      '用户“帮我完成第6步”正在执行。实际27项NumPy缓存合成测试、39项stdlib/mock训练测试exit0；独立50项源码静态检查通过，两类分列，原scope文字补充见test_scope_clarification.json。原价cache一次准备已exit0，18994144行/3039063040字节，21原P2黄金X逐字节一致、原7源和7标签SHA保持，0模型/0fit。\n'
      '共同selection_lock bf498dd21c0cd92f27473d97013332f1d6b41f8afd60c3efbdecc0ea91152126在任何43模型/分数前一次冻结ED4相对价/lr.001/原argmax/代表42；两cache/norm/config/七源同锁。control43永久预约且admit实际exit0后，唯一真实训练exec92822启动，capture16884/launcher49084/实际worker14400分列；root独占原session后续poll，其他agent不poll或另开训练。当前仅第1轮TRAIN、无完整43DEV成绩，不代表整步完成。\n'
      'control真实terminal、两新CP CPU-only容器验收、D-only全epoch缓存审计实际成功后，才允许candidate43永久预约及同seed43冷启动；最多两新run/max10/pat3，各483620update，旧两完整item11epoch531982update保持。每组terminal保存不可变budget_at_completion_snapshot，后续不将旧live budget SHA正常变化当篡改。\n'
      '稳定DEV采用门仅两预定seed的candidate-control精确Accuracy均>=0且至少一正；43Acc tie即使F1/Recall下降仍按预登记通过，代表始终42，43负回退原WN42且ED4另存不稳定选项。不加偏置/新候选/新数据/17—19评价，不重开旧P3/P5，历史quality=false/control覆盖未知/flag64保持。ED7未授权。\n\n')
    before=STAGE/'handoff_control43_started';before.mkdir(exist_ok=False)
    for name in ('AGENTS.md','PROGRESS.md'):
        p=ROOT/name;(before/name).write_bytes(p.read_bytes());p.write_text(heading+p.read_text('utf-8-sig'),'utf-8')
    print(json.dumps({'handoff':'control43_started','session':92822,'automatic_next_step':False}))

def postrun_tests_evidence():
    review=read(STAGE/'postrun_source_review.json');assert review['passed']
    for r in review['source_bindings'].values():
        assert ref(ROOT/r['path'])['sha256']==r['sha256']
    path=STAGE/'postrun_helpers_tests_execution_receipt.json';receipt=read(path)
    assert receipt['phase']=='postrun_helpers_tests' and receipt['exit_code']==0 and receipt['entry_unchanged']
    assert receipt['sources_before']==receipt['sources_after']
    err=Path(receipt['stderr']['path']);assert ref(err)['sha256']==receipt['stderr']['sha256']
    msg=err.read_text('utf-8');match=re.search(r'Ran (\d+) tests? in ',msg)
    assert match and int(match.group(1))==32 and re.search(r'\nOK\s*$',msg)
    put(STAGE/'postrun_helpers_tests_evidence.json',{'passed':True,'actual_exit_code':0,
        'separate_synthetic_test_count':32,'no_real_data_or_ML':True,'actual_receipt':ref(path),
        'sources_before':receipt['sources_before'],'sources_after':receipt['sources_after'],
        'test_source':receipt['entry_before'],'scope':'32 independent NumPy/stdlib synthetic helper tests; no real payload or model operations'})
    assert read(STAGE/'finalizer_tests_evidence.json')['passed']
    print(json.dumps({'postrun_helpers_actual_tests':32,'finalizer_actual_tests':17,'passed':True}))

def checkpoint_register(slot):
    assert read(STAGE/'postrun_helpers_tests_evidence.json')['passed']
    review=read(STAGE/'postrun_source_review.json');assert review['passed']
    for r in review['source_bindings'].values():assert ref(ROOT/r['path'])['sha256']==r['sha256']
    trial=STAGE/'trials'/slot;receipt=read(trial/'training_execution_receipt.json')
    assert receipt['exit_code']==0 and receipt['entry_unchanged'] and receipt['phase']=='training' and receipt['slot_id']==slot
    acceptance=bound(receipt['outputs']['training_acceptance']);assert acceptance['status']=='completed'
    assert acceptance['selection_lock_binding']==ref(STAGE/'selection_lock.json')
    assert ref(trial/'training_registration.json')==receipt['input']
    reg={'version':'TCN-ED6-bounded-checkpoint-registration-1.0','slot_id':slot,'registered_utc':now(),
      'source_binding':ref(STAGE/'checkpoint_validation.py'),
      'training_receipt_binding':ref(trial/'training_execution_receipt.json'),
      'training_registration_binding':ref(trial/'training_registration.json'),
      'checkpoint_bindings':acceptance['checkpoint_bindings'],
      'limits':{'new_checkpoint_CPU_loads':2,'models':0,'optimizers':0,'forwards':0,'backwards':0,'optimizer_updates':0},
      'postrun_source_review_binding':ref(STAGE/'postrun_source_review.json'),
      'actual_helper_tests_binding':ref(STAGE/'postrun_helpers_tests_evidence.json')}
    put(trial/'checkpoint_validation_registration.json',reg)
    print(json.dumps({'checkpoint_registration':ref(trial/'checkpoint_validation_registration.json'),'slot':slot},ensure_ascii=True))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['init','test_evidence','preparation','lock','control43','candidate43','handoff_running','postrun_tests_evidence','checkpoint_control43','checkpoint_candidate43']);args=a.parse_args()
    {'init':init,'test_evidence':test_evidence,'preparation':preparation,'lock':lock,
     'control43':lambda:reserve('control43'),'candidate43':lambda:reserve('candidate43'),'handoff_running':handoff_running,
     'postrun_tests_evidence':postrun_tests_evidence,'checkpoint_control43':lambda:checkpoint_register('control43'),
     'checkpoint_candidate43':lambda:checkpoint_register('candidate43')}[args.phase]()
