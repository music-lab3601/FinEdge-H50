"""Single-round E board orchestration; effects injected for offline testing.

All gates run before opening SSH. Never attaches an old owner, terminates a
process, resets the PL or releases an SDK allocation, even on reporting errors.
"""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import time
from shared.model_paths import ROOT
from software.runtime.delivery import read_json, write_json, sha256
from experiments.reference.runtime.udw_e_release import load_permit, require_confirmation
from experiments.reference.runtime.udw_e_preflight import capture_boot
from experiments.reference.runtime.udw_e_candidate import CandidateRunner, CandidateSession
from experiments.reference.runtime.udw_lab import RpcTransport, digest
from experiments.reference.runtime.udw_reports import RunReports


BOOT_FILE='artifacts/fpga_user_ddr_window/boot_20261024_02/BOOT.bin'
OWNER_FILE='build_udw_lab/Release/finedge_udw_owner.exe'


class JournalRpc(RpcTransport):
    def __init__(self,owner_file,log):super().__init__(owner_file);self.log=log

    def call(self,op,**values):
        begin=time.perf_counter_ns();row=dict(op=op,begin_ns=begin)
        for key in ('offset','size','value','window_id'):
            if key in values:row[key]=values[key]
        if 'hex' in values:row['payload_sha256']=digest(bytes.fromhex(values['hex']))
        try:
            result=super().call(op,**values);row['ok']=True
            if op=='predict':row['prediction']=result
            if op=='status':row['regs']=result['regs']
            return result
        except Exception as error:row.update(ok=False,error=str(error));raise
        finally:
            row['end_ns']=time.perf_counter_ns()
            self.log.write(json.dumps(row)+'\n')


class RealEffects:
    is_simulated=False

    @contextmanager
    def connection(self):
        # Deferred imports/calls: denied gates do not even construct an SSH client.
        from experiments.reference.runtime.board import connect, command
        with connect() as client:
            yield lambda text:command(client,text)

    def start_owner(self,out,rpc_log):
        flags=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
        wall_started=time.time_ns()
        with (out/'daemon_console.log').open('ab') as console:
            daemon=subprocess.Popen([sys.executable,str(ROOT/'scripts/udw_owner_daemon.py'),str(out)],
                stdin=subprocess.DEVNULL,stdout=console,stderr=console,cwd=ROOT,creationflags=flags)
        # Persist identity immediately; never kill on timeout/report failure.
        write_json(out/'new_daemon_started.json',dict(daemon_pid=daemon.pid,owner_must_be_retained=True,
            spawn_started_unix_ns=wall_started,spawn_returned_unix_ns=time.time_ns()))
        deadline=time.monotonic()+45
        while not (out/'owner.json').exists():
            if daemon.poll() is not None:raise RuntimeError('Owner initialization refused; inspect new logs')
            if time.monotonic()>deadline:raise TimeoutError('Owner readiness timeout; no retry or termination')
            time.sleep(.1)
        ready=read_json(out/'owner_ready.json');owner=read_json(out/'owner.json')
        transport=JournalRpc(out/'owner.json',rpc_log)
        info=transport.call('info')
        return dict(transport=transport,info=info,ready=ready,
                    public_owner=dict(daemon_pid=owner['daemon_pid'],sdk_owner_pid=owner['sdk_owner_pid']))


def ensure_service(execute,label):
    def checked(command):
        r=execute(command)
        if r['exit_code']:raise RuntimeError('Service command failed: '+command)
        return r['stdout']
    query="ss -ltnp | grep ':9981 ' || true"
    if checked(query).strip():return dict(already_listening=True)
    # Do not background an entire `mkdir && command` AND-list: the wrapper
    # shell can retain the SSH output channel while waiting for the server.
    checked('mkdir -p /root/finedge_host_demo')
    pid=checked(f'nohup icraft-serve > /root/finedge_host_demo/e_serve_{label}.log 2>&1 < /dev/null & echo $!').strip()
    if not pid.isdigit() or int(pid)<=0:raise RuntimeError('Service launch returned no PID; no owner started')
    for _ in range(20):
        if checked(query).strip():return dict(started=True,service_pid=int(pid))
        time.sleep(.25) # service readiness only, never DDR visibility evidence
    raise TimeoutError('icraft-serve readiness timeout')


def validate_owner_ready(opened,expected_versions):
    r=opened['ready'];info=opened['info']
    if (not r.get('ok') or r.get('startup_gate')!='fresh_observed_before_Session_apply_and_malloc'
        or r.get('size')!=2*1024*1024 or info.get('size')!=2*1024*1024 or info.get('used')!=0
        or info.get('base')!=r.get('base')):
        raise ValueError('New frozen owner startup/allocation evidence differs')
    if r.get('versions')!=expected_versions or info.get('versions')!=expected_versions:
        raise ValueError('Device/core version differs from reviewed baseline')
    if not any('HardOp' in b['op_type'] and 'ZG330' in b['backend'] for b in r.get('bindings',[])):
        raise ValueError('No actual ZG330 HardOp binding')
    for offset in (4,8,12,0x70,0x60,0x64,0x68,0x6c,0x54,0xc0,0x8c):
        expected=0x55445731 if offset==0xc0 else 2 if offset==0x8c else 0
        if r.get('startup_registers',{}).get(str(offset))!=expected:
            raise ValueError('Incomplete/dirty pre-allocation startup capture')


def run_board_round(*,arm,label,bundle,review_path,confirmation,previous_result_path,out,effects=None):
    if arm not in ('S','H') or not re.fullmatch(r'[A-Za-z0-9_]{1,64}',label):raise ValueError('Single S/H arm and safe label required')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    reports=RunReports(out);effects=effects or RealEffects();begin=time.perf_counter_ns()
    report=dict(status='preparing',arm=arm,label=label,board_accessed=False,
        transport_substitute=effects.is_simulated,old_owner_attached=False,old_owner_terminated=False,
        owner_must_be_retained=True,strict_visibility_proven=False,performance_acceptance=False,
        input_preparation=bundle['provenance'])
    session=None;runner=None
    def save():reports.save(report)
    save()
    with (out/'rpc_timing.jsonl').open('a',encoding='utf-8',buffering=1) as rpc_log, (out/'driver_journal.jsonl').open('a',encoding='utf-8',buffering=1) as driver_log:
        try:
            permit=load_permit(review_path,arm)
            previous=require_confirmation(confirmation,permit,arm=arm,label=label,previous_result_path=previous_result_path)
            old=read_json(previous);oldboot=old.get('preflight',old.get('board_preflight',{})).get('boot_id')
            if not oldboot:raise ValueError('Previous run has no saved boot identity')
            expected=dict(BOOT_sha256=permit.reviewed_files[BOOT_FILE],owner_sha256=permit.reviewed_files[OWNER_FILE],
                ZG_json_sha256=permit.reviewed_files['artifacts/icraft/tcn_ed4_ZG.json'],
                ZG_raw_sha256=permit.reviewed_files['artifacts/icraft/tcn_ed4_ZG.raw'],SDK_version='3.36.1')
            for name,h in bundle['provenance']['input_hashes'].items():
                if sha256(ROOT/name)!=h:raise ValueError('Prepared input changed: '+name)
            authority=read_json(permit.decision_path)
            report.update(execution_record_sha256=permit.decision_sha256,
                execution_authority=authority.get('authority_kind','independent_review'),
                review_deferred=authority.get('review_deferred',False),
                independent_review_passed=authority.get('authority_kind')!='direct_user_deferred_review',
                confirmation=confirmation,previous_result_sha256=sha256(previous))
            save()
            report['connection_attempted']=True;report['board_accessed']=not effects.is_simulated
            with effects.connection() as execute:
                preflight=capture_boot(execute,label=label,previous_boot_ids=[oldboot],expected_BOOT_sha256=expected['BOOT_sha256'])
                report['preflight']=preflight;save()
                report['service']=ensure_service(execute,label);save()
                # After preflight, before ANY new owner/model/arena allocation.
                permit.validate(arm)
                context=dict(expected,BOOT_sha256=preflight['actual_BOOT_sha256'],SDK_version=preflight['SDK_version'],
                    boot_id=preflight['boot_id'],previous_boot_id=oldboot,physical_stop_confirmed=True,
                    offline_only=False,board_permit=permit,transport_substitute=effects.is_simulated)
                def owner_factory():
                    nonlocal session
                    report['owner_creation_attempted']=True;save()
                    opened=effects.start_owner(out,rpc_log)
                    report['public_owner']=opened['public_owner'];report['owner_ready']=opened['ready'];save()
                    validate_owner_ready(opened,read_json(permit.decision_path)['expected_device_versions'])
                    def journal(row):driver_log.write(json.dumps(row)+'\n')
                    session=CandidateSession(opened['transport'],opened['info']['base'],opened['info']['size'],
                        run_id=time.time_ns(),generation=time.time_ns()//100,journal=journal)
                    return session
                def observe(stage,state):
                    report['phase']=stage;report['runner']=state;save()
                runner=CandidateRunner(bundle['normalization'],arm,observer=observe)
                report['runner']=runner.run(bundle['batches'],bundle['references'],context=context,
                    expected_context=expected,session_factory=owner_factory)
                if report['runner']['status'] not in ('bounded_board_candidate_verified','orchestration_substitute_passed'):
                    raise RuntimeError(report['runner'].get('error','Candidate failed'))
                permit.validate(arm)
                for name,h in bundle['provenance']['input_hashes'].items():
                    if sha256(ROOT/name)!=h:raise ValueError('Input changed during board round')
                report['status']='orchestration_substitute_passed' if effects.is_simulated else 'bounded_E_board_round_verified'
        except Exception as error:
            report.update(status='failed',error=f'{type(error).__name__}: {error}',owner_must_be_retained=True)
            if runner is not None:
                runner.fail(error);report['runner']=runner.report
        finally:
            # Save only already-captured host bytes. No extra SDK call on failure.
            if session is not None:
                regions=[]
                for i,(offset,image) in enumerate(session.expected_regions):
                    path=out/f'region_{i:03d}_captured.bin';path.write_bytes(image)
                    regions.append(dict(offset=offset,bytes=len(image),sha256=sha256(path),file=path.name))
                report['captured_regions']=regions
            report['orchestration_total_ns']=time.perf_counter_ns()-begin
            report['total_including_input_preparation_ns']=report['orchestration_total_ns']+bundle['provenance']['preparation_ns']
            reports.save(report,final=True)
    return report
