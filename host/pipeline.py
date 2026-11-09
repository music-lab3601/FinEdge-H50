from shared.paths import environment
"""Exclusive GUI runner for the accepted queue32 hardware. UART is management;
raw market data and FPGA results use the physical PL network. No CPU model.
"""
from pathlib import Path
import argparse, ast, json, os, shlex, shutil, subprocess, sys, tarfile, time, uuid
import numpy as np
from shared.paths import ROOT as P, work_path, load_settings, resolve_path
from shared.config import board_config
U=Path(__file__).resolve().parent;G=work_path('board');Q=work_path('runtime')
SENDER=U/'market_source/sender.py'
sys.dont_write_bytecode=True
from shared.io import read, sha
from host.telemetry.storage import atomic_json
from shared.resources import Queue, ResourceLock
from board.management.serial_console import Console
from board.management.console_login import ensure_shell
from host.telemetry.telemetry import ROW
from host.telemetry.dispatch_timing import checked_speed, wall_duration_ns
ANALYZER=P/'experiments/analysis/analyze_case.py'

def checked(c,command,timeout=30):
    v=c.run(command,timeout)
    if v['exit_code']:raise RuntimeError(v['stdout'][-1600:])
    return v['stdout']

def remote_json(c,path):
    code='import pathlib;p=pathlib.Path('+repr(path)+');print(p.read_text() if p.exists() else "null")'
    return json.loads(checked(c,'python3 -c '+shlex.quote(code)))

def qualify(request):
    p=Path(request['candidate_manifest'])
    if sha(p)!=request['candidate_manifest_sha256']:raise ValueError('GUI run manifest changed')
    for name,digest in read(p)['files'].items():
        if sha(name)!=digest:raise ValueError('GUI run dependency changed: '+name)
    if read(p)['loaded_firmware']!=read(G/'state/loaded_firmware.json'):raise ValueError('Recorded firmware boot changed')

def progress(d,stage,message,**kw):
    atomic_json(d/'ui_progress.json',dict(stage=stage,message=message,updated=time.time(),**kw))

def execute(path):
    m=read(path);d=Path(m['directory']);remote=m['remote'];c=None;sent=None
    result=dict(status='failed',owner_started=False,remote=remote,automatic_retry=False)
    qualify(dict(candidate_manifest=str(path),candidate_manifest_sha256=sha(path)))
    end=time.monotonic()+2
    while not any(r['state']=='active' and r.get('child_pid')==os.getpid() and r['candidate_manifest']==str(path) for r in Queue()._load()['requests']):
        if time.monotonic()>end:raise RuntimeError('Not the exclusive G board queue child')
        time.sleep(.01)
    try:
        c=Console(load_settings()['board']['serial_port']);ensure_shell(c);boot=checked(c,'cat /proc/sys/kernel/random/boot_id').strip()
        if boot!=m['loaded_firmware']['boot_id']:raise ValueError('Board boot identity changed')
        result['boot_id']=boot;progress(d,'uploading','正在通过PS串口准备板端程序；行情使用PL千兆网口。')
        result['upload']=c.put(m['archive'],remote+'.tar.gz',timeout=180)
        checked(c,'mkdir -p /root/finedge_integrated/runs; mkdir '+shlex.quote(remote))
        checked(c,'tar -xzf '+shlex.quote(remote+'.tar.gz')+' -C '+shlex.quote(remote))
        progress(d,'compiling','正在准备板端原生调度和显示采样。')
        text=checked(c,'gcc -O3 -shared -fPIC -Wall -Wextra -pthread -o '+shlex.quote(remote+'/libintegrated.so')+' '+shlex.quote(remote+'/integrated_native.c'),60)
        (d/'compile.log').write_text(text,encoding='utf-8')
        result['native_library_sha256']=checked(c,'sha256sum '+shlex.quote(remote+'/libintegrated.so')).split()[0]
        argv=['python3','-u',remote+'/integrated_owner.py','--directory',remote,'--session',str(m['session']),'--first',str(m['fixture']['first_raw_ordinal']),'--last',str(m['fixture']['last_raw_ordinal']),'--audit-inputs','0','--slots',str(m['slots']),'--max-age-us',str(m['max_age_us']),'--h50-records',str(m['h50_records']),'--trace-capacity',str(m['fixture']['opportunities']),'--runtime-ms',str(wall_duration_ns(m['fixture']['duration_ns'],m['speed'])//1_000_000+120000)]
        checked(c,'cd '+shlex.quote(remote)+' && nohup '+shlex.join(argv)+' >owner.log 2>&1 < /dev/null &')
        result['owner_started']=True;atomic_json(d/'result.json',result)
        progress(d,'qualifying','正在核验常驻模型、32槽映射与不同输入切换；随后初始化订单状态。')
        end=time.monotonic()+240
        while True:
            ready=remote_json(c,remote+'/ready.json')
            if ready is not None:atomic_json(d/'ready.json',ready);break
            failure=remote_json(c,remote+'/result.json')
            if failure is not None:atomic_json(d/'board_result.json',failure);raise RuntimeError('板端启动核验失败：'+str(failure.get('error')))
            if time.monotonic()>end:raise TimeoutError('板端初始化超时')
            time.sleep(.5)
        progress(d,'running',f"板卡已就绪，正在恢复订单前缀并按{m['speed']}倍历史事件时钟派发行情。")
        args=[sys.executable,'-B',str(SENDER),'--fixture','integrated','--release',m['fixture_path'],'--host',load_settings()['board']['device_ip'],'--local',load_settings()['board']['host_ip'],'--session',str(m['session']),'--out',str(d/'sender'),'--speed',str(m['speed']),'--rate','0']
        with (d/'sender.log').open('xb') as log:
            sent=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);end=time.monotonic()+wall_duration_ns(m['fixture']['duration_ns'],m['speed'])/1e9+120
            while sent.poll() is None:
                try:
                    snapshot=remote_json(c,remote+'/ui_telemetry.json')
                    if snapshot:
                        atomic_json(d/'ui_telemetry.json',snapshot)
                        with (d/'ui_history.jsonl').open('a',encoding='utf-8') as history:
                            history.write(json.dumps(dict(PC_observed_ns=time.perf_counter_ns(),snapshot=snapshot))+'\n')
                except Exception as display_error:
                    # Display is an observer. A slow/temporarily locked display
                    # file must never interrupt the independent finite source.
                    errors=result.setdefault('display_sampling_errors',[])
                    if len(errors)<16:errors.append(type(display_error).__name__+': '+str(display_error))
                    result['display_sampling_error_count']=result.get('display_sampling_error_count',0)+1
                if time.monotonic()>end:raise TimeoutError('行情片段未在规定时间内结束')
                time.sleep(.3)
        result['source_return_code']=sent.returncode
        progress(d,'draining','行情派发结束，正在排空已接收窗口并保存结果。')
        checked(c,'touch '+shlex.quote(remote+'/finish.request'))
        end=time.monotonic()+20
        while True:
            board=remote_json(c,remote+'/result.json')
            if board is not None:break
            if time.monotonic()>end:raise TimeoutError('板端排空超时')
            time.sleep(.1)
        atomic_json(d/'board_result.json',board)
        progress(d,'collecting','预测已返回，正在通过串口保存板端计时与核验记录。')
        for name in ('traces.bin','audited_inputs.bin','owner.log','dual_slot_qualification.json','allocation.json','resident_mappings.json','startup_inputs.json','startup_input_attempts.jsonl','mmu_attempts.jsonl'):
            exists=checked(c,'test -f '+shlex.quote(remote+'/'+name)+' && echo yes || echo no').strip()
            if exists=='yes':
                size=int(checked(c,'stat -c %s '+shlex.quote(remote+'/'+name)).strip())
                c.get(remote+'/'+name,d/name,timeout=min(1800,max(180,60+size/2500)))
        if sent.returncode or board['status']!='finished_parked':raise RuntimeError('行情或板端运行失败，已保存现场。')
        result.update(status='completed_pending_content_analysis',firmware=m['loaded_firmware'],board=board)
    except BaseException as exc:
        result['error']=type(exc).__name__+': '+str(exc)
        if c and result['owner_started']:
            try:
                if not (d/'owner.log').exists():c.get(remote+'/owner.log',d/'owner.log')
                for name in ('allocation.json','known_failure.json','mmu_failure.json','mmu_expected.bin','mmu_actual.bin','mmu_attempts.jsonl','dual_slot_qualification.json','startup_inputs.json','startup_input_attempts.jsonl','startup_input_failure.json','startup_input_expected.bin','startup_input_actual.bin','native_slot_failure.json','native_input_expected.bin','native_input_actual.bin'):
                    if checked(c,'test -f '+shlex.quote(remote+'/'+name)+' && echo yes || echo no').strip()=='yes':c.get(remote+'/'+name,d/name)
            except Exception as secondary:result['collection_error']=str(secondary)
    finally:
        if sent and sent.poll() is None:sent.terminate();sent.wait(timeout=5)
        if c:c.close()
        result['owner_released']=False;atomic_json(d/'result.json',result)
    return 0 if result['status']=='completed_pending_content_analysis' else 1

def run_tool(d,name,args,timeout=700):
    with (d/(name+'.log')).open('xb') as f:p=subprocess.run([sys.executable,'-B',*map(str,args)],stdout=f,stderr=subprocess.STDOUT,timeout=timeout,env=environment())
    if p.returncode:raise RuntimeError(name+'失败，详情：'+str(d/(name+'.log')))

def prepare_fixture(d,minutes,cfg):
    if type(minutes)!=int or minutes not in (1,2,5):raise ValueError('时长仅支持1、2、5分钟。')
    if sha(cfg['reference'])!=cfg['reference_sha256']:raise ValueError('盘口显示参考已变化。')
    data=read(cfg['duration_data']);fixture=dict(data['ranges'][str(minutes)])
    rows=np.memmap(cfg['reference'],dtype=ROW,mode='r');n=fixture['eligible_rows']
    if len(rows)<n+50:raise ValueError('时长参考片段长度不足。')
    if np.any(rows['segment'][:n]!=rows['segment'][0]) or np.any(np.diff(rows['row'][:n])!=1):raise ValueError('演示片段必须连续且同段。')
    path=d/'fixture.json';atomic_json(path,dict(schema='G_integrated_finite_fixture_v1',raw_source=cfg['raw_source'],raw_source_sha256=cfg['raw_source_sha256'],fixture=fixture,display_reference=cfg['reference'],display_reference_sha256=cfg['reference_sha256']))
    return path,fixture

def synchronize_observed_boot(d,cfg):
    """A user may have rebooted since the last GUI. Observe; never invent a
    physical confirmation. A normal verified software reboot still follows.
    """
    from board.management.board_lifecycle import process_exists
    q=Queue()
    pending=[r for r in q._load()['requests'] if r['resource']=='board' and r['state'] in ('active','needs_reconciliation','finished_pending_quiescence')]
    if any(r.get('child_pid') and process_exists(r['child_pid']) for r in pending):raise RuntimeError('板卡有正在运行的G任务，请等待该任务结束。')
    with ResourceLock(q.locks,'board'):
        c=Console(load_settings()['board']['serial_port'])
        try:
            ensure_shell(c)
            boot=checked(c,'cat /proc/sys/kernel/random/boot_id').strip();previous=read(G/'state/loaded_firmware.json') if (G/'state/loaded_firmware.json').exists() else {}
            if boot==previous.get('boot_id'):return
            mount='/mnt/fe_gui_'+uuid.uuid4().hex[:8]
            existing=checked(c,'findmnt -rn -S /dev/mmcblk0p1 -o TARGET || true').strip()
            if existing and ('\n' in existing or not existing.startswith('/')):raise ValueError('启动分区挂载位置不明确。')
            target=existing or mount;mounted=False
            try:
                if not existing:checked(c,'mkdir '+mount+' && mount -o ro /dev/mmcblk0p1 '+mount);mounted=True
                actual=checked(c,'sha256sum '+shlex.quote(target+'/BOOT.bin')).split()[0]
            finally:
                if mounted:checked(c,'umount '+mount)
            known={cfg['BOOT_sha256']:cfg['hardware_release'],cfg['rollback_BOOT_sha256']:cfg['rollback_hardware_release']}
            if actual not in known:raise RuntimeError('检测到未知SD固件，已保留现场。')
            observed_release=known[actual]
            evidence=dict(status='observed_new_OS_boot_and_matching_SD_image',previous=previous,boot_id=boot,actual_SD_BOOT_sha256=actual,physical_power_cycle_asserted=False,next_operation='verified_normal_software_reboot')
            path=d/'observed_boot.json';atomic_json(path,evidence)
            previous.update(boot_id=boot,boot_sha256=actual,build=read(observed_release)['source_build'],release=observed_release,evidence=str(path),physical_cold_start=False)
            atomic_json(G/'state/loaded_firmware.json',previous)
        finally:c.close()

def run(minutes=5,slots=32,max_age_us=2000,on_directory=lambda d:None,note=lambda s:None,*,speed=1):
    checked_speed(speed)
    cfg=board_config();d=work_path('runs')/('run_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:5]);d.mkdir(parents=True);on_directory(d)
    from tools.board_bundle import preflight
    preflight(cfg)
    fixture_path,fixture=prepare_fixture(d,minutes,cfg)
    if slots not in (2,4,8,16,32) or not 0<=max_age_us<=10000:raise ValueError('无效队列配置。')
    result=dict(status='preparing',directory=str(d),CPU_reference=False,Host_simulation=False,FPGA_book=True,GUI_reference_only=True)
    atomic_json(d/'showcase_result.json',result);request=None
    try:
        note('正在软件重启板卡并建立新会话，请保持PS串口与PL网线连接。')
        progress(d,'recovering','正在正常软件重启并核对固件，无需拔电。')
        synchronize_observed_boot(d,cfg)
        loaded=read(G/'state/loaded_firmware.json')
        operation='reboot' if loaded['boot_sha256']==cfg['BOOT_sha256'] else 'deploy'
        args=[P/'board/management/board_lifecycle.py','--operation',operation]
        if operation=='deploy':
            deployment=d/'firmware_release';deployment.mkdir()
            for name in ('BOOT.bin','BOOT.bin.gz','release.json'):shutil.copyfile(Path(cfg['hardware_release']).parent/name,deployment/name)
            args+=['--release',deployment/'release.json']
        run_tool(d,'prepare_board',args,1900)
        loaded=read(G/'state/loaded_firmware.json')
        if loaded['boot_sha256']!=cfg['BOOT_sha256']:raise ValueError('当前固件不是整数预处理与整窗FIFO版本。')
        from tools.board_bundle import prepare_bundle
        bundle=d/'bundle';base=prepare_bundle(bundle,cfg)
        for src,name in [(P/'board/runtime/integrated_owner.py','integrated_owner.py'),(P/'board/runtime/startup_input.py','startup_input.py'),(P/'board/native/integrated_native.c','integrated_native.c')]:shutil.copyfile(src,bundle/name)
        for name in ('dual_slot_input.i8','dual_slot_logits.f32'):shutil.copyfile(resolve_path(load_settings()['inputs']['startup'])/name,bundle/name)
        for p in bundle.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
        base['files']={p.name:sha(p) for p in bundle.iterdir() if p.is_file()};atomic_json(bundle/'config.json',base)
        archive=d/'runtime.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for p in bundle.iterdir():tar.add(p,arcname=p.name)
        m=dict(directory=str(d),remote='/root/finedge_integrated/runs/'+d.name,archive=str(archive),session=uuid.uuid4().int&((1<<63)-1) or 1,fixture=fixture,fixture_path=str(fixture_path),loaded_firmware=loaded,functional=False,audit_inputs=0,rate=0,speed=speed,slots=slots,max_age_us=max_age_us,h50_records=50,GUI_monitor=True,CPU_reference=False,Host_simulation=False,reference=cfg['reference'],reference_sha256=cfg['reference_sha256'],preprocessing_version=cfg['preprocessing_version'],transport_version=cfg['transport_version'],full_snapshot_RAM_copies=0,per_window_ARM_input_copy=False)
        build=read(work_path('native/source_build.json'))
        m['files']={str(p):sha(p) for p in (archive,Path(__file__).resolve(),fixture_path,SENDER,P/'host/telemetry/storage.py',P/'board/management/console_login.py',P/'host/telemetry/dispatch_timing.py',work_path('native/source_build.json'),Path(build['executable']),P/'config/board_demo.json',ANALYZER,Path(cfg['candidate_manifest']),Path(cfg['hardware_release']),P/'board/management/serial_console.py')}
        manifest=d/'run_manifest.json';atomic_json(manifest,m)
        q=Queue();request=q.submit(resource='board',stage='tensor_transport',candidate_manifest=manifest,candidate_sha256=sha(manifest),command=[sys.executable,'-B',str(Path(__file__).resolve()),'--execute',str(manifest)])
        atomic_json(d/'queue_request.json',request)
        outcome=q.run(request['request_id'],qualify=qualify,timeout_seconds=wall_duration_ns(fixture['duration_ns'],speed)/1e9+1800)
        if outcome['return_code']!=0:raise RuntimeError(read(d/'result.json').get('error') or '板测失败，现场已保留。')
        note('正在核验逐窗身份、计数及端到端时间。')
        progress(d,'analyzing','正在核验窗口身份、过期计数与真实端到端延时。')
        run_tool(d,'analysis',[ANALYZER,'--directory',d,'--oracle',cfg['reference']],180)
        analysis=read(d/'analysis.json');source=read(d/'sender/source/source.json')
        market_duration=fixture['duration_ns'];requested=wall_duration_ns(market_duration,speed);elapsed=source['schedule_finished_ns']-source['anchor_ns']
        if source['speed']!=speed or source['requested_duration_ns']!=requested:raise ValueError('Native source dispatch speed/duration differs from selected configuration')
        if elapsed<requested:raise ValueError('Source stopped before selected market duration')
        analysis.update(requested_duration_seconds=requested/1e9,historical_market_duration_seconds=market_duration/1e9,actual_source_clock_seconds=elapsed/1e9,speed=speed,selected_minutes=minutes)
        atomic_json(d/'analysis.json',analysis);result.update(status='passed',analysis=analysis)
    except BaseException as exc:
        result.update(status='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        if request:
            try:
                progress(d,'recovering','正在软件重启回收会话，保留本轮全部结果。')
                run_tool(d,'cleanup',[P/'board/management/board_lifecycle.py','--operation','reboot'],180)
                result['cleanup']='software_reboot_verified'
            except Exception as exc:result.update(status='failed',cleanup_error=str(exc),cleanup='requires_recovery')
        atomic_json(d/'showcase_result.json',result)
    if result['status']!='passed':raise RuntimeError(result.get('error') or result.get('cleanup_error'))
    atomic_json(work_path('dashboard/artifacts/CURRENT.json'),dict(directory=str(d)))
    return d

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--execute',type=Path);a=ap.parse_args()
    if not a.execute:ap.error('Use dashboard.py for an interactive run')
    raise SystemExit(execute(a.execute.resolve()))
