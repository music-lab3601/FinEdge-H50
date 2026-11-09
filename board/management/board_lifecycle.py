from shared.paths import environment
"""G-only deployment/recovery through the PS console; normal reboot preferred.

Deployment is a queued board operation. Recovery holds the same physical lock,
then reconciles an earlier parked owner only after a different boot and clean
controller were actually observed. Neither path frees a live SDK allocation.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

from shared.paths import work_path, load_settings
ROOT = work_path('board')
SCRIPTS = Path(__file__).resolve().parent

from shared.io import PROJECT, atomic_json, read, sha
from shared.resources import Queue, ResourceLock, now


def process_exists(pid):
    """Use the OS API; lifecycle recovery must not need an optional package."""
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        k.OpenProcess.restype=wintypes.HANDLE
        k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        k.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=k.OpenProcess(0x1000,False,pid)
        if not handle:
            return ctypes.get_last_error()!=87  # ERROR_INVALID_PARAMETER: PID absent
        try:
            code=wintypes.DWORD()
            if not k.GetExitCodeProcess(handle,ctypes.byref(code)):
                raise OSError(ctypes.get_last_error(),'GetExitCodeProcess')
            return code.value==259  # STILL_ACTIVE
        finally:k.CloseHandle(handle)
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    except PermissionError:return True
    return True


def qualify(request):
    path = Path(request['candidate_manifest'])
    if sha(path) != request['candidate_manifest_sha256']:
        raise ValueError('Lifecycle manifest changed')
    for name, digest in read(path)['files'].items():
        if sha(name) != digest:
            raise ValueError('Lifecycle dependency changed: ' + name)


def run_tool(argv, log):
    with log.open('xb') as stream:
        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=1800, env=environment())
    if result.returncode:
        raise RuntimeError('Lifecycle step failed; see ' + str(log))


def observe(port):
    
    from board.management.serial_console import Console
    import shlex
    console = Console(port)
    try:
        code = """import os,json,mmap,struct
f=os.open('/dev/mem',os.O_RDONLY|os.O_SYNC)
m=mmap.mmap(f,1048576,prot=mmap.PROT_READ,offset=0x40000000)
print(json.dumps(dict(boot_id=open('/proc/sys/kernel/random/boot_id').read().strip(),registers={hex(i):struct.unpack_from('<I',m,i)[0] for i in (0xd0000,0xd0004,0xd0008)})))
"""
        result = console.run('python3 -c ' + shlex.quote(code), 15)
        if result['exit_code']:
            raise RuntimeError('Controller read failed: ' + result['stdout'])
        return json.loads(result['stdout'])
    finally:
        console.close()


def perform(plan):
    out = Path(plan['directory'])
    result = dict(status='failed',physical_power_cycle=False,operation=plan['operation'])
    try:
        before = observe(plan['serial_port'])
        atomic_json(out / 'before.json', before)
        if plan['operation'] == 'deploy':
            run_tool([sys.executable, str(SCRIPTS / 'stage_serial_boot.py'),
                      '--release', plan['release']], out / 'stage.log')
            stage = read(Path(plan['release']).parent / 'serial_stage/result.json')
            if stage['status'] != 'staged_verified_not_loaded':
                raise RuntimeError('BOOT staging not complete')
            atomic_json(out / 'stage_result.json', stage)
        run_tool([sys.executable, str(SCRIPTS / 'reboot_g_generated.py'),
                  '--serial-port', plan['serial_port'], '--output', str(out / 'reboot')], out / 'reboot.log')
        reboot = read(out / 'reboot/result.json')
        after = observe(plan['serial_port'])
        if (reboot['status'] != 'new_G_boot_and_clean_controller_verified'
                or before['boot_id'] == after['boot_id'] or reboot['after'] != after['boot_id']
                or after['registers'] != {'0xd0000':0x31424746,'0xd0004':1,'0xd0008':0}):
            raise RuntimeError('Different boot and clean G firmware not proven')
        release = read(plan['release'])
        loaded = dict(status='G_firmware_boot_and_clean_registers_verified',boot_id=after['boot_id'],
                      boot_sha256=release['boot_sha256'],build=release['source_build'],
                      evidence=str(out / 'result.json'),release=plan['release'],
                      physical_cold_start=False,verified_utc=now())
        if plan['operation'] == 'reboot':
            previous = plan['previous_loaded']
            if previous['boot_id'] != before['boot_id'] or previous['boot_sha256'] != loaded['boot_sha256']:
                raise RuntimeError('Recovery did not start from the recorded firmware boot')
        result.update(status='passed',before=before,after=after,loaded_firmware=loaded)
        atomic_json(out / 'result.json', result)
        atomic_json(ROOT / 'state/loaded_firmware.json', loaded)
    except BaseException as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        atomic_json(out / 'result.json', result)
        raise
    return result


def reconcile(request, out, result):
    evidence = dict(request_id=request['request_id'],child_access_stopped=True,resource_quiescent=True,
                    method='software_reboot',observed_utc=now(),observations=[
                        'Lifecycle child exited; serial console closed.',
                        'Normal Linux reboot changed boot identity from '+result['before']['boot_id']+' to '+result['after']['boot_id'],
                        'G magic/ABI and zero controller flags observed after reboot; earlier owner cannot survive reboot.',
                        'Saved lifecycle evidence: '+str(out / 'result.json')])
    path = out / ('reconcile_' + request['request_id'] + '.json')
    atomic_json(path, evidence)
    Queue().reconcile(request['request_id'], path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--operation',choices=['deploy','reboot'])
    ap.add_argument('--release',type=Path)
    ap.add_argument('--serial-port',default=load_settings()['board']['serial_port'])
    ap.add_argument('--execute',type=Path)
    a=ap.parse_args()
    queue=Queue()
    if a.execute:
        path=a.execute.resolve();qualify(dict(candidate_manifest=str(path),candidate_manifest_sha256=sha(path)))
        end=time.monotonic()+2
        while not any(r['state']=='active' and r.get('child_pid')==os.getpid() and r['candidate_manifest']==str(path)
                      for r in queue._load()['requests']):
            if time.monotonic()>end:raise RuntimeError('Not the active G queue child')
            time.sleep(.01)
        print(json.dumps(perform(read(path))),flush=True);return
    if not a.operation:ap.error('--operation required')
    previous=read(ROOT/'state/loaded_firmware.json') if a.operation=='reboot' else None
    release=(a.release.resolve() if a.release else Path(previous['release']))
    rel=read(release)
    if (rel['status']!='packaged_pending_board_validation' or sha(release.parent/'BOOT.bin')!=rel['boot_sha256']
            or sha(release.parent/'BOOT.bin.gz')!=rel['compressed_sha256']):
        raise ValueError('Signed-off release identity failed')
    out=ROOT/'artifacts'/('board_'+a.operation+'_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
    files=[Path(__file__),release,release.parent/'BOOT.bin',release.parent/'BOOT.bin.gz',
           SCRIPTS/'reboot_g_generated.py',SCRIPTS/'stage_serial_boot.py',SCRIPTS/'serial_console.py']
    plan=dict(operation=a.operation,directory=str(out),release=str(release),serial_port=a.serial_port,
              previous_loaded=previous,files={str(p):sha(p) for p in files})
    path=out/'manifest.json';atomic_json(path,plan)
    print(json.dumps(dict(directory=str(out),operation=a.operation)),flush=True)
    if a.operation=='deploy':
        request=queue.submit(resource='board',stage='small_benchmark',candidate_manifest=path,
                             candidate_sha256=sha(path),command=[sys.executable,str(Path(__file__).resolve()),'--execute',str(path)])
        atomic_json(out/'queue_request.json',request)
        outcome=queue.run(request['request_id'],qualify=qualify,timeout_seconds=1900)
        if outcome['return_code']!=0:raise RuntimeError('Deployment failed; retain queue quarantine/evidence')
        result=read(out/'result.json');reconcile(request,out,result)
    else:
        pending=[r for r in queue._load()['requests'] if r['resource']=='board' and r['state'] in
                 ('active','needs_reconciliation','finished_pending_quiescence')]
        # This is explicit recovery of a parked run, never a concurrent test.
        for request in pending:
            if request.get('child_pid') and process_exists(request['child_pid']):
                raise RuntimeError('Prior board child still exists; inspect before recovery')
        with ResourceLock(queue.locks,'board'):
            result=perform(plan)
        for request in pending:reconcile(request,out,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
