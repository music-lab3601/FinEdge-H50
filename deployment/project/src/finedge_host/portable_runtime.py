"""Self-contained release preparation. No historical sessions or tool installs."""
from pathlib import Path
import json
import secrets
import shutil
import shlex
import re
import time
import uuid
import paramiko

from .board import connect, command
from .delivery import read_json, write_json, sha256, safe_path
from .paths import ROOT


def prepare(mode, source_names):
    from .native_showcase import MODEL_LABELS, validate_normalization
    from .showcase_data import data_binding
    validate_normalization()
    release = read_json(ROOT/'distribution/submission.json')
    template = ROOT/'distribution/board_templates'/mode
    cfg = read_json(template/'runtime.json')
    if mode not in MODEL_LABELS or cfg['model_mode'] != mode:
        raise ValueError('Portable model mode mismatch')
    if cfg['data_binding'] != data_binding():
        raise ValueError('Portable dataset/runtime version changed')
    if cfg['expected_BOOT_sha256'] != release['BOOT_sha256']:
        raise ValueError('Portable BOOT identity mismatch')
    identity = read_json(ROOT/'artifacts/tenth/model_identity.json')[mode]
    for name, key in (('tcn_ed4_ZG.json', 'json_sha256'), ('tcn_ed4_ZG.raw', 'raw_sha256')):
        if sha256(template/'files'/name) != identity[key]:
            raise ValueError('Qualified compiled model changed')
    # Verify the complete public template before making a fresh private attempt.
    for name, digest in cfg['remote_hashes'].items():
        if sha256(safe_path(template/'files', name)) != digest:
            raise ValueError('Portable board runtime changed: '+name)
    out = ROOT/'artifacts/shared_input_fast'/('portable_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    for name in cfg['remote_hashes']:
        target = safe_path(out, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template/'files'/name, target)
    names = (*source_names, 'src/finedge_host/portable_runtime.py', 'src/finedge_host/board.py',
        'src/finedge_host/showcase_web.py', 'src/finedge_host/showcase_monitor.py',
        'src/finedge_host/web/index.html', 'src/finedge_host/web/app.js', 'src/finedge_host/web/app.css')
    write_json(out/'showcase_contract.json', dict(schema='native-showcase-v2',
        data_source=cfg['data_binding'], model_mode=mode, model_label=MODEL_LABELS[mode],
        source_hashes={name: sha256(ROOT/name) for name in dict.fromkeys(names)},
        cpu_reference_enabled=False, host_simulation_enabled=False, full_tensor_readback=False,
        numeric_policy='TF32 reassociation experimental; old logits tolerance not passed' if mode == 'fused' else 'old compiled outputs preserved in qualification'))
    cfg.update(token=secrets.token_hex(32), run_id=int.from_bytes(uuid.uuid4().bytes[:8], 'little') or 1,
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    cfg['remote_hashes'] = {p.relative_to(out).as_posix(): sha256(p) for p in out.rglob('*') if p.is_file()}
    write_json(out/'private_config.json', cfg)
    return out


def checked(ssh, cmd, timeout=30):
    result = command(ssh, cmd, timeout)
    if result['exit_code']:
        raise RuntimeError(result['stderr'][:500] or 'Remote command failed')
    return result['stdout'].strip()


def inspect_parked_clients(ssh, clients, boot):
    """Recognize completed matching native owners left by another PC.

    Never reclaim allocations: only allow an ordinary whole-board reboot after
    service hash, process identity, current boot and parked status agree.
    """
    pids = sorted(set(map(int, re.findall(r'pid=(\d+)', clients))))
    template = ROOT/'distribution/board_templates/fused'
    cfg = read_json(template/'runtime.json')
    expected = dict(pids=pids, boot=boot, server=cfg['server_sha256'],
                    service=sha256(template/'files/shared_fast_service.py'),
                    BOOT=cfg['expected_BOOT_sha256'])
    script = '''
import hashlib,json,os
from pathlib import Path
expected=EXPECTED
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
owners=[]
for pid in expected['pids']:
 p=Path('/proc')/str(pid)
 executable=(p/'exe').resolve()
 if executable.name=='icraft-serve':
  if sha(executable)!=expected['server']:raise ValueError('Unexpected SDK server')
  continue
 args=(p/'cmdline').read_bytes().decode().strip('\\0').split('\\0')
 if '--config' not in args:raise ValueError('Unknown SDK process')
 config=Path(args[args.index('--config')+1])
 service=config.parent/'shared_fast_service.py'
 if str(service) not in args or sha(service)!=expected['service']:raise ValueError('Unknown native service')
 cfg=json.loads(config.read_text())
 status=json.loads((config.parent/'status.json').read_text())
 if cfg.get('schema')!='finedge-native-showcase-v2' or cfg.get('boot_id')!=expected['boot'] or cfg.get('BOOT_sha256')!=expected['BOOT']:raise ValueError('Foreign native identity')
 if status.get('pid')!=pid or not status.get('status','').endswith('_parked'):raise ValueError('Native work is not parked')
 owners.append({'pid':pid,'status':status['status'],'boot_id':expected['boot']})
if not owners:raise ValueError('No matching parked native owner')
print(json.dumps({'safe_for_ordinary_reboot':True,'owners':owners}))
'''.replace('EXPECTED', repr(expected))
    return json.loads(checked(ssh, 'python3 -c '+shlex.quote(script)))


def reboot(out):
    """Ordinary reboot of a clean/known parked owner, without legacy records."""
    with connect() as ssh:
        before = checked(ssh, 'cat /proc/sys/kernel/random/boot_id')
        clients = checked(ssh, "ss -tnp state established | grep ':9981' || true")
        known = []
        for path in (ROOT/'artifacts/shared_input_fast').glob('*/private_session.json'):
            cfg = read_json(path)
            if cfg.get('boot_id') == before:
                status = checked(ssh, 'cat '+cfg['remote']+'/status.json 2>/dev/null || true')
                if status:
                    from .native_showcase import restartable
                    if not restartable(cfg, json.loads(status)):
                        raise ValueError('板端已有未结束任务，请先在原电脑停止并保存。')
                    known.append(cfg)
        if clients and not known:
            try:
                migration = inspect_parked_clients(ssh, clients, before)
            except Exception as exc:
                raise ValueError('SDK连接不属于已验证的结束会话，请先在原电脑停止并保存；不会中断未知任务。') from exc
            write_json(Path(out)/'migration_observation.json', migration)
        write_json(Path(out)/'reboot.json', dict(before_boot_id=before,
            method='ordinary_software_reboot', physical_power_cycle=False, portable=True))
        checked(ssh, "nohup sh -c 'sleep 1; /sbin/reboot' > /tmp/finedge_reboot.log 2>&1 < /dev/null &")
    deadline = time.monotonic()+150
    while time.monotonic() < deadline:
        time.sleep(2)
        try:
            with connect() as ssh:
                new = checked(ssh, 'cat /proc/sys/kernel/random/boot_id')
                if new != before:
                    return new
        except (OSError, EOFError, paramiko.SSHException):
            pass
    raise TimeoutError('板卡正常重启后未在150秒内恢复，请检查网络和电源。')
