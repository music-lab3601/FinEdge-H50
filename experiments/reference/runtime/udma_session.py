"""Local preparation, one-shot deployment and read-only evidence collection."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import time
import uuid

from software.runtime.delivery import read_json, write_json, sha256
from shared.model_paths import ROOT, WORKSPACE
from experiments.reference.runtime.udma_probe import ARENA, FILL, cases, pattern, digest

BOOT_SHA = 'f9ea130dd1761015f5d8bc79846fdfb79161da9d43c332d48d6061a777fa0dec'
PREVIOUS_BOOT = '81ffdc1d-66eb-478d-88e9-f1b2f8937572'
CONFIRMATION = '已为本次udma实验物理断电重启'
FILES = ('finedge_host/__init__.py','finedge_host/udma_probe.py',
         'finedge_host/udw_probe_stop.py','udma_probe_service.py')


def prepare(directory):
    directory = Path(directory).resolve()
    # Recover hashes from the already verified matching ARM installation package.
    evidence = ROOT/'artifacts/fpga_shared_tensor/copy_prepare_20261025_02/packet/probe/sdk_interface.json'
    interface = read_json(evidence)
    package = WORKSPACE/'Icraft_setup/Icraft_3.36.1_onchip.deb'
    if sha256(package)!=interface['package_sha256']:
        raise ValueError('Matching ARM package identity differs')
    stub = ROOT/'artifacts/stage2/xrt.pyi'
    signature = 'def copyFrom(self, dest_offset: int, src_chunk: MemChunk, src_offset: int, byte_size: int) -> None:'
    if signature not in stub.read_text(encoding='utf-8'):
        raise ValueError('Missing copyFrom Python signature evidence')
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'finedge_host').mkdir()
    (directory/FILES[0]).write_bytes(b'')
    for filename in FILES[1:]:
        source = ROOT/('src/'+filename if filename.startswith('finedge_host/') else 'scripts/'+filename)
        (directory/filename).write_bytes(source.read_bytes())
    write_json(directory/'bundle.json',dict(schema='udma-transfer-bundle-v1',
        created_utc=datetime.now(timezone.utc).isoformat(), expected_BOOT_sha256=BOOT_SHA,
        previous_boot_id=PREVIOUS_BOOT, arena_bytes=ARENA, allocation_count=3,
        cases=cases(), NPU_calls=0, RTL_commands=0, reused_ranges=0,
        ARM_extensions={n:interface['ARM_extensions'][n] for n in ('xir','xrt')},
        SDK_package_sha256=sha256(package), SDK_evidence_sha256=sha256(evidence),
        Python_stub_sha256=sha256(stub),
        remote_hashes={name:sha256(directory/name) for name in FILES}))
    return directory/'bundle.json'


def validate(path):
    path = Path(path).resolve()
    cfg = read_json(path)
    if (cfg.get('schema')!='udma-transfer-bundle-v1' or cfg.get('cases')!=cases()
        or cfg.get('arena_bytes')!=ARENA or cfg.get('allocation_count')!=3
        or cfg.get('NPU_calls')!=0 or cfg.get('RTL_commands')!=0 or cfg.get('reused_ranges')!=0
        or set(cfg['remote_hashes'])!=set(FILES) or cfg.get('expected_BOOT_sha256')!=BOOT_SHA):
        raise ValueError('Unexpected frozen probe scope')
    for name,expected in cfg['remote_hashes'].items():
        if sha256(path.parent/name)!=expected:
            raise ValueError('Frozen probe file changed: '+name)
    return cfg


def deploy(path, confirmation):
    if confirmation!=CONFIRMATION:
        raise ValueError('本次实验需要新的物理断电重启确认，不能复用旧回复。')
    path = Path(path).resolve()
    cfg = validate(path)
    with (path.parent/'deployment_attempt.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(confirmation=confirmation,bundle_sha256=sha256(path),
                       utc=datetime.now(timezone.utc).isoformat()),stream,ensure_ascii=False)
    from experiments.reference.runtime.board import connect, command
    from experiments.reference.runtime.udw_e_preflight import capture_boot
    label = 'udma_'+uuid.uuid4().hex[:12]
    remote = '/root/finedge_'+label
    previous = {PREVIOUS_BOOT,cfg['previous_boot_id']}
    for root in ('artifacts/fpga_live_demo','artifacts/fpga_udma'):
        for old in (ROOT/root).glob('*/private_session.json'):
            previous.add(read_json(old)['boot_id'])
    with connect() as connection:
        def checked(cmd):
            result = command(connection,cmd)
            if result['exit_code']:
                raise RuntimeError(str(result))
            return result['stdout'].strip()
        observation = capture_boot(lambda cmd:command(connection,cmd),label=label,
            previous_boot_ids=previous,expected_BOOT_sha256=BOOT_SHA)
        write_json(path.parent/'preflight.json',observation)
        # Atomic per-boot ownership marker for this experiment; never reclaimed.
        checked('mkdir /run/finedge_udma_owner')
        checked('mkdir '+shlex.quote(remote)+' '+shlex.quote(remote+'/finedge_host'))
        session = dict(cfg, remote=remote,boot_id=observation['boot_id'],
                       bundle_sha256=sha256(path),bundle_path=str(path))
        write_json(path.parent/'private_session.json',session)
        with connection.open_sftp() as sftp:
            for name in FILES:
                sftp.put(str(path.parent/name),remote+'/'+name)
            sftp.put(str(path.parent/'private_session.json'),remote+'/config.json')
        code = "import pathlib,json,hashlib;p=pathlib.Path("+repr(remote)+");c=json.loads((p/'config.json').read_text());print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in c['remote_hashes']}))"
        hashes = json.loads(checked('python3 -c '+shlex.quote(code)))
        if hashes!=cfg['remote_hashes']:
            raise ValueError('Uploaded files differ')
        write_json(path.parent/'remote_hashes.json',hashes)
        checked('python3 -m compileall -q '+shlex.quote(remote))
        checked('nohup env OMP_NUM_THREADS=1 PYTHONPATH='+shlex.quote(remote)+
            ' python3 '+shlex.quote(remote+'/udma_probe_service.py')+' --config '+
            shlex.quote(remote+'/config.json')+' > '+shlex.quote(remote+'/service.log')+' 2>&1 < /dev/null &')
    return path.parent/'private_session.json'


def collect(session_path, destination):
    from experiments.reference.runtime.board import connect
    cfg = read_json(session_path)
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    with connect() as connection, connection.open_sftp() as sftp:
        # Completed/failed status is written after result and snapshots.
        with sftp.open(cfg['remote']+'/status.json') as stream:
            status = json.load(stream)
        write_json(destination/'status.json',status)
        if not status['status'].endswith('_parked'):
            return status
        for name in sftp.listdir(cfg['remote']):
            if name in ('result.json','binding.json','trace.jsonl','service.log','stop.json') or name.endswith('.bin'):
                sftp.get(cfg['remote']+'/'+name,str(destination/name))
    return status


def verify(directory, bundle_path):
    """Rebuild expected images independently of the board's pass flags."""
    directory = Path(directory)
    cfg = validate(bundle_path)
    result = read_json(directory/'result.json')
    if result['status']!='passed' or result.get('bundle_sha256')!=sha256(bundle_path):
        raise ValueError('Experiment failed or wrong frozen bundle')
    expected = {n:bytearray([fill])*ARENA for n,fill in FILL.items()}
    checked = 0
    def check(filename, image):
        nonlocal checked
        actual = (directory/filename).read_bytes()
        if actual!=bytes(image) or digest(actual)!=result['snapshot_hashes'][filename]:
            raise ValueError('Snapshot differs: '+filename)
        checked += 1
    for name in FILL:
        check(name+'.initial_read.bin',expected[name])
    if len(result['cases'])!=len(cfg['cases']) or result['allocated_bytes']!=3*ARENA:
        raise ValueError('Case/allocation accounting mismatch')
    for case,row in zip(cfg['cases'],result['cases']):
        if any(row.get(k)!=v for k,v in case.items()) or row['payload_sha256']!=digest(pattern(case)):
            raise ValueError('Case identity mismatch')
        for name,suffix in (('ps_source','ps_read'),('pl','pl_read'),('ps_return','ps_return_read')):
            expected[name][case[name]:case[name]+case['size']] = pattern(case)
            check('case%02d.%s.bin'%(case['index'],suffix),
                  expected[name][case['slot']:case['slot']+case['slot_size']])
    for name in FILL:
        check(name+'.final_read.bin',expected[name])
    report = dict(status='passed',full_snapshot_checks=checked,cases=len(result['cases']),
                  allocated_bytes=result['allocated_bytes'],result_sha256=sha256(directory/'result.json'),
                  strict_visibility_proven=False,DMA_implementation_proven=False,
                  full_day_complete=False)
    write_json(directory/'verification.json',report)
    return report
