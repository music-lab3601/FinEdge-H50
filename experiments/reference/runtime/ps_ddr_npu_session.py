"""Freeze and operate a finite, single-owner compatibility probe.

Preparation and verification are local. Deployment is one-shot, requires a new
physical restart attestation, and never reclaims an existing board allocation.
This does not enable the new PSW bitstream or authorize long-running reuse.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import time
import uuid

import numpy as np

from software.runtime.delivery import read_json, write_json, sha256
from shared.model_paths import ROOT, WORKSPACE, MANIFEST_SHA
from experiments.reference.runtime.ps_ddr_npu import ARENA, SLOT, TENSOR_BYTES, COUNT, FILL, OFFSETS, SCHEMA, DEVICE_URL, VERSIONS, STARTUP, digest

BOOT_SHA = 'f9ea130dd1761015f5d8bc79846fdfb79161da9d43c332d48d6061a777fa0dec'
PREVIOUS_BOOT = 'bb3523cb-e4db-4ced-9a6e-f360007d134c'
CONFIRMATION = '已为本次统一所有者实验物理断电重启'
FILES = ('finedge_host/__init__.py', 'finedge_host/ps_ddr_npu.py',
         'finedge_host/udw_probe_stop.py', 'ps_ddr_npu_probe_service.py',
         'tcn_ed4_ZG.json', 'tcn_ed4_ZG.raw', 'golden4.f32', 'reference4.json')
LIBRARIES = {'libaxizg330aiu.so', 'libsocketzg330aiu.so', 'libicraft_xrt.so',
             'libicraft_hostbackend.so', 'libicraft_zg330backend.so', 'libicraft_xir.so'}
UDMA_RANGE = dict(base=0x30100000,bytes=134217728)
HOST_REFERENCE = ROOT/'artifacts/reports/npu_verification.json'


def prepare(directory):
    directory = Path(directory).resolve()
    interface_path = ROOT/'artifacts/fpga_shared_tensor/copy_prepare_20261025_02/packet/probe/sdk_interface.json'
    interface = read_json(interface_path)
    package = WORKSPACE/'Icraft_setup/Icraft_3.36.1_onchip.deb'
    if sha256(package) != interface['package_sha256']:
        raise ValueError('Matching ARM SDK package changed')
    board_evidence = ROOT/'artifacts/fpga_ps_ddr/audit_20261025_01/board_sdk_hashes_readonly.json'
    hashes_read = read_json(board_evidence)
    if hashes_read['exit_code'] != 0:
        raise ValueError('Installed library observation failed')
    installed = json.loads(hashes_read['stdout'])
    if installed['missing'] or {Path(n).name for n in installed['files']} != LIBRARIES:
        raise ValueError('Installed library evidence incomplete')
    report_path = HOST_REFERENCE
    report = read_json(report_path)
    manifest = ROOT/'vendor/ed4_model/package_manifest.json'
    golden = ROOT/'vendor/ed4_model/golden/inputs.f32'
    if (sha256(manifest) != MANIFEST_SHA or report['model_manifest_sha256'] != MANIFEST_SHA
            or sha256(golden) != read_json(manifest)['files']['golden/inputs.f32']['sha256']
            or report['status'] != 'passed'):
        raise ValueError('Current model/golden reference identity changed')
    model = ROOT/'artifacts/icraft/tcn_ed4_ZG.json'
    raw = model.with_suffix('.raw')
    if sha256(model) != report['compiled_json_sha256'] or sha256(raw) != report['compiled_raw_sha256']:
        raise ValueError('Compiled model differs from Host reference')
    comparisons = report['comparisons'][:COUNT]
    if [r['input_index'] for r in comparisons] != list(range(COUNT)):
        raise ValueError('Four ordered Host references required')
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'finedge_host').mkdir()
    (directory/FILES[0]).write_bytes(b'')
    for name in FILES[1:4]:
        source = ROOT/('src/'+name if name.startswith('finedge_host/') else 'scripts/'+name)
        (directory/name).write_bytes(source.read_bytes())
    (directory/model.name).write_bytes(model.read_bytes())
    (directory/raw.name).write_bytes(raw.read_bytes())
    (directory/'golden4.f32').write_bytes(golden.read_bytes()[:COUNT*TENSOR_BYTES])
    write_json(directory/'reference4.json', dict(
        compiled_json_sha256=sha256(model), compiled_raw_sha256=sha256(raw),
        inputs_sha256=sha256(directory/'golden4.f32'),
        comparisons=[dict(input_index=r['input_index'], host=dict(logits=r['host']['logits'])) for r in comparisons]))
    write_json(directory/'bundle.json', dict(schema=SCHEMA,
        created_utc=datetime.now(timezone.utc).isoformat(), expected_BOOT_sha256=BOOT_SHA,
        previous_boot_id=PREVIOUS_BOOT, model=model.name, inputs='golden4.f32', references='reference4.json',
        arena_bytes=ARENA, allocation_count=3, count=COUNT, warmups=0, RTL_commands=0,
        reused_device_ranges=0, device_url=DEVICE_URL, expected_udma_range=UDMA_RANGE,
        ARM_extensions=interface['ARM_extensions'], SDK_libraries=installed['files'],
        SDK_package_sha256=sha256(package), SDK_interface_sha256=sha256(interface_path),
        library_observation_sha256=sha256(board_evidence), Host_reference_sha256=sha256(report_path),
        golden_source_sha256=sha256(golden), model_manifest_sha256=MANIFEST_SHA,
        remote_hashes={n:sha256(directory/n) for n in FILES}))
    validate(directory/'bundle.json')
    return directory/'bundle.json'


def validate(path):
    path = Path(path).resolve()
    cfg = read_json(path)
    required = dict(schema=SCHEMA, expected_BOOT_sha256=BOOT_SHA, previous_boot_id=PREVIOUS_BOOT,
        model='tcn_ed4_ZG.json', inputs='golden4.f32', references='reference4.json',
        arena_bytes=ARENA, allocation_count=3, count=COUNT, warmups=0, RTL_commands=0,
        reused_device_ranges=0, device_url=DEVICE_URL, expected_udma_range=UDMA_RANGE)
    if any(cfg.get(k) != v for k,v in required.items()) or set(cfg.get('remote_hashes',{})) != set(FILES):
        raise ValueError('Frozen compatibility probe scope differs')
    if set(cfg.get('ARM_extensions',{})) != {'xir','xrt','host_backend','zg330backend'}:
        raise ValueError('Complete ARM binding hashes required')
    libs = cfg.get('SDK_libraries',{})
    if set(libs) != {'/usr/lib/aarch64-linux-gnu/'+n for n in LIBRARIES}:
        raise ValueError('Exact installed library set required')
    for expected in [*libs.values(), *cfg['ARM_extensions'].values(), *cfg['remote_hashes'].values()]:
        if not isinstance(expected,str) or not re.fullmatch(r'[0-9a-f]{64}',expected):
            raise ValueError('Invalid hash')
    for name,expected in cfg['remote_hashes'].items():
        actual = (path.parent/name).resolve()
        if path.parent not in actual.parents or sha256(actual) != expected:
            raise ValueError('Frozen file changed: '+name)
    ref = read_json(path.parent/cfg['references'])
    for k,name in [('compiled_json_sha256','tcn_ed4_ZG.json'),('compiled_raw_sha256','tcn_ed4_ZG.raw'),('inputs_sha256','golden4.f32')]:
        if ref.get(k) != cfg['remote_hashes'][name]:
            raise ValueError('Reference content is not bound to frozen input/model')
    values = np.fromfile(path.parent/'golden4.f32',dtype='<f4')
    if values.size != COUNT*4000 or not np.isfinite(values).all():
        raise ValueError('Four finite input tensors required')
    if [r['input_index'] for r in ref['comparisons']] != list(range(COUNT)):
        raise ValueError('Golden reference indices changed')
    refs = np.asarray([r['host']['logits'] for r in ref['comparisons']],dtype=np.float64)
    if refs.shape != (COUNT,3) or not np.isfinite(refs).all():
        raise ValueError('Four finite Host outputs required')
    return cfg


def pre_hardware_failure_kind(result, old_session):
    """Only import failure or TCP init refusal before any server RPC can recover."""
    required = dict(status='failed',boot_id=old_session['boot_id'],bundle_sha256=old_session['bundle_sha256'],
        persistent_refs=0,user_device_allocations=0,user_host_input_allocations=0,NPU_calls=0,
        completed_invocations=0,active_invocation_refs=0,versions={},startup={},bindings=[],metadata={})
    if (any(result.get(k)!=v for k,v in required.items())
            or result.get('stop',{}).get('SDK_returns_retained')!=0
            or result.get('allocation_attempts')!={'device':0,'host_input':0}):
        raise ValueError('Hardware ownership may exist; no same-boot retry')
    stage=result.get('stop',{}).get('current_stage')
    if (stage=='created' and result.get('error')=='generic_type: type "PtrType" is already registered!'
            and 'modules = load_modules(cfg)' in result.get('traceback','')):
        return 'python_import'
    if (stage=='device.open:before' and result.get('error')=='[irpc::port::client::tcp::init] connect failed'
            and 'xrt.Device.Open(DEVICE_URL)' in result.get('traceback','')):
        return 'tcp_init_before_server_RPC'
    raise ValueError('Failure is not a recognized pre-hardware error; no same-boot retry')


def deploy(path, confirmation, *, pre_sdk_failure_session=None):
    if confirmation != CONFIRMATION:
        raise ValueError('本次统一所有者实验需要新的物理断电重启确认，不能复用旧回复。')
    path = Path(path).resolve()
    cfg = validate(path)
    # Exclusive creation before any board connection: never silently retry a partially executed owner.
    with (path.parent/'deployment_attempt.json').open('x',encoding='utf-8') as f:
        json.dump(dict(confirmation=confirmation,bundle_sha256=sha256(path),
            utc=datetime.now(timezone.utc).isoformat()),f,ensure_ascii=False)
    from experiments.reference.runtime.board import connect, command
    from experiments.reference.runtime.udw_e_preflight import capture_boot
    label = 'psnpu_'+uuid.uuid4().hex[:12]
    remote = '/root/finedge_'+label
    previous = {PREVIOUS_BOOT}
    for base in ('fpga_live_demo','fpga_udma','fpga_ps_ddr'):
        for old in (ROOT/'artifacts'/base).glob('*/private_session.json'):
            previous.add(read_json(old)['boot_id'])
    with connect() as connection:
        def checked(cmd):
            result = command(connection,cmd)
            if result['exit_code']:
                raise RuntimeError(str(result))
            return result['stdout'].strip()
        recovery = None
        if pre_sdk_failure_session is not None:
            old = read_json(pre_sdk_failure_session)
            old_remote = old.get('remote','')
            if not re.fullmatch(r'/root/finedge_psnpu_[0-9a-f]{12}',old_remote):
                raise ValueError('Recognized failed probe path required')
            old_bundle = Path(pre_sdk_failure_session).parent/'bundle.json'
            old_cfg = validate(old_bundle)
            if sha256(old_bundle)!=old['bundle_sha256']:
                raise ValueError('Failed probe bundle identity changed')
            failed = json.loads(checked('cat '+shlex.quote(old_remote+'/result.json')))
            failure_kind=pre_hardware_failure_kind(failed,old)
            if checked('cat /proc/sys/kernel/random/boot_id')!=old['boot_id']:
                raise ValueError('Failure continuation requires the same attested boot')
            # No RPC/SDK returns, and no device FDs/maps. Retain failed process.
            pid=failed.get('pid')
            if type(pid) is not int or pid<=1:raise ValueError('Invalid failed owner PID')
            code="""import pathlib,json,hashlib,os
p=pathlib.Path(%r); pid=%d; failure_kind=%r
expected=%r
assert {n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in expected}==expected
assert (p/'pre_device_open.json').exists()==(failure_kind=='tcp_init_before_server_RPC')
assert not (p/'trace.jsonl').exists() or (p/'trace.jsonl').stat().st_size==0
fds={x.name:os.readlink(str(x)) for x in pathlib.Path('/proc/'+str(pid)+'/fd').iterdir()}
assert fds['0']=='/dev/null'
assert fds['1']==str(p/'service.log') and fds['2']==str(p/'service.log')
extra=[v for k,v in fds.items() if k not in ('0','1','2')]
assert all(v.startswith('socket:[') and v.endswith(']') for v in extra)
if failure_kind=='python_import':assert not extra
inodes={v[8:-1] for v in extra}
for table in ('tcp','tcp6'):
 for line in pathlib.Path('/proc/net/'+table).read_text().splitlines()[1:]:
  fields=line.split();assert fields[9] not in inodes
assert '/dev/' not in pathlib.Path('/proc/'+str(pid)+'/maps').read_text()
cmd=pathlib.Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(b'\\0')
assert str(p/'ps_ddr_npu_probe_service.py').encode() in cmd
print(json.dumps(dict(fds=fds,source_hashes_verified=True,no_SDK_trace=True,no_connected_TCP=True,no_device_maps=True)))
"""%(old_remote,pid,failure_kind,old_cfg['remote_hashes'])
            observed=json.loads(checked('python3 -c '+shlex.quote(code)))
            if failure_kind=='tcp_init_before_server_RPC' and checked("ss -ltnp | grep ':9981 ' || true"):
                raise ValueError('Server state changed; missing-server recovery evidence no longer holds')
            previous.discard(old['boot_id'])
            recovery=dict(previous_session=str(Path(pre_sdk_failure_session).resolve()),
                boot_id=old['boot_id'],failed_result=failed,process_observation=observed,
                failure_kind=failure_kind,previous_remote=old_remote,
                rationale='Failure precedes hardware RPC; no device allocation is reused',
                failed_owner_retained=True)
            write_json(path.parent/'pre_sdk_recovery.json',recovery)
        preflight = capture_boot(lambda cmd:command(connection,cmd),label=label,
            previous_boot_ids=previous,expected_BOOT_sha256=BOOT_SHA)
        if recovery and preflight['boot_id']!=recovery['boot_id']:
            raise ValueError('Boot changed during explicit pre-SDK recovery')
        write_json(path.parent/'preflight.json',preflight)
        # Refuse any known owner marker; never remove one or terminate an owner.
        checked('test ! -e /run/finedge_udma_owner')
        if recovery:
            checked('test -d /run/finedge_ps_ddr_npu_owner && mkdir /run/finedge_ps_ddr_npu_owner/recover_'+recovery['previous_remote'].rsplit('_',1)[1])
        else:
            checked('mkdir /run/finedge_ps_ddr_npu_owner')
        checked('mkdir '+shlex.quote(remote)+' '+shlex.quote(remote+'/finedge_host'))
        listening=checked("ss -ltnp | grep ':9981 ' || true")
        if not listening:
            checked('nohup icraft-serve > '+shlex.quote(remote+'/serve.log')+' 2>&1 < /dev/null &')
            for _ in range(20):
                listening=checked("ss -ltnp | grep ':9981 ' || true")
                if listening:break
                time.sleep(.1)
            if not listening:raise RuntimeError('icraft-serve did not listen; probe owner not started')
        preflight['SDK_server_listener']=listening
        write_json(path.parent/'preflight.json',preflight)
        session = dict(cfg,boot_id=preflight['boot_id'],preflight=preflight,remote=remote,
                       bundle_sha256=sha256(path),bundle_path=str(path))
        if recovery:session['pre_sdk_recovery']=recovery
        write_json(path.parent/'private_session.json',session)
        with connection.open_sftp() as sftp:
            for name in FILES:
                sftp.put(str(path.parent/name),remote+'/'+name)
            sftp.put(str(path.parent/'private_session.json'),remote+'/config.json')
        code = "import pathlib,json,hashlib;p=pathlib.Path("+repr(remote)+");c=json.loads((p/'config.json').read_text());print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in c['remote_hashes']}))"
        remote_hashes = json.loads(checked('python3 -c '+shlex.quote(code)))
        if remote_hashes != cfg['remote_hashes']:
            raise ValueError('Uploaded files differ')
        write_json(path.parent/'uploaded_hashes.json',remote_hashes)
        checked('python3 -m compileall -q '+shlex.quote(remote))
        checked('nohup env OMP_NUM_THREADS=1 PYTHONPATH='+shlex.quote(remote)+
            ' python3 '+shlex.quote(remote+'/ps_ddr_npu_probe_service.py')+' --config '+
            shlex.quote(remote+'/config.json')+' > '+shlex.quote(remote+'/service.log')+' 2>&1 < /dev/null &')
    return path.parent/'private_session.json'


def collect(session_path,destination):
    from experiments.reference.runtime.board import connect
    cfg = read_json(session_path)
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    with connect() as connection,connection.open_sftp() as sftp:
        with sftp.open(cfg['remote']+'/status.json') as f:
            status = json.load(f)
        write_json(destination/'status.json',status)
        if not status['status'].endswith('_parked'):
            return status
        for name in sftp.listdir(cfg['remote']):
            if name in ('result.json','trace.jsonl','service.log','stop.json','pre_device_open.json') or (name.endswith('.bin') and '/' not in name):
                sftp.get(cfg['remote']+'/'+name,str(destination/name))
    return status


def verify(directory,bundle_path):
    """Reconstruct all eighteen byte snapshots and all four expected outputs locally."""
    directory,bundle_path = Path(directory),Path(bundle_path)
    cfg = validate(bundle_path)
    session = read_json(bundle_path.parent/'private_session.json')
    preflight = read_json(bundle_path.parent/'preflight.json')
    boot = session.get('boot_id')
    if (not isinstance(boot,str) or str(uuid.UUID(boot)) != boot or boot == PREVIOUS_BOOT
            or preflight.get('boot_id') != boot or session.get('preflight') != preflight
            or preflight.get('actual_BOOT_sha256') != BOOT_SHA or preflight.get('SDK_version') != '3.36.1'
            or session.get('bundle_sha256') != sha256(bundle_path)):
        raise ValueError('Missing matching fresh-boot deployment/preflight identity')
    result = read_json(directory/'result.json')
    if sha256(HOST_REFERENCE)!=cfg.get('Host_reference_sha256'):
        raise ValueError('Original same-model Host/socket evidence changed')
    known=read_json(HOST_REFERENCE)
    if known.get('status')!='passed':raise ValueError('Prior deployment evidence not passed')
    expected_bindings=[dict(op_id=x['op_id'],op=x['op_type'],backend=x['backend'])
                       for x in known['backend_info']['socket']['bindings']]
    expected_hardops=[x['op_id'] for x in expected_bindings if 'ZG330' in x['backend'] and 'HardOp' in x['op']]
    if not expected_hardops or result.get('bindings')!=expected_bindings or result.get('hardop_ids')!=expected_hardops:
        raise ValueError('Actual execution bindings differ from independent same-model socket reference')
    required = dict(status='passed',bundle_sha256=sha256(bundle_path),blocked=False,
        NPU_calls=COUNT,completed_invocations=COUNT,active_invocation_refs=0,
        user_device_allocations=3,user_device_bytes=3*ARENA,warmups=0,
        RTL_commands=0,reused_device_ranges=0,device_url=DEVICE_URL,
        boot_id=boot,versions=VERSIONS,ps_range=UDMA_RANGE,owner_retained=True)
    if any(result.get(k) != v for k,v in required.items()) or len(result.get('cases',[])) != COUNT:
        raise ValueError('Failed, incomplete or wrong-scope compatibility result')
    if (not result.get('hardop_ids') or len(set(result['hardop_ids'])) != len(result['hardop_ids'])
            or result.get('failed_invocation') is not None):
        raise ValueError('Incomplete NPU completion ledger')
    expected_startup = {str(offset):0x55445731 if offset==0xC0 else 2 if offset==0x8C else 0 for offset in STARTUP}
    if result.get('startup') != expected_startup or set(result.get('metadata',{})) != set(FILL):
        raise ValueError('Clean ABI or allocation evidence missing')
    for name,row in result['metadata'].items():
        region = 'plddr' if name=='pl' else 'udma'
        address = row.get('address')
        if (type(address) is not int or address < 0 or address%64 or address+ARENA>2**32
                or row.get('bytes') != ARENA or row.get('region') != region
                or row.get('pointer_type') not in ('ADDR','BOTH') or row.get('auto_free') is not False
                or row.get('owner_device') != DEVICE_URL):
            raise ValueError('Wrong allocation ownership or extent')
        if region=='udma' and not UDMA_RANGE['base']<=address<=UDMA_RANGE['base']+UDMA_RANGE['bytes']-ARENA:
            raise ValueError('PS allocation outside verified CMA extent')
    a,b = [result['metadata'][n]['address'] for n in ('ps_source','ps_return')]
    if max(a,b)<min(a+ARENA,b+ARENA):
        raise ValueError('PS allocations overlap')
    expected = {name:bytearray([fill])*ARENA for name,fill in FILL.items()}
    inputs = (bundle_path.parent/cfg['inputs']).read_bytes()
    references = read_json(bundle_path.parent/cfg['references'])['comparisons']
    checked = []
    def check(tag,image):
        filename = tag+'.bin'
        actual = (directory/filename).read_bytes()
        if actual != bytes(image) or digest(actual) != result['snapshot_hashes'].get(filename):
            raise ValueError('Independent byte comparison failed: '+filename)
        checked.append(filename)
    for name in FILL:
        check(name+'.initial_read',expected[name])
    errors = []
    for index,row in enumerate(result['cases']):
        payload = inputs[index*TENSOR_BYTES:(index+1)*TENSOR_BYTES]
        base = index*SLOT
        offsets = {name:base+off for name,off in OFFSETS.items()}
        if (row.get('input_index') != index or row.get('offsets') != offsets
                or row.get('tensor_sha256') != digest(payload) or row.get('hardops') != result['hardop_ids']):
            raise ValueError('Window identity or callback accounting differs')
        for name,suffix in [('ps_source','ps_read'),('pl','pl_read'),('ps_return','ps_return_read')]:
            expected[name][offsets[name]:offsets[name]+TENSOR_BYTES] = payload
            check('case%02d.%s'%(index,suffix),expected[name][base:base+SLOT])
        actual = np.asarray(row['logits'],dtype=np.float64)
        reference = np.asarray(references[index]['host']['logits'],dtype=np.float64)
        if (actual.shape != (3,) or not np.isfinite(actual).all()
                or not np.allclose(actual,reference,rtol=1e-3,atol=1e-4)
                or int(actual.argmax()) != int(reference.argmax()) or row.get('class_index') != int(actual.argmax())):
            raise ValueError('Independent compiled Host comparison failed')
        errors.append(float(np.max(np.abs(actual-reference))))
    for name in FILL:
        check(name+'.final_read',expected[name])
    if set(checked) != set(result['snapshot_hashes']):
        raise ValueError('Unexpected/missing evidence snapshot')
    report = dict(status='passed',full_snapshot_checks=len(checked),NPU_calls=COUNT,
        compiled_Host_max_abs_error=max(errors),result_sha256=sha256(directory/'result.json'),
        full_day_complete=False,hardware_pool_reuse_validated=False,
        strict_visibility_proven=False,throughput_benchmark=False)
    write_json(directory/'verification.json',report)
    return report
