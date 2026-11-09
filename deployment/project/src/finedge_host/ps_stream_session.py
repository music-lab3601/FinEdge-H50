"""Host-side frozen stream deployment and a single pinned SSH transport.

Preparation is local. Deployment requires a new physical restart and never
changes BOOT, resets FPGA, terminates an owner or retries an uncertain request.
"""
from __future__ import annotations

import ast
import hashlib
from datetime import datetime,timezone
import json
from importlib.metadata import version
from pathlib import Path
import re
import secrets
import shlex
import time
import sys
import uuid

from .delivery import read_json,write_json,sha256
from .paths import ROOT,PACKAGE,MANIFEST_SHA
from .ps_stream_platform import SCHEMA,BOOT_SHA,SOURCES,FILES,MAX_WINDOWS,validate_files,PATH_SCHEMA,PATH_SOURCES,file_set,DATA_SCHEMA,DATA_BINDING
from .ps_window_runtime import ARENA_BYTES

PREVIOUS_BOOT='1180828d-c371-49bb-918a-2fb5fd519c57'
CONFIRMATION='已为本次全日流式实验物理断电重启'
ACCEPTED_RESULT='d0e5bb49288468608bfc7f11e4b332a0dd659e8f921c5058510901e3b86bc651'
HOST_SOURCES=('ps_stream_session.py','ps_stream_client.py','ps_stream_replay.py','ps_stream_display.py',
    'ps_stream_protocol.py','fpga_streaming.py','udw_e_frontend.py','ddr_transport_contract.py',
    'raw_itch.py','itch.py','stream.py','backends.py','market.py','demo_rate.py',
    'gui.py','market_view.py','delivery.py','paths.py','board.py','udw_e_preflight.py',
    'ps_stream_platform.py','ps_window_runtime.py','ps_stream_server.py')
HOST_ENTRIES=('ps_stream.py','start_ps_stream_gui.ps1')
PATH_HOST_SOURCES=('ps_path_profile.py','ps_path_runtime.py')
PATH_BASELINE=dict(manifest_sha256='13fb3191db0d2cfea920c06b5a1ed88bb05cd7d458b6c3fbf0922d142b22f581',
    final_result_sha256='ad9d61f12ac4bd49fd0456b94fcfa642cbdf55b5708fe5fc75fc7d375e7f0d70',
    audit_sha256='1a855edd878274b8585a8d1383f7265202894261a301e446e2d3eb5d0272d668')


def runtime_versions():
    return dict(python=sys.version,packages={name:version(name) for name in ('numpy','torch','paramiko')})


def prepare(directory,*,path_profile=None,measurement='phases',sample_stride=64,dataset_binding=None):
    directory=Path(directory).resolve()
    if dataset_binding is not None and (dataset_binding != DATA_BINDING or path_profile is not None):
        raise ValueError('Unknown or mixed explicit dataset stream scope')
    profile=None
    if path_profile is not None:
        from .ps_path_profile import profile_config
        profile=profile_config(path_profile,measurement,sample_stride)
    previous=ROOT/'artifacts/fpga_ps_ddr/window_prepare_20261025_01'
    if sha256(previous/'collection_01/result.json')!=ACCEPTED_RESULT or read_json(previous/'acceptance.json')['status']!='finite_100_window_hardware_acceptance_passed':
        raise ValueError('Accepted PSW1 hardware result required')
    if sha256(previous/'bundle.json')!=read_json(previous/'collection_01/result.json')['bundle_sha256']:
        raise ValueError('Accepted PSW hardware bundle changed')
    original=read_json(previous/'bundle.json')
    if original['expected_BOOT_sha256']!=BOOT_SHA:raise ValueError('Accepted firmware differs')
    report_path=ROOT/'artifacts/reports/npu_verification.json'
    report=read_json(report_path)
    model=ROOT/'artifacts/icraft/tcn_ed4_ZG.json'
    if (report['status']!='passed' or report['model_manifest_sha256']!=MANIFEST_SHA
            or sha256(model)!=report['compiled_json_sha256'] or sha256(model.with_suffix('.raw'))!=report['compiled_raw_sha256']):
        raise ValueError('Compiled model reference differs')
    if sha256(PACKAGE/'package_manifest.json')!=MANIFEST_SHA:raise ValueError('Model delivery manifest changed')
    golden=PACKAGE/'golden/inputs.f32'
    manifest=read_json(PACKAGE/'package_manifest.json')
    if sha256(golden)!=manifest['files']['golden/inputs.f32']['sha256']:raise ValueError('Golden source changed')
    cache=previous/'cache_contract.json'
    if sha256(cache)!=original['cache_contract_sha256']:raise ValueError('Cache evidence changed')
    directory.mkdir(parents=True,exist_ok=False);(directory/'finedge_host').mkdir()
    (directory/'finedge_host/__init__.py').write_bytes(b'')
    source_names=SOURCES+(PATH_SOURCES if profile is not None else ())
    host_names=HOST_SOURCES+(PATH_HOST_SOURCES if profile is not None else ())
    for name in source_names:
        data=(ROOT/'src'/name).read_bytes()
        ast.parse(data.decode('utf-8'),feature_version=(3,8))
        (directory/name).write_bytes(data)
    entry=ROOT/'scripts/ps_stream_service.py'
    ast.parse(entry.read_text(encoding='utf-8'),feature_version=(3,8))
    (directory/entry.name).write_bytes(entry.read_bytes())
    for source in (model,model.with_suffix('.raw'),cache):(directory/source.name).write_bytes(source.read_bytes())
    (directory/'golden1.f32').write_bytes(golden.read_bytes()[:16000])
    write_json(directory/'golden_reference.json',dict(input_sha256=sha256(directory/'golden1.f32'),
        compiled_json_sha256=sha256(model),compiled_raw_sha256=sha256(model.with_suffix('.raw')),
        logits=report['comparisons'][0]['host']['logits']))
    # PC-only prefix references are from an independently accepted 100-window run.
    prefix=ROOT/'artifacts/fpga_ps_ddr/vectors_100_20261025_01/references.json'
    refs=read_json(prefix)
    if (sha256(prefix)!=original['remote_hashes']['references.json']
            or refs.get('compiled_json_sha256')!=sha256(model)
            or refs.get('compiled_raw_sha256')!=sha256(model.with_suffix('.raw'))
            or refs.get('model_manifest_sha256')!=MANIFEST_SHA):
        raise ValueError('Host prefix is not bound to accepted model/reference')
    (directory/'host_prefix_reference.json').write_bytes(prefix.read_bytes())
    host_hashes={name:sha256(ROOT/'src/finedge_host'/name) for name in host_names}
    for folder,names,root in (('host_sources',host_names,ROOT/'src/finedge_host'),
                              ('host_entries',HOST_ENTRIES,ROOT/'scripts')):
        (directory/folder).mkdir()
        for name in names:(directory/folder/name).write_bytes((root/name).read_bytes())
    cfg=dict(schema=PATH_SCHEMA if profile is not None else DATA_SCHEMA if dataset_binding is not None else SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),
        previous_boot_id=PREVIOUS_BOOT,expected_BOOT_sha256=BOOT_SHA,arena_bytes=ARENA_BYTES,
        max_windows=10000 if profile is not None else DATA_BINDING['windows'] if dataset_binding is not None else MAX_WINDOWS,warmups=10,golden_checks=1,max_RSS_kib=393216,
        listen_host='127.0.0.1',listen_port=9982,expected_udma_range=original['expected_udma_range'],
        ARM_extensions=original['ARM_extensions'],SDK_libraries=original['SDK_libraries'],
        server_sha256=original['server_sha256'],cache_contract_sha256=sha256(cache),
        model_manifest_sha256=MANIFEST_SHA,accepted_hardware_result_sha256=ACCEPTED_RESULT,
        run_id=int.from_bytes(uuid.uuid4().bytes[:8],'little') or 1,
        remote_hashes={name:sha256(directory/name) for name in set(FILES)|set(PATH_SOURCES if profile is not None else ())},host_hashes=host_hashes,
        host_entry_hashes={name:sha256(ROOT/'scripts'/name) for name in HOST_ENTRIES},
        host_runtime=runtime_versions(),
        host_prefix_reference_sha256=sha256(prefix),hardware_full_day_validated=False)
    if dataset_binding is not None: cfg['dataset_binding']=dict(DATA_BINDING)
    if profile is not None:
        baseline=ROOT/'artifacts/fpga_ps_ddr/stream_hardware_20261025_02'
        cfg.update(path_profile=profile,path_baseline=dict(
            manifest_sha256=sha256(baseline/'manifest.json'),
            final_result_sha256=sha256(baseline/'board_final/result.json'),
            audit_sha256=sha256(baseline/'audit_partial_40221.json')),
            path_experiment_device_approved=False)
    write_json(directory/'bundle.json',cfg)
    validate_bundle(directory/'bundle.json')
    return directory/'bundle.json'


def validate_bundle(path,*,check_host_sources=True):
    path=Path(path).resolve();cfg=read_json(path)
    validate_files(cfg,path.parent)
    if cfg['schema']==PATH_SCHEMA and (cfg.get('path_baseline')!=PATH_BASELINE
            or cfg.get('path_experiment_device_approved') is not False):
        raise ValueError('Candidate must bind accepted40221 baseline and remain closed pending separate release')
    host_names=HOST_SOURCES+(PATH_HOST_SOURCES if cfg['schema']==PATH_SCHEMA else ())
    if (cfg.get('previous_boot_id')!=PREVIOUS_BOOT or cfg.get('model_manifest_sha256')!=MANIFEST_SHA
            or cfg.get('accepted_hardware_result_sha256')!=ACCEPTED_RESULT
            or set(cfg.get('host_hashes',{}))!=set(host_names)
            or set(cfg.get('host_entry_hashes',{}))!=set(HOST_ENTRIES)
            or sha256(path.parent/'host_prefix_reference.json')!=cfg.get('host_prefix_reference_sha256')):
        raise ValueError('Stream provenance or host source set differs')
    references=read_json(path.parent/'host_prefix_reference.json')
    if (references.get('compiled_json_sha256')!=cfg['remote_hashes']['tcn_ed4_ZG.json']
            or references.get('compiled_raw_sha256')!=cfg['remote_hashes']['tcn_ed4_ZG.raw']
            or references.get('model_manifest_sha256')!=MANIFEST_SHA):
        raise ValueError('Frozen Host prefix/model identity differs')
    for folder,hashes in (('host_sources',cfg['host_hashes']),('host_entries',cfg['host_entry_hashes'])):
        for name,h in hashes.items():
            if sha256(path.parent/folder/name)!=h:raise ValueError('Frozen host snapshot changed: '+name)
    if check_host_sources:
        if cfg.get('host_runtime')!=runtime_versions():raise ValueError('Host environment changed after stream freeze')
        for name,h in cfg['host_hashes'].items():
            if sha256(ROOT/'src/finedge_host'/name)!=h:raise ValueError('Host code changed after stream freeze: '+name)
        for name,h in cfg['host_entry_hashes'].items():
            if sha256(ROOT/'scripts'/name)!=h:raise ValueError('Host entry changed after stream freeze: '+name)
    return cfg


def deploy(bundle,confirmation,*,release=None,power_confirmation=None,user_authorization=None):
    if release is None and confirmation!=CONFIRMATION:raise ValueError('New physical restart required for this stream owner')
    bundle=Path(bundle).resolve();cfg=validate_bundle(bundle)
    admission=None;proof={}
    if cfg['schema']==PATH_SCHEMA:
        if release is None or power_confirmation is None:
            raise ValueError('Candidate is local-only until execution authorization and fresh physical confirmation')
        from .ps_path_profile import validate_candidate_release
        for original,name in ((release,'deployment_release.json'),(power_confirmation,'physical_confirmation.json'),
                              (user_authorization,'deployment_user_authorization.json')):
            if original is not None:proof[name]=Path(original).read_bytes()
        auth=proof.get('deployment_user_authorization.json')
        admission=validate_candidate_release(cfg,sha256(bundle),json.loads(proof['deployment_release.json']),
            hashlib.sha256(proof['deployment_release.json']).hexdigest(),json.loads(proof['physical_confirmation.json']),
            user_authorization=json.loads(auth) if auth is not None else None,
            user_authorization_sha=hashlib.sha256(auth).hexdigest() if auth is not None else None)
    elif confirmation!=CONFIRMATION:
        raise ValueError('New physical restart required for this stream owner')
    with (bundle.parent/'deployment_attempt.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(confirmation=confirmation,bundle_sha256=sha256(bundle)),stream,ensure_ascii=False)
    if admission is not None:
        # Bind the admitted proof locally for every later attach. No approval
        # or physical event is synthesized here; preserve the exact inputs.
        for name,raw in proof.items():
            with (bundle.parent/name).open('xb') as stream:stream.write(raw)
    from .board import connect,command
    from .udw_e_preflight import capture_boot
    label='stream_'+uuid.uuid4().hex[:12];remote='/root/finedge_'+label
    previous={PREVIOUS_BOOT}
    if admission is not None:previous.add(admission['previous_boot_id'])
    for category in ('fpga_live_demo','fpga_udma','fpga_ps_ddr'):
        for p in (ROOT/'artifacts'/category).glob('*/private_session.json'):previous.add(read_json(p)['boot_id'])
    with connect() as connection:
        def checked(cmd):
            r=command(connection,cmd)
            if r['exit_code']:raise RuntimeError(str(r))
            return r['stdout'].strip()
        preflight=capture_boot(lambda cmd:command(connection,cmd),label=label,
            previous_boot_ids=previous,expected_BOOT_sha256=BOOT_SHA)
        write_json(bundle.parent/'preflight.json',preflight)
        for marker in ('finedge_udma_owner','finedge_ps_ddr_npu_owner','finedge_ps_window_owner','finedge_ps_stream_owner'):
            checked('test ! -e /run/'+marker)
        checked('mkdir /run/finedge_ps_stream_owner')
        checked('mkdir '+remote+' '+remote+'/finedge_host')
        if checked('sha256sum /usr/bin/icraft-serve').split()[0]!=cfg['server_sha256']:raise ValueError('SDK server changed')
        if checked("ss -ltnp | grep ':9982 ' || true"):raise ValueError('Stream port already occupied')
        listening=checked("ss -ltnp | grep ':9981 ' || true")
        if not listening:
            checked('nohup icraft-serve > '+remote+'/serve.log 2>&1 < /dev/null &')
            for _ in range(20):
                listening=checked("ss -ltnp | grep ':9981 ' || true")
                if listening:break
                time.sleep(.1)
            if not listening:raise RuntimeError('SDK server did not start')
        session=dict(cfg,boot_id=preflight['boot_id'],preflight=preflight,remote=remote,
            bundle_sha256=sha256(bundle),bundle_path=str(bundle),token=secrets.token_hex(32))
        if admission is not None:session['path_execution_admission']=admission
        private=bundle.parent/'private_session.json';write_json(private,session)
        with connection.open_sftp() as sftp:
            for name in file_set(cfg):sftp.put(str(bundle.parent/name),remote+'/'+name)
            sftp.put(str(private),remote+'/config.json')
        code="import pathlib,json,hashlib;p=pathlib.Path("+repr(remote)+");c=json.loads((p/'config.json').read_text());print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in c['remote_hashes']}))"
        uploaded=json.loads(checked('python3 -c '+shlex.quote(code)))
        if uploaded!=cfg['remote_hashes']:raise ValueError('Stream upload differs')
        write_json(bundle.parent/'uploaded_hashes.json',uploaded)
        checked('python3 -m compileall -q '+remote)
        checked('nohup env OMP_NUM_THREADS=1 PYTHONPATH='+remote+' python3 '+remote+
            '/ps_stream_service.py --config '+remote+'/config.json > '+remote+'/service.log 2>&1 < /dev/null &')
    return private


def validate_session(path,*,check_host_sources=True):
    path=Path(path).resolve();session=read_json(path)
    cfg=validate_bundle(path.parent/'bundle.json',check_host_sources=check_host_sources)
    if (any(session.get(k)!=v for k,v in cfg.items()) or session.get('bundle_sha256')!=sha256(path.parent/'bundle.json')
            or not re.fullmatch('/root/finedge_stream_[0-9a-f]{12}',session.get('remote',''))
            or not re.fullmatch('[0-9a-f]{64}',session.get('token',''))
            or Path(session.get('bundle_path','')).resolve()!=(path.parent/'bundle.json').resolve()
            or session.get('boot_id')==PREVIOUS_BOOT or str(uuid.UUID(session.get('boot_id','')))!=session['boot_id']):
        raise ValueError('Frozen stream session identity differs')
    if (session['preflight']['boot_id']!=session['boot_id']
            or session['preflight']['actual_BOOT_sha256']!=BOOT_SHA):raise ValueError('Session boot preflight differs')
    if cfg['schema']==PATH_SCHEMA:
        from .ps_path_profile import validate_candidate_release
        release=path.parent/'deployment_release.json';physical=path.parent/'physical_confirmation.json'
        authorization=path.parent/'deployment_user_authorization.json'
        admitted=validate_candidate_release(cfg,sha256(path.parent/'bundle.json'),read_json(release),sha256(release),read_json(physical),
            user_authorization=read_json(authorization) if authorization.exists() else None,
            user_authorization_sha=sha256(authorization) if authorization.exists() else None)
        if session.get('path_execution_admission')!=admitted or session['boot_id']==admitted['previous_boot_id']:
            raise ValueError('Candidate execution admission or fresh boot differs')
    return session


class ManagedStreamClient:
    """The only production transport. Closing network never frees board memory."""
    hardware=True

    def __init__(self,session_path,log_path=None):
        from .board import connect,command
        from .ps_stream_client import StreamClient
        self.path=Path(session_path).resolve();self.session=validate_session(self.path)
        self.connection=self.channel=self.client=None;self.closed=False
        # A replay is one-shot even when its outcome is ambiguous. No auto-retry.
        with (self.path.parent/'replay_attempt.json').open('x',encoding='utf-8') as stream:
            json.dump(dict(bundle_sha256=self.session['bundle_sha256'],utc=datetime.now(timezone.utc).isoformat()),stream)
        try:
            self.connection=connect()
            observed=command(self.connection,'cat /proc/sys/kernel/random/boot_id')
            if observed['exit_code'] or observed['stdout'].strip()!=self.session['boot_id']:raise ValueError('Board boot changed before stream attach')
            deadline=time.monotonic()+90
            while True:
                with self.connection.open_sftp() as sftp:
                    try:
                        with sftp.open(self.session['remote']+'/status.json') as f:state=json.load(f)
                    except FileNotFoundError:state={'status':'starting'}
                if state.get('status')=='ready':break
                if state.get('status','').endswith('_parked') or state.get('status')=='streaming':
                    raise RuntimeError('Stream is unavailable or already consumed: '+state.get('status','unknown'))
                if time.monotonic()>=deadline:raise TimeoutError('Stream service did not become ready')
                time.sleep(.2)
            if state.get('boot_id')!=self.session['boot_id'] or state.get('bundle_sha256')!=self.session['bundle_sha256']:
                raise ValueError('Ready service identity differs')
            transport=self.connection.get_transport();transport.set_keepalive(15)
            self.channel=transport.open_channel('direct-tcpip',('127.0.0.1',9982),('127.0.0.1',0),timeout=15)
            self.channel.settimeout(120)
            identity=expected_service_identity(self.session)
            self.client=StreamClient(self.channel.recv,self.channel.send,token=self.session['token'],
                window_budget=execution_window_budget(self.session),expected_boot_sha256=BOOT_SHA,
                expected_boot_id=self.session['boot_id'],expected_identity=identity)
        except BaseException:
            try:self.close()
            except BaseException:pass  # preserve the original admission/transport failure
            raise

    def hello(self):return self.client.hello()
    def submit(self,*args,**kwargs):return self.client.submit(*args,**kwargs)
    def finish(self):return self.client.finish()
    def is_alive(self):
        return not self.closed and self.connection.get_transport().is_active() and not self.channel.closed
    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.channel is not None:self.channel.close()
        finally:
            if self.connection is not None:self.connection.close()
    def disconnect(self):self.close()


def expected_service_identity(session):
    identity=dict(bundle_sha256=session['bundle_sha256'],
        compiled_json_sha256=session['remote_hashes']['tcn_ed4_ZG.json'],
        compiled_raw_sha256=session['remote_hashes']['tcn_ed4_ZG.raw'])
    if session['schema']==PATH_SCHEMA:
        identity.update(schema=PATH_SCHEMA,path_profile=dict(session['path_profile']))
    return identity


def execution_window_budget(session):
    if session['schema']!=PATH_SCHEMA:return session['max_windows']
    value=session['path_execution_admission']['run_windows']
    if type(value) is not int or value not in (100,10000) or value>session['max_windows']:
        raise ValueError('Candidate finite execution budget differs')
    return value


def client_factory(session_path,log_path=None):return ManagedStreamClient(session_path,log_path)


def collect(session_path,destination):
    from .board import connect
    session=validate_session(session_path,check_host_sources=False);destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    with connect() as connection,connection.open_sftp() as sftp:
        with sftp.open(session['remote']+'/status.json') as f:status=json.load(f)
        write_json(destination/'status.json',status)
        allowed={'startup.json','result.json','service.log','pre_device_open.json','stop.json','failure_context.json'}
        for name in sftp.listdir(session['remote']):
            if name in allowed or re.fullmatch(r'failure_[a-z_]+[0-9]*\.bin',name):
                sftp.get(session['remote']+'/'+name,str(destination/name))
    return status
