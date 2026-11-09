"""Prepare locally, then deploy once after a fresh human cold-start confirmation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
import shlex
import shutil
import socket
import struct
import time
import uuid

import numpy as np

from software.runtime.delivery import read_json, sha256, write_json
from shared.model_paths import ROOT, PACKAGE
from experiments.reference.runtime.ddr_transport_contract import pack_record
from experiments.reference.runtime.udw_board_bridge import digest

BOOT_SHA = 'f9ea130dd1761015f5d8bc79846fdfb79161da9d43c332d48d6061a777fa0dec'
LAST_BOOT = '59d54d66-3526-4865-8829-5ff468578b4d'
CONFIRMATION = '已为本次FPGA演示物理断电重启'


def golden_input():
    return np.fromfile(PACKAGE/'golden/inputs.f32', dtype='<f4').reshape(32,1,40,100)[0]


def golden_payload(tensor):
    return b''.join(pack_record(tuple(int(w) for w in tensor[0,:,i].view('<u4')),
                    (10000,i,i+1,i), True) for i in range(100))


def prepare_bundle(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    pkg = read_json(PACKAGE/'package_manifest.json')
    for name in ('normalization.json','golden/inputs.f32','model_cpu.ts'):
        if sha256(PACKAGE/name) != pkg['files'][name]['sha256']:
            raise ValueError('Model delivery modified: '+name)
    (directory/'finedge_host').mkdir()
    (directory/'finedge_host/__init__.py').write_bytes(b'')
    for name in ('udw_board_bridge','udw_lab','ddr_transport_contract','fpga_demo_owner','udw_probe_stop'):
        # Retain the accepted runtime source; make type annotations safe on ARM Python3.8.
        source = ROOT/'src/finedge_host'/f'{name}.py'
        (directory/'finedge_host'/f'{name}.py').write_bytes(b'from __future__ import annotations\n'+source.read_bytes())
    for name in ('fpga_demo_service.py','udw_board_bridge_service.py'):
        shutil.copyfile(ROOT/'scripts'/name, directory/name)
    baseline = read_json(ROOT/'artifacts/reports/npu_verification.json')
    reference_path = ROOT/'artifacts/fpga_live_demo/reference_50/reference.json'
    reference = read_json(reference_path)
    if reference.get('status') != 'completed' or len(reference['windows']) != 50:
        raise ValueError('Build the one-time 50-window Host reference first')
    for ext in ('json','raw'):
        source = ROOT/f'artifacts/icraft/tcn_ed4_ZG.{ext}'
        if sha256(source) != baseline[f'compiled_{ext}_sha256']:
            raise ValueError('Compiled model reference differs')
        if sha256(source) != reference['compiled_model_hashes'][source.name]:
            raise ValueError('Cached live Host reference uses a different model')
        shutil.copyfile(source, directory/source.name)
    shutil.copyfile(reference_path,directory/'reference.json')
    golden = golden_input()
    manifest = dict(schema='finite-fpga-demo-v1', created_utc=datetime.now(timezone.utc).isoformat(),
                    expected_BOOT_sha256=BOOT_SHA, previous_boot=LAST_BOOT, max_windows=50,
                    warmups=10, arena_used_bytes=899328, reserve_bytes=4096,
                    golden_sha256=digest(golden.tobytes()), golden_wdm_sha256=digest(golden_payload(golden)),
                    golden_logits=baseline['comparisons'][0]['host']['logits'],
                    source_hashes={str(p.relative_to(ROOT)).replace('\\','/'): sha256(p) for p in
                        [ROOT/'src/finedge_host'/f'{n}.py' for n in
                         ('fpga_demo_replay','fpga_demo_session','gui','demo_rate','raw_itch','udw_e_frontend','stream')]},
                    remote_hashes={str(p.relative_to(directory)).replace('\\','/'):sha256(p)
                                   for p in directory.rglob('*') if p.is_file()})
    write_json(directory/'bundle.json', manifest)
    return directory/'bundle.json'


def validate_bundle(path):
    path = Path(path).resolve()
    cfg = read_json(path)
    if cfg.get('schema') != 'finite-fpga-demo-v1' or cfg['max_windows'] != 50:
        raise ValueError('Unknown demo bundle')
    for root, entries in ((path.parent,cfg['remote_hashes']), (ROOT,cfg['source_hashes'])):
        for name, expected in entries.items():
            target = (root/name).resolve()
            if not target.is_relative_to(root.resolve()) or sha256(target) != expected:
                raise ValueError('Frozen demo source differs: '+name)
    return cfg


def deploy_bundle(path, confirmation):
    """No board I/O until exact, newly supplied human text; consume attempt once."""
    if confirmation != CONFIRMATION:
        raise ValueError('需要本次物理断电重启确认；旧实验回复不能复用。')
    path = Path(path).resolve()
    cfg = validate_bundle(path)
    directory = path.parent
    with (directory/'deployment_attempt.json').open('x',encoding='utf-8') as f:
        json.dump(dict(human_response=confirmation, utc=datetime.now(timezone.utc).isoformat(),
                       bundle_sha256=sha256(path)),f,ensure_ascii=False)
    from experiments.reference.runtime.board import connect, command, HOST
    from experiments.reference.runtime.udw_e_preflight import capture_boot
    label = 'demo_'+uuid.uuid4().hex[:12]
    remote = '/root/finedge_'+label
    previous = [cfg['previous_boot']]
    # Successful earlier demo boots also cannot be automatically reused.
    for old in (ROOT/'artifacts/fpga_live_demo').glob('*/private_session.json'):
        previous.append(read_json(old)['boot_id'])
    with connect() as connection:
        def checked(cmd):
            result = command(connection,cmd)
            if result['exit_code']: raise RuntimeError(str(result))
            return result['stdout'].strip()
        observation = capture_boot(lambda cmd:command(connection,cmd),label=label,
                                   previous_boot_ids=previous,expected_BOOT_sha256=BOOT_SHA)
        write_json(directory/'preflight.json',observation)
        if checked("ss -ltnp | grep ':9992 ' || true"):
            raise ValueError('Demo port occupied; no reclaim or reconnect')
        checked('mkdir '+shlex.quote(remote))
        checked('mkdir '+shlex.quote(remote+'/finedge_host'))
        session = dict(cfg, host=HOST, port=9992, token=secrets.token_hex(32), boot_id=observation['boot_id'],
                       run_id=int(time.time()), remote=remote, package_sha256=sha256(path),
                       bundle_path=str(path), confirmation=confirmation)
        write_json(directory/'private_session.json',session)
        with connection.open_sftp() as sftp:
            for name in cfg['remote_hashes']:
                sftp.put(str(directory/name),remote+'/'+name)
            sftp.put(str(directory/'private_session.json'),remote+'/config.json')
            sftp.chmod(remote+'/config.json',0o600)
        # Validate files again on ARM before starting any SDK owner.
        verified = json.loads(checked('python3 -c '+shlex.quote(
            "import hashlib,json,pathlib;p=pathlib.Path("+repr(remote)+");c=json.loads((p/'config.json').read_text());"
            "print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in c['remote_hashes']}))")))
        if verified != cfg['remote_hashes']: raise ValueError('Remote file hashes differ')
        write_json(directory/'remote_hashes.json',verified)
        checked('python3 -m compileall -q '+shlex.quote(remote))
        if not checked("ss -ltnp | grep ':9981 ' || true"):
            checked('nohup icraft-serve > '+shlex.quote(remote+'/serve.log')+' 2>&1 < /dev/null &')
            ready_deadline=time.monotonic()+15
            while not checked("ss -ltnp | grep ':9981 ' || true"):
                if time.monotonic()>=ready_deadline:raise TimeoutError('icraft-serve did not listen; owner not started')
                time.sleep(.25)
        checked('nohup env OMP_NUM_THREADS=1 PYTHONPATH='+shlex.quote(remote)+' python3 '+
                shlex.quote(remote+'/fpga_demo_service.py')+' --config '+shlex.quote(remote+'/config.json')+
                ' > '+shlex.quote(remote+'/service.log')+' 2>&1 < /dev/null &')
        deadline = time.monotonic()+45
        while time.monotonic()<deadline:
            result = command(connection,'cat '+shlex.quote(remote+'/status.json'))
            if result['exit_code']==0:
                status = json.loads(result['stdout'])
                write_json(directory/'deployment_status.json',status)
                if status['status']=='ready': return directory/'private_session.json'
                raise RuntimeError('Board service quarantined: '+str(status))
            time.sleep(.5)
        raise TimeoutError('No ready response. Do not retry or terminate board owner.')


class DemoClient:
    def __init__(self, session_path, wire_log):
        self.cfg = read_json(session_path)
        validate_bundle(self.cfg['bundle_path'])
        if sha256(self.cfg['bundle_path']) != self.cfg['package_sha256']:
            raise ValueError('Session bundle binding differs')
        # Persist BEFORE connect: even an ambiguous connect failure cannot retry.
        with Path(session_path).with_name('client_claim.json').open('x',encoding='utf-8') as f:
            json.dump(dict(utc=datetime.now(timezone.utc).isoformat(),wire_log=str(wire_log)),f)
        self.sock = socket.create_connection((self.cfg['host'],self.cfg['port']),timeout=30)
        self.sock.settimeout(30)
        self.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        self.file = self.sock.makefile('rb')
        self.log = Path(wire_log)
        self.sequence = 0

    def call(self, op, payload=b'', **values):
        self.sequence += 1
        header = dict(token=self.cfg['token'], sequence=self.sequence, op=op,
                      bytes=len(payload), sha256=digest(payload), **values)
        encoded = json.dumps(header,separators=(',',':')).encode()
        packet = struct.pack('<I',len(encoded))+encoded+payload
        started = time.perf_counter_ns()
        self.sock.sendall(packet)
        raw = self.file.readline(100000)
        elapsed = time.perf_counter_ns()-started
        if not raw.endswith(b'\n'): raise EOFError('Board disconnected / truncated response; no retry')
        response = json.loads(raw)
        with self.log.open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(op=op,sequence=self.sequence,payload_bytes=len(payload),
                payload_sha256=digest(payload),rtt_ns=elapsed,response=response))+'\n')
        if not response.get('ok') or response.get('sequence')!=self.sequence:
            raise RuntimeError(str(response))
        return dict(response['result'],rtt_ns=elapsed)

    def disconnect(self):
        self.file.close()
        self.sock.close()


def collect_board_logs(session_path, directory):
    """SFTP files only after finish/disconnect; never queries SDK or signals owner."""
    from experiments.reference.runtime.board import connect
    cfg = read_json(session_path)
    out = Path(directory)/'board'
    out.mkdir()
    with connect() as connection, connection.open_sftp() as sftp:
        for name in sftp.listdir(cfg['remote']):
            if name in ('status.json','responses.jsonl','events.json','service.log') or re.fullmatch(r'slot_\d{3}\.bin',name):
                sftp.get(cfg['remote']+'/'+name,str(out/name))
