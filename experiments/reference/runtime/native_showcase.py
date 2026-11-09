"""Native board deployment and live adapter for the existing product replay.

The desktop keeps raw reconstruction/normalization/metadata/plots; the board
keeps the shared FPGA window, resident NPU and C binary stream. No model or
prebuilt tensor is evaluated on the desktop in this module.
"""
from collections import deque
from pathlib import Path
import json,struct,sys,time,uuid
import numpy as np
from software.runtime.delivery import read_json, write_json, sha256
from shared.model_paths import ROOT, PACKAGE, MANIFEST_SHA
from experiments.reference.runtime.board import connect, command
from experiments.reference.runtime.ps_stream_protocol import read_exact, write_exact
from experiments.reference.runtime.ps_window_runtime import selected_endpoints

REQUEST=struct.Struct('<4I');RESPONSE=struct.Struct('<5I');PACKET=struct.Struct('<17Q3fI')
MAGIC_IN=0x3253444e;MAGIC_OUT=0x3252444e
MODEL_LABELS={'fused':'快速融合图（实验：logits略有变化）','strict':'原输出一致图'}
SOURCE_NAMES=('src/finedge_host/native_showcase.py','src/finedge_host/showcase_replay.py',
    'src/finedge_host/showcase_gui.py','src/finedge_host/ps_stream_display.py',
    'scripts/launch_showcase.py','scripts/native_showcase_service.py',
    'native/native_showcase_record.c','native/native_showcase_stream.c')

def restartable(prior,status):
    return status['status'].endswith('_parked') or (
        prior.get('schema')=='finedge-native-showcase-v2' and status.get('protocol')=='NDS2'
        and status['status']=='ready' and status.get('connected') is False)

def validate_normalization():
    manifest=PACKAGE/'package_manifest.json'
    if sha256(manifest)!=MANIFEST_SHA:raise ValueError('Delivered model manifest changed')
    if sha256(PACKAGE/'normalization.json')!=read_json(manifest)['files']['normalization.json']['sha256']:
        raise ValueError('Frozen normalization changed')

def validate_native_session(session):
    session=Path(session);cfg=read_json(session);contract=read_json(session.parent/'showcase_contract.json')
    binding=read_json(session.parent/'showcase_session_binding.json')
    if (binding!=dict(session_sha256=sha256(session),contract_sha256=sha256(session.parent/'showcase_contract.json'))
            or cfg.get('schema')!='finedge-native-showcase-v2'
            or cfg.get('model_mode')!=contract.get('model_mode') or cfg.get('data_binding')!=contract.get('data_source')):
        raise ValueError('Frozen native session binding changed')
    for name,digest in contract['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Frozen native source changed: '+name)
    validate_normalization()
    return cfg

def helpers():
    if (ROOT/'distribution/submission.json').is_file():
        from experiments.reference.runtime import portable_runtime
        return None, portable_runtime
    
    import tenth,shared_fast
    return tenth,shared_fast

def prepare_native_showcase(mode='fused'):
    if mode not in MODEL_LABELS:raise ValueError('Unknown native model selection')
    if (ROOT/'distribution/submission.json').is_file():
        from experiments.reference.runtime.portable_runtime import prepare
        return prepare(mode, SOURCE_NAMES)
    validate_normalization()
    tenth,_=helpers()
    release=read_json(ROOT/'artifacts/tenth/build_20261027_02/physical_release.json')
    if not release.get('physical_signoff'):raise ValueError('Native FPGA physical release missing')
    model=ROOT/('artifacts/subhalf/'+('polyphase_20261027_01' if mode=='fused' else 'polyphase_taps_20261027_01')+'/icraft/polyphase_ZG.json')
    identity=read_json(ROOT/'artifacts/tenth/model_identity.json')[mode]
    if sha256(model)!=identity['json_sha256'] or sha256(model.with_suffix('.raw'))!=identity['raw_sha256']:
        raise ValueError('Qualified compiled model changed')
    label='showcase_native_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    out=tenth.prepare(label,model,release['boot_sha256'],fused=mode=='fused')
    cfg=read_json(out/'private_config.json')
    for k in ('same_model_reference','original_compiled_reference','first_window_logits'):cfg.pop(k,None)
    cfg.update(schema='finedge-native-showcase-v2',model_mode=mode,
        max_windows=cfg['data_binding']['expected_counts']['windows'],startup_reference_enabled=False)
    c=out/'shared_fast_mmio.c'
    with c.open('a',encoding='utf-8') as f:
        for name in ('native_showcase_record.c','native_showcase_stream.c'):f.write('\n'+(ROOT/'native'/name).read_text(encoding='utf-8'))
    (out/'shared_fast_service.py').write_bytes((ROOT/'scripts/native_showcase_service.py').read_bytes())
    sources={name:sha256(ROOT/name) for name in SOURCE_NAMES}
    write_json(out/'showcase_contract.json',dict(schema='native-showcase-v2',data_source=cfg['data_binding'],
        model_mode=mode,model_label=MODEL_LABELS[mode],source_hashes=sources,
        cpu_reference_enabled=False,host_simulation_enabled=False,full_tensor_readback=False,
        numeric_policy='TF32 reassociation experimental; old logits tolerance not passed' if mode=='fused' else 'old compiled outputs preserved in qualification'))
    cfg['remote_hashes']={str(p.relative_to(out)).replace('\\','/'):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='private_config.json'}
    write_json(out/'private_config.json',cfg)
    return out

def deploy_native_showcase(out,emit=lambda message:None):
    out=Path(out);_,fast=helpers();cfg=read_json(out/'private_config.json')
    contract=read_json(out/'showcase_contract.json')
    for name,digest in contract['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Frozen showcase source changed: '+name)
    # A running GUI must not be interrupted by another launch/reboot.
    with connect() as ssh:
        before=fast.checked(ssh,'cat /proc/sys/kernel/random/boot_id')
        for p in (ROOT/'artifacts/shared_input_fast').glob('*/private_session.json'):
            prior=read_json(p)
            if prior.get('boot_id')!=before:continue
            status=fast.checked(ssh,'cat '+prior['remote']+'/status.json 2>/dev/null || true')
            if status and not restartable(prior,json.loads(status)):
                raise RuntimeError('板端会话仍在运行，请先在原窗口停止并保存。')
    emit('正在软件重启板卡，建立新的模型和共享缓冲会话…')
    boot=fast.reboot(out)
    from experiments.reference.runtime.udw_e_preflight import capture_boot
    remote='/root/finedge_native_'+uuid.uuid4().hex[:12]
    with connect() as ssh:
        observation=capture_boot(lambda cmd:command(ssh,cmd),label=out.name,previous_boot_ids={before},
            expected_BOOT_sha256=cfg['expected_BOOT_sha256'])
        cfg.update(boot_id=boot,remote=remote,BOOT_sha256=observation['actual_BOOT_sha256'],bundle_sha256=sha256(out/'private_config.json'))
        path=out/'private_session.json';write_json(path,cfg)
        write_json(out/'showcase_session_binding.json',dict(session_sha256=sha256(path),contract_sha256=sha256(out/'showcase_contract.json')))
        fast.checked(ssh,'mkdir '+remote+' '+remote+'/finedge_host')
        with ssh.open_sftp() as sftp:
            for name in cfg['remote_hashes']:sftp.put(str(out/name),remote+'/'+name)
            sftp.put(str(path),remote+'/config.json')
        emit('正在构建原生控制器并加载板载模型…')
        fast.checked(ssh,'cd '+remote+' && (ulimit -v 131072; gcc -O3 -shared -fPIC shared_fast_mmio.c -o libshared_fast.so)',timeout=30)
        fast.checked(ssh,'nohup icraft-serve > '+remote+'/sdk.log 2>&1 < /dev/null &')
        fast.checked(ssh,'nohup env OMP_NUM_THREADS=1 PYTHONPATH='+remote+' python3 '+remote+'/shared_fast_service.py --config '+remote+'/config.json > '+remote+'/service.log 2>&1 < /dev/null &')
        for _ in range(160):
            state=fast.checked(ssh,'cat '+remote+'/status.json 2>/dev/null || true')
            if state:
                status=json.loads(state)['status']
                if status=='ready':return path
                if status.startswith('failed'):
                    with ssh.open_sftp() as sftp:
                        for name in ('result.json','service.log'):
                            try:sftp.get(remote+'/'+name,str(out/('board_'+name)))
                            except OSError:pass
                    raise RuntimeError('板端初始化失败，详情已保留于 '+str(out))
            time.sleep(.25)
    raise TimeoutError('板端原生服务未就绪；请保留本次目录，不自动重试。')

def decode_packet(raw,startup=False):
    v=PACKET.unpack(raw);t=v[8:15]
    return dict(window_id=v[16],start=list(v[:4]),end=list(v[4:8]),logits=list(v[17:20]),class_index=v[20],
        backend='board_socket_npu',whole_predict_ns=t[2]+t[3],board_pipeline_ns=t[6],
        startup_included=startup,tensor_sha256=None,integrity_mode='native_identity_and_completion_v2',
        stages_ns=dict(record_DMA=t[0],fpga_window=t[1],npu_wait=t[2],output_DMA=t[3],rearm=t[4]),
        request_id=v[15],explicit_full_window_copy_bytes=0)

class NativeShowcaseClient:
    hardware=True
    native=True
    def __init__(self,session,log_path=None,*,transport=None):
        self.session=Path(session);self.cfg=validate_native_session(self.session) if transport is None else read_json(self.session)
        if self.cfg.get('schema')!='finedge-native-showcase-v2':raise ValueError('Native showcase session required')
        self.model_mode=self.cfg['model_mode'];self.failed=False;self.finished=False;self.primed=False
        self.frame_seq=1;self.batch_seq=0;self.windows=self.records=0;self.history=deque(maxlen=100);self.hello_done=False
        self.ssh=self.channel=None;self.log_path=Path(log_path) if log_path else None
        if transport is None:
            self.ssh=connect()
            self.channel=self.ssh.get_transport().open_channel('direct-tcpip',('127.0.0.1',9984),('127.0.0.1',0))
            self.channel.settimeout(60);self.read,self.write=self.channel.recv,self.channel.send
        else:self.read,self.write=transport
    def json_response(self):
        n,=struct.unpack('<I',read_exact(self.read,4))
        if not 0<n<=262144:raise ValueError('Native response length')
        return json.loads(read_exact(self.read,n))
    def hello(self):
        if self.hello_done or self.failed or self.finished:raise ValueError('Native HELLO is one-shot')
        write_exact(self.write,self.cfg['token'].encode('ascii'));r=self.json_response()
        if r!={'protocol':'NDS2',**{k:self.cfg[k] for k in ('bundle_sha256','boot_id','BOOT_sha256')}}:raise ValueError('Native session identity changed')
        self.hello_done=True
        return dict(protocol='NDS2',model_mode=self.model_mode,model_label=MODEL_LABELS[self.model_mode],
            native_shared_window=True,full_tensor_readback=False,**{k:r[k] for k in ('boot_id','BOOT_sha256')})
    def frame(self,payload):
        if self.failed or self.finished or not self.hello_done:raise ValueError('Native session unavailable')
        expected,history=selected_endpoints(payload,self.history) if payload else ([],self.history)
        count=len(payload)//256
        begin=time.perf_counter_ns()
        write_exact(self.write,REQUEST.pack(MAGIC_IN,self.frame_seq,count,len(expected))+payload)
        magic,seq,got_records,got_windows,flags=RESPONSE.unpack(read_exact(self.read,RESPONSE.size))
        if (magic,seq,got_records,got_windows)!=(MAGIC_OUT,self.frame_seq,count,len(expected)) or flags not in (0,1):raise ValueError('Native response frame mismatch')
        if flags and (self.primed or got_windows!=1):raise ValueError('Unexpected native startup flag')
        if got_windows and not self.primed and not flags:raise ValueError('Missing native startup flag')
        rows=[decode_packet(read_exact(self.read,PACKET.size),bool(flags)) for _ in range(got_windows)]
        elapsed=time.perf_counter_ns()-begin
        for i,(row,(first,last)) in enumerate(zip(rows,expected),1):
            if row['window_id']!=self.windows+i or row['start']!=list(first) or row['end']!=list(last):raise ValueError('Native window identity mismatch')
            if not np.isfinite(row['logits']).all() or row['class_index']!=int(np.argmax(row['logits'])):raise ValueError('Invalid native logits')
            row['batch_roundtrip_ns']=elapsed
        self.windows+=len(rows);self.records+=count;self.history=history;self.frame_seq+=1
        if rows:self.primed=True
        return rows,elapsed
    def submit(self,payload):
        try:
            if not payload or len(payload)%256 or len(payload)>8192:raise ValueError('Native input frame size')
            rows=[];elapsed=0
            # Warmup ends exactly at the first emitting record, even after
            # earlier short segments. No 1000-window or same-segment cap.
            if not self.primed:
                emit=[i+256 for i in range(0,len(payload),256) if struct.unpack_from('<I',payload,i+32)[0]]
                split=emit[0] if emit else len(payload)
                got,ns=self.frame(payload[:split]);rows.extend(got);elapsed+=ns
                payload=payload[split:]
            if payload:
                got,ns=self.frame(payload);rows.extend(got);elapsed+=ns
            self.batch_seq+=1
            return dict(seq=self.batch_seq,windows=rows,data=dict(native=True,model_mode=self.model_mode,batch_roundtrip_ns=elapsed))
        except BaseException:
            self.failed=True;raise
    def finish(self):
        try:
            self.frame(b'');r=self.json_response()
            if r.get('status')!='finished' or r.get('windows')!=self.windows or r.get('records')!=self.records:raise ValueError('Native finish/count mismatch')
            self.finished=True;return r
        except BaseException:self.failed=True;raise
    def disconnect(self):
        if self.channel:self.channel.close()
        if self.ssh:
            destination=self.log_path.parent if self.log_path else self.session.parent
            errors=[]
            with self.ssh.open_sftp() as sftp:
                for name in ('result.json','startup.json','startup_predictions.jsonl','native_predictions.bin','service.log'):
                    for attempt in range(15):
                        try:sftp.get(self.cfg['remote']+'/'+name,str(destination/('board_'+name)));break
                        except OSError:
                            if attempt==14:errors.append(name)
                            time.sleep(.1)
            if errors:write_json(destination/'board_collection_missing.json',dict(files=errors))
            self.ssh.close();self.ssh=None

def native_showcase_client(session,log_path=None):return NativeShowcaseClient(session,log_path)
