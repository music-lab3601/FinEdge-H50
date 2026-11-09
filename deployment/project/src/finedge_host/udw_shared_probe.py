"""Independent two-input device probe. Default preparation is entirely local."""
from datetime import datetime
import hashlib
import re

import numpy as np

from .ddr_transport_contract import pack_record,pack_output
from .udw_shared_tensor import LeaseLedger
from .udw_probe_stop import ProbeStop

SCOPE=dict(schema='shared-device-probe-v1',known_windows=1,FPGA_records=100,FPGA_windows=1,
           NPU_calls=2,arena_bytes=2097152,known_allocation_bytes=65536,quota_bytes=1048576,
           reuse=False,reset=False,retry=False,control='axi_registers',Cast='preserved')
FINDINGS={'SHARED_REVIEW_01','SHARED_REVIEW_02','SHARED_PROBE_REVIEW_01'}
DEVICE_VERSIONS={'device':'25122301','icore':'FMSHZGV3TECH-AID - 24160628'}


def packet_files(root,review):
    files=review.get('packet_hashes',{})
    required={'src/finedge_host/udw_shared_tensor.py','src/finedge_host/udw_shared_probe.py',
        'src/finedge_host/udw_probe_stop.py','src/finedge_host/udw_shared_probe_owner.py',
        'src/finedge_host/udw_board_bridge.py','src/finedge_host/udw_lab.py',
        'src/finedge_host/ddr_transport_contract.py','src/finedge_host/udw_e_preflight.py',
        'src/finedge_host/delivery.py','src/finedge_host/paths.py','scripts/udw_shared_probe_check.py',
        'artifacts/icraft/tcn_ed4_ZG.json','artifacts/icraft/tcn_ed4_ZG.raw',
        'probe/golden.f32','probe/expected.json','probe/previous.json'}
    if not required<=set(files):raise ValueError('Incomplete reviewed probe packet')
    root=root.resolve()
    for name,h in files.items():
        path=root/name
        if root not in path.resolve().parents:raise ValueError('Probe packet path escapes root')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=h:raise ValueError('Probe packet changed: '+name)


def validate_request(root,review,confirmation,label,review_sha256):
    if not re.fullmatch(r'[A-Za-z0-9_]{1,40}',label):raise ValueError('Safe unique probe label required')
    if (review.get('device_probe_approved') is not True or not FINDINGS<=set(review.get('closed_findings',[]))
        or review.get('scope')!=SCOPE or review.get('expected_device_versions')!=DEVICE_VERSIONS):
        raise ValueError('Shared probe review/scope is not approved')
    packet_files(root,review)
    if (confirmation.get('source')!='human_user_current_session'
        or confirmation.get('physical_power_removed') is not True
        or confirmation.get('linux_boot_complete') is not True
        or confirmation.get('label')!=label or not confirmation.get('verbatim_reply','').strip()
        or not re.fullmatch(r'[0-9a-f]{64}',review_sha256)
        or confirmation.get('review_sha256')!=review_sha256):
        raise ValueError('Fresh human confirmation bound to this probe required')
    # Paths/hashes bind the human bookkeeping to this exact release/previous boot.
    import json
    previous=json.loads((root/'probe/previous.json').read_text())
    if confirmation.get('previous_result_sha256')!=previous['original_result_sha256']:
        raise ValueError('Confirmation predecessor differs')
    return previous


def golden_records(tensor):
    x=np.frombuffer(tensor,dtype='<f4').reshape(40,100)
    if not np.isfinite(x).all():raise ValueError('Nonfinite golden input')
    ids=[(10000,i,i+1,i) for i in range(100)]
    payload=b''.join(pack_record(tuple(int(w) for w in x[:,i].copy().view('<u4')),ids[i],i==99)
                     for i in range(100))
    return payload,ids[0],ids[-1]


def known_sdk_input(owner,tensor):
    """A distinct, retained allocation; no PL command before this comparison."""
    stop=owner.probe_stop
    stop.checkpoint('known.before');owner.ensure_active()
    payload,first,last=golden_records(tensor)
    region=stop.call('known.region',lambda:owner.device.defaultMemRegion())
    chunk=stop.call('known.malloc',lambda:region.malloc(65536,False,64),owner.known_probe_chunks.append)
    base=int(chunk.begin.addr())
    if int(chunk.byte_size)!=65536 or not (base+65536<=owner.base or owner.base+owner.size<=base):
        raise ValueError('Known SDK allocation overlaps FPGA arena or has wrong extent')
    lease=LeaseLedger(boot_id=owner.info['boot_id'],allocation_id='known:'+str(owner.run_id),
        base=base,size=65536,quota=65536,run_id=owner.run_id,generation=owner.generation,retained_owner=chunk)
    owner.known_probe_ledgers.append(lease)
    stop.bind_ledger(lease)
    key=lease.reserve_batch(len(payload),[(first,last)],1)[0];lease.sdk_write_may_start([key])
    image=(owner.GUARD+payload+owner.GUARD*2+
           pack_output(tensor,first,last,run_id=owner.run_id,request_id=1,window_index=0,generation=owner.generation)+owner.GUARD)
    stop.call('known.write',lambda:chunk.write(0,np.frombuffer(image,dtype=np.uint8),len(image)))
    got=np.empty(len(image),dtype=np.uint8)
    stop.call('known.read_before',lambda:chunk.read(got,0,len(image)))
    if got.tobytes()!=image:raise ValueError('SDK known-input readback differs')
    prior=(owner.chunk,owner.ledger,owner.pending,owner.pending_index)
    try:
        owner.chunk,owner.ledger,owner.pending,owner.pending_index=chunk,lease,[key],0
        result=stop.call('known.predict',lambda:owner.predict_local(tensor))
    finally:owner.chunk,owner.ledger,owner.pending,owner.pending_index=prior
    got=np.empty(len(image),dtype=np.uint8)
    stop.call('known.read_after',lambda:chunk.read(got,0,len(image)))
    if got.tobytes()!=image:raise ValueError('Known SDK slot changed during inference')
    owner.known_probe_images.append(image)
    result['known_SDK_io']=dict(allocation_bytes=65536,slot_bytes=len(image),
        explicit_write_bytes=len(image),explicit_read_bytes=2*len(image),SDK_internal_transfer_bytes=None)
    return result


def run_probe(root,review,confirmation,label,*,review_sha256,execute,owner_factory,record,stop=None):
    """Caller retains owner_factory's instance even on partial initialization."""
    stop=stop if stop is not None else ProbeStop()
    stop.checkpoint('probe.before')
    previous=validate_request(root,review,confirmation,label,review_sha256)
    from .udw_e_preflight import capture_boot
    observation=capture_boot(lambda command:stop.call('preflight.shell',lambda:execute(command)),
        label='share_'+label,previous_boot_ids=[previous['boot_id']],
        expected_BOOT_sha256=review['expected_BOOT_sha256'],expected_SDK='3.36.1')
    record('preflight',observation)
    stop.checkpoint('preflight.recorded')
    packet_files(root,review) # source TOCTOU check immediately before owner factory
    owner=stop.call('owner.create',owner_factory,stop.bind_owner)
    owner.probe_stop=stop
    owner.known_probe_chunks=[];owner.known_probe_ledgers=[];owner.known_probe_images=[]
    try:
        packet_files(root,review)
        info=stop.call('owner.initialize',lambda:owner.initialize(root/'artifacts/icraft/tcn_ed4_ZG.json',
            observation['boot_id'],int(datetime.now().timestamp()*1000000),control='axi_registers'))
        stop.bind_ledger(owner.ledger)
        if info.get('versions')!=review['expected_device_versions']:raise ValueError('Device/core differs')
        record('initialized',info)
        stop.checkpoint('initialized.recorded')
        import json
        tensor=(root/'probe/golden.f32').read_bytes()
        expected=json.loads((root/'probe/expected.json').read_text())['logits']
        def verify(result):
            values=np.asarray(result['logits'])
            if (result['tensor_sha256']!=hashlib.sha256(tensor).hexdigest()
                or not np.allclose(values,expected,rtol=1e-3,atol=1e-4)
                or values.argmax()!=np.asarray(expected).argmax()):
                raise ValueError('Shared probe input/numerical reference mismatch')
        known=known_sdk_input(owner,tensor);verify(known);record('known_SDK_passed',known)
        stop.checkpoint('known.recorded')
        packet_files(root,review)
        payload,_,_=golden_records(tensor)
        pl=stop.call('PL.submit',lambda:owner.submit_local(payload,False))
        if len(pl['predictions'])!=1:raise ValueError('FPGA 100-to-1 output count differs')
        verify(pl['predictions'][0]);record('FPGA_passed',pl)
        stop.checkpoint('FPGA.recorded')
        if owner.call_count!=2:raise ValueError('Unexpected probe NPU call count')
        packet_files(root,review)
        stop.checkpoint('probe.before_return')
        return dict(status='bounded_shared_device_probe_verified',known=known,FPGA=pl,
            NPU_calls=2,owner_retained=True,strict_visibility_proven=False,zero_copy=False)
    except BaseException as error:
        owner.quarantine(error)
        for ledger in owner.known_probe_ledgers:ledger.stop(error)
        raise
