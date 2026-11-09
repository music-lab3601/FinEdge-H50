"""Two-phase Host copy diagnostic. An old shared-probe release never opens it."""
from datetime import datetime
import re

import numpy as np

from .delivery import read_json, sha256
from .udw_shared_probe import DEVICE_VERSIONS, FINDINGS as OLD_FINDINGS, SCOPE as OLD_SCOPE
from .udw_shared_probe import packet_files as old_packet_files, golden_records
from .udw_shared_tensor import LeaseLedger
from .udw_probe_stop import ProbeStop
from .udw_e_preflight import capture_boot
from .ddr_transport_contract import pack_output

SCOPE = dict(OLD_SCOPE, schema='host-copy-device-probe-v1',
             copy_api='Tensor.to(HostDevice.MemRegion())', verify_copy_bytes=16000,
             Cast_return_contract='L02_saved_adapted_dtype',
             performance_experiment=False)
FINDINGS = OLD_FINDINGS | {'SHARED_DEVICE_REVIEW_01'}


def packet_files(root, review):
    required = {'src/finedge_host/udw_copy_probe.py', 'src/finedge_host/udw_copy_probe_owner.py',
                'scripts/udw_copy_probe_check.py', 'probe/sdk_interface.json', 'probe/cast_contract.json'}
    if not required <= set(review.get('packet_hashes', {})):
        raise ValueError('Incomplete copy diagnostic packet')
    old_packet_files(root, review)


def validate_request(root, review, confirmation, label, review_sha256):
    if not re.fullmatch(r'[A-Za-z0-9_]{1,40}', label):
        raise ValueError('Safe unique copy probe label required')
    if (review.get('device_probe_approved') is not True
            or not FINDINGS <= set(review.get('closed_findings', []))
            or review.get('scope') != SCOPE or review.get('expected_device_versions') != DEVICE_VERSIONS):
        raise ValueError('Copy diagnostic review/scope is not approved')
    packet_files(root, review)
    if (confirmation.get('source') != 'human_user_current_session'
            or confirmation.get('physical_power_removed') is not True
            or confirmation.get('linux_boot_complete') is not True
            or confirmation.get('label') != label or not confirmation.get('verbatim_reply', '').strip()
            or not re.fullmatch(r'[0-9a-f]{64}', review_sha256)
            or confirmation.get('review_sha256') != review_sha256):
        raise ValueError('Fresh human confirmation bound to copy probe required')
    previous = read_json(root / 'probe/previous.json')
    if confirmation.get('previous_result_sha256') != previous['original_result_sha256']:
        raise ValueError('Copy probe predecessor differs')
    return previous


def known_sdk_input(owner, tensor):
    owner.trace.context = {'phase': 'known_SDK'}
    owner.ensure_active()
    payload, first, last = golden_records(tensor)
    region = owner.sdk('known.region', owner.device.defaultMemRegion)
    chunk = owner.sdk('known.malloc', lambda: region.malloc(65536, False, 64), owner.known_probe_chunks.append)
    ptr = owner.sdk('known.begin', lambda: chunk.begin)
    kind = owner.sdk('known.ptype', ptr.ptype)
    if kind != owner.xrt.PtrType.ADDR:
        raise ValueError('Known source requires ADDR allocation')
    base = owner.sdk('known.base', lambda: int(ptr.addr()))
    size = owner.sdk('known.size', lambda: int(chunk.byte_size))
    owner.trace.event('known_allocation', dict(base=base, bytes=size, ptype='ADDR',
        allocation_id='known:' + str(owner.run_id), boot_id=owner.info['boot_id'],
        main_arena_base=owner.base, main_arena_bytes=owner.size))
    if size != 65536 or not (base+size <= owner.base or owner.base+owner.size <= base):
        raise ValueError('Known SDK allocation overlaps FPGA arena or has wrong extent')
    lease = LeaseLedger(boot_id=owner.info['boot_id'], allocation_id='known:' + str(owner.run_id),
        base=base, size=size, quota=size, run_id=owner.run_id, generation=owner.generation, retained_owner=chunk)
    owner.known_probe_ledgers.append(lease)
    owner.stop_control.bind_ledger(lease)
    key = lease.reserve_batch(len(payload), [(first, last)], 1)[0]
    lease.sdk_write_may_start([key])
    image = (owner.GUARD + payload + owner.GUARD*2 + pack_output(tensor, first, last,
        run_id=owner.run_id, request_id=1, window_index=0, generation=owner.generation) + owner.GUARD)
    owner.trace.event('known_slot_io_plan', dict(slot_bytes=len(image), payload_bytes=len(payload),
        logical_tensor_bytes=16000, write_API_bytes=len(image), each_full_read_API_bytes=len(image),
        SDK_internal_transfer_bytes=None, device_bus_bytes=None))
    owner.sdk('known.write', lambda: chunk.write(0, np.frombuffer(image, dtype=np.uint8), len(image)))
    data = np.empty(len(image), dtype=np.uint8)
    owner.sdk('known.read_before', lambda: chunk.read(data, 0, len(image)))
    if data.tobytes() != image:
        raise ValueError('SDK known-input readback differs')
    prior = owner.chunk, owner.ledger, owner.pending, owner.pending_index
    try:
        owner.chunk, owner.ledger, owner.pending, owner.pending_index = chunk, lease, [key], 0
        result = owner.predict_local(tensor)
    finally:
        owner.chunk, owner.ledger, owner.pending, owner.pending_index = prior
    owner.sdk('known.read_after', lambda: chunk.read(data, 0, len(image)))
    if data.tobytes() != image:
        raise ValueError('Known SDK slot changed during inference')
    owner.known_probe_images.append(image)
    result['known_SDK_io'] = dict(allocation_bytes=size, slot_bytes=len(image),
        explicit_write_bytes=len(image), explicit_read_bytes=2*len(image),
        Tensor_to_logical_bytes=16000, Host_copy_verify_API_read_bytes=16000,
        SDK_internal_transfer_bytes=None, device_bus_bytes=None)
    return result


def run_probe(root, review, confirmation, label, *, review_sha256, execute, owner_factory, record, stop=None):
    stop = stop if stop is not None else ProbeStop()
    stop.checkpoint('copy_probe.before')
    previous = validate_request(root, review, confirmation, label, review_sha256)
    observation = capture_boot(lambda command: stop.call('preflight.shell', lambda: execute(command)),
        label='copy_' + label, previous_boot_ids=[previous['boot_id']],
        expected_BOOT_sha256=review['expected_BOOT_sha256'], expected_SDK='3.36.1')
    record('preflight', observation)
    stop.checkpoint('preflight.recorded')
    packet_files(root, review)
    owner = stop.call('owner.create', owner_factory, stop.bind_owner)
    owner.known_probe_chunks = []; owner.known_probe_ledgers = []; owner.known_probe_images = []
    try:
        info = stop.call('owner.initialize', lambda: owner.initialize(root/'artifacts/icraft/tcn_ed4_ZG.json',
            observation['boot_id'], int(datetime.now().timestamp()*1000000), control='axi_registers'))
        stop.bind_ledger(owner.ledger)
        if info.get('versions') != DEVICE_VERSIONS:
            raise ValueError('Device/core differs')
        record('initialized', info)
        stop.checkpoint('initialized.recorded')
        packet_files(root, review)
        tensor = (root/'probe/golden.f32').read_bytes()
        expected = read_json(root/'probe/expected.json')['logits']

        def verify(result):
            values = np.asarray(result['logits'])
            if (result['tensor_sha256'] != sha256(root/'probe/golden.f32')
                    or not np.allclose(values, expected, rtol=1e-3, atol=1e-4)
                    or values.argmax() != np.asarray(expected).argmax()):
                raise ValueError('Copy diagnostic numerical/input reference mismatch')

        known = known_sdk_input(owner, tensor)
        verify(known)
        record('known_SDK_passed', known)
        stop.checkpoint('known.recorded')
        packet_files(root, review)
        owner.trace.context = {'phase': 'FPGA'}
        payload, _, _ = golden_records(tensor)
        pl = stop.call('PL.submit', lambda: owner.submit_local(payload, False))
        if len(pl['predictions']) != 1:
            raise ValueError('FPGA 100-to-1 count differs')
        verify(pl['predictions'][0])
        record('FPGA_passed', pl)
        stop.checkpoint('FPGA.recorded')
        if owner.call_count != 2:
            raise ValueError('Unexpected copy diagnostic NPU call count')
        packet_files(root, review)
        return dict(status='bounded_explicit_copy_diagnostic_verified', known=known, FPGA=pl,
            NPU_calls=2, owner_retained=True, strict_visibility_proven=False, zero_copy=False,
            performance_experiment=False, SDK_internal_transfer_bytes=None)
    except BaseException as error:
        stop.isolate(error)
        raise
