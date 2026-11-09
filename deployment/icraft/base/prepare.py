"""Offline PTQ dataset from a provenance-bound current independent trace."""
import json
import shutil
import sys
import numpy as np
from deployment.icraft.base.common import ROOT, PROJECT, PACKAGE, read, save, sha

TRACE = np.dtype([('wdm', 'V256'), ('raw', '<u8', (40,)), ('opportunity', '<u8'),
                  *[(key, '<u8') for key in ('due','published','received','bookdone','enqueued','dispatch','available','backlog')],
                  ('result', 'V152')])
assert TRACE.itemsize == 800


def normalize(raw, normalization):
    x = np.array(raw, dtype=np.float64, copy=True)
    mid = (x[:, 0] + x[:, 2]) / 2.0
    x[:, 0::2] = (x[:, 0::2] - mid[:, None]) / mid[:, None]
    return ((x - np.array(normalization['mean'])) / np.array(normalization['std'])).astype('<f4')


def endpoints():
    # Frozen before looking at evaluation logits or labels.
    return np.linspace(99, 4095, 256, dtype=np.int64), np.linspace(5000, 10048, 512, dtype=np.int64)


def validate_footprints(ids, selected, horizon=50):
    for end in selected:
        footprint = ids[end-99:end+horizon+1]
        if len(footprint) != 150 or not np.all(footprint[:, 0] == footprint[0, 0]):
            raise ValueError('Incomplete/cross-segment input+label footprint')
        if not np.all(np.diff(footprint[:, 1].astype(np.int64)) == 1):
            raise ValueError('Nonconsecutive original profile rows')
        if not np.all(np.diff(footprint[:, 2].astype(np.int64)) > 0):
            raise ValueError('Original raw identities regressed')


def main():
    out = ROOT / 'data'
    out.mkdir(exist_ok=False)
    source = PROJECT / 'versions/network_fast_v1/artifacts/run_20261028_183114_latest'
    refs = [PACKAGE/'model_cpu.ts', PACKAGE.parent/'normalization.json', PACKAGE/'package_manifest.json',
            PACKAGE/'golden/inputs.f32', PACKAGE/'golden/script_batch1.f32',PACKAGE/'golden/script_batch32.f32',PROJECT/'software/runtime/cpu.py',
            source/'records.bin', source/'contract.json', source/'analysis.json',
            PROJECT/'versions/realtime_integration_v1/contracts/r1/contract.json',
            PROJECT/'artifacts/subhalf/polyphase_20261027_01/polyphase.onnx',
            PROJECT/'artifacts/export/tcn_ed4.onnx']
    hashes = {str(p): sha(p) for p in refs}
    contract = read(source/'contract.json')
    expected_source = '0e69b858781b8c7d82e4563b5098f70ff45b02a17c8c3a88aa8880c430ba2fd7'
    if contract['cache']['source_sha256'] != expected_source:
        raise ValueError('Saved trace source date/provenance differs')
    raw_cache = PROJECT / 'versions/network_fast_v1/artifacts/raw_mbo56.bin'
    if sha(raw_cache) != contract['cache']['raw_sha256']:
        raise ValueError('Original MBO cache hash differs')
    trace = np.fromfile(source/'records.bin', dtype=TRACE)
    ids = np.stack([np.frombuffer(x.tobytes(), dtype='<u8', count=4) for x in trace['wdm']])
    rows = normalize(trace['raw'], read(PACKAGE.parent/'normalization.json'))
    saved_rows = np.stack([np.frombuffer(x.tobytes(), dtype='<f4', count=40, offset=64) for x in trace['wdm']])
    if not np.array_equal(rows, saved_rows):
        raise ValueError('Independent frozen preprocessing differs from saved stream')
    cal, evaluation = endpoints()
    validate_footprints(ids, np.concatenate((cal, evaluation)))
    if int(cal.max())+50 >= int(evaluation.min())-99:
        raise ValueError('Calibration and evaluation footprints overlap')
    np.save(out/'rows.f32.npy', rows)
    np.save(out/'raw.u64.npy', trace['raw'])
    np.save(out/'identities.u64.npy', ids)
    groups = {}
    for name, selected in [('calibration', cal), ('evaluation', evaluation)]:
        windows = np.stack([rows[e-99:e+1].T for e in selected]).astype('<f4')
        mid2 = [int(trace['raw'][e][0]) + int(trace['raw'][e][2]) for e in selected]
        future = [int(trace['raw'][e+50][0]) + int(trace['raw'][e+50][2]) for e in selected]
        labels = np.array([2 if b>a else 0 if b<a else 1 for a,b in zip(mid2,future)], dtype=np.int64)
        np.save(out/f'{name}.inputs.npy', windows)
        np.save(out/f'{name}.labels.npy', labels)
        np.save(out/f'{name}.endpoints.npy', selected)
        items = []
        if name == 'calibration':
            (out/'calibration_bin').mkdir()
        for i,(end,window) in enumerate(zip(selected, windows)):
            binary = f'calibration_bin/{i:04d}.bin' if name=='calibration' else None
            if binary:
                window.tofile(out/binary)
            items.append(dict(index=i, endpoint_accepted_index=int(end),
                              input_accepted_range=[int(end)-99,int(end)],
                              label_accepted_index=int(end)+50,
                              first_identity=ids[end-99].tolist(),last_identity=ids[end].tolist(),
                              future_identity=ids[end+50].tolist(),
                              input_sha256=__import__('hashlib').sha256(window.tobytes()).hexdigest(),
                              calibration_file=binary))
        groups[name] = dict(count=len(selected), class_counts=np.bincount(labels,minlength=3).tolist(),items=items)
    (out/'forward_list.txt').write_text(''.join(f'{i:04d}.bin\n' for i in range(len(cal))),encoding='utf-8')
    save(out/'manifest.json', dict(schema='D_isolated_ptq_data_v1',status='prepared',
        purpose='offline_exploratory_same_day_engineering_screening',date='2025-12-15',symbol='AAPL',
        training_dates_available=False,original_validation_alignment=False,
        source_sha256=expected_source, source_trace=str(source/'records.bin'),
        source_trace_hash=hashes[str(source/'records.bin')],source_hashes=hashes,
        source_records=len(trace), normalization_bit_equal=True,
        interface_contract_version='finedge_realtime_r1',
        interface_contract_sha256=hashes[str(PROJECT/'versions/realtime_integration_v1/contracts/r1/contract.json')],
        label_rule='integer ask1+bid1 at endpoint+50 versus endpoint; no threshold; same original segment',
        calibration_max_footprint_index=int(cal.max())+50,
        evaluation_min_footprint_index=int(evaluation.min())-99,
        unused_records_between_footprints=int(evaluation.min())-99-(int(cal.max())+50)-1,
        declared_endpoint_gap_records=int(evaluation.min()-cal.max()-1),groups=groups,
        quality_budget_predeclared=dict(accuracy_drop_percentage_points_max=1.,macro_f1_drop_percentage_points_max=1.,
            candidates_max=2,base_int8_max=1,mixed_extra_max=1)))
    model_dir = ROOT/'model'; model_dir.mkdir()
    shutil.copyfile(PROJECT/'artifacts/subhalf/polyphase_20261027_01/polyphase.onnx',model_dir/'tcn_polyphase.onnx')
    shutil.copyfile(PACKAGE.parent/'normalization.json',model_dir/'normalization.json')
    shutil.copyfile(PROJECT/'versions/realtime_integration_v1/contracts/r1/contract.json',ROOT/'contracts/core_r1.json')
    save(ROOT/'artifacts/progress.json',dict(stage='data_ready',calibration=len(cal),evaluation=len(evaluation),
        fp32_source_model='delivered TorchScript',candidate_graph='same 39-node polyphase ONNX for TF32 and INT8',board_accessed=False))
    print(json.dumps(dict(status='prepared',calibration=len(cal),evaluation=len(evaluation),rows=len(trace))))


if __name__=='__main__':main()
