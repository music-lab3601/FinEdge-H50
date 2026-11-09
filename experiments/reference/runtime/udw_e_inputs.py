"""Prepare immutable fixed E inputs/references before any board access."""
from copy import deepcopy
from time import perf_counter_ns
import numpy as np
from shared.model_paths import ROOT, PACKAGE, RAW, MANIFEST_SHA
from software.runtime.delivery import read_json, sha256
from software.runtime.stream import Record
from experiments.reference.runtime.udw_lab import digest
from experiments.reference.runtime.udw_e_frontend import CandidateFrontend
from experiments.reference.runtime.udw_e_offline import build_references
from experiments.reference.runtime.udw_e_release import PLAN
from experiments.reference.runtime.udw_measurement_plan import validate_plan
from software.runtime.backends import OnnxBackend


def prepare_inputs():
    begin=perf_counter_ns();plan_path=ROOT/PLAN;plan=read_json(plan_path);validate_plan(plan)
    if sha256(PACKAGE/'package_manifest.json')!=MANIFEST_SHA:raise ValueError('Model manifest changed')
    original=read_json(PACKAGE/'package_manifest.json')
    for name in ('normalization.json','golden/inputs.f32'):
        if sha256(PACKAGE/name)!=original['files'][name]['sha256']:raise ValueError('Original model data changed: '+name)
    hashes=dict(plan['input_files_sha256'])
    extra=[PACKAGE/'golden/inputs.f32',RAW/'window_inputs.f32le.bin',
        ROOT/'artifacts/reports/npu_verification.json',
        ROOT/'artifacts/fpga_udw_runtime/numeric_20261024_01/result.json',
        ROOT/'artifacts/export/tcn_ed4.onnx',ROOT/'artifacts/icraft/tcn_ed4_ZG.json',
        ROOT/'artifacts/icraft/tcn_ed4_ZG.raw',plan_path]
    for p in extra:hashes[str(p.relative_to(ROOT))]=sha256(p)
    for name,h in hashes.items():
        if sha256(ROOT/name)!=h:raise ValueError('Fixed input changed: '+name)
    model=plan['arms'][0]['model_hashes']
    for suffix in ('json','raw'):
        if sha256(ROOT/f'artifacts/icraft/tcn_ed4_ZG.{suffix}')!=model[suffix]:raise ValueError('ZG model changed')
    if sha256(ROOT/'artifacts/export/tcn_ed4.onnx')!=plan['arms'][2]['model_hashes']['onnx']:
        raise ValueError('ONNX model changed')
    base=ROOT/'artifacts/fpga_udw_runtime/local_review_20261024_01'
    norm=read_json(PACKAGE/'normalization.json');frontend=CandidateFrontend(norm,'H')
    records=[];batches=[]
    for i in range(1,9):
        meta=read_json(base/f'batch_{i:03d}_metadata.json')
        raw=np.fromfile(base/f'batch_{i:03d}_raw_book.u64le.bin',dtype='<u8').reshape(-1,40)
        if len(raw)!=64 or len(meta)!=64:raise ValueError('Fixed 64-record batches required')
        batch=[];wire=[]
        for values,row in zip(raw,meta,strict=True):
            record=Record(values.copy(),deepcopy(row['metadata']))
            record.values.flags.writeable=False
            batch.append((record,tuple(row['identity'])));records.append(record)
            wire.append(frontend.pack(frontend.push(record,row['identity'])))
        if b''.join(wire)!=(base/f'batch_{i:03d}_input.bin').read_bytes():raise ValueError('Raw/WDM mismatch')
        batches.append(batch)
    fixed=np.concatenate((np.fromfile(PACKAGE/'golden/inputs.f32',dtype='<f4').reshape(32,1,40,100),
        np.fromfile(RAW/'window_inputs.f32le.bin',dtype='<f4').reshape(6,1,40,100)))
    fixed=[x.tobytes() for x in fixed]
    refs=read_json(ROOT/'artifacts/reports/npu_verification.json')['comparisons']
    logits={digest(x):{'npu':r['host']['logits']} for x,r in zip(fixed,refs,strict=True)}
    reference=build_references(records,norm,fixed,logits)
    numerical=read_json(ROOT/'artifacts/fpga_udw_runtime/numeric_20261024_01/result.json')['comparisons']
    for row in numerical:
        tensor=reference['windows'][tuple(row['last'])]['tensor']
        reference['logits'].setdefault(digest(tensor),{})['npu']=row['Host']['logits']
    cpu=OnnxBackend();cpu_calls=0
    for tensor in fixed+[reference['windows'][key]['tensor'] for key in reference['expected_ids']]:
        result=cpu.predict(cpu_calls+1,np.frombuffer(tensor,dtype='<f4').reshape(1,40,100));cpu_calls+=1
        reference['logits'].setdefault(digest(tensor),{})['cpu']=result['logits']
    if reference['expected_records']!=448 or [list(x) for x in reference['expected_ids']]!=plan['schedule']['selected_identities']:
        raise ValueError('Fixed E workload differs from references')
    for name,h in hashes.items():
        if sha256(ROOT/name)!=h:raise ValueError('Input changed during preparation: '+name)
    return dict(plan=plan,batches=batches,references=reference,normalization=norm,
        provenance=dict(input_hashes=hashes,preparation_ns=perf_counter_ns()-begin,
                        CPU_reference_calls=cpu_calls,records_prepared=512,windows_to_submit=22,
                        source='reviewed_D_raw_book_snapshots_not_network_ITCH'))
