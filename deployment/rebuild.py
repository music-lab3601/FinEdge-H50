"""Rebuild native INT8 deployment from raw MBO, frozen weights and public source."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from shared.io import atomic_json, sha
from shared.paths import ConfigurationError, ROOT, environment, load_settings

ROW = np.dtype([('identity','<u8',(4,)),('raw','<u8',(40,)),('normalized','<f4',(40,))])


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def windows_from_rows(rows, endpoints):
    """Verify full input/H50 identities before materializing channel-first tensors."""
    tensors, labels = [], []
    for endpoint in endpoints:
        footprint = rows[endpoint-99:endpoint+51]
        ids = footprint['identity']
        if len(footprint)!=150 or not np.all(ids[:,0]==ids[0,0]):
            raise ValueError('Input/H50 footprint crosses a segment or file boundary')
        if not np.all(np.diff(ids[:,1].astype('i8'))==1) or not np.all(np.diff(ids[:,2].astype('i8'))>0):
            raise ValueError('Profile rows or source ordinals are discontinuous')
        tensors.append(footprint['normalized'][:100].T.copy())
        current = int(footprint['raw'][99,0])+int(footprint['raw'][99,2])
        future = int(footprint['raw'][149,0])+int(footprint['raw'][149,2])
        labels.append(2 if future>current else 0 if future<current else 1)
    return np.asarray(tensors,dtype='<f4'), np.asarray(labels,dtype='<i8')


def verify_normalized_rows(rows, normalization):
    mean = np.asarray(normalization['mean'],dtype='f8')
    std = np.asarray(normalization['std'],dtype='f8')
    for first in range(0,len(rows),8192):
        part = rows[first:first+8192]
        x = part['raw'].astype('f8')
        midpoint = (x[:,0]+x[:,2])*.5
        x[:,::2] = (x[:,::2]-midpoint[:,None])/midpoint[:,None]
        expected = ((x-mean)/std).astype('<f4')
        if not np.array_equal(expected,part['normalized']):
            raise ValueError('C++ reconstruction differs from frozen float64 preprocessing')


def error_metrics(reference, actual):
    if reference.shape!=actual.shape or not np.isfinite(actual).all():
        raise ValueError('Output shape or finite-value contract differs')
    return dict(samples=len(actual),max_abs=float(np.max(np.abs(actual-reference))),
                mean_abs=float(np.mean(np.abs(actual-reference))),
                class_changes=int(np.count_nonzero(reference.argmax(1)!=actual.argmax(1))),
                bit_equal=reference.tobytes()==actual.tobytes())


def quality(logits, labels):
    predictions = logits.argmax(1)
    matrix = np.zeros((3,3),dtype='i8')
    np.add.at(matrix,(labels,predictions),1)
    denominator = matrix.sum(0)+matrix.sum(1)
    f1 = np.divide(2*np.diag(matrix),denominator,out=np.zeros(3),where=denominator>0)
    return dict(correct=int(np.count_nonzero(predictions==labels)),samples=len(labels),
                accuracy=float(np.mean(predictions==labels)),macro_f1=float(f1.mean()),
                confusion=matrix.tolist())


def validate_native_model(model):
    entry = model['ops'][0]['outputs'][0]
    dtype = entry['dtype']
    if dtype['shape']!=[1,3,1,100,16] or dtype['element_dtype']['storage_dtype']!='@sint(8)':
        raise ValueError('Native input ABI differs')
    body = model['ops'][1:-1]
    if any(op['_type_key']!='icraft::xir::HardOp' or op.get('compile_target')!='@zhuget(330)' for op in body):
        raise ValueError('Generated compute body contains a non-ZG330 operator')
    layers = sum(op.get('layer_count',0) for op in body)
    if layers!=30 or entry['v_id']!=0 or model['ops'][-1]['inputs'][0]['v_id']!=250:
        raise ValueError('Layer count or board binding IDs differ')
    return dict(input_shape=dtype['shape'],input_bytes=4800,input_id=0,output_id=250,
                layers=layers,hardware_operations=len(body),compute_target='ZG330',
                compiler_version=model['icraft_version'])


class Rebuild:
    def __init__(self, directory, settings):
        self.directory = directory
        self.state = dict(status='running',created_utc=datetime.now(timezone.utc).isoformat(),
                          stages=[],board_access=False,profile='repo://contracts/model/rebuild_profile.json')
        self.config = directory/'configuration.json'
        settings['paths']['work_root'] = str(directory)
        atomic_json(self.config, settings)
        self.env = environment()
        self.env['FINEDGE_CONFIG'] = str(self.config)
        self.env['OMP_NUM_THREADS'] = self.env['MKL_NUM_THREADS'] = '1'
        self.logs = directory/'logs'
        self.logs.mkdir()

    def save(self):
        atomic_json(self.directory/'result.json',self.state)

    def run(self, name, command, timeout=1800):
        started = time.monotonic()
        path = self.logs/(name+'.log')
        with path.open('xb') as log:
            result = subprocess.run(list(map(str,command)),cwd=self.directory,env=self.env,
                                    stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        self.state['stages'].append(dict(stage=name,exit_code=result.returncode,
                                        seconds=time.monotonic()-started,log=path.relative_to(self.directory).as_posix(),
                                        log_sha256=sha(path)))
        self.save()
        print(name+' '+str(result.returncode),flush=True)
        if result.returncode:
            raise RuntimeError('Rebuild stage failed. '+str(path))
        return path

    def module(self, name, module, *arguments, timeout=1800):
        return self.run(name,[sys.executable,'-B','-m',module,*arguments],timeout)

    def host(self, name, model, inputs, integer=True):
        stage = 'native' if integer else 'base'
        executable = 'd_integer_host.exe' if integer else 'd_host_tools.exe'
        output = self.directory/'verification'/name
        output.mkdir(parents=True)
        self.run(name,[self.directory/'model_build'/stage/'build/Release'/executable,
                       '--mode','forward','--json',model,'--raw',model.with_suffix('.raw'),
                       '--inputs',inputs,'--out',output])
        return np.fromfile(output/'logits.f32',dtype='<f4').reshape(-1,3)


def rebuild_model(raw_mbo, work_directory):
    settings = load_settings()
    directory = Path(work_directory).resolve()
    raw_mbo = Path(raw_mbo).resolve()
    profile = read(ROOT/'contracts/model/rebuild_profile.json')
    if directory.exists():
        raise FileExistsError('Choose a new rebuild directory. '+str(directory))
    if not raw_mbo.is_file() or sha(raw_mbo)!=profile['raw_mbo_sha256']:
        raise ConfigurationError('Raw MBO must match the frozen rebuild profile SHA256')
    for tool in ('icraft','cmake'):
        if not settings['tools'][tool]:
            raise ConfigurationError('Configure --'+tool+' before the rebuild')
    directory.mkdir(parents=True)
    build = Rebuild(directory,settings)
    build.state['raw_mbo'] = dict(path=str(raw_mbo),sha256=sha(raw_mbo),bytes=raw_mbo.stat().st_size)
    build.state['source_files'] = {p.relative_to(ROOT).as_posix():sha(p) for folder in (
        'deployment','models','contracts/model') for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    build.save()
    try:
        return execute_rebuild(build,raw_mbo,profile)
    except BaseException as exc:
        build.state.update(status='failed',error=type(exc).__name__+': '+str(exc),automatic_retry=False)
        build.save()
        raise


def execute_rebuild(build, raw_mbo, profile):
    directory = build.directory
    base = directory/'model_build/base'
    mixed = directory/'model_build/mixed'
    native = directory/'model_build/native'
    for stage in ('base','mixed','native'):
        build.module('build_'+stage,'deployment.commands','tools','--stage',stage)
    data = directory/'data'
    data.mkdir()
    rows_path = data/'reconstructed_rows.bin'
    build.run('reconstruct',[mixed/'build/Release/d_fresh_book.exe',raw_mbo,rows_path,str(profile['reconstructed_records'])])
    if sha(rows_path)!=profile['reconstructed_rows_sha256']:
        raise ValueError('Reconstructed row bytes differ from the frozen engineering profile')
    rows = np.memmap(rows_path,dtype=ROW,mode='r')
    verify_normalized_rows(rows,read(ROOT/'models/normalization.json'))
    groups = {}
    for group in ('calibration','evaluation'):
        spec = profile[group]
        endpoints = np.linspace(spec['first_endpoint'],spec['last_endpoint'],spec['count'],dtype=np.int64)
        values, labels = windows_from_rows(rows,endpoints)
        path = data/(group+'.npy')
        np.save(path,values)
        np.save(data/(group+'_labels.npy'),labels)
        if sha(path)!=spec['inputs_sha256']:
            raise ValueError('Reconstructed '+group+' windows differ from frozen identities')
        groups[group] = dict(path=str(path),sha256=sha(path),windows=len(values),endpoints=endpoints.tolist())
    del rows
    atomic_json(data/'manifest.json',dict(raw_sha256=sha(raw_mbo),rows_sha256=sha(rows_path),
                normalization_sha256=sha(ROOT/'models/normalization.json'),normalized_rows_checked=profile['reconstructed_records'],
                groups=groups,profile_sha256=sha(ROOT/'contracts/model/rebuild_profile.json')))
    build.state['data'] = groups
    build.save()
    build.module('export','deployment.commands','export')
    build.module('polyphase','deployment.commands','polyphase','--label','rebuild')
    graph = directory/'runtime/subhalf/rebuild/polyphase.onnx'
    import onnx
    candidate = onnx.load(graph)
    frozen = onnx.load(ROOT/'models/onnx/tcn_polyphase.onnx')
    if candidate.graph.SerializeToString()!=frozen.graph.SerializeToString():
        raise ValueError('Regenerated polyphase graph differs from frozen deployment graph')
    (base/'model').mkdir(exist_ok=True)
    shutil.copyfile(graph,base/'model/tcn_polyphase.onnx')
    build.module('calibration','deployment.commands','calibration','--input',data/'calibration.npy')
    build.module('tf32','deployment.commands','compile','--kind','tf32')
    build.module('int8','deployment.commands','compile','--kind','int8')
    (mixed/'artifacts').mkdir(exist_ok=True)
    build.module('diagnostics','deployment.icraft.mixed.diagnose')
    build.module('mixed','deployment.commands','compile','--kind','mixed')
    build.module('native','deployment.commands','compile','--kind','native')
    folded = native/'artifacts/int8_folded/tcn_d_integer_folded_quantized.json'
    source = mixed/'artifacts/mixed_tf32_compat/tcn_d_quantized.json'
    audit_path = build.run('fold_audit',[native/'build/Release/d_audit_weights.exe',source,source.with_suffix('.raw'),
                                        folded,folded.with_suffix('.raw'),ROOT/'models/npu/scales.f32'])
    build.state['fold_audit'] = read(audit_path)
    build.module('codec','deployment.icraft.native.screen3')
    new_model = native/'artifacts/int8_physical/d3p_ZG.json'
    model = read(new_model)
    model_interface = validate_native_model(model)
    values = np.load(data/'evaluation.npy',allow_pickle=False)
    from deployment.icraft.native.codec3 import encode
    integer_input = data/'evaluation.i8'
    integer_input.write_bytes(b''.join(encode(value[None]) for value in values))
    new_logits = build.host('new_native',new_model,integer_input)
    frozen_logits = build.host('frozen_native',ROOT/'models/npu/model.json',integer_input)
    reproduction = error_metrics(frozen_logits,new_logits)
    if not reproduction['bit_equal']:
        np.save(directory/'verification/frozen_native/different_rows.npy',np.flatnonzero(np.any(new_logits!=frozen_logits,axis=1)))
        raise ValueError('Rebuilt and frozen native Host outputs differ')
    from software.runtime.cpu import Predictor
    identity = read(ROOT/'models/identity.json')
    predictor = Predictor(ROOT/'models/cpu',identity['CPU_manifest_sha256'])
    cpu_logits = np.concatenate([predictor.predict_normalized(values[i:i+32])['logits'] for i in range(0,len(values),32)])
    np.save(directory/'verification/cpu_logits.npy',cpu_logits)
    labels = np.load(data/'evaluation_labels.npy',allow_pickle=False)
    reference_quality, native_quality = quality(cpu_logits,labels),quality(new_logits,labels)
    drops = dict(accuracy_percentage_points=(reference_quality['accuracy']-native_quality['accuracy'])*100,
                 macro_f1_percentage_points=(reference_quality['macro_f1']-native_quality['macro_f1'])*100)
    if drops['accuracy_percentage_points']>profile['accuracy_drop_percentage_points_max'] or drops['macro_f1_percentage_points']>profile['macro_f1_drop_percentage_points_max']:
        raise ValueError('Frozen quantization quality budget exceeded')
    result_model = dict(json_sha256=sha(new_model),raw_sha256=sha(new_model.with_suffix('.raw')),
                        frozen_json_byte_equal=sha(new_model)==identity['NPU_model_sha256'],
                        frozen_raw_byte_equal=sha(new_model.with_suffix('.raw'))==identity['NPU_raw_sha256'],
                        **model_interface,path=str(new_model))
    build.state.update(status='passed',model=result_model,reproduction=reproduction,
                       error_vs_cpu=error_metrics(cpu_logits,new_logits),quality_cpu=reference_quality,
                       quality_native=native_quality,quality_drop=drops,
                       source_files_unchanged=all(sha(ROOT/name)==digest for name,digest in build.state['source_files'].items()),
                       raw_source_unchanged=sha(raw_mbo)==profile['raw_mbo_sha256'],
                       quantization='INT8 native input; frozen mixed TF32/INT8 body',board_validation='pending')
    if not build.state['source_files_unchanged'] or not build.state['raw_source_unchanged']:
        raise ValueError('Source files changed during rebuild')
    build.save()
    return build.state
