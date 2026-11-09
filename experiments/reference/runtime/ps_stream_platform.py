"""Frozen PSW stream admission; Python 3.8, no device access during validation."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from experiments.reference.runtime.ps_window_runtime import ARENA_BYTES

SCHEMA = 'psw-stream-session-v1'
BOOT_SHA = '1cf2a4bfa294a33251124db8831eb6d1a1cf3f440ff9853ae6e8bd8df605c828'
SDK_DIR = Path('/usr/lib/aarch64-linux-gnu')
MODULE_ORDER = ('xir','xrt','host_backend','zg330backend')
LIBRARIES = {'libaxizg330aiu.so','libsocketzg330aiu.so','libicraft_xrt.so',
    'libicraft_hostbackend.so','libicraft_zg330backend.so','libicraft_xir.so'}
SOURCES = ('finedge_host/ps_window_runtime.py','finedge_host/ps_npu_runtime.py',
    'finedge_host/ps_window_probe.py','finedge_host/udw_probe_stop.py',
    'finedge_host/ps_stream_protocol.py','finedge_host/ps_stream_server.py',
    'finedge_host/ps_stream_platform.py')
FILES = {'finedge_host/__init__.py',*SOURCES,'ps_stream_service.py',
    'tcn_ed4_ZG.json','tcn_ed4_ZG.raw','golden1.f32','golden_reference.json','cache_contract.json'}
MAX_WINDOWS = 1126606
DATA_SCHEMA = 'psw-stream-dataset-v1'
DATA_BINDING = dict(dataset_id='databento20251215',
    source_sha256='0e69b858781b8c7d82e4563b5098f70ff45b02a17c8c3a88aa8880c430ba2fd7',
    status_sha256='c1d0ee67f1bc5bdab40c4b497684ef056f205b12f2d790bd24756f58e3891fe3',
    profile='AAPL-DBN-development-20251215-v1', frames=5867517,records=5125446,windows=5125347,
    original_validation_alignment=False)
PATH_SCHEMA = 'psw-stream-pathopt-v1'
PATH_SOURCES = ('finedge_host/ps_path_profile.py', 'finedge_host/ps_path_runtime.py')


def file_set(cfg):
    if cfg.get('schema') == DATA_SCHEMA:
        if cfg.get('dataset_binding') != DATA_BINDING or 'path_profile' in cfg:
            raise ValueError('Explicit dataset scope differs or selects optimization profile')
        return FILES
    if cfg.get('schema') == SCHEMA:
        if 'path_profile' in cfg or 'dataset_binding' in cfg: raise ValueError('Legacy schema cannot select another profile')
        return FILES
    if cfg.get('schema') == PATH_SCHEMA:
        if 'dataset_binding' in cfg: raise ValueError('Optimization schema cannot select dataset profile')
        from experiments.reference.runtime.ps_path_profile import validate_profile
        validate_profile(cfg.get('path_profile'))
        return FILES | set(PATH_SOURCES)
    raise ValueError('Unknown explicit stream schema')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path=Path(path)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    tmp.replace(path)


def validate_files(cfg, directory):
    directory=Path(directory).resolve()
    files = file_set(cfg)
    expected=dict(schema=cfg['schema'],expected_BOOT_sha256=BOOT_SHA,arena_bytes=ARENA_BYTES,
        max_windows=10000 if cfg['schema']==PATH_SCHEMA else DATA_BINDING['windows'] if cfg['schema']==DATA_SCHEMA else MAX_WINDOWS,warmups=10,golden_checks=1,listen_host='127.0.0.1',listen_port=9982,
        expected_udma_range={'base':0x30100000,'bytes':134217728},max_RSS_kib=393216)
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Frozen PSW full-day stream scope differs')
    if (set(cfg.get('remote_hashes',{}))!=files or set(cfg.get('ARM_extensions',{}))!=set(MODULE_ORDER)
            or set(cfg.get('SDK_libraries',{}))!={'/usr/lib/aarch64-linux-gnu/'+n for n in LIBRARIES}
            or type(cfg.get('run_id')) is not int or not 0<cfg['run_id']<2**64):
        raise ValueError('Complete frozen manifest and uint64 run identity required')
    for group in ('remote_hashes','ARM_extensions','SDK_libraries'):
        if any(not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) for h in cfg[group].values()):
            raise ValueError('Concrete file hashes required')
    for name,h in cfg['remote_hashes'].items():
        p=(directory/name).resolve()
        if directory not in p.parents or sha(p)!=h:
            raise ValueError('Frozen stream file changed: '+name)
    cache=json.loads((directory/'cache_contract.json').read_text(encoding='utf-8'))
    if (sha(directory/'cache_contract.json')!=cfg.get('cache_contract_sha256')
            or cache.get('status')!='static_chain_verified' or cache.get('strict_visibility_proven') is not False
            or cache.get('libraries')!=cfg['SDK_libraries'] or cache.get('server_sha256')!=cfg.get('server_sha256')):
        raise ValueError('Cache contract does not bind installed binaries')
    ref=json.loads((directory/'golden_reference.json').read_text(encoding='utf-8'))
    if (ref.get('compiled_json_sha256')!=sha(directory/'tcn_ed4_ZG.json')
            or ref.get('compiled_raw_sha256')!=sha(directory/'tcn_ed4_ZG.raw')
            or ref.get('input_sha256')!=sha(directory/'golden1.f32')):
        raise ValueError('Golden reference is not bound to model/input')
    tensor=(directory/'golden1.f32').read_bytes()
    values=np.asarray(ref['logits'])
    if len(tensor)!=16000 or not np.isfinite(np.frombuffer(tensor,dtype='<f4')).all() or values.shape!=(3,) or not np.isfinite(values).all():
        raise ValueError('Finite golden tensor and three logits required')
    return tensor,ref


def verify_platform(cfg, boot_id):
    if (cfg.get('boot_id')!=boot_id or cfg['preflight']['boot_id']!=boot_id
            or cfg['preflight']['actual_BOOT_sha256']!=BOOT_SHA or cfg['preflight']['SDK_version']!='3.36.1'
            or not re.fullmatch('[0-9a-f]{64}',cfg.get('token',''))):
        raise ValueError('Fresh stream session preflight and private authentication required')
    version=subprocess.check_output(['dpkg-query','-W','-f=${Version}','icraft'],text=True).strip()
    if not re.fullmatch(r'3\.36\.1(?:[-+].*)?',version):raise ValueError('SDK version differs')
    for name,h in cfg['SDK_libraries'].items():
        path=Path(name)
        if path.parent!=SDK_DIR or sha(path)!=h:raise ValueError('SDK library differs: '+name)
    if sha('/usr/bin/icraft-serve')!=cfg['server_sha256']:raise ValueError('Server binary differs')
    servers=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:executable=(process/'exe').resolve(strict=True)
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if executable.name!='icraft-serve':continue
        argv=(process/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        if (executable!=Path('/usr/bin/icraft-serve') or len(argv)!=1
                or Path(os.fsdecode(argv[0])).name!='icraft-serve' or sha(process/'exe')!=cfg['server_sha256']):
            raise ValueError('Exactly one default synchronous SDK server required')
        servers.append(int(process.name))
    if len(servers)!=1:raise ValueError('Exactly one default synchronous SDK server required')
    sysfs=Path('/sys/class/u-dma-buf/udmabuf0')
    values={k:int((sysfs/k).read_text().strip(),0) for k in ('phys_addr','size','dma_coherent','sync_mode')}
    if values!={'phys_addr':0x30100000,'size':134217728,'dma_coherent':0,'sync_mode':1}:
        raise ValueError('Actual PS extent/cache configuration differs')
    return dict(udma=values,synchronous_server_pid=servers[0])


def load_modules(cfg):
    sys.path.insert(0,str(SDK_DIR))
    modules={}
    for name in MODULE_ORDER:
        module=importlib.import_module(name)
        if Path(module.__file__).parent!=SDK_DIR or sha(module.__file__)!=cfg['ARM_extensions'][name]:
            raise ValueError('ARM extension identity differs: '+name)
        modules[name]=module
    return modules
