from shared.paths import resolve_path
from pathlib import Path
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
from shared.paths import ROOT as PROJECT, work_path
SOURCE = Path(__file__).resolve().parent
ROOT = work_path('model_build/base')
PACKAGE = PROJECT / 'models/cpu'
ICRAFT = Path(str(resolve_path('tool://icraft/')))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    from shared.paths import read_config
    return read_config(path)


def save(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('D output escaped its owned directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def local_env():
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               TEMP=str(ROOT / '.tmp'), TMP=str(ROOT / '.tmp'))
    env['PATH'] = str(ICRAFT / 'bin') + os.pathsep + env.get('PATH', '')
    return env
