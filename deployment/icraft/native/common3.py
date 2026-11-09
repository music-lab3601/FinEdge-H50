from shared.paths import resolve_path
from pathlib import Path
import hashlib,json,os,sys
sys.dont_write_bytecode=True
from shared.paths import ROOT as PROJECT, work_path
SOURCE=Path(__file__).resolve().parent
ROOT=work_path('model_build/native');BASE=work_path('model_build/base');V2=work_path('model_build/mixed')
SDK=Path(str(resolve_path('tool://icraft/')))
def read(p):
    from shared.paths import read_config
    return read_config(p)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,value):
    p=Path(p)
    if not p.resolve().is_relative_to(ROOT):raise ValueError('Output outside D v3')
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.writing')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(t,p)
def env():
    e=os.environ.copy();e.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TEMP=str(ROOT/'.tmp'),TMP=str(ROOT/'.tmp'))
    e['PATH']=str(SDK/'bin')+os.pathsep+e.get('PATH','');return e
