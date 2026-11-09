from pathlib import Path
import hashlib,json,os,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
PROJECT=BASE.parents[1]
SDK=Path('@ICRAFT_INSTALL@/CLI v3.36.1')
PYTHON=BASE/'.venv/Scripts/python.exe'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not p.resolve().is_relative_to(ROOT):raise ValueError('Output escaped improvement scope')
    temp=p.with_name(p.name+'.writing');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,p)
def env():
    e=os.environ.copy();e.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TEMP=str(ROOT/'.tmp'),TMP=str(ROOT/'.tmp'))
    e['PATH']=str(SDK/'bin')+os.pathsep+e.get('PATH','');return e
