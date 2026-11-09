import subprocess
from datetime import datetime
from common3 import ROOT,BASE,PROJECT,SDK,save,env
def main():
    cmake=BASE/'.venv/Scripts/cmake.exe';out=ROOT/'build'
    (ROOT/'artifacts').mkdir(parents=True,exist_ok=True)
    (ROOT/'.tmp').mkdir(exist_ok=True)
    cmds=[[str(cmake),'-S',str(ROOT/'cpp'),'-B',str(out),'-G','Visual Studio 17 2022','-A','x64',
           '-DCMAKE_GENERATOR_INSTANCE='+str(PROJECT/'.tools/vs_buildtools'),'-DCMAKE_PREFIX_PATH='+str(SDK)],
          [str(cmake),'--build',str(out),'--config','Release','--parallel','1']]
    with (ROOT/'artifacts'/('build_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.log')).open('xb') as log:
        for cmd in cmds:
            p=subprocess.run(cmd,cwd=ROOT,env=env(),stdout=log,stderr=subprocess.STDOUT)
            if p.returncode:raise RuntimeError('Build failed; see log')
    print('Direct-byte Host utility built')
if __name__=='__main__':main()
