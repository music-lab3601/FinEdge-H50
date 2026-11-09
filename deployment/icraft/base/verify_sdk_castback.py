import subprocess
import shutil
from deployment.icraft.base.common import ROOT, read, save, local_env, sha

def main():
    old=read(ROOT/'golden/command.json')['command']
    out=ROOT/'golden/cast_back';out.mkdir(exist_ok=False)
    cmd=list(old);cmd[cmd.index('--out')+1]=str(out)
    with (out/'sdk.log').open('wb') as stream:
        p=subprocess.run(cmd,cwd=out,env=local_env(),stdout=stream,stderr=subprocess.STDOUT)
    save(out/'command.json',dict(command=cmd,exit_code=p.returncode,
        executable_sha256=sha(ROOT/'build/Release/d_host_tools.exe'),board_accessed=False))
    if p.returncode:raise RuntimeError('SDK cast-back failed')
    if (out/'encoded.sqb').read_bytes()!=(ROOT/'golden/encoded.sqb').read_bytes():
        raise ValueError('Adding cast-back changed golden encoding')
    shutil.copyfile(out/'sdk_dequantized.f32',ROOT/'golden/sdk_dequantized.f32')
    print('SDK cast-back captured')

if __name__=='__main__':main()
