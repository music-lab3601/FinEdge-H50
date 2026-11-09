"""Verify successful compile path from source-only sibling without old failures."""
import shutil,subprocess,sys
from deployment.icraft.native.common3 import ROOT, BASE, read, save, sha, env

def main():
    dest=BASE/'lowbit_native_rebuild_check_20261030';dest.mkdir(exist_ok=False)
    for name in ['cpp','scripts']:shutil.copytree(ROOT/name,dest/name,ignore=shutil.ignore_patterns('__pycache__'))
    results=[]
    for script in ['build_tools.py','rebuild_native_models.py']:
        cmd=[sys.executable,str(dest/'scripts'/script)]
        with (dest/(script+'.log')).open('wb') as log:r=subprocess.run(cmd,cwd=dest,env=env(),stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(command=cmd,exit_code=r.returncode))
        if r.returncode:
            save(ROOT/'artifacts/rebuild_check.json',dict(status='failed',commands=results,output=str(dest)))
            raise RuntimeError('Independent source rebuild failed')
    matches=[]
    for m in read(ROOT/'contracts/frozen_candidate.json')['models'].values():
        for key in ['json','raw']:
            p=ROOT/m[key];q=dest/m[key];same=sha(p)==sha(q)
            matches.append(dict(path=m[key],original_sha256=sha(p),rebuilt_sha256=sha(q),bit_equal=same))
    save(ROOT/'artifacts/rebuild_check.json',dict(status='passed' if all(r['bit_equal'] for r in matches) else 'compiled_hash_differences_need_review',
        commands=results,files=matches,output=str(dest),fresh_data_read=False,board_accessed=False))
    if not all(r['bit_equal'] for r in matches):raise RuntimeError('Rebuild hash mismatch')
    print('Source-only sibling rebuild matches both JSON/RAW pairs bit for bit')

if __name__=='__main__':main()
