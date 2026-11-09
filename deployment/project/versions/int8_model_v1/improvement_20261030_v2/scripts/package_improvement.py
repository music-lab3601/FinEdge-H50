import hashlib,json,os,platform,re,shutil,subprocess,sys,zipfile
from common2 import ROOT,BASE,PROJECT,read,save,sha,env


def main():
    archive=ROOT/'D_precision_improvement_20261030_v2.zip'
    if archive.exists():raise ValueError('Release already exists')
    # Parent v1 remains valid, including all 445 payload hashes.
    parent=read(BASE/'candidate.json')
    for r in parent['files']:
        if sha(BASE/r['path'])!=r['sha256']:raise ValueError('Parent release was changed')
    lograw=(ROOT/'artifacts/tests.log').read_bytes()
    log=lograw.decode('utf-16' if lograw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
    if not re.search(r'Ran 10 tests',log) or not re.search(r'\nOK\s*$',log):raise ValueError('Tests not passed')
    save(ROOT/'artifacts/local_test_report.json',dict(status='passed',tests=10,log_sha256=sha(ROOT/'artifacts/tests.log'),
        parent_v1_payloads_verified=len(parent['files']),board_accessed=False,
        quality_status=read(ROOT/'quality_report.json')['status']))
    shutil.copyfile(BASE/'requirements-lock.txt',ROOT/'requirements-lock.txt')
    resources=[BASE/'data/calibration.inputs.npy',BASE/'data/manifest.json',BASE/'model/tcn_polyphase.onnx',
        BASE/'artifacts/compile/shared/tcn_d_optimized.json',BASE/'artifacts/compile/shared/tcn_d_optimized.raw',
        BASE/'artifacts/compile/int8/tcn_d_ZG.json',BASE/'artifacts/compile/int8/tcn_d_ZG.raw',
        BASE/'artifacts/compile/tf32/tcn_d_ZG.json',BASE/'artifacts/compile/tf32/tcn_d_ZG.raw',
        BASE/'build/Release/d_host_tools.exe',PROJECT/'vendor/ed4_model/model_cpu.ts',
        PROJECT/'vendor/ed4_model/normalization.json',PROJECT/'versions/network_fast_v1/native/fast.cpp']
    save(ROOT/'source_manifest.json',dict(date='2026-10-30',resources={str(p):sha(p) for p in resources},
        parent_candidate_sha256=sha(BASE/'candidate.json'),parent_unchanged=True,
        fresh_data_sha256=sha(ROOT/'data/manifest.json'),board_accessed=False,
        reproduction_requires_current_D_parent_resources=True))
    payload=[]
    excluded={'release_manifest.json','candidate_revision.json'}
    for directory,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in ('build','.tmp','__pycache__')]
        for name in files:
            p=__import__('pathlib').Path(directory)/name
            if name in excluded or name.endswith(('.zip','.zip.sha256')) or name in ('offline_rebuilt_rows.bin','package_verification.json','probe_receipt.json'):continue
            payload.append(p)
    payload.extend([ROOT/'build/Release/d_layer_dump.exe',ROOT/'build/Release/d_fresh_book.exe'])
    payload=sorted(set(payload))
    save(ROOT/'candidate_revision.json',dict(schema='D_versioned_candidate_revision_v2',agent='D',stage='int8',
        revision='improvement_20261030_v2',parent_candidate_sha256=sha(BASE/'candidate.json'),
        core_contract_sha256=sha(BASE/'contracts/core_r1.json'),
        status='offline_quality_passed_board_pending',
        intake_note='Preserve v1 root candidate.json. G registers this versioned increment in its own candidate registry.',
        files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in payload],
        local_test_reports=['artifacts/local_test_report.json'],
        entrypoints={'probe':['{python}','{candidate_root}/scripts/probe_improvement.py']},
        board_request='board_handoff/request.json',input_contract='input_contract.json',
        limitations=['Mixed INT8+TF32, not pure INT8','same-day finite new holdout','no board latency measured','16KB FP32 full-window input']))
    ship=payload+[ROOT/'candidate_revision.json']
    save(ROOT/'release_manifest.json',dict(schema='D_precision_improvement_release_v2',date='2026-10-30',
        files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in ship],
        excluded_large_generated_file='data/offline_rebuilt_rows.bin (76.9MB; reproduce from hashed current raw cache)',
        manifest_excludes_itself=True))
    probe=subprocess.run([sys.executable,str(ROOT/'scripts/probe_improvement.py')],cwd=ROOT,env=env(),capture_output=True,text=True,encoding='utf-8',check=True)
    save(ROOT/'artifacts/probe_receipt.json',json.loads(probe.stdout))
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in ship+[ROOT/'release_manifest.json',ROOT/'artifacts/probe_receipt.json']:
            z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('ZIP CRC failure')
        for p in ship:
            if hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()!=sha(p):raise ValueError('ZIP hash failure')
    digest=sha(archive);archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n',encoding='utf-8')
    result=dict(status='passed',files=len(ship)+2,tests=10,parent_v1_unchanged=True,board_accessed=False,
        quality_screen_passed=True,zip=str(archive),zip_bytes=archive.stat().st_size,zip_sha256=digest)
    save(ROOT/'artifacts/package_verification.json',result)
    # Additive discovery index; never overwrite a hashed v1 payload.
    index=BASE/'CURRENT_IMPROVEMENT.json'
    if index.exists():raise ValueError('An improvement index already exists; inspect before changing it')
    index.write_text(json.dumps(dict(directory=str(ROOT),quality_report=str(ROOT/'quality_report.json'),
        input_contract=str(ROOT/'input_contract.json'),candidate_revision=str(ROOT/'candidate_revision.json'),
        package=str(archive),package_sha256=digest,status='offline_quality_passed_board_pending'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(probe.stdout);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
