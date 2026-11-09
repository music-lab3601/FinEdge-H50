"""Versioned D release. Preserve all failures, old releases and device state."""
import hashlib,json,os,re,shutil,subprocess,sys,zipfile
from common3 import ROOT,BASE,V2,PROJECT,read,save,sha,env
from prepare_holdout3 import frozen_check

def main():
    archive=ROOT/'D_native_INT8_input_20261030_v3.zip'
    if archive.exists():raise FileExistsError('Release already exists')
    f=frozen_check()
    # Preserve accepted releases; explicit paths only, never the archive tree.
    preserved={}
    for root,name in [(BASE,'candidate.json'),(V2,'release_manifest.json')]:
        m=read(root/name)
        for r in m['files']:
            p=(root/r['path']).resolve()
            if not p.is_relative_to(root) or sha(p)!=r['sha256']:raise ValueError('Prior release changed '+str(p))
        preserved[root.name]=dict(manifest=str(root/name),sha256=sha(root/name),files_verified=len(m['files']))
    failures=[]
    for path in sorted((ROOT/'artifacts').rglob('stage_results.json')):
        n=read(path);bad=[s for s in n.get('stages',[]) if s.get('exit_code',0)]
        if bad:failures.append(dict(path=path.relative_to(ROOT).as_posix(),stages=bad))
    for path in [ROOT/'short/result.json',ROOT/'artifacts/decoder_int8/stage_results.json']:
        if path.exists():
            n=read(path)
            if n.get('exit_code',0):failures.append(dict(path=path.relative_to(ROOT).as_posix(),result=n))
    failures.append(dict(path='golden/verification.json',result=read(ROOT/'golden/verification.json'),
        resolution='SDK FP32->unit INT8 Cast uses floor(float32(x+0.5f)); corrected before fresh evaluation; passed evidence in golden_v2/'))
    save(ROOT/'failure_ledger.json',dict(schema='D_lowbit_failure_ledger_v3',failures=failures,
        successful_path='artifacts/int8_physical/d3p_ZG.json',
        compiler_root_cause='Unknown; native blocked entry avoids failing prefix transforms; short path did not resolve crash',
        precision_search='No additional protected-layer/weight training search after G v2 acceptance'))
    resources=[PROJECT/'AGENTS.md'] if (PROJECT/'AGENTS.md').exists() else []
    resources += [PROJECT.parent/'AGENTS.md',PROJECT/'vendor/ed4_model/model_cpu.ts',PROJECT/'vendor/ed4_model/normalization.json',
        PROJECT/'vendor/ed4_model/golden/inputs.f32',BASE/'model/tcn_polyphase.onnx',BASE/'requirements-lock.txt',
        V2/'data/inputs.npy',V2/'data/labels.npy',V2/'data/manifest.json',V2/'build/Release/d_fresh_book.exe',
        V2/'artifacts/mixed_tf32_compat/tcn_d_quantized.json',V2/'artifacts/mixed_tf32_compat/tcn_d_quantized.raw',
        V2/'artifacts/mixed_tf32_compat/tcn_d_ZG.json',V2/'artifacts/mixed_tf32_compat/tcn_d_ZG.raw',
        BASE/'artifacts/compile/tf32/tcn_d_quantized.json',BASE/'artifacts/compile/tf32/tcn_d_quantized.raw',
        BASE/'artifacts/compile/tf32/tcn_d_ZG.json',BASE/'artifacts/compile/tf32/tcn_d_ZG.raw',BASE/'build/Release/d_host_tools.exe',
        PROJECT/'versions/realtime_integration_v1/contracts/r1/contract.json',
        PROJECT/'versions/realtime_integration_v1/artifacts/D_v2_acceptance_20261030_214231/README.md']
    save(ROOT/'source_manifest.json',dict(resources={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in resources},
        raw_mbo_cache=read(ROOT/'data/manifest.json')['raw_cache_sha256'],raw_original_dbn=read(ROOT/'data/manifest.json')['source_sha256'],
        core_contract='81ffd0d31b348281160ca2d8f0d94ec2f9b5cc511cfb19425727b9584928a73c',
        sdk='Icraft 3.36.1',sdk_path='@ICRAFT_INSTALL@/CLI v3.36.1',cmake=str(BASE/'.venv/Scripts/cmake.exe'),
        msvc=str(PROJECT/'.tools/vs_buildtools'),preserved_releases=preserved,board_accessed=False,
        outside_task_writes=False,additional_D_owned_source_only_rebuild=read(ROOT/'artifacts/rebuild_check.json')['output']))
    shutil.copyfile(BASE/'requirements-lock.txt',ROOT/'requirements-lock.txt')
    # Required regression after strengthening constant binding; capture actual log.
    cmd=[sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v']
    r=subprocess.run(cmd,cwd=ROOT,env=env(),capture_output=True,text=True,encoding='utf-8')
    log=r.stdout+r.stderr;(ROOT/'artifacts/final_tests.log').write_text(log,encoding='utf-8')
    if r.returncode or not re.search(r'Ran 14 tests',log) or not re.search(r'\nOK\s*$',log):raise RuntimeError('Tests failed')
    save(ROOT/'artifacts/local_test_report.json',dict(status='passed',tests=14,command=cmd,exit_code=r.returncode,
        log_sha256=sha(ROOT/'artifacts/final_tests.log'),prior_releases=preserved,
        source_only_rebuild_bit_equal=read(ROOT/'artifacts/rebuild_check.json')['status']=='passed',
        board_accessed=False,performance_measured=False))
    q=read(ROOT/'quality_report.json')
    save(ROOT/'contracts/published.json',dict(schema='D_input_contract_publication_v3',version='D.int8.native.r3',
        input_contract='../input_contract.json',sha256=sha(ROOT/'input_contract.json'),
        frozen_candidate_sha256=sha(ROOT/'contracts/frozen_candidate.json'),quality_report_sha256=sha(ROOT/'quality_report.json'),
        golden_sha256=sha(ROOT/'golden_v2/encoded.bin'),status=q['status'],G_adopted=False,board_tested=False))
    excluded={'release_manifest.json','candidate_revision.json','package_verification.json','probe_receipt.json'}
    payload=[]
    for directory,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in ('build','.tmp','__pycache__')]
        for name in files:
            p=__import__('pathlib').Path(directory)/name
            if name in excluded or name.endswith(('.zip','.zip.sha256')) or name=='offline_rebuilt_rows.bin':continue
            payload.append(p)
    payload += [ROOT/'build/Release/d_integer_host.exe',ROOT/'build/Release/d_fold_scale.exe',ROOT/'build/Release/d_audit_weights.exe']
    payload=sorted(set(payload))
    save(ROOT/'candidate_revision.json',dict(schema='D_versioned_candidate_revision_v3',agent='D',stage='int8',
        revision='lowbit_entry_20261030_v3',parent_candidate_sha256=sha(BASE/'candidate.json'),parent_v2_revision_sha256=sha(V2/'candidate_revision.json'),
        core_contract_sha256=read(ROOT/'input_contract.json')['base_contract_sha256'],status=q['status'],
        intake_note='New D-owned version only; G approves native INT8 ABI, registry and board tests. v1/v2 remain unchanged.',
        files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in payload],
        local_test_reports=['artifacts/local_test_report.json'],entrypoints={'probe':['{python}','{candidate_root}/scripts/probe3.py']},
        board_request='board_handoff/request.json',input_contract='input_contract.json',
        limitations=['same-day engineering evaluation','lossy mixed model, not full INT8 compute','board timing/correctness pending','native padded input4800B, not dense4000B']))
    ship=payload+[ROOT/'candidate_revision.json']
    save(ROOT/'release_manifest.json',dict(schema='D_native_int8_release_v3',
        files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in ship],
        excluded_large_file=dict(path='data/offline_rebuilt_rows.bin',bytes=(ROOT/'data/offline_rebuilt_rows.bin').stat().st_size,
            sha256=sha(ROOT/'data/offline_rebuilt_rows.bin'),reproduce='scripts/prepare_holdout3.py with hashed current raw cache and accepted D v2 offline extractor'),
        excluded_directories=['build except3 Release exe','.tmp','__pycache__'],manifest_excludes_itself=True))
    p=subprocess.run([sys.executable,str(ROOT/'scripts/probe3.py')],cwd=ROOT,env=env(),capture_output=True,text=True,encoding='utf-8')
    if p.returncode:raise RuntimeError('Release probe failed '+p.stderr)
    receipt=json.loads(p.stdout);save(ROOT/'artifacts/probe_receipt.json',receipt)
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in ship+[ROOT/'release_manifest.json',ROOT/'artifacts/probe_receipt.json']:z.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('ZIP CRC failed')
        for path in ship:
            if hashlib.sha256(z.read(path.relative_to(ROOT).as_posix())).hexdigest()!=sha(path):raise ValueError('ZIP content mismatch')
    digest=sha(archive);archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n',encoding='utf-8')
    result=dict(status='passed',files=len(ship)+2,tests=14,board_accessed=False,performance_measured=False,
        zip=str(archive),bytes=archive.stat().st_size,sha256=digest,prior_releases_unchanged=True,source_rebuild_bit_equal=True)
    save(ROOT/'artifacts/package_verification.json',result)
    print(json.dumps(receipt,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
