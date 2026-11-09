"""Freeze D-only handoff, validate it using G's read-only intake, and ZIP it."""
from pathlib import Path
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from deployment.icraft.base.common import ROOT, PROJECT, PACKAGE, ICRAFT, read, save, sha, local_env


def files_to_ship():
    skip_dirs={'.venv','.cache','.tmp','__pycache__','build'}
    skip_files={'candidate.json','release_manifest.json'}
    for directory,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in skip_dirs]
        for name in sorted(files):
            p=Path(directory)/name
            if name in skip_files or name.endswith(('.zip','.zip.sha256')) or p==ROOT/'artifacts/final_verification.json':continue
            yield p
    yield ROOT/'build/Release/d_host_tools.exe'


def main():
    if (ROOT/'candidate.json').exists():raise ValueError('Release already frozen; do not replace its manifest')
    manifest=read(ROOT/'data/manifest.json')
    for name,digest in manifest['source_hashes'].items():
        if sha(name)!=digest:raise ValueError('Read-only source changed: '+name)
    vendor=read(PACKAGE/'package_manifest.json')
    package_files=[]
    for rel,meta in vendor['files'].items():
        p=(PACKAGE/rel).resolve()
        if not p.is_relative_to(PACKAGE.resolve()) or 'history' in p.relative_to(PACKAGE).parts:
            raise ValueError('Unexpected package payload path')
        if p.stat().st_size!=meta['bytes'] or sha(p)!=meta['sha256']:
            raise ValueError('Authoritative package differs: '+rel)
        package_files.append(dict(path=rel,bytes=meta['bytes'],sha256=meta['sha256']))
    extra=[PROJECT/'software/runtime/cpu.py',PACKAGE/'golden/script_batch32.f32',ICRAFT/'bin/icraft.exe',
           ICRAFT/'bin/icraft_xir.dll',ICRAFT/'bin/icraft_xrt.dll',ICRAFT/'bin/icraft_hostbackend.dll',
           ICRAFT/'docs/quantizer/zhuge/zhuge.html']
    save(ROOT/'source_manifest.json',dict(
        schema='D_read_only_source_provenance_v1',date='2026-10-30',
        initial_references=manifest['source_hashes'],initial_references_unchanged=True,
        extra_references={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in extra},
        authoritative_package_payloads_verified=len(package_files),package_files=package_files,
        training_dates_missing=['2025-12-08','2025-12-09','2025-12-10','2025-12-11','2025-12-12'],
        reads_archived_sources=False,writes_outside_task_project_files=False,board_accessed=False))
    save(ROOT/'contract_ack.json',dict(agent='D',status='accepted',contract_sha256=sha(ROOT/'contracts/core_r1.json'),
        notes='r1 raw/book semantics adopted read-only. D_input_r1 is a separate actual quantization contract; no new raw/book wire ABI.'))
    c=read(ROOT/'input_contract.json')
    c['input'].update(padding='no external tensor padding; downstream time dimension is aligned separately',
        allocation_alignment_bytes=None,
        allocation_requirement='Use SDK allocator/binding and G/F verified DMA alignment; no hardcoded device address is supplied by D.')
    save(ROOT/'input_contract.json',c)
    save(ROOT/'contracts/published.json',dict(contract='input_contract.json',sha256=sha(ROOT/'input_contract.json'),
        verification='golden/codec_verification.json',board_accessed=False))
    subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'report.py')],cwd=ROOT,env=local_env(),check=True)

    lograw=(ROOT/'artifacts/tests.log').read_bytes()
    log=lograw.decode('utf-16' if lograw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
    count=int(re.search(r'Ran (\d+) tests?',log).group(1))
    if count!=15 or not re.search(r'\nOK\s*$',log):raise ValueError('Saved actual unit tests did not pass')
    save(ROOT/'artifacts/local_test_report.json',dict(status='passed',tests_run=count,
        scope='implementation/interface/isolation tests, not INT8 task quality or hardware acceptance',
        log='artifacts/tests.log',log_sha256=sha(ROOT/'artifacts/tests.log'),
        sdk_codec=read(ROOT/'golden/codec_verification.json'),
        fp32_equivalence_status=read(ROOT/'artifacts/reference/report.json')['status'],
        int8_quality_budget_passed=False,board_tested=False))
    save(ROOT/'artifacts/failure_ledger.json',dict(
        setup_failures=[dict(stage='reference preparation',evidence='artifacts/reference_first_attempt_failure.json',resolved=True),
                        dict(stage='C++ build',error='SDK Tensor::write requires char*, not const char*',
                             resolved=True,fix='Use mutable input buffer pointer; subsequent SDK builds passed.')],
        unresolved=[dict(kind='model_quality',evidence='quality_report.json',
                         summary='Ordinary INT8 exceeds both predeclared 1pp loss budgets.'),
                    dict(kind='board_acceptance',status='not_run_by_D',handoff='board_handoff/request.json')]))
    commands=[['scripts/prepare.py'],['scripts/reference.py'],['scripts/build_host.ps1'],
        ['scripts/compile_models.py','--kind','tf32','--shared'],
        ['scripts/compile_models.py','--kind','int8'],
        ['scripts/run_host.py','--kind','int8','--group','calibration'],
        ['scripts/run_host.py','--kind','tf32','--group','golden'],
        ['scripts/freeze_candidate.py'],
        *[['scripts/run_host.py','--kind',k,'--group',g] for k,g in
          [('int8','evaluation'),('int8','golden'),('tf32','evaluation'),('tf32','calibration')]],
        ['scripts/publish_contract.py'],['scripts/make_codec_golden.py'],
        ['scripts/verify_sdk_castback.py'],['scripts/verify_codec.py'],['scripts/report.py']]
    save(ROOT/'commands_run.json',dict(working_directory=str(ROOT),
        python=str(Path(__import__('sys').executable)),steps=commands,
        notes=['Build script was rerun after compile signature repair and after codec-only extension.',
               'Reference and Host scripts operate offline. Actual individual Host argv/logs are in artifacts/host/*/*/*/command.json.',
               'The code preserves frozen output directories and refuses silent replacement; these are actual generation commands, not an idempotent overwrite script.',
               'Current release can be reproduced/read-verified using scripts/probe.py and the unit tests.'],
        python_version=platform.python_version(),board_calls=0))
    (ROOT/'contracts/d_input_contract.h').write_text('''#pragma once
#include <stdint.h>
/* Full generated graph ABI: no direct INT8 binding has been verified. */
#define D_INPUT_BATCH 1
#define D_INPUT_FEATURES 40
#define D_INPUT_HISTORY 100
#define D_FULL_GRAPH_INPUT_BYTES 16000u
#define D_LOGICAL_INTERNAL_INT8_BYTES 4000u
#define D_FIRST_GENERATED_INTEGER_BYTES 4480u
#define D_INTERNAL_INT8_STEP_F32_BITS 1025970844u
#define D_TF32_LAYERS_PER_FORWARD 26u
#define D_INT8_LAYERS_PER_FORWARD 29u
#define D_INT8_QUALITY_BUDGET_PASSED 0
/* Read input_contract.json for layout, precision and hashes. */
''',encoding='utf-8')
    goldenfiles={p.relative_to(ROOT/'golden').as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
                 for p in (ROOT/'golden').rglob('*') if p.is_file() and p.name!='release_manifest.json'}
    save(ROOT/'golden/release_manifest.json',dict(scope='logical codec vectors; not a verified NPU device binding',files=goldenfiles))
    # Freeze only explicit files. Local virtual environments/tool caches are not handed to G.
    payload=sorted(set(files_to_ship()))
    # Receipt files are created after this candidate and intentionally remain outside its file inventory.
    payload=[p for p in payload if p.name not in ('probe_receipt.json','G_intake_receipt.json')]
    candidate=dict(schema='finedge_candidate_v1',agent='D',stage='int8',contract_sha256=sha(ROOT/'contracts/core_r1.json'),
        files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in payload],
        local_test_reports=['artifacts/local_test_report.json'],
        entrypoints={'probe':['{python}','{candidate_root}/scripts/probe.py']},
        limitations=['INT8 quality budget FAILED; no production promotion.',
                     'No board access or actual NPU/DDR latency measurement by D.',
                     'Full generated input is FP32 16000 bytes, not INT8 4000 bytes.',
                     'Same-day isolated engineering evaluation; not original validation reproduction.'])
    save(ROOT/'candidate.json',candidate)
    probe=subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'probe.py')],cwd=ROOT,env=local_env(),
                         capture_output=True,text=True,encoding='utf-8',check=True)
    save(ROOT/'artifacts/probe_receipt.json',json.loads(probe.stdout))
    
    from integration.intake import inspect_candidate
    checked=inspect_candidate(ROOT/'candidate.json',project=PROJECT,contract_digest=candidate['contract_sha256'])
    save(ROOT/'artifacts/G_intake_receipt.json',checked)
    ship=payload+[ROOT/'candidate.json',ROOT/'artifacts/probe_receipt.json',ROOT/'artifacts/G_intake_receipt.json']
    save(ROOT/'release_manifest.json',dict(schema='D_release_manifest_v1',date='2026-10-30',
        quality_status=read(ROOT/'quality_report.json')['status'],candidate_sha256=sha(ROOT/'candidate.json'),
        manifest_excludes_itself=True,
        files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in ship]))
    archive=ROOT/'D_INT8_handoff_20261030_v1.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in ship+[ROOT/'release_manifest.json']:
            z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failed')
        for p in ship:
            if hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()!=sha(p):raise ValueError('ZIP file hash differs')
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8')
    save(ROOT/'artifacts/final_verification.json',dict(status='passed',scope='local_handoff_integrity_only',
        candidate_files=len(payload),released_files=len(ship)+1,unit_tests=count,
        G_intake=checked['state'],int8_quality_passed=False,board_accessed=False,
        zip=str(archive),zip_bytes=archive.stat().st_size,zip_sha256=sha(archive)))
    print(probe.stdout)
    print(json.dumps(read(ROOT/'artifacts/final_verification.json'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
