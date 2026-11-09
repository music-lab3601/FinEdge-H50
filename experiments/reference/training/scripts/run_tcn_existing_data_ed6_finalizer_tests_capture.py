"""One root-only real execution of the separate, pure ED6 finalizer tests."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'artifacts/tcn_existing_data_optimization_2026-10-17/step06_replication'
def ref(p):
    with p.open('rb')as f:d=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':d}
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def put(p,v):
    with p.open('x',encoding='utf-8')as f:
        json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
    a=argparse.ArgumentParser();a.add_argument('--test-sha256',required=True);a.add_argument('--implementation-sha256',required=True);args=a.parse_args()
    test=ROOT/'scripts/test_finalize_tcn_existing_data_ed6.py';implementation=ROOT/'scripts/finalize_tcn_existing_data_ed6.py'
    before=ref(test);ib=ref(implementation)
    assert before['sha256']==args.test_sha256 and ib['sha256']==args.implementation_sha256
    assert not (STAGE/'finalizer_tests_execution_receipt.json').exists()
    started=now();begin=time.perf_counter();command=[sys.executable,'-u','-B',str(test)]
    put(STAGE/'finalizer_tests_capture_reservation.json',{'permanent_single_use':True,'started_utc':started,'entry_before':before,'implementation_before':ib})
    out=STAGE/'finalizer_tests.stdout.log';err=STAGE/'finalizer_tests.stderr.log'
    with out.open('xb')as o,err.open('xb')as e:
        child=subprocess.Popen(command,cwd=ROOT,stdout=o,stderr=e,env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1'))
        code=child.wait()
    after=ref(test);ia=ref(implementation)
    receipt=STAGE/'finalizer_tests_execution_receipt.json'
    put(receipt,{'phase':'finalizer_tests','started_utc':started,'ended_utc':now(),'wall_seconds':time.perf_counter()-begin,
        'exit_code':code,'entry_before':before,'entry_after':after,'entry_unchanged':before==after,
        'implementation_before':ib,'implementation_after':ia,'capture_pid':os.getpid(),'launcher_pid':child.pid,
        'actual_worker_pid_independently_observed':False,'command':command,'stdout':ref(out),'stderr':ref(err),'capture_source':ref(Path(__file__))})
    match=re.search(r'Ran (\d+) tests? in ',err.read_text('utf-8'))
    passed=code==0 and before==after and ib==ia and match is not None and int(match.group(1))==17 and re.search(r'\nOK\s*$',err.read_text('utf-8'))is not None
    put(STAGE/'finalizer_tests_evidence.json',{'passed':passed,'actual_exit_code':code,'tests_run':int(match.group(1))if match else 0,
         'test_source_binding':before,'implementation_before':ib,'implementation_after':ia,'actual_receipt':ref(receipt),
         'scope':'17 separate stdlib synthetic tests; no real market streams, model, optimizer or checkpoint deserialization'})
    print(json.dumps({'passed':passed,'actual_exit_code':code,'receipt':ref(receipt)}))
    return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
