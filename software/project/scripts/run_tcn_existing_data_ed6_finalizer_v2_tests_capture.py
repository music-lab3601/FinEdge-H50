"""One real execution of independent ED6 publication-v2 synthetic tests."""
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
def put(p,v):
    with p.open('x',encoding='utf-8')as f:
        json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def main():
    a=argparse.ArgumentParser();a.add_argument('--test-sha256',required=True);a.add_argument('--implementation-sha256',required=True);a.add_argument('--expected-tests',required=True,type=int);args=a.parse_args()
    assert 1<=args.expected_tests<=200
    test=ROOT/'scripts/test_finalize_tcn_existing_data_ed6_v2.py';impl=ROOT/'scripts/finalize_tcn_existing_data_ed6_v2.py'
    before=ref(test);ib=ref(impl);assert before['sha256']==args.test_sha256 and ib['sha256']==args.implementation_sha256
    assert not (STAGE/'finalizer_v2_tests_execution_receipt.json').exists()
    started=now();begin=time.perf_counter();command=[sys.executable,'-u','-B',str(test)]
    put(STAGE/'finalizer_v2_tests_capture_reservation.json',{'permanent_single_use':True,'started_utc':started,'entry_before':before,'implementation_before':ib,'expected_tests':args.expected_tests})
    out=STAGE/'finalizer_v2_tests.stdout.log';err=STAGE/'finalizer_v2_tests.stderr.log'
    with out.open('xb')as o,err.open('xb')as e:
        child=subprocess.Popen(command,cwd=ROOT,stdout=o,stderr=e,env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1'));code=child.wait()
    after=ref(test);ia=ref(impl);receipt=STAGE/'finalizer_v2_tests_execution_receipt.json'
    put(receipt,{'phase':'finalizer_v2_tests','started_utc':started,'ended_utc':now(),'wall_seconds':time.perf_counter()-begin,
        'exit_code':code,'entry_before':before,'entry_after':after,'entry_unchanged':before==after,
        'implementation_before':ib,'implementation_after':ia,'capture_pid':os.getpid(),'launcher_pid':child.pid,
        'actual_worker_pid_independently_observed':False,'command':command,'stdout':ref(out),'stderr':ref(err),'capture_source':ref(Path(__file__))})
    message=err.read_text('utf-8');match=re.search(r'Ran (\d+) tests? in ',message)
    count=int(match.group(1))if match else 0
    passed=code==0 and before==after and ib==ia and count==args.expected_tests and re.search(r'\nOK\s*$',message)is not None
    put(STAGE/'finalizer_v2_tests_evidence.json',{'passed':passed,'actual_exit_code':code,'tests_run':count,'expected_tests':args.expected_tests,
        'test_source_binding':before,'implementation_before':ib,'implementation_after':ia,'actual_receipt':ref(receipt),
        'scope':'Separate publication-v2 stdlib synthetic fixtures; no actual market streams, model, optimizer or CP loads'})
    print(json.dumps({'passed':passed,'actual_exit_code':code,'actual_tests':count,'receipt':ref(receipt)}))
    return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
