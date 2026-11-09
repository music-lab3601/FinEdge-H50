"""Read small ED6 runtime logs only; never poll a process or access data/model."""
from pathlib import Path
import argparse
import json
import ctypes
from ctypes import wintypes

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'artifacts/tcn_existing_data_optimization_2026-10-17/step06_replication'
def power_status():
    class Power(ctypes.Structure):
        _fields_=[('ACLineStatus',wintypes.BYTE),('BatteryFlag',wintypes.BYTE),
                  ('BatteryLifePercent',wintypes.BYTE),('SystemStatusFlag',wintypes.BYTE),
                  ('BatteryLifeTime',wintypes.DWORD),('BatteryFullLifeTime',wintypes.DWORD)]
    api=ctypes.WinDLL('kernel32',use_last_error=True).GetSystemPowerStatus
    api.argtypes=[ctypes.POINTER(Power)];api.restype=wintypes.BOOL;value=Power()
    if not api(ctypes.byref(value)):return {'available':False,'win_error':ctypes.get_last_error()}
    return {'available':True,'ACLineStatus':int(value.ACLineStatus),
            'battery_percent':int(value.BatteryLifePercent)if value.BatteryLifePercent<=100 else None,
            'read_only':True}
def main():
    a=argparse.ArgumentParser();a.add_argument('--slot',choices=['control43','candidate43'],required=True);args=a.parse_args()
    trial=STAGE/'trials'/args.slot;log=trial/'training.stdout.log';latest=None;power=None
    if log.exists():
        with log.open('rb')as f:
            f.seek(max(0,log.stat().st_size-1024**2));lines=f.read().decode('utf-8',errors='replace').splitlines()
        for line in reversed(lines):
            try:v=json.loads(line)
            except ValueError:continue
            if latest is None and v.get('phase')in ('TRAIN','DEV'):latest=v
            if power is None and v.get('phase')=='temporary_power_request':power=v
            if latest is not None and power is not None:break
    history=trial/'run/history.json'
    h=json.loads(history.read_text('utf-8'))if history.exists()else []
    selected=None
    status=trial/'run/status.json'
    if status.exists():selected=json.loads(status.read_text('utf-8')).get('selected_epoch')
    receipt=trial/'training_execution_receipt.json'
    actual=json.loads(receipt.read_text('utf-8'))if receipt.exists()else None
    print(json.dumps({'slot':args.slot,'latest':latest,'last_observed_power_event':power,'current_power_status':power_status(),'completed_epochs':len(h),'selected_epoch':selected,
        'epochs':[{'epoch':r['epoch'],'accuracy':r['development_validation']['accuracy'],
                   'macro_f1':r['development_validation']['macro_f1'],'recall':r['development_validation']['recall'],
                   'improved':r['improved'],'wait':r['wait']}for r in h],
        'actual_training_receipt_exists':actual is not None,'actual_exit_code':actual.get('exit_code')if actual else None,
        'read_scope':'small existing D stdout/history/status/receipt metadata only; no process poll or ML'}))
if __name__=='__main__':main()
