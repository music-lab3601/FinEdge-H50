"""W3 acquisition/audit entry. No automatic approval or formal training."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from software.data.w1 import ROOT, config, head, save, stamp
from software.data.w3_download import fetch

REPORTS=ROOT/'artifacts/w3_2026-10-03'
BASE='https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/'


def draft():
    return json.loads((Path(__file__).resolve().parents[2]/'config/research/w3_protocol_draft.json').read_text(encoding='utf-8'))


def candidate(date):
    d=draft(); c=config()
    if date not in d['dates']: raise ValueError('Date outside reviewed proposal')
    filename='S'+date[5:7]+date[8:10]+date[2:4]+'-v50.txt.gz'
    item=next(v for v in c['candidate_only'] if v['filename']==filename)
    return {**item, 'url':BASE+filename, 'date':date}


def require_acquisition(date):
    if date not in draft().get('approved_acquisition_dates', []):
        raise ValueError('STOP: acquisition scope is not confirmed; ask Codex to record your scope decision first')


def pointer(date): return REPORTS/f'latest_{date}.json'


def rebuild(date, smoke=False):
    from software.data import w3_engine as engine
    require_acquisition(date)
    c=config(); item=candidate(date)
    path=Path(c['data_root'])/'raw'/item['filename']
    sidecar=path.with_name(path.name+'.download.json')
    if not sidecar.exists(): raise ValueError('Download and verify this date first')
    verified=json.loads(sidecar.read_text(encoding='utf-8'))
    if verified.get('status')!='size_sha_gzip_verified': raise ValueError('Download not verified')
    if verified['identity']['url']!=item['url'] or verified['identity']['bytes']!=item['expected_bytes']:
        raise ValueError('Download identity does not match selected date')
    cfg=json.loads((Path(__file__).resolve().parents[2]/'config/research/w2_audit.json').read_text(encoding='utf-8'))
    cfg.update(symbol=draft()['symbol'], trading_date=date, expected_sample_messages=None,
               expected_sample_target_messages=None, version='W3-audit-1.0',
               purpose='Multi-date engineering audit; not automatically training-ready',
               count_reference_source='No independent published count for this date; structural/lifecycle/reference checks only')
    c['sample']={**item,'reference_sha256':verified['sha256'], 'reference_sha256_source':verified['sha_provenance']}
    engine.CONTEXT=(c,cfg)
    folder=engine.execute('smoke' if smoke else 'full', Path(c['data_root'])/'derived/w3'/date)
    result=engine.check(folder)
    if not smoke:
        save(pointer(date), {'date':date, 'folder':str(folder), 'check':result, 'at_utc':stamp()})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['status','probe','download','smoke','rebuild','check'])
    p.add_argument('--date')
    args=p.parse_args(); d=draft(); c=config()
    if args.action=='status':
        result={'phase':'W3 materials/preparation; acceptance NOT complete', 'data_root':c['data_root'],
                'free_gib':round(shutil.disk_usage(c['data_root']).free/1024**3,2),
                'approved_acquisition_dates':d.get('approved_acquisition_dates',[]),
                'label_horizon':d['label_horizon'],'label_threshold':d['label_threshold'],
                'formal_training_protocol_frozen':d['formal_training_protocol_frozen'],
                'dates':[{'date':date,'raw_present':(Path(c['data_root'])/'raw'/candidate(date)['filename']).exists(),
                          'audit_pointer_present':pointer(date).exists()} for date in d['dates']]}
    else:
        if not args.date: p.error('--date required')
        item=candidate(args.date)
        if args.action=='probe':
            result=head(item['url']); save(REPORTS/f'probe_{args.date}.json',result)
        elif args.action=='download':
            require_acquisition(args.date)
            first=d['dates'][0]
            if args.date!=first and not pointer(first).exists():
                raise ValueError('Audit the first date successfully before downloading later dates')
            result=fetch(item['url'],Path(c['data_root'])/'raw'/item['filename'],item['expected_bytes'])
        elif args.action in ('smoke','rebuild'):
            result=rebuild(args.date,args.action=='smoke')
        else:
            from software.data import w3_engine
            run=json.loads(pointer(args.date).read_text(encoding='utf-8'))
            result=w3_engine.check(run['folder'])
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    try: main()
    except KeyboardInterrupt: print('Interrupted; preserve .part and rerun the same command.');sys.exit(130)
    except Exception as e: print('ERROR:',e);sys.exit(1)
