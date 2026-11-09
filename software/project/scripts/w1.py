"""W1 acquisition kit. No model training or order-book certification.

Network requests use certificate-verified HTTPS. A complete file is published
only after size and reference SHA256 checks. gzip/ITCH checks are separate.
"""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'artifacts' / 'w1_2026-10-02'
CHUNK = 1024 * 1024
LENGTHS = dict(S=12,R=39,H=25,Y=20,L=26,V=35,W=12,K=28,J=35,h=21,A=36,F=40,E=31,C=36,X=23,D=19,U=35,P=44,Q=40,B=19,I=50,N=20)

def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def config():
    return json.loads((ROOT / 'configs/w1_data.json').read_text(encoding='utf-8-sig'))

def head(url):
    request = urllib.request.Request(url, method='HEAD', headers={'Accept-Encoding':'identity', 'User-Agent':'FinEdge-LOB-W1/1.0'})
    with urllib.request.urlopen(request, timeout=30) as r:
        return {'status':r.status,'url':r.url,'bytes':int(r.headers['Content-Length']) if r.headers.get('Content-Length') else None,
                'etag':r.headers.get('ETag'),'last_modified':r.headers.get('Last-Modified'),'accept_ranges':r.headers.get('Accept-Ranges')}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*CHUNK), b''):
            h.update(block)
    return h.hexdigest()

def raw_path(data_root, cfg):
    return Path(data_root) / 'raw' / cfg['sample']['filename']

def download_one(url, target, expected_bytes, expected_sha, *, allow_http_test=False, retries=3):
    """Resumable fixed-object fetch; tests may use an explicit local HTTP server."""
    target=Path(target)
    if not url.startswith('https://') and not (allow_http_test and url.startswith('http://127.0.0.1:')):
        raise ValueError('Only verified HTTPS is permitted')
    if target.exists():
        if target.stat().st_size != expected_bytes or sha256(target) != expected_sha:
            raise ValueError('Existing final file differs; it was NOT overwritten. Check its source.')
        return {'status':'already_present_verified','bytes':expected_bytes,'sha256':expected_sha,'path':str(target)}
    target.parent.mkdir(parents=True, exist_ok=True)
    part=target.with_name(target.name+'.part')
    meta=target.with_name(target.name+'.download.json')
    identity={'url':url,'expected_bytes':expected_bytes,'expected_sha256':expected_sha}
    if part.exists():
        if not meta.exists():
            raise ValueError('Partial file has no identity metadata; refusing unsafe resume')
        old=json.loads(meta.read_text(encoding='utf-8'))
        if any(old.get(k)!=v for k,v in identity.items()):
            raise ValueError('Partial file belongs to a different source; refusing resume')
    else:
        save(meta,{**identity,'created_utc':stamp()})
    start=time.monotonic()
    last_print=start
    for attempt in range(1,retries+1):
        offset=part.stat().st_size if part.exists() else 0
        if offset>expected_bytes:
            raise ValueError('Partial file is larger than the expected object')
        if offset==expected_bytes:
            break
        free=shutil.disk_usage(target.parent).free
        if free < expected_bytes-offset+2*1024**3:
            raise ValueError('Insufficient free space (must also retain 2 GiB safety reserve)')
        headers={'Accept-Encoding':'identity','User-Agent':'FinEdge-LOB-W1/1.0'}
        if offset:headers['Range']=f'bytes={offset}-'
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as r:
                if r.headers.get('Content-Encoding','identity')!='identity':
                    raise ValueError('Unexpected HTTP content encoding')
                if offset:
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''))
                    if r.status!=206 or not match or tuple(map(int,match.groups()))!=(offset,expected_bytes-1,expected_bytes):
                        raise ValueError('Server did not honor exact resume range; partial file kept unchanged. Use restart action.')
                elif r.status!=200:
                    raise ValueError(f'Expected HTTP 200, received {r.status}')
                length=r.headers.get('Content-Length')
                if length and int(length)!=expected_bytes-offset:
                    raise ValueError('HTTP content length differs from manifest; refusing to append')
                with part.open('ab' if offset else 'wb') as f:
                    received=offset
                    while True:
                        block=r.read(CHUNK)
                        if not block:break
                        if received+len(block)>expected_bytes:
                            raise ValueError('Response exceeds expected object size')
                        f.write(block);received+=len(block)
                        if time.monotonic()-last_print>10:
                            print(f'DOWNLOAD {received}/{expected_bytes} bytes ({received/expected_bytes:.1%})',flush=True)
                            last_print=time.monotonic()
                    f.flush();os.fsync(f.fileno())
                if received!=expected_bytes:
                    raise OSError(f'Incomplete response: {received}/{expected_bytes}; .part retained')
            break
        except ValueError:
            raise
        except Exception as e:
            print(f'Download attempt {attempt} failed: {e}; resumable .part retained',flush=True)
            if attempt==retries:raise
            time.sleep(min(attempt*2,5))
    if not part.exists() or part.stat().st_size!=expected_bytes:
        raise ValueError('Download is incomplete')
    digest=sha256(part)
    if digest!=expected_sha:
        raise ValueError('SHA256 mismatch; .part retained and no final file published')
    part.rename(target)
    result={'status':'downloaded_size_sha_verified','bytes':expected_bytes,'sha256':digest,'path':str(target),'seconds':round(time.monotonic()-start,2)}
    save(meta,{**identity,**result,'completed_utc':stamp()})
    return result

def frames(stream, limit):
    counts=collections.Counter();events=[];first=None;last=None
    for index in range(limit):
        header=stream.read(2)
        if not header:break
        if len(header)!=2:raise ValueError('Truncated length prefix')
        length=int.from_bytes(header,'big')
        payload=stream.read(length)
        if len(payload)!=length:raise ValueError('Truncated message')
        if length<11:raise ValueError('Invalid common header length')
        typ=chr(payload[0])
        if typ not in LENGTHS or LENGTHS[typ]!=length:
            raise ValueError(f'Unknown type or mismatched ITCH5 length: {typ!r}/{length}')
        timestamp=int.from_bytes(payload[5:11],'big')
        if timestamp>=86_400*10**9:raise ValueError('Timestamp outside a trading day')
        if first is None:first=timestamp
        last=timestamp;counts[typ]+=1
        if typ=='S':events.append({'message_index':index+1,'event_code':chr(payload[11]),'timestamp_ns':timestamp})
    return {'count':sum(counts.values()),'types':dict(counts),'first_timestamp_ns':first,'last_timestamp_ns':last,'system_events_in_prefix':events}

def probe(cfg):
    result={'checked_utc':stamp(),'sample':None,'official_checksum':None,'candidates':[],'large_file_downloaded':False}
    try:result['sample']=head(cfg['sample']['url'])
    except Exception as e:result['sample']={'error':str(e)}
    try:
        with urllib.request.urlopen(cfg['sample']['official_checksum_url'],timeout=20) as r:
            result['official_checksum']={'status':r.status,'text':r.read(4096).decode('ascii')}
    except Exception as e:result['official_checksum']={'error':str(e),'status':'not_available'}
    base=cfg['sample']['url'].rsplit('/',1)[0]+'/'
    for candidate in cfg['candidate_only']:
        try:result['candidates'].append({'filename':candidate['filename'],**head(base+candidate['filename'])})
        except Exception as e:result['candidates'].append({'filename':candidate['filename'],'error':str(e)})
    save(REPORTS/'source_probe.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['sample'].get('bytes')!=cfg['sample']['expected_bytes']:raise ValueError('Sample HEAD not validated')

def smoke(cfg):
    """Fetch at most 1 MiB compressed prefix; never label it a complete dataset."""
    import zlib
    count=CHUNK
    request=urllib.request.Request(cfg['sample']['url'],headers={'Range':f'bytes=0-{count-1}','Accept-Encoding':'identity'})
    with urllib.request.urlopen(request,timeout=30) as r:
        if r.status!=206 or r.headers.get('Content-Range')!=f'bytes 0-{count-1}/{cfg["sample"]["expected_bytes"]}':
            raise ValueError('Bounded range request was not honored; body was not downloaded')
        prefix=r.read(count+1)
    if len(prefix)!=count:raise ValueError('Prefix byte count mismatch')
    REPORTS.mkdir(parents=True,exist_ok=True)
    prefix_path=Path(cfg['data_root'])/'probes'/'sample_prefix_1MiB.gz.part'
    prefix_path.parent.mkdir(parents=True,exist_ok=True)
    prefix_path.write_bytes(prefix)
    decompressor=zlib.decompressobj(16+zlib.MAX_WBITS)
    raw=decompressor.decompress(prefix,16*CHUNK)
    import io
    result=frames(io.BytesIO(raw),1000)
    result.update({'scope':'first 1000 messages from 1 MiB compressed prefix only','prefix_path':str(prefix_path),'complete_gzip_checked':False,'full_day_parsed':False,'order_book_reconstructed':False})
    save(REPORTS/'prefix_smoke.json',result)
    print(json.dumps(result,indent=2))

def verify_file(path,cfg):
    if not path.exists():raise FileNotFoundError(f'No complete file at {path}; run download or provide -ExistingFile')
    started=time.monotonic()
    if path.stat().st_size!=cfg['sample']['expected_bytes']:raise ValueError('File size does not match the W1 sample')
    print('VERIFY SHA256: reading the complete compressed file...',flush=True)
    digest=sha256(path)
    if digest!=cfg['sample']['reference_sha256']:raise ValueError('SHA256 differs from teammate report reference')
    unpacked=0;last_progress=time.monotonic()
    print('VERIFY GZIP: streaming to EOF and checking CRC; no uncompressed copy will be saved...',flush=True)
    with gzip.open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*CHUNK),b''):
            unpacked+=len(chunk)
            if time.monotonic()-last_progress>10:
                print(f'GZIP checked {unpacked} uncompressed bytes',flush=True);last_progress=time.monotonic()
    with gzip.open(path,'rb') as f:prefix=frames(f,100_000)
    result={'verified_utc':stamp(),'path':str(path.resolve()),'compressed_bytes':path.stat().st_size,'sha256':digest,
            'sha_source':cfg['sample']['reference_sha256_source'],'full_gzip_crc_eof_passed':True,'uncompressed_bytes_streamed':unpacked,
            'prefix_framing':prefix,'full_day_itch_validation':False,'order_book_reconstructed':False,'training_ready':False,
            'seconds':round(time.monotonic()-started,2)}
    save(REPORTS/'file_verification.json',result)
    print(json.dumps(result,indent=2))

def environment(data_root):
    result={'checked_utc':stamp(),'python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),
            'data_root':str(data_root),'torch_smoke':'not_run','synthetic_only':True}
    existing=Path(data_root)
    while not existing.exists() and existing!=existing.parent:existing=existing.parent
    if existing.exists():result['data_drive_free_gib']=round(shutil.disk_usage(existing).free/1024**3,2)
    nvidia=shutil.which('nvidia-smi')
    if nvidia:result['nvidia_smi']=subprocess.run([nvidia,'--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    try:
        import torch
        result.update(torch_version=torch.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available())
        if not torch.cuda.is_available():raise RuntimeError('PyTorch cannot access CUDA')
        torch.manual_seed(42)
        model=torch.nn.Sequential(torch.nn.Conv1d(40,32,5,padding=2),torch.nn.ReLU(),torch.nn.Conv1d(32,32,3,padding=1),torch.nn.ReLU(),torch.nn.Conv1d(32,64,3,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool1d(1),torch.nn.Flatten(),torch.nn.Linear(64,3)).cuda()
        x=torch.randn(16,40,100,device='cuda');y=torch.arange(16,device='cuda')%3
        torch.cuda.synchronize();started=time.perf_counter()
        output=model(x);loss=torch.nn.functional.cross_entropy(output,y);loss.backward();torch.cuda.synchronize()
        if not torch.isfinite(output).all() or not torch.isfinite(loss):raise RuntimeError('Non-finite synthetic smoke result')
        result.update(torch_smoke='passed_cuda_forward_backward_synthetic',output_shape=list(output.shape),gpu_name=torch.cuda.get_device_name(0),smoke_seconds=round(time.perf_counter()-started,4),weights_saved=False,
                      note='Random synthetic environment test only, no market training or NPU evidence. Smoke network parameters are not frozen.')
    except Exception as e:result['torch_error']=str(e)
    save(REPORTS/'environment.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['torch_smoke']!='passed_cuda_forward_backward_synthetic':raise RuntimeError('Training environment not yet ready; see environment.json')

def status():
    for filename in ['source_probe.json','prefix_smoke.json','download_result.json','file_verification.json','environment.json']:
        p=REPORTS/filename
        print(f'{filename}: {"PRESENT (read scope inside)" if p.exists() else "MISSING"}')
    print('W1 does NOT certify full-day order-book correctness, final dataset, trained model, or hardware milestones.')

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['probe','smoke','download','verify','environment','status','restart'])
    p.add_argument('--data-root');p.add_argument('--existing-file')
    args=p.parse_args();cfg=config();data_root=args.data_root or cfg['data_root'];path=raw_path(data_root,cfg)
    REPORTS.mkdir(parents=True,exist_ok=True)
    if args.action=='probe':probe(cfg)
    elif args.action=='smoke':smoke(cfg)
    elif args.action=='environment':environment(data_root)
    elif args.action=='verify':verify_file(Path(args.existing_file) if args.existing_file else path,cfg)
    elif args.action=='status':status()
    elif args.action=='restart':
        # Preserve a failed/unsupported-resume partial; never recursively delete data.
        part=path.with_name(path.name+'.part')
        if part.exists():
            archive=part.with_name(part.name+'.kept-'+str(time.time_ns()));part.rename(archive);print(f'Preserved old partial: {archive}')
        print('Run download again to start a fresh transfer. No final file deleted.')
    else:
        sample=cfg['sample'];meta=head(sample['url'])
        if meta['bytes']!=sample['expected_bytes']:raise ValueError('Remote size differs from manifest')
        save(REPORTS/'download_head.json',meta)
        result=download_one(sample['url'],path,sample['expected_bytes'],sample['reference_sha256'])
        save(REPORTS/'download_result.json',{**result,'checked_utc':stamp(),'official_checksum_verified':False,'sha_source':sample['reference_sha256_source']})
        print(json.dumps(result,indent=2))

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('Interrupted. Any .part is retained for verified resume.',file=sys.stderr);sys.exit(130)
    except Exception as e:
        save(REPORTS/'last_error.json',{'at_utc':stamp(),'error':str(e),'action':sys.argv[1] if len(sys.argv)>1 else None})
        print(f'ERROR: {e}',file=sys.stderr);sys.exit(1)
