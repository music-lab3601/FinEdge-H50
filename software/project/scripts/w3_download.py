"""W3 bounded-range acquisition; local SHA is NOT a publisher checksum."""
import gzip
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
from w1 import head, save, sha256, stamp

BLOCK = 32 * 1024**2
RESERVE = 15 * 1024**3


def verify_gzip(path):
    count = 0
    with gzip.open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            count += len(block)
    return count


def fetch(url, target, expected_bytes, *, reserve=RESERVE, allow_local_test=False):
    if not (url.startswith('https://') or (allow_local_test and url.startswith('http://127.0.0.1:'))):
        raise ValueError('Verified HTTPS required')
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = head(url)
    if metadata['bytes'] != expected_bytes:
        raise ValueError('Source length differs from reviewed manifest')
    identity = {k: metadata[k] for k in ('url', 'bytes', 'etag', 'last_modified')}
    if not identity['etag'] and not identity['last_modified']:
        raise ValueError('No stable remote identity available for resume')
    sidecar = target.with_name(target.name+'.download.json')
    part = target.with_name(target.name+'.part')
    previous = json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() else {}
    if target.exists():
        if previous.get('identity') != identity or previous.get('status') != 'size_sha_gzip_verified':
            raise ValueError('Existing file has no matching verified manifest; preserved')
        if target.stat().st_size != expected_bytes or sha256(target) != previous['sha256']:
            raise ValueError('Existing file differs; preserved')
        return previous
    if part.exists() and previous.get('identity') != identity:
        raise ValueError('Partial file identity missing/changed; preserved, do not append')
    offset = part.stat().st_size if part.exists() else 0
    if offset > expected_bytes:
        raise ValueError('Partial file is too large')
    if shutil.disk_usage(target.parent).free < expected_bytes-offset+reserve:
        raise ValueError('Not enough space while keeping 15 GiB reserve')
    save(sidecar, {'identity': identity, 'status': 'partial', 'at_utc': stamp()})
    started = time.monotonic()
    while offset < expected_bytes:
        end = min(offset+BLOCK, expected_bytes)-1
        for attempt in range(3):
            try:
                # Cache key includes range; servers/proxies must still return the exact range.
                request_url = url+('?' if '?' not in url else '&')+f'finedge_range={offset}-{end}'
                headers = {'Range': f'bytes={offset}-{end}', 'Accept-Encoding': 'identity',
                           'If-Range': identity['etag'] or identity['last_modified']}
                with urllib.request.urlopen(urllib.request.Request(request_url, headers=headers), timeout=60) as r:
                    if r.status != 206 or r.headers.get('Content-Range') != f'bytes {offset}-{end}/{expected_bytes}':
                        raise ValueError('Exact range not honored; partial preserved')
                    if r.headers.get('Content-Encoding', 'identity') != 'identity':
                        raise ValueError('Unexpected HTTP encoding')
                    for key, header in [('etag', 'ETag'), ('last_modified', 'Last-Modified')]:
                        if identity[key] and r.headers.get(header) != identity[key]:
                            raise ValueError('Remote identity changed during download')
                    block = r.read(end-offset+2)
                    if len(block) != end-offset+1:
                        raise OSError('Short/oversized range; current block not appended')
                break
            except ValueError:
                raise
            except Exception:
                if attempt == 2: raise
                time.sleep(1)
        if shutil.disk_usage(target.parent).free < len(block)+reserve:
            raise ValueError('Disk reserve reached; resume after freeing space')
        with part.open('ab') as f:
            f.write(block); f.flush(); os.fsync(f.fileno())
        offset = end+1
        print(f'DOWNLOAD {offset:,}/{expected_bytes:,} ({offset/expected_bytes:.1%})', flush=True)
    print('Checking full gzip EOF/CRC and local SHA256...', flush=True)
    unpacked = verify_gzip(part)
    digest = sha256(part)
    result = {'identity': identity, 'status': 'size_sha_gzip_verified', 'sha256': digest,
              'publisher_checksum_verified': False, 'sha_provenance': 'locally computed after HTTPS transfer and gzip CRC',
              'uncompressed_bytes': unpacked, 'path': str(target), 'seconds': round(time.monotonic()-started, 2), 'at_utc': stamp()}
    # Record verification first so an interruption after rename is recoverable.
    save(sidecar, result)
    part.rename(target)
    return result
