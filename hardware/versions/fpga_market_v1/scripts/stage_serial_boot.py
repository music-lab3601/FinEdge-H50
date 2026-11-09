"""Stage a validated BOOT through PS UART, with SD + PC recovery copies."""
from pathlib import Path
import argparse,hashlib,json,shlex,time,uuid
from serial_console import Console
V=Path(__file__).resolve().parents[1];P=V.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--release',type=Path,required=True);a=ap.parse_args()
    release=json.loads(a.release.read_text());boot=a.release.parent/'BOOT.bin';gz=a.release.parent/'BOOT.bin.gz'
    if release['status']!='packaged_pending_board_validation' or sha(boot)!=release['boot_sha256'] or sha(gz)!=release['compressed_sha256']:raise RuntimeError('Package identity')
    if any(float(release['slack_ns'][k])<0 for k in ('max','min')):raise RuntimeError('Setup/hold must close')
    out=a.release.parent/'serial_stage';out.mkdir();tag=uuid.uuid4().hex[:12]
    mount='/mnt/finedge_market_'+tag;remote='/root/finedge_market/BOOT_'+tag+'.bin.gz'
    report=dict(status='uploading',new_BOOT_sha256=sha(boot),BOOT_replaced=False,board_rebooted=False)
    def save():(out/'result.json').write_text(json.dumps(report,indent=2))
    c=Console();mounted=False;save()
    def checked(command,timeout=60):
        r=c.run(command,timeout)
        if r['exit_code']:raise RuntimeError(r['stdout'])
        return r['stdout']
    try:
        report['before_boot_id']=checked('cat /proc/sys/kernel/random/boot_id').strip()
        if checked('findmnt -rn -S /dev/mmcblk0p1 -o TARGET || true').strip():raise RuntimeError('SD boot partition already mounted')
        c.put(gz,remote,timeout=1200)
        checked('mkdir '+mount+' && mount /dev/mmcblk0p1 '+mount);mounted=True
        old=checked('sha256sum '+mount+'/BOOT.bin').split()[0];report['old_BOOT_sha256']=old
        backup='BOOT_before_market_'+tag+'.bin'
        checked('test ! -e '+mount+'/'+backup+' && cp '+mount+'/BOOT.bin '+mount+'/'+backup+' && sync')
        if checked('sha256sum '+mount+'/'+backup).split()[0]!=old:raise RuntimeError('SD recovery SHA')
        report['SD_backup']=backup
        # Locate a byte-identical already saved current-project BOOT. If none
        # exists, fetch the board's backup through UART rather than assume it.
        local=None
        preferred=P/'artifacts/tenth/build_20261027_02/boot/BOOT.bin'
        candidates=[preferred,*P.joinpath('artifacts').glob('*/boot/BOOT.bin')]
        loaded_path=P/'versions/realtime_integration_v1/state/loaded_firmware.json'
        if loaded_path.is_file():
            loaded=json.loads(loaded_path.read_text())
            if loaded.get('boot_sha256')==old and loaded.get('release'):
                known_boot=(Path(loaded['release']).parent/'BOOT.bin').resolve()
                if known_boot.is_relative_to(P.resolve()):candidates.insert(0,known_boot)
        for p in candidates:
            if p.is_file() and sha(p)==old:local=p;break
        recovery=out/'BOOT_before_market.bin'
        if local:recovery.write_bytes(local.read_bytes())
        else:c.get(mount+'/'+backup,recovery,timeout=1200)
        if sha(recovery)!=old:raise RuntimeError('PC recovery SHA')
        report['PC_backup']=str(recovery);report['status']='backups_verified';save()
        pending=mount+'/BOOT.pending_'+tag
        code='import gzip,pathlib,hashlib;d=gzip.decompress(pathlib.Path('+repr(remote)+').read_bytes());assert hashlib.sha256(d).hexdigest()=='+repr(report['new_BOOT_sha256'])+';f=open('+repr(pending)+',"xb");f.write(d);f.flush();f.close()'
        checked('python3 -c '+shlex.quote(code))
        checked('mv '+pending+' '+mount+'/BOOT.bin && sync')
        report['BOOT_replaced']=True;save()
        if checked('sha256sum '+mount+'/BOOT.bin').split()[0]!=report['new_BOOT_sha256']:raise RuntimeError('Active BOOT SHA')
        checked('umount '+mount);mounted=False
        report['status']='staged_verified_not_loaded';save();print(json.dumps(report))
    except BaseException as e:report.update(status='failed',error=str(e));raise
    finally:
        if mounted:
            try:checked('sync && umount '+mount)
            except BaseException as e:report['unmount_error']=str(e)
        save();c.close()
if __name__=='__main__':main()
