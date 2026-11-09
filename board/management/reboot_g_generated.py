"""Normal software reboot and PS-console login; never calls Device.reset."""
import os
from pathlib import Path
import argparse,json,uuid
import sys

from board.management.serial_console import Console
from shared.paths import load_settings

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--serial-port',default=load_settings()['board']['serial_port'])
    ap.add_argument('--npu-only',action='store_true',help='Keep existing BOOT; identify new Linux boot only, no PL-market admission')
    a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);c=Console(a.serial_port);report=dict(status='requested',physical_cold_start=False)
    try:
        r=c.run('cat /proc/sys/kernel/random/boot_id')
        if r['exit_code']:raise RuntimeError('Cannot identify previous boot')
        report['before']=r['stdout'].strip()
        r=c.run("nohup sh -c 'sleep 2; sync; /sbin/reboot' > /root/finedge_g_bench/reboot.log 2>&1 < /dev/null &")
        if r['exit_code']:raise RuntimeError('Reboot request rejected')
        c.serial.reset_input_buffer();c.pending=b''
        raw,_=c.until(rb'(?:^|\n)[^\n]{1,80} login: ?',60)
        (a.output/'boot_console.txt').write_bytes(raw)
        c.serial.write(b'root\r');c.until(rb'Password: ?',5)
        c.serial.write(os.environ['FINEDGE_BOARD_PASSWORD'].encode('utf-8') + b'\r');c.until(rb'root@[^ \r\n]+:[^\r\n]*# ?',10)
        r=c.run('cat /proc/sys/kernel/random/boot_id')
        if r['exit_code']:raise RuntimeError('Cannot identify new boot')
        report['after']=r['stdout'].strip()
        if report['after']==report['before']:raise RuntimeError('Boot identity did not change')
        if a.npu_only:
            report.update(status='new_boot_verified_NPU_component_only',PL_market_firmware_verified=False)
            return
        # Read only the expected new controller, without opening an SDK Session.
        code="import os,mmap,struct,json;f=os.open('/dev/mem',os.O_RDONLY|os.O_SYNC);m=mmap.mmap(f,1048576,prot=mmap.PROT_READ,offset=0x40000000);print(json.dumps({hex(a):struct.unpack_from('<I',m,a)[0] for a in (0xd0000,0xd0004,0xd0008)}))"
        import shlex
        r=c.run('python3 -c '+shlex.quote(code),timeout=10)
        if r['exit_code']:raise RuntimeError('Controller observation failed')
        report['registers']=json.loads(r['stdout'])
        if report['registers']['0xd0000']!=0x31424746 or report['registers']['0xd0004']!=1 or report['registers']['0xd0008']&15:
            raise RuntimeError('Expected fresh G firmware state')
        report['status']='new_G_boot_and_clean_controller_verified'
    except BaseException as e:report.update(status='failed',error=str(e));raise
    finally:
        (a.output/'result.json').write_text(json.dumps(report,indent=2));c.close()
    print(json.dumps(report))
if __name__=='__main__':main()
