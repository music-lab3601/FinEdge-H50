"""E startup observation using an injected SSH command executor.

This module imports no network client. Tests exercise the same read/mount/hash
flow with a command substitute. No board executor is released this round.
"""
import re
import uuid


def capture_boot(execute,*,label,previous_boot_ids,expected_BOOT_sha256,expected_SDK='3.36.1'):
    if not re.fullmatch(r'[A-Za-z0-9_]{1,64}',label):raise ValueError('Safe unique preflight label required')
    if not re.fullmatch(r'[0-9a-f]{64}',expected_BOOT_sha256):raise ValueError('Expected frozen BOOT hash')
    transcript=[]
    def checked(command):
        result=execute(command);transcript.append(dict(command=command,result=result))
        if result['exit_code']!=0:raise ValueError('Preflight command failed: '+command)
        return result['stdout'].strip()
    boot=checked('cat /proc/sys/kernel/random/boot_id')
    if str(uuid.UUID(boot))!=boot or boot in previous_boot_ids:raise ValueError('New boot_id required')
    version=checked("dpkg-query -W -f='${Version}' icraft")
    if not re.fullmatch(re.escape(expected_SDK)+r'(?:[-+].*)?',version):raise ValueError('SDK version mismatch')
    if checked("ss -tnp state established | grep ':9981' || true"):
        raise ValueError('Existing SDK client; do not reclaim/attach')
    if checked('findmnt -rn -S /dev/mmcblk0p1 -o TARGET || true'):
        raise ValueError('SD partition already mounted; no changes made')
    mount='/mnt/finedge_e_'+label;mounted=False
    try:
        checked(f'test ! -e {mount} && mkdir {mount}')
        checked(f'mount -o ro /dev/mmcblk0p1 {mount}');mounted=True
        info=checked(f'findmnt -rn -T {mount} -o SOURCE,FSTYPE,OPTIONS').split()
        if len(info)!=3 or info[:2]!=['/dev/mmcblk0p1','vfat'] or 'ro' not in info[2].split(','):
            raise ValueError('Expected read-only SD boot mount')
        actual=checked(f'sha256sum {mount}/BOOT.bin').split()[0]
        if actual!=expected_BOOT_sha256:raise ValueError('Actual SD BOOT differs from frozen candidate')
    finally:
        if mounted:checked(f'umount {mount}')
    return dict(boot_id=boot,actual_BOOT_sha256=actual,SDK_version=expected_SDK,
                SDK_package_version=version,transcript=transcript,
                strict_visibility_proven=False)


def initialize_after_preflight(execute,owner_factory,*,physical_stop_confirmed,**kwargs):
    """No factory/Session/arena creation until fresh observations all pass.

    The factory must start the unchanged accepted C++ owner, whose separate
    initialize_owner gate checks clean UDW registers BEFORE Session/allocation.
    """
    if physical_stop_confirmed is not True:raise ValueError('Human physical stop confirmation required')
    observation=capture_boot(execute,**kwargs)
    owner=owner_factory()
    return observation,owner
