"""Board administration for the explicitly selected local development board."""
import base64
import hashlib
import os
import json
from pathlib import Path
import shlex
import time

import paramiko

from .delivery import sha256, write_json
from .paths import ARTIFACTS, WORKSPACE, ROOT

_settings_path = ROOT/'config/board.json'
_settings = json.loads(_settings_path.read_text(encoding='utf-8-sig')) if _settings_path.is_file() else {}
HOST = os.environ.get('FINEDGE_BOARD_HOST', _settings.get('host', '192.168.126.49'))
HOST_KEY = os.environ.get('FINEDGE_BOARD_HOST_KEY', _settings.get('ssh_sha256', 'IQGq+fgZLG+sC77GCs+cZuntFUYB1mcNM4POwA4ZzLc')).removeprefix('SHA256:').rstrip('=')
USERNAME = os.environ.get('FINEDGE_BOARD_USER', _settings.get('username', 'root'))


class PinnedHostKey(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        fingerprint = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
        if hostname != HOST or fingerprint != HOST_KEY:
            raise paramiko.SSHException('Board SSH identity differs from the initial local-board observation')


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(PinnedHostKey())
    # Public factory credential for this development board; never logged or saved.
    password = os.environ.get('FINEDGE_BOARD_PASSWORD', 'fmsh')
    client.connect(HOST, username=USERNAME, password=password, look_for_keys=False,
                   allow_agent=False, timeout=8, auth_timeout=8)
    return client


def command(client, text, timeout=30):
    _, stdout, stderr = client.exec_command(text, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    code = stdout.channel.recv_exit_status()
    return {'exit_code': code, 'stdout': out, 'stderr': err}


def preflight():
    with connect() as client:
        result = command(client, 'uname -a; dpkg-query -W "*icraft*" "*Icraft*" 2>/dev/null; '
                         'command -v icraft-serve; ls -l /dev/udmabuf*; ss -ltnp | grep :9981 || true')
    report = {'host': HOST, 'ssh_host_key_sha256': HOST_KEY, 'inspection': result}
    write_json(ARTIFACTS / 'board' / 'preflight.json', report)
    return report


def install_runtime():
    setup = ROOT/'board/packages' if (ROOT/'board/packages').is_dir() else WORKSPACE/'Icraft_setup'
    paths = [setup / 'Icraft_3.36.1_onchip.deb', setup / 'CustomOp_3.36.1_onchip.deb']
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError('Matching 3.36.1 onchip packages are missing')
    steps = []
    with connect() as client:
        result = command(client, 'mkdir -p /root/finedge_host_demo/packages; dpkg-query -W > /root/finedge_host_demo/packages/dpkg_before.txt')
        steps.append({'prepare': result})
        with client.open_sftp() as sftp:
            for p in paths:
                remote = '/root/finedge_host_demo/packages/' + p.name
                sftp.put(str(p), remote)
                verify = command(client, 'sha256sum ' + shlex.quote(remote) + '; dpkg-deb -f ' + shlex.quote(remote) + ' Package Version Architecture')
                if verify['exit_code'] or not verify['stdout'].startswith(sha256(p)) or 'arm64' not in verify['stdout'] or '3.36.1' not in verify['stdout']:
                    raise ValueError('Uploaded package identity/architecture mismatch: ' + p.name)
                steps.append({'file': p.name, 'sha256': sha256(p), 'control': verify['stdout']})
        files = ' '.join(shlex.quote('/root/finedge_host_demo/packages/' + p.name) for p in paths)
        result = command(client, 'dpkg -i ' + files, timeout=120)
        steps.append({'install': result})
        write_json(ARTIFACTS / 'board' / 'installation.json', {'steps': steps, 'status': 'passed' if result['exit_code'] == 0 else 'failed'})
        if result['exit_code']:
            raise RuntimeError('Board runtime installation failed; see installation.json')
        check = command(client, 'command -v icraft-serve; icraft-serve --version 2>&1 || true; dpkg-query -W "*icraft*" "*Icraft*" 2>/dev/null || true')
        write_json(ARTIFACTS / 'board' / 'runtime_version.json', check)
    return preflight()


def start_service():
    with connect() as client:
        result = command(client, 'if ss -ltnp | grep -q ":9981 "; then echo ALREADY_LISTENING; '
                         'else mkdir -p /root/finedge_host_demo; '
                         'nohup icraft-serve > /root/finedge_host_demo/serve.log 2>&1 < /dev/null & echo $!; fi')
        time.sleep(1)
        check = command(client, 'ss -ltnp | grep :9981; tail -15 /root/finedge_host_demo/serve.log')
        report = {'start': result, 'check': check, 'host': HOST}
        write_json(ARTIFACTS / 'board' / 'service.json', report)
        if ':9981' not in check['stdout']:
            raise RuntimeError('icraft-serve did not start; see board/service.json')
        return report

