"""Tool-independent CI checks in an isolated configuration and output directory."""
from datetime import datetime, timezone
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time

from contracts.market.finedge_contract import decode_event
from shared.io import atomic_json
from shared.paths import ROOT


def run_step(name, command, output, env, results):
    row = dict(name=name, status='running')
    results.append(row)
    started = time.monotonic()
    try:
        with (output / 'logs' / (name + '.log')).open('wb') as log:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=900)
        row['return_code'] = process.returncode
        if process.returncode:
            raise RuntimeError(name + ' failed. See ' + str(output / 'logs' / (name + '.log')))
        row['status'] = 'passed'
        print(name + ' passed', flush=True)
    except Exception:
        row['status'] = 'failed'
        raise
    finally:
        row['seconds'] = time.monotonic() - started


def check_native_source(output, env):
    build = json.loads((output / 'work/native/source_build.json').read_text('utf-8'))
    raw = (output / 'inputs/market/raw56.bin').read_bytes()[:180 * 56]
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    receiver.bind(('127.0.0.1', 0))
    receiver.settimeout(10)
    command = [build['executable'], '--raw', str(output / 'inputs/market/raw56.bin'),
               '--first', '1', '--last', '180', '--session', '1', '--host', '127.0.0.1',
               '--port', str(receiver.getsockname()[1]), '--out', str(output / 'native_protocol'),
               '--schedule', 'historical', '--speed', '100', '--packetization', 'single',
               '--core', '0', '--lead-ms', '20']
    child = None
    try:
        with (output / 'logs/native_protocol.log').open('wb') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            for index in range(180):
                packet = decode_event(receiver.recvfrom(2048)[0])
                if (packet['session_id'], packet['packet_seq'], packet['first_raw_ordinal'], packet['records']) != (1, index + 1, index + 1, 1):
                    raise ValueError('Native source packet identity mismatch')
                if packet['raw'] != raw[index * 56:(index + 1) * 56]:
                    raise ValueError('Native source raw bytes mismatch')
            if child.wait(timeout=10):
                raise RuntimeError('Native source returned failure')
        result = json.loads((output / 'native_protocol/source.json').read_text('utf-8'))
        if result['status'] != 'passed' or result['events_submitted'] != 180:
            raise ValueError('Native source event count mismatch')
    finally:
        receiver.close()
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
    return dict(status='passed', UDP_events=180, identity_and_raw_bytes_equal=True)


def main():
    output = ROOT / '.local/ci' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    (output / 'logs').mkdir(parents=True)
    config = output / 'config.json'
    atomic_json(config, dict(
        paths=dict(work_root=str(output / 'work'), data_root=str(output / 'data'),
                   vendor_root=str(output / 'vendor')),
        tools=dict(cmake=shutil.which('cmake')),
        inputs=dict(demo_profile=str(output / 'inputs/manifest.json'))))
    env = dict(os.environ, FINEDGE_CONFIG=str(config), PYTHONPATH=str(ROOT),
               PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    env.pop('FINEDGE_BOARD_PASSWORD', None)
    cli = [sys.executable, '-B', str(ROOT / 'finedge.py')]
    profile = str(output / 'inputs/manifest.json')
    stages = [
        ('dependencies', [sys.executable, '-m', 'pip', 'check']),
        ('checkout', cli + ['check', 'checkout']),
        ('templates', cli + ['configure', '--render']),
        ('imports', cli + ['check', 'imports']),
        ('generate', cli + ['example', 'generate', '--directory', str(output / 'inputs')]),
        ('verify', cli + ['example', 'verify', '--profile', profile]),
        ('tests', cli + ['check', 'tests']),
        ('CPU_model', cli + ['cpu', 'check']),
        ('CPU_network', cli + ['cpu', 'smoke', '--records', '160']),
        ('public_CPU', cli + ['example', 'cpu', '--profile', profile, '--directory',
                              str(output / 'CPU_replay'), '--records', '160', '--speed', '20']),
        ('board_bundle', cli + ['example', 'bundle', '--profile', profile,
                                '--directory', str(output / 'ARM_bundle')]),
        ('ONNX_export', cli + ['model', 'export']),
        ('causal_graph', cli + ['model', 'polyphase', '--label', 'ci', '--preserve-taps']),
        ('ROM_generation', [sys.executable, '-B', '-m', 'deployment.quantization.generate_fixed_rom',
                            '--out', str(output / 'ROM')]),
        ('native_build', cli + ['native']),
    ]
    result = dict(status='running', steps=[], board_access=False,
                  market_data='synthetic', proprietary_tools_required=False,
                  GitHub_hosted_run=bool(os.environ.get('GITHUB_ACTIONS')))
    print('CI output ' + str(output), flush=True)
    try:
        for name, command in stages:
            run_step(name, command, output, env, result['steps'])
        names = ['fixed_parameters.vh'] + ['fixed_lane' + str(i) + '.hex' for i in range(4)]
        for name in names:
            if (output / 'ROM/rtl' / name).read_bytes() != (ROOT / 'hardware/rom' / name).read_bytes():
                raise ValueError('Regenerated ROM differs. ' + name)
        result['ROM_files_byte_equal'] = len(names)
        result['native_protocol'] = check_native_source(output, env)
        test_log = (output / 'logs/tests.log').read_text('utf-8')
        count = int(re.search(r'Ran (\d+) tests', test_log).group(1))
        skipped = re.search(r'OK \(skipped=(\d+)\)', test_log)
        skips = int(skipped.group(1)) if skipped else 0
        result['tests'] = dict(executed=count, passed=count - skips, skipped=skips, failed=0)
        result['tests']['skipped_cases'] = [dict(test=match.group(1), reason=match.group(2))
            for match in re.finditer(r"^(.+?) \.\.\. skipped ['\"](.+?)['\"]$", test_log, re.MULTILINE)]
        if len(result['tests']['skipped_cases']) != skips:
            raise ValueError('Skipped-test accounting differs from unittest summary')
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error=type(exc).__name__ + '. ' + str(exc))
        print(result['error'], file=sys.stderr)
    finally:
        atomic_json(output / 'result.json', result)
    print(json.dumps(result, indent=2))
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
