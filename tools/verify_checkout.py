"""Verify the files actually tracked by Git, their identities and public metadata."""
from shared.paths import resolve_path
from pathlib import Path
import argparse
import ast
import hashlib
import json
import pickletools
import re
import shutil
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_PATH = re.compile(r'(?<![A-Za-z0-9_])[A-Za-z]:[\\/]+')
SECRET_RULES = {
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'service_key': re.compile(r'\bsk-[A-Za-z0-9_-]{24,}\b'),
    'aws_access_key': re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
    'slack_token': re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{20,}\b'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}
TEXT_SUFFIXES = {'.py', '.c', '.cpp', '.h', '.hpp', '.sv', '.v', '.vh', '.svh', '.xdc',
                 '.tcl', '.xpr', '.xci', '.json', '.toml', '.csv', '.md', '.txt',
                 '.ps1', '.bat', '.cmd', '.sh', '.in', '.cmake', '.html', '.css', '.js', '.yml'}
RELATIVE_BUILD = 'hardware/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358'
ETHERNET = 'hardware/versions/realtime_integration_v1/board_sources/vendor/verilog-ethernet'
REQUIRED = [
    'LICENSE', 'THIRD_PARTY_NOTICES.md', '.gitignore', '.gitattributes',
    'models/identity.json', 'models/cpu/model_cpu.ts', 'models/cpu/native/best.pt',
    'models/cpu/runtime.py', 'models/cpu/normalization.json', 'models/cpu/package_manifest.json',
    'models/npu/model.json', 'models/npu/model.raw', 'models/npu/input_contract.json',
    'models/npu/scales.f32', 'models/onnx/tcn_ed4.onnx', 'models/onnx/tcn_polyphase.onnx',
    'software/project/scripts/tcn_existing_data_ed4_train.py',
    'software/project/scripts/tcn_existing_data_ed4_features.py',
    'software/project/scripts/tcn_research_p5_model.py', 'software/project/scripts/tcn_candidate.py',
    'deployment/project/src/finedge_host/export_model.py',
    'deployment/project/scripts/build_polyphase_model.py',
    'deployment/project/versions/int8_model_v1/scripts/compile_models.py',
    'deployment/project/versions/int8_model_v1/improvement_20261030_v2/scripts/compile_mixed.py',
    'deployment/project/versions/int8_model_v1/lowbit_entry_20261030_v3/scripts/rebuild_native_models.py',
    'third_party/lightweight-charts_LICENSE.txt', 'third_party/lightweight-charts_NOTICE.txt',
    'external/dependencies.json', 'provenance/source_map.json',
]
REQUIRED += [ETHERNET + '/' + name for name in (
    'COPYING', 'AUTHORS', 'lib/axis/syn/vivado/axis_async_fifo.tcl',
    'syn/vivado/rgmii_phy_if.tcl', 'syn/vivado/eth_mac_fifo.tcl',
    'syn/vivado/eth_mac_1g_rgmii.tcl', 'syn/vivado/eth_mac.tcl')]
REQUIRED += [RELATIVE_BUILD + '/project/rtl/bench/' + name for name in (
    'g_bench_endpoint.sv', 'fb_book_core.sv', 'fp_fixed_record.sv', 'fp_fixed_scalar.sv',
    'wt_history.sv', 'wt_async_fifo.sv', 'tt_native_packer.sv', 'g_payload_bridge.sv',
    'g_queue_scheduler.sv', 'g_predict_control.sv', 'g_predict_path.sv',
    'g_result_packet.sv', 'fixed_parameters.vh',
    'fixed_lane0.hex', 'fixed_lane1.hex', 'fixed_lane2.hex', 'fixed_lane3.hex')]

# Required identities follow the recorded relocation map.
LAYOUT = json.loads((ROOT/'provenance/layout_map.json').read_text(encoding='utf-8'))['files']
REQUIRED = [LAYOUT.get(name,name) for name in REQUIRED]
REQUIRED += ['finedge.py', 'config/default.json', 'models/normalization.json',
             'hardware/project/reference.xpr', 'tools/configuration.py', 'requirements.txt']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_text(text, location, findings):
    for index, line in enumerate(text.splitlines(), 1):
        if WINDOWS_PATH.search(line):
            findings.append(dict(path=location, line=index, rule='machine_absolute_path'))
        for name, pattern in SECRET_RULES.items():
            if pattern.search(line):
                # The matched value is deliberately excluded from all reports.
                findings.append(dict(path=location, line=index, rule=name))


def scan_blob(path, relative, findings):
    if path.suffix in {'.pt', '.ts'} and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if '/data/' in name:
                    continue  # Tensor storage is numerical data, not metadata.
                data = archive.read(name)
                location = relative + '!' + name
                if name.endswith('.pkl'):
                    try:
                        for opcode, value, offset in pickletools.genops(data):
                            if isinstance(value, str):
                                scan_text(value, location + '@' + str(offset), findings)
                    except (ValueError, UnicodeError):
                        findings.append(dict(path=location, rule='unparsed_pickle_metadata'))
                elif name.endswith(('.py', '.json', '.txt')):
                    scan_text(data.decode('utf-8'), location, findings)
    else:
        for match in re.findall(rb'[\x20-\x7e]{8,}', path.read_bytes()):
            scan_text(match.decode('ascii'), relative + '!printable_metadata', findings)


def scan_python_literals(tree, location, findings):
    names = {'password', 'passwd', 'pwd', 'api_key', 'client_secret',
             'secret_key', 'access_token', 'board_password'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            value = node.value.value
            if not isinstance(value, (str, bytes)) or not value:
                continue
            for target in node.targets:
                name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else ''
                if name.lower() in names:
                    findings.append(dict(path=location, line=node.lineno, rule='literal_credential_assignment'))
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            if node.value in ('fmsh' + chr(13), b'fmsh' + bytes([13])) and '/tests/' not in location:
                findings.append(dict(path=location, line=node.lineno, rule='default_runtime_password'))


def verify(git):
    process = subprocess.run([git, '-C', str(ROOT), 'ls-files', '-z'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    tracked = {name for name in process.stdout.decode('utf-8').split('\0') if name}
    errors = []
    findings = []
    templates = []
    for name in REQUIRED:
        if name not in tracked or not (ROOT / name).is_file():
            errors.append('Required file missing from Git: ' + name)
    manifest = json.loads((ROOT / 'provenance/release_files.json').read_text(encoding='utf-8'))
    expected = {row['path']: row for row in manifest['files']}
    for name, row in expected.items():
        path = ROOT / name
        if name not in tracked or not path.is_file() or sha(path) != row['sha256']:
            errors.append('File manifest mismatch: ' + name)
    extras = tracked - set(expected) - {'provenance/release_files.json'}
    errors.extend('Unlisted tracked file: ' + name for name in sorted(extras))
    for name in sorted(tracked):
        path = ROOT / name
        if not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            errors.append('File escapes repository: ' + name)
            continue
        if path.stat().st_size > 100 * 1024**2:
            errors.append('Oversized Git blob: ' + name)
        if path.suffix.lower() in {'.edif', '.edf', '.dcp', '.dll', '.so', '.bit', '.elf'}:
            errors.append('Vendor/build binary requires separate distribution review: ' + name)
        is_text = path.suffix.lower() in TEXT_SUFFIXES or path.name in {'LICENSE', 'COPYING', 'AUTHORS', 'CMakeLists.txt', '.gitignore', '.gitattributes'}
        if is_text:
            try:
                text = path.read_text(encoding='utf-8-sig')
            except UnicodeError:
                errors.append('Unreadable text: ' + name)
                continue
            scan_text(text, name, findings)
            tokens = set(re.findall(r'@[A-Z][A-Z0-9_]*@', text))
            if tokens:
                templates.append(dict(path=name, tokens=sorted(tokens)))
            if path.suffix == '.py':
                try:
                    tree = ast.parse(text)
                    scan_python_literals(tree, name, findings)
                except SyntaxError as error:
                    errors.append('Python syntax: ' + name + ' line ' + str(error.lineno))
            if path.suffix == '.json':
                try:
                    json.loads(text)
                except ValueError:
                    errors.append('JSON syntax: ' + name)
        else:
            scan_blob(path, name, findings)
    # Resolve the source project inputs against tracked files or explicit external assets.
    external = json.loads((ROOT / 'external/dependencies.json').read_text(encoding='utf-8'))
    declared = {row['target'] for row in external['files']}
    xpr = ROOT / 'hardware/project/reference.xpr'
    xpr_inputs = 0
    for item in ET.parse(xpr).findall('.//File'):
        reference=item.get('Path','')
        if reference.startswith('repo://'):
            name=reference.removeprefix('repo://')
            if name not in tracked or not (ROOT/name).is_file():
                errors.append('XPR repository input missing: '+name)
        elif reference.startswith('vendor://'):
            if reference not in declared:
                errors.append('Undeclared XPR vendor input: '+reference)
        else:
            errors.append('XPR input must use a declared resource URI: '+reference)
        xpr_inputs += 1
    for row in json.loads((ROOT/'provenance/source_map.json').read_text(encoding='utf-8')):
        path=ROOT/row['path']
        if row['path'] not in tracked or not path.is_file() or sha(path)!=row['public_sha256']:
            errors.append('Source provenance mismatch: '+row['path'])
    identity = json.loads((ROOT / 'models/identity.json').read_text(encoding='utf-8'))
    for filename, key in (('models/cpu/package_manifest.json', 'CPU_manifest_sha256'),
                          ('models/normalization.json', 'normalization_sha256'),
                          ('models/npu/model.json', 'NPU_model_sha256'), ('models/npu/model.raw', 'NPU_raw_sha256')):
        if sha(ROOT / filename) != identity[key]:
            errors.append('Model identity differs: ' + filename)
    cpu = json.loads((ROOT / 'models/cpu/package_manifest.json').read_text(encoding='utf-8'))
    for name, reference in cpu['files'].items():
        if not (ROOT / 'models/cpu' / name).is_file() or sha(ROOT / 'models/cpu' / name) != reference['sha256']:
            errors.append('CPU model payload differs: ' + name)
    return dict(status='passed' if not errors and not findings else 'failed',
                tracked_files=len(tracked), required_files=len(REQUIRED),
                manifest_files=len(expected), XPR_inputs_accounted_for=xpr_inputs,
                external_files_declared=len(declared), configuration_templates=templates,
                sensitive_findings=findings, errors=errors,
                verification_scope='Git collection, identities, metadata and declared dependencies',
                full_build_verified=False, board_access=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--git', default=shutil.which('git'))
    args = parser.parse_args(argv)
    if not args.git:
        parser.error('Install Git or specify --git')
    result = verify(args.git)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
