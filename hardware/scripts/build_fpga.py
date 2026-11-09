"""Resolve, stage and build the H50 vendor-integrated FPGA project."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from shared.io import atomic_json
from shared.paths import ConfigurationError, ROOT, environment, load_settings, resolve_path, tool_executable, work_path
from shared.resources import ResourceLock
from tools.configuration import URI


def required_assets():
    rows = json.loads((ROOT / 'external/dependencies.json').read_text('utf-8'))['files']
    return [row for row in rows if row['target'].startswith(('vendor://fpga/', 'vendor://boot/'))]


def import_assets(source):
    """Import exact manifest inputs from an extracted original delivery tree."""
    source = source.resolve(strict=True)
    inputs = []
    for row in required_assets():
        path = source / row['original_target'].removeprefix('hardware/')
        if not path.is_file() or path.stat().st_size != row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ConfigurationError('Original vendor input missing or changed: ' + row['target'])
        target = resolve_path(row['target'])
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
            raise ConfigurationError('Existing vendor cache differs: ' + row['target'])
        inputs.append((path, target, row))
    for path, target, row in inputs:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
            raise ConfigurationError('Vendor import verification failed: ' + row['target'])
    result = dict(status='passed', source=str(source), checked=len(inputs),
                  files=[dict(target=row['target'], sha256=row['sha256']) for _, _, row in inputs], board_access=False)
    atomic_json(resolve_path('vendor://import_receipt.json'), result)
    return result


def dependencies():
    required = required_assets()
    missing, changed = [], []
    for row in required:
        path = resolve_path(row['target'])
        if not path.is_file():
            missing.append(row['target'])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            changed.append(row['target'])
    project_refs = [item.attrib['Path'] for item in ET.parse(ROOT / 'hardware/project/reference.xpr').findall('.//File')]
    missing += [ref for ref in project_refs if not resolve_path(ref).is_file() and ref not in missing]
    return dict(status='passed' if not missing and not changed else 'missing_or_changed_dependencies',
                checked=len(required), project_inputs=len(project_refs), missing=sorted(set(missing)),
                changed=changed, board_access=False)


def restore_vendor_outputs(directory):
    """Restore manifest IP outputs removed by Vivado reset_run/target generation."""
    directory = directory.resolve(strict=True)
    if not (directory / 'staging.json').is_file():
        raise ConfigurationError('Vendor restoration requires a staged build')
    restored = []
    checkpoints = []
    for row in required_assets():
        if not row['target'].startswith('vendor://fpga/'):
            continue
        target = Path(replace_value(row['target'], directory))
        if not target.exists():
            source = resolve_path(row['target'])
            if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
                raise ConfigurationError('Vendor cache identity changed: ' + row['target'])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            restored.append(row['target'])
        if target.suffix == '.dcp':
            if hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
                raise ConfigurationError('Vendor checkpoint identity changed: ' + row['target'])
            checkpoints.append(row['target'])
    receipt = dict(status='passed', restored=restored, verified_IP_checkpoints=checkpoints, board_access=False)
    atomic_json(directory / ('vendor_outputs_' + time.strftime('%Y%m%d_%H%M%S') + '.json'), receipt)
    return receipt


def stage(directory):
    directory = directory.resolve()
    if not directory.as_posix().isascii() or any(c.isspace() for c in directory.as_posix()):
        raise ConfigurationError('Use an ASCII FPGA build directory without spaces')
    if directory.exists():
        raise FileExistsError('Choose a new build directory. ' + str(directory))
    status = dependencies()
    if status['status'] != 'passed':
        raise ConfigurationError('FPGA vendor inputs missing or changed. Run finedge.py fpga --check; '
                                 'supply the exact external/dependencies.json assets under --vendor-root')
    directory.mkdir(parents=True)
    for folder in ('hardware', 'third_party/verilog-ethernet'):
        shutil.copytree(ROOT / folder, directory / folder, ignore=shutil.ignore_patterns('__pycache__'))
    # Vendor hooks infer the project root from <project>/<name>.runs/<run>.
    # Their RTL, IP repository and patch files must share that project root.
    shutil.copytree(resolve_path('vendor://fpga'), directory / 'project')

    def replace(match):
        return replace_value(match.group(), directory)

    for path in directory.rglob('*'):
        if path.is_file() and path.suffix in {'.sv', '.v', '.xdc', '.tcl', '.xpr', '.xci', '.xml', '.bd'}:
            try:
                text = path.read_text('utf-8-sig')
            except UnicodeError:
                continue
            if not URI.search(text):
                continue
            if path.suffix in {'.xpr', '.xci', '.xml', '.bd'} and text.lstrip().startswith('<'):
                # XML references may contain spaces, including "PS_AI IP".
                document = ET.fromstring(text)
                changed = False
                for element in document.iter():
                    for key, value in list(element.attrib.items()):
                        if value.startswith(('repo://', 'vendor://', 'work://', 'data://', 'tool://')):
                            element.set(key, replace_value(value, directory))
                            changed = True
                if changed:
                    ET.ElementTree(document).write(path, encoding='utf-8', xml_declaration=True)
            else:
                rewritten = URI.sub(replace, text)
                if rewritten != text:
                    path.write_text(rewritten, encoding='utf-8')
    project = directory / 'project/fpai_demo_vivado.xpr'
    document = ET.parse(directory / 'hardware/project/reference.xpr')
    document.getroot().set('Path', project.as_posix())
    for option in document.findall('.//Option'):
        if option.get('Name') == 'IPRepoPath':
            option.set('Val', (directory / 'project/PS_AI IP').as_posix())
    document.write(project, encoding='utf-8', xml_declaration=True)
    patch_config = directory / 'project/ip_patch/ip_patch.cfg'
    patch_config.write_text('vvd_path ' + tool_executable('vivado').parent.parent.as_posix() + '\n', encoding='utf-8')
    tcl = (directory / 'hardware/scripts/build.tcl').read_text('utf-8')
    tcl = tcl.replace('__PROJECT__', project.as_posix()).replace('__BUILD__', directory.as_posix())
    (directory / 'build.tcl').write_text(tcl, encoding='utf-8')
    atomic_json(directory / 'staging.json', dict(status='staged', dependency_check=status, board_access=False))
    return project


def replace_value(reference, directory):
    if reference.startswith(('repo://hardware/', 'repo://third_party/')):
        return (directory / reference.removeprefix('repo://')).as_posix()
    if reference.startswith('vendor://fpga/'):
        return (directory / 'project' / reference.removeprefix('vendor://fpga/')).as_posix()
    if reference.startswith('work://'):
        return (directory / reference.removeprefix('work://')).as_posix()
    return resolve_path(reference).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--package-boot', action='store_true')
    parser.add_argument('--import-source', type=Path, help='Extracted original delivery root containing the manifest original_target paths')
    parser.add_argument('--restore-vendor', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.restore_vendor:
        receipt = restore_vendor_outputs(args.restore_vendor)
        print(json.dumps(dict(status=receipt['status'], restored=len(receipt['restored']),
                              verified_IP_checkpoints=len(receipt['verified_IP_checkpoints']))))
        return 0
    if args.import_source:
        result = import_assets(args.import_source)
        print(json.dumps(dict(status=result['status'], imported=result['checked'], board_access=False)))
        return 0
    if args.check:
        status = dependencies()
        print(json.dumps(status, indent=2))
        return int(status['status'] != 'passed')
    directory = (args.work_dir or work_path('fpga/build_' + time.strftime('%Y%m%d_%H%M%S'))).resolve()
    project = stage(directory)
    if args.prepare_only:
        print(json.dumps(dict(status='staged', project=str(project), board_access=False)))
        return 0
    settings = load_settings()
    vivado = tool_executable('vivado')
    for name in ('procise', 'jfm'):
        if not settings['tools'][name] or not resolve_path('tool://' + name + '/').is_dir():
            raise ConfigurationError('Configure --' + name + ' before running the vendor build')
    env = environment()
    env.update(FINEDGE_PYTHON=sys.executable, FINEDGE_VIVADO=str(vivado),
               FMSH_PROCISE_PATH=str(resolve_path('tool://procise/')), JFM_PATH=str(resolve_path('tool://jfm/')))
    procise = resolve_path('tool://procise/')
    env['PATH'] = os.pathsep.join([str(procise / 'dll'), str(procise / 'bin'), env['PATH']])
    env.pop('TCL_LIBRARY', None)
    state = dict(status='running', project=str(project), board_access=False)
    atomic_json(directory / 'result.json', state)
    with ResourceLock(work_path('state/locks'), 'full_vivado'), (directory / 'build.log').open('xb') as log:
        result = subprocess.run([str(vivado), '-mode', 'batch', '-source', str(directory / 'build.tcl')],
                                cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
    complete = result.returncode == 0 and (directory / 'market_routed.dcp').is_file()
    state.update(status='build_finished_requires_validation' if complete else 'failed', return_code=result.returncode)
    atomic_json(directory / 'result.json', state)
    atomic_json(directory / 'execution_result.json', state)
    if not complete:
        raise RuntimeError('FPGA build failed. Inspect ' + str(directory / 'build.log'))
    if args.package_boot:
        subprocess.run([sys.executable, '-B', '-m', 'hardware.scripts.package_board_generated', '--build', str(directory),
                        '--cdc-recipe', str(directory / 'hardware/constraints/network_cdc.tcl')], env=env, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
