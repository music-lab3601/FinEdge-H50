"""Resolve, stage and build the H50 vendor-integrated FPGA project."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from shared.io import atomic_json
from shared.paths import ConfigurationError, ROOT, environment, load_settings, resolve_path, tool_executable, work_path
from tools.configuration import URI


def dependencies():
    rows = json.loads((ROOT / 'external/dependencies.json').read_text('utf-8'))['files']
    required = [row for row in rows if row['target'].startswith('vendor://') and
                not row['target'].startswith('vendor://source_reference/')]
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
    shutil.copytree(resolve_path('vendor://fpga'), directory / 'vendor/fpga')

    def replace(match):
        reference = match.group()
        if reference.startswith(('repo://hardware/', 'repo://third_party/')):
            return (directory / reference.removeprefix('repo://')).as_posix()
        if reference.startswith('vendor://fpga/'):
            return (directory / 'vendor' / reference.removeprefix('vendor://')).as_posix()
        if reference.startswith('work://'):
            return (directory / reference.removeprefix('work://')).as_posix()
        return resolve_path(reference).as_posix()

    for path in directory.rglob('*'):
        if path.is_file() and path.suffix in {'.sv', '.v', '.xdc', '.tcl', '.xpr', '.xci', '.xml', '.bd'}:
            try:
                text = path.read_text('utf-8-sig')
            except UnicodeError:
                continue
            path.write_text(URI.sub(replace, text), encoding='utf-8')
    project = directory / 'project/fpai_demo_vivado.xpr'
    project.parent.mkdir()
    document = ET.parse(directory / 'hardware/project/reference.xpr')
    document.getroot().set('Path', project.as_posix())
    for option in document.findall('.//Option'):
        if option.get('Name') == 'IPRepoPath':
            option.set('Val', (directory / 'vendor/fpga/PS_AI IP').as_posix())
    document.write(project, encoding='utf-8', xml_declaration=True)
    tcl = (directory / 'hardware/scripts/build.tcl').read_text('utf-8')
    tcl = tcl.replace('__PROJECT__', project.as_posix()).replace('__BUILD__', directory.as_posix())
    (directory / 'build.tcl').write_text(tcl, encoding='utf-8')
    atomic_json(directory / 'staging.json', dict(status='staged', dependency_check=status, board_access=False))
    return project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--package-boot', action='store_true')
    args = parser.parse_args()
    if args.check:
        status = dependencies()
        print(json.dumps(status, indent=2))
        return int(status['status'] != 'passed')
    directory = args.work_dir or work_path('fpga/build_' + time.strftime('%Y%m%d_%H%M%S'))
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
    state = dict(status='running', project=str(project), board_access=False)
    atomic_json(directory / 'result.json', state)
    with (directory / 'build.log').open('xb') as log:
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
