"""Stage a portable FPGA tree and run a fresh build with selected tools."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from configure_package import relocate_project

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def stage_project(source, workspace):
    workspace = workspace.resolve()
    if not workspace.as_posix().isascii() or any(character.isspace() for character in workspace.as_posix()):
        raise ValueError('Vivado 2018.3 requires an ASCII build path without spaces, for example @EXTERNAL_ROOT@/H50_01')
    if workspace.exists():
        raise FileExistsError('Select a new empty build path: ' + str(workspace))
    project = workspace / 'finedge_host_demo'
    shutil.copytree(source, project)
    relocate_project(project)
    save(workspace / 'source_origin.json', dict(source=str(source), staged_project=str(project), board_access=False))
    return project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-boot', action='store_true')
    parser.add_argument('--work-dir', type=Path, required=True,
                        help='New ASCII build directory without spaces')
    parser.add_argument('--prepare-only', action='store_true', help='Stage source without launching Vivado')
    args = parser.parse_args()
    paths = read(ROOT / '03_检查与配置/PATH_MAP.json')
    configuration = read(ROOT / '03_检查与配置/local_configuration.json')
    if not args.prepare_only:
        for name in ('python', 'vivado', 'procise', 'jfm'):
            if not configuration.get(name) or not Path(configuration[name]).exists():
                raise ValueError('Configure --python, --vivado, --procise and --jfm first')
    project = stage_project(ROOT / paths['package_project_root'], args.work_dir)
    integration = project / 'versions/realtime_integration_v1'
    build = project / paths['build']
    if args.prepare_only:
        print(json.dumps(dict(status='source_staged', project=str(project), board_access=False)))
        return
    environment = dict(os.environ, JFM_PATH=configuration['jfm'],
                       FMSH_PROCISE_PATH=configuration['procise'], FINEDGE_PYTHON=configuration['python'],
                       FINEDGE_VIVADO=configuration['vivado'], PYTHONIOENCODING='utf-8')
    environment['PATH'] = (str(Path(configuration['procise']) / 'dll') + os.pathsep +
                           str(Path(configuration['procise']) / 'bin') + os.pathsep + environment['PATH'])
    sources = {path.relative_to(project).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (build / 'project/rtl/bench').rglob('*') if path.is_file()}
    state = dict(status='running', project=str(build / 'project/fpai_demo_vivado.xpr'),
                 started_ns=time.time_ns(), source_sha256=sources, board_access=False)
    save(build / 'fresh_source_build.json', state)
    with (build / 'fresh_source_build.log').open('xb') as log:
        child = subprocess.run([configuration['vivado'], '-mode', 'batch', '-source', str(build / 'build.tcl')],
                               cwd=build, env=environment, stdout=log, stderr=subprocess.STDOUT)
    text = (build / 'fresh_source_build.log').read_text(encoding='utf-8', errors='replace')
    complete = (child.returncode == 0 and 'MARKET_BUILD_DONE_REQUIRES_VALIDATION' in text
                and (build / 'market_routed.dcp').is_file())
    state.update(status='build_finished_requires_validation' if complete else 'failed',
                 return_code=child.returncode, ended_ns=time.time_ns())
    save(build / 'fresh_source_build.json', state)
    if not complete:
        raise RuntimeError('FPGA source build failed. Inspect ' + str(build / 'fresh_source_build.log'))
    save(build / 'result.json', state)
    save(build / 'execution_result.json', dict(status=state['status'], return_code=0))
    if args.package_boot:
        subprocess.run([configuration['python'], '-B', str(integration / 'scripts/package_board_generated.py'),
                        '--build', str(build), '--cdc-recipe', str(build / 'network_cdc_signoff_mcp_v5.tcl')],
                       env=environment, cwd=build, check=True)
    print(json.dumps(dict(status=state['status'], board_access=False, BOOT_packaged=args.package_boot,
                          build_directory=str(build))))


if __name__ == '__main__':
    main()
