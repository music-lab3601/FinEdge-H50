"""Build the checked-in Icraft C++ utilities into the configured work tree."""
import json
import subprocess

from shared.io import atomic_json, sha
from shared.paths import ConfigurationError, ROOT, environment, resolve_path, tool_executable, work_path


def build_stage(stage):
    if stage not in ('base', 'mixed', 'native'):
        raise ValueError('Unknown Icraft helper stage')
    sdk = resolve_path('tool://icraft/')
    if not sdk.is_dir():
        raise ConfigurationError('Configure --icraft before building the SDK utilities')
    cmake = tool_executable('cmake')
    source = ROOT / 'deployment/icraft' / stage / 'cpp'
    output = work_path('model_build/' + stage + '/build')
    output.mkdir(parents=True, exist_ok=True)
    (output.parent/'.tmp').mkdir(exist_ok=True)
    commands = [[str(cmake), '-S', str(source), '-B', str(output), '-DCMAKE_PREFIX_PATH=' + str(sdk)],
                [str(cmake), '--build', str(output), '--config', 'Release', '--parallel', '1']]
    with (output / 'build.log').open('wb') as log:
        for command in commands:
            result = subprocess.run(command, env=environment(), stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('SDK utility build failed. ' + str(output / 'build.log'))
    report = dict(status='built', stage=stage,
                  source_sha256={p.name: sha(p) for p in source.iterdir() if p.is_file()}, board_access=False)
    atomic_json(output / 'result.json', report)
    print(json.dumps(report, indent=2))
    return report
