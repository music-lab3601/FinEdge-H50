"""Build the finite native market source and record the resulting executable."""
import argparse
import json
import subprocess

from shared.io import atomic_json, sha
from shared.paths import ROOT, environment, tool_executable, work_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generator', help='Optional CMake generator')
    args = parser.parse_args()
    cmake = tool_executable('cmake')
    output = work_path('native')
    output.mkdir(parents=True, exist_ok=True)
    command = [str(cmake), '-S', str(ROOT / 'host/market_source/native'), '-B', str(output / 'build')]
    if args.generator:
        command += ['-G', args.generator]
    for cmd in (command, [str(cmake), '--build', str(output / 'build'), '--config', 'Release']):
        subprocess.run(cmd, env=environment(), check=True)
    binaries = [p for p in (output / 'build').rglob('finedge_market_source*') if p.is_file()
                and p.name in ('finedge_market_source', 'finedge_market_source.exe')]
    if len(binaries) != 1:
        raise RuntimeError('Expected one native market-source executable')
    result = dict(status='built', executable=str(binaries[0]), sha256=sha(binaries[0]),
                  source_sha256=sha(ROOT / 'host/market_source/native/fair_source.cpp'), board_access=False)
    atomic_json(output / 'source_build.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
