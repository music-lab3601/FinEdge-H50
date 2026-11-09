"""Run the four shipped synthetic RTL benches with Vivado XSim."""
import argparse
from pathlib import Path
import subprocess
import time

from shared.io import atomic_json
from shared.paths import ROOT, environment, tool_executable, work_path


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    bin_path = tool_executable('vivado').parent
    suffix = '.bat' if (bin_path / 'xvlog.bat').exists() else ''
    output = work_path('simulation/' + time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / ('hardware/rtl/' + name) for name in (
        'common/e_reset_sync.sv', 'common/b1_mailbox.sv', 'common/wt_async_fifo.sv',
        'window/wt_history.sv', 'window/tt_native_packer.sv', 'memory/g_payload_bridge.sv')]
    markers = dict(fifo='WT_FIFO_PASS', window_bridge='WT_BRIDGE_PASS', history='WT_HISTORY_PASS',
                   window_bridge_earliest='WT_VERIFY_MCP_PASS response_delay=0')
    results = []
    for name, marker in markers.items():
        bench = ROOT / ('tests/rtl/tb_' + name + '.sv')
        inputs = sources + ([ROOT / 'tests/rtl/tb_window_bridge.sv'] if name.endswith('earliest') else []) + [bench]
        commands = [[str(bin_path / ('xvlog' + suffix)), '-sv', '-i', str(ROOT / 'hardware/rom'), *map(str, inputs)],
                    [str(bin_path / ('xelab' + suffix)), '-mt', '2', 'tb_' + name, '-s', 'run_' + name],
                    [str(bin_path / ('xsim' + suffix)), 'run_' + name, '-runall', '-log', name + '.log']]
        with (output / (name + '_console.log')).open('wb') as log:
            for command in commands:
                subprocess.run(command, cwd=output, env=environment(), stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=240)
        if marker not in (output / (name + '.log')).read_text('utf-8', errors='replace'):
            raise RuntimeError('RTL pass marker missing. ' + name)
        results.append(dict(bench=name, status='passed'))
        print(name + ' passed', flush=True)
    atomic_json(output / 'result.json', dict(status='passed', tests=results, board_access=False))
    print('RTL logs ' + str(output))


if __name__ == '__main__':
    main()
