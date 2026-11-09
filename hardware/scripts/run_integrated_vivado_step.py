"""Run a generated Vivado step while its parent project is closed.

The normal generated WSH wrapper did not start a child on this Windows host.
This explicit launcher records the actual PID, exit and completion marker.
The G full-build supervisor owns the resource lock for the entire invocation.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

from shared.io import atomic_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--directory', type=Path, required=True)
    ap.add_argument('--step', choices=['synth_1', 'impl_1'], required=True)
    a = ap.parse_args()
    out = a.directory.resolve()
    if not (out / 'staging.json').is_file():
        raise ValueError('Not a staged FinEdge build directory')
    run = out / 'project/fpai_demo_vivado.runs' / a.step
    if (run / 'direct_run_result.json').exists():
        raise ValueError('Step is one use; prepare a new build after failure')
    script = run / 'ai7030_edif_top.tcl'
    if not script.is_file():
        raise ValueError('Generated step script is missing')
    original_script = script
    script = run / 'G_explicit_step.tcl'
    vivado = Path(os.environ['FINEDGE_VIVADO'])
    tool_scripts = vivado.parent.parent / 'scripts'
    preflight = 'set_param general.maxThreads 2\n'
    # Vivado 2018.3 on this Windows host can report an existing Tcl hook as
    # missing after checkpoint loading. Read the staged scripts before design
    # initialization, as well as the tool retarget scripts used below.
    input_scripts = sorted({path for folder in (
        out / 'project/ip_patch', out / 'hardware/scripts', out / 'hardware/constraints',
        out / 'third_party/verilog-ethernet') for path in folder.rglob('*.tcl')})
    for path in input_scripts:
        preflight += 'set f [open {' + path.as_posix() + '} r]; puts "G_INPUT_PREFLIGHT ' + path.name + ' [string length [read $f]]"; close $f\n'
    for relative in ('rt/data/retarget/retarget_vhdl.tcl', 'rt/data/retarget/retarget_verilog.tcl', 'rt/data/unimacro/unimacro_vhdl.tcl', 'rt/data/unimacro/unimacro_verilog.tcl'):
        path = tool_scripts / relative
        if path.is_file():
            preflight += 'set f [open {' + path.as_posix() + '} r]; puts "G_TOOL_PREFLIGHT [string length [read $f]]"; close $f\n'
    script.write_text(preflight + 'source {' + original_script.as_posix() + '}\n', encoding='utf-8')
    command = [str(vivado), '-mode', 'batch',
               '-source', script.name, '-log', 'direct_vivado.log', '-journal', 'direct_vivado.jou']
    state = dict(status='starting', step=a.step, command=command, parent_pid=os.getpid(),
                 parent_project_closed=True, board_access=False)
    atomic_json(run / 'direct_run_result.json', state)
    started = time.monotonic_ns()
    with (run / 'direct_console.txt').open('xb') as log:
        child = subprocess.Popen(command, cwd=run, stdout=log, stderr=subprocess.STDOUT)
        state.update(status='running', child_pid=child.pid)
        atomic_json(run / 'direct_run_result.json', state)
        document = ET.Element('ProcessHandle', Version='1', Minor='0')
        ET.SubElement(document, 'Process', Command='vivado.bat', Owner=os.environ.get('USERNAME', ''),
                      Host=os.environ.get('COMPUTERNAME', ''), Pid=str(child.pid))
        ET.ElementTree(document).write(run / '.vivado.begin.rst', encoding='UTF-8', xml_declaration=True)
        code = child.wait()
    expected = ([run / '__synthesis_is_complete__', run / 'ai7030_edif_top.dcp']
                if a.step == 'synth_1' else [run / '.write_bitstream.end.rst', run / 'ai7030_edif_top.bit'])
    complete = code == 0 and all(p.exists() for p in expected)
    (run / ('.vivado.end.rst' if complete else '.vivado.error.rst')).touch(exist_ok=False)
    state.update(status='completed' if complete else 'failed', return_code=code,
                 completion_markers={p.name: p.exists() for p in expected}, elapsed_ns=time.monotonic_ns()-started)
    atomic_json(run / 'direct_run_result.json', state)
    print(json.dumps(state), flush=True)
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
