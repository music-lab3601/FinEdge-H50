"""Resolve source inputs used by the packaged FPGA build, without running tools."""
from pathlib import Path
import argparse
import glob
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / '01_工程源码/finedge_host_demo'
BUILD = PROJECT / 'versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358'
ETHERNET = PROJECT / 'versions/realtime_integration_v1/board_sources/vendor/verilog-ethernet'
INPUT_SUFFIXES = {'.tcl', '.sv', '.v', '.vh', '.svh', '.hex', '.xpr', '.xci', '.edif', '.xdc'}
ABSOLUTE_INPUT = re.compile(r'''[A-Za-z]:[/\\][^{}"'\r\n]+''')


def audit(configured=False):
    errors = []
    checked = set()

    def check(reference, origin):
        reference = reference.replace('\\\\', '/').replace('\\', '/')
        # Source suffixes distinguish input dependencies from report outputs.
        if Path(reference).suffix.lower() not in INPUT_SUFFIXES or '$' in reference:
            return
        marker = '/finedge_host_demo/'
        if marker not in reference:
            return
        suffix = reference.rsplit(marker, 1)[1]
        target = PROJECT / suffix
        if configured and reference != target.as_posix():
            errors.append('Foreign or repeated project path in ' + str(origin) + ': ' + reference)
        if not target.resolve().is_relative_to(PROJECT.resolve()):
            errors.append('Input escapes project: ' + reference)
        elif not glob.glob(str(target)):
            errors.append('Missing build input in ' + str(origin) + ': ' + suffix)
        checked.add(suffix)

    for path in BUILD.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix.lower() in {'.tcl', '.sv', '.v', '.vh', '.svh'}:
            try:
                text = path.read_text(encoding='utf-8')
            except UnicodeError:
                continue  # Encrypted vendor netlists are resolved through XPR.
            for line in text.splitlines():
                if line.lstrip().startswith(('//', '#', '*')):
                    continue
                for reference in ABSOLUTE_INPUT.findall(line):
                    check(reference, path.relative_to(BUILD))
    recipe = (BUILD / 'network_cdc_signoff_mcp_v5.tcl').read_text(encoding='utf-8')
    for relative in re.findall(r'\$eth/([^\s{}]+\.tcl)', recipe):
        path = ETHERNET / relative
        checked.add(path.relative_to(PROJECT).as_posix())
        if not path.is_file():
            errors.append('Missing Ethernet CDC Tcl: ' + relative)
    xpr = BUILD / 'project/fpai_demo_vivado.xpr'
    document = ET.parse(xpr)
    for item in document.findall('.//File'):
        reference = item.get('Path', '').replace('$PPRDIR', str(xpr.parent))
        reference = reference.replace('$PSRCDIR', str(xpr.parent / 'fpai_demo_vivado.srcs'))
        if '$' not in reference and not Path(reference).is_file():
            errors.append('Missing XPR file: ' + reference)
        checked.add(reference)
    for relative in ('hardware/wujing_fpai/vendor/BOOT_Gen/FSBL251210_ddr400_demo.out',
                     'hardware/wujing_fpai/vendor/BOOT_Gen/bl31.elf',
                     'hardware/wujing_fpai/vendor/BOOT_Gen/u-boot',
                     'artifacts/fpga_reference_20261021/boot_candidate_UNVERIFIED_20261021/manifest.json'):
        if not (PROJECT / relative).is_file():
            errors.append('Missing BOOT input: ' + relative)
        checked.add(relative)
    return dict(status='passed' if not errors else 'failed', checked_inputs=len(checked),
                board_access=False, full_source_build=False, errors=errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configured', action='store_true')
    args = parser.parse_args()
    result = audit(args.configured)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(result['errors']))
