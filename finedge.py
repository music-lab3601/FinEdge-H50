"""Single entry point for the public H50 source tree."""
import argparse
import json
import subprocess
import sys

from shared.paths import ConfigurationError, ROOT, environment, load_settings, save_configuration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    groups = parser.add_subparsers(dest='command', required=True)
    configure = groups.add_parser('configure', help='Save machine settings in ignored .local/config.json')
    for section, keys in load_settings().items():
        for key, value in keys.items():
            configure.add_argument('--' + key.replace('_', '-'), dest=section + '.' + key,
                                   type=int if type(value) is int else str)
    configure.add_argument('--render', action='store_true', help='Resolve all 57 migrated templates')
    check = groups.add_parser('check', help='Check source collection or run local tests')
    check.add_argument('--git', help='Git executable when absent from PATH')
    check.add_argument('scope', choices=['checkout', 'config', 'tests', 'imports'], default='config', nargs='?')
    cpu = groups.add_parser('cpu', help='Frozen CPU model checks and network smoke loop')
    cpu.add_argument('action', choices=['check', 'smoke'], default='check', nargs='?')
    cpu.add_argument('--records', type=int, default=160)
    demo = groups.add_parser('demo', help='Launch the local dashboard')
    demo.add_argument('--review', help='Directory of a saved board run')
    demo.add_argument('--no-browser', action='store_true')
    for name, description in [('data', 'ITCH reconstruction'), ('train', 'Registered H50 training'),
                              ('native', 'Build the native network source'),
                              ('simulate', 'Run four RTL simulations'),
                              ('example', 'Generate and verify public synthetic inputs'),
                              ('fpga', 'Materialize or build the vendor FPGA project'),
                              ('model', 'Export ONNX, transform or compile')]:
        p = groups.add_parser(name, help=description)
        p.add_argument('arguments', nargs=argparse.REMAINDER)
    raw = list(sys.argv[1:] if argv is None else argv)
    modules = {'data': 'software.data.w2', 'train': 'software.training.tcn_existing_data_ed4_train',
               'native': 'tools.build_native', 'simulate': 'tools.simulate',
               'fpga': 'hardware.scripts.build_fpga', 'model': 'deployment.commands',
               'example': 'tools.synthetic_inputs'}
    if raw and raw[0] in modules:
        return subprocess.call([sys.executable, '-B', '-m', modules[raw[0]], *raw[1:]],
                               cwd=ROOT, env=environment())
    args = parser.parse_args(raw)
    try:
        if args.command == 'configure':
            changes = {}
            for name, value in vars(args).items():
                if '.' in name and value is not None:
                    section, key = name.split('.')
                    changes.setdefault(section, {})[key] = value
            if changes:
                save_configuration(changes)
            from tools.configuration import render_templates
            print(json.dumps(render_templates() if args.render else load_settings(), indent=2))
            return 0
        if args.command == 'check':
            if args.scope == 'config':
                from tools.configuration import check_configuration
                result = check_configuration()
                print(json.dumps(result, indent=2))
                return int(result['status'] != 'passed')
            if args.scope == 'imports':
                from tools.configuration import check_imports
                result = check_imports()
                print(json.dumps(result, indent=2))
                return int(result['status'] != 'passed')
            if args.scope == 'tests':
                return subprocess.call([sys.executable, '-B', '-m', 'unittest', 'discover',
                                        '-s', 'tests', '-t', '.'], cwd=ROOT, env=environment())
            from tools.verify_checkout import main as verify
            return verify(['--git',args.git] if args.git else [])
        if args.command == 'cpu':
            if args.action == 'check':
                from tools.check_cpu_model import main as check_model
                return check_model()
            from software.runtime.network_smoke import run
            print(json.dumps(run(args.records), indent=2))
            return 0
        if args.command == 'demo':
            from host.dashboard.app import main as dashboard
            arguments = (['--review', args.review] if args.review else [])
            if args.no_browser:
                arguments.append('--no-browser')
            return dashboard(arguments)
    except (ConfigurationError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print('FinEdge error. ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
