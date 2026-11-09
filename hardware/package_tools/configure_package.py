"""Configure this extraction in one pass, without opening a board connection."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {'.py', '.tcl', '.xpr', '.json', '.xci', '.xml', '.bd',
                 '.bif', '.sv', '.v', '.vh', '.svh', '.xdc'}
# Consume previous, even repeatedly expanded, roots. Substitution is single-pass.
PROJECT_PREFIX = re.compile(
    r'''[A-Za-z]:[\\/]+[^"'\r\n{}<>]*[\\/]finedge_host_demo(?=[\\/])''')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relocate_text(text, project_root):
    return PROJECT_PREFIX.sub(lambda match: project_root.as_posix(), text)


def relocate_project(project):
    changed = 0
    for path in project.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES or path.name == 'model.json':
            continue
        try:
            original = path.read_text(encoding='utf-8')
        except UnicodeError:
            continue
        rewritten = relocate_text(original, project)
        if rewritten != original:
            path.write_text(rewritten, encoding='utf-8')
            changed += 1
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    for name in ('vivado', 'procise', 'jfm', 'raw', 'display', 'ranges', 'selftests'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    bundled = ROOT / '04_运行数据'
    if (bundled / 'manifest.json').is_file():
        args.raw = args.raw or bundled / 'market/raw_mbo56_prefix.bin'
        args.display = args.display or bundled / 'display/rows.bin'
        args.ranges = args.ranges or bundled / 'ranges.json'
        args.selftests = args.selftests or bundled / 'startup'
    paths = read(ROOT / '03_检查与配置/PATH_MAP.json')
    project = ROOT / paths['package_project_root']
    integration = project / 'versions/realtime_integration_v1'
    showcase = project / paths['showcase']
    build = project / paths['build']
    model = project / paths['model_bundle']
    statefile = ROOT / '03_检查与配置/local_configuration.json'
    previous = read(statefile) if statefile.exists() else {}
    tools = {'python': args.python.resolve().as_posix()}
    for name in ('vivado', 'procise', 'jfm'):
        value = getattr(args, name)
        tools[name] = value.resolve().as_posix() if value else previous.get(name, '')
        if value and not value.exists():
            raise FileNotFoundError(str(value))
    for value in (args.raw, args.display, args.ranges):
        if value and not value.is_file():
            raise FileNotFoundError(str(value))
    changed = relocate_project(project)
    config = read(showcase / 'config.json')
    config.update(hardware_release=(showcase / 'firmware/release.json').as_posix(), bundle=(model).as_posix(),
                  candidate_manifest=(project / 'versions/window_stream_v1/candidate_manifest.json').as_posix(),
                  rollback_hardware_release=(showcase / 'firmware/release.json').as_posix(),
                  rollback_BOOT_sha256=config['BOOT_sha256'],
                  seed_run=(showcase / 'artifacts/no_bundled_records').as_posix(),
                  initial_record=(showcase / 'artifacts/no_bundled_records').as_posix())
    for value, key in ((args.raw, 'raw_source'), (args.display, 'reference'), (args.ranges, 'duration_data')):
        if value:
            config[key] = value.resolve().as_posix()
    for value, key in ((args.raw, 'raw_source_sha256'), (args.display, 'reference_sha256')):
        if value:
            config[key] = sha(value.resolve())
    save(showcase / 'config.json', config)
    if args.selftests:
        sizes = {'golden_input.i8': 4800, 'golden_logits.f32': 12,
                 'dual_slot_input.i8': 9600, 'dual_slot_logits.f32': 24}
        for name, size in sizes.items():
            path = args.selftests.resolve() / name
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError('Missing/invalid startup fixture: ' + str(path))
        for name in sizes:
            target = model / name if name.startswith('golden_') else integration / 'queued_v2/reference' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.selftests.resolve() / name, target)
        model_config = read(model / 'config.json')
        model_config['files'] = {path.name: sha(path) for path in sorted(model.iterdir())
                                 if path.is_file() and path.name != 'config.json'}
        save(model / 'config.json', model_config)
    loaded = integration / 'state/loaded_firmware.json'
    if not loaded.exists():
        save(loaded, dict(status='unobserved_submission_bootstrap', boot_id='UNOBSERVED',
                          boot_sha256=config['BOOT_sha256'], build=(build).as_posix(),
                          release=(showcase / 'firmware/release.json').as_posix(), physical_cold_start=False))
    (showcase / 'artifacts').mkdir(exist_ok=True)
    save(statefile, dict(project_root=project.as_posix(), **tools, board_access=False))
    print(json.dumps(dict(status='paths_configured', changed_source_files=changed,
                          board_access=False, project_root=str(project),
                          external_inputs_complete=all(Path(config[key]).is_file()
                                                      for key in ('raw_source', 'reference', 'duration_data'))),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
