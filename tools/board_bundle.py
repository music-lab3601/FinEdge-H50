"""Prepare the standalone Python 3.8/ARM runtime without workstation packages."""
import os
import ast
from pathlib import Path
import shutil

from shared.io import read, sha
from shared.paths import ConfigurationError, ROOT, load_settings, resolve_path, work_path


def preflight(config):
    startup = Path(config.get('startup') or resolve_path(load_settings()['inputs']['startup']))
    if config.get('demo_profile'):
        from tools.synthetic_inputs import load_profile
        load_profile(config['demo_profile'])
        if sha(config['demo_profile']) != config['demo_profile_sha256']:
            raise ConfigurationError('Synthetic demo profile changed after configuration')
    paths = [Path(config[key]) for key in ('raw_source', 'reference', 'duration_data', 'hardware_release')]
    paths += [startup / name for name in ('golden_input.i8', 'golden_logits.f32',
                                         'dual_slot_input.i8', 'dual_slot_logits.f32')]
    paths += [work_path('native/source_build.json')]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ConfigurationError('Board demo inputs missing. Configure --replay-raw, --replay-rows, '
                                 '--replay-ranges, --startup and --firmware-release. Build native source first.\n'
                                 + '\n'.join(missing))
    if not os.environ.get('FINEDGE_BOARD_PASSWORD'):
        raise ConfigurationError('Set FINEDGE_BOARD_PASSWORD in this terminal for the PS console login')
    release = read(config['hardware_release'])
    if release['boot_sha256'] != config['BOOT_sha256']:
        raise ConfigurationError('Configured firmware identity differs from this H50 demo')
    for key in ('raw_source', 'reference'):
        if sha(config[key]) != config[key + '_sha256']:
            raise ConfigurationError('Replay identity differs. ' + key)


def prepare_bundle(directory, config):
    directory.mkdir()
    startup = Path(config.get('startup') or resolve_path(load_settings()['inputs']['startup']))
    source = read(ROOT / 'models/npu/bundle_config.json')
    expected = dict(source['files'])
    if config.get('demo_profile'):
        from tools.synthetic_inputs import load_profile
        load_profile(config['demo_profile'])
        if sha(config['demo_profile']) != config['demo_profile_sha256']:
            raise ValueError('Synthetic profile changed before bundle preparation')
        expected.update(config['startup_files'])
    files = {'model.json': ROOT / 'models/npu/model.json',
             'model.raw': ROOT / 'models/npu/model.raw',
             'configure_phy.py': ROOT / 'board/management/configure_phy.py',
             'd_npu_native.c': ROOT / 'board/native/d_npu_native.c',
             **{name: startup / name for name in ('golden_input.i8', 'golden_logits.f32')}}
    for name, path in files.items():
        if name in ('model.json', 'model.raw', 'golden_input.i8', 'golden_logits.f32'):
            if sha(path) != expected[name]:
                raise ValueError('Board bundle input identity differs. ' + name)
        shutil.copyfile(path, directory / name)
    return {key: value for key, value in source.items() if key != 'files'}


def prepare_integrated_bundle(directory, config):
    """Assemble the same standalone ARM runtime used by the board runner."""
    base = prepare_bundle(directory, config)
    if config.get('demo_profile'):
        base.update(dataset=config['dataset'], startup_reference_backend='Icraft Host', board_validation='deferred')
    for name in ('integrated_owner.py', 'startup_input.py'):
        shutil.copyfile(ROOT / 'board/runtime' / name, directory / name)
    shutil.copyfile(ROOT / 'board/native/integrated_native.c', directory / 'integrated_native.c')
    startup = Path(config.get('startup') or resolve_path(load_settings()['inputs']['startup']))
    for name in ('dual_slot_input.i8', 'dual_slot_logits.f32'):
        path = startup / name
        if config.get('startup_files') and sha(path) != config['startup_files'][name]:
            raise ValueError('Synthetic dual-slot startup input changed')
        shutil.copyfile(path, directory / name)
    for path in directory.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
    base['files'] = {path.name: sha(path) for path in directory.iterdir() if path.is_file()}
    from shared.io import atomic_json
    atomic_json(directory / 'config.json', base)
    return base
