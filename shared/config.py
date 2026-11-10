"""Resolve one board-demo configuration from the repository and local settings."""
from shared.io import read
from shared.paths import ROOT, load_settings, resolve_path, work_path


def board_config(profile=None):
    config = read(ROOT / 'config/board_demo.json')
    settings = load_settings()
    inputs = settings['inputs']
    config.update(settings['board'])
    config.update(raw_source=str(resolve_path(inputs['replay_raw'])),
                  reference=str(resolve_path(inputs['replay_rows'])),
                  duration_data=str(resolve_path(inputs['replay_ranges'])),
                  bundle=str(ROOT / 'models/npu'),
                  hardware_release=str(resolve_path(inputs['firmware_release'])),
                  rollback_hardware_release=str(resolve_path(inputs['firmware_release'])),
                  candidate_manifest=str(ROOT / 'contracts/release/h50_candidate.json'),
                  seed_run=str(work_path('runs/empty')), initial_record=str(work_path('runs/empty')))
    config['startup'] = str(resolve_path(inputs['startup']))
    config['dataset'] = dict(data_kind='historical', symbol='AAPL', date='2025-12-15', label='历史市场回放')
    selected = profile or inputs.get('demo_profile')
    if selected:
        from tools.synthetic_inputs import load_profile, STARTUP
        from shared.io import sha
        profile_path = resolve_path(selected)
        manifest, files = load_profile(profile_path)
        config.update(raw_source=str(files['raw_source']), reference=str(files['reference']),
                      duration_data=str(files['duration_data']), startup=str(files['golden_input'].parent),
                      raw_source_sha256=manifest['files']['raw_source']['sha256'],
                      reference_sha256=manifest['files']['reference']['sha256'],
                      demo_start_ordinal=21, demo_profile=str(profile_path), demo_profile_sha256=sha(profile_path),
                      startup_files={filename: manifest['files'][key]['sha256'] for key, filename in STARTUP.items()},
                      dataset=dict(data_kind='synthetic', symbol='SYNTH', date=manifest['date'], label='公开合成订单'))
    release = resolve_path(inputs['firmware_release'])
    if release.is_file():
        config['BOOT_sha256'] = read(release)['boot_sha256']
        if config['BOOT_sha256'] != read(ROOT / 'config/board_demo.json')['BOOT_sha256']:
            config['candidate_manifest'] = str(release)
    rollback = resolve_path(inputs.get('rollback_firmware_release') or inputs['firmware_release'])
    config['rollback_hardware_release'] = str(rollback)
    if rollback.is_file():
        config['rollback_BOOT_sha256'] = read(rollback)['boot_sha256']
    return config
