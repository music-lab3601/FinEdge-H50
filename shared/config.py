"""Resolve one board-demo configuration from the repository and local settings."""
from shared.io import read
from shared.paths import ROOT, load_settings, resolve_path, work_path


def board_config():
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
    return config
