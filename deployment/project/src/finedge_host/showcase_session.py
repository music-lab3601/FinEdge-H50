"""Freeze product entry points alongside the unchanged admitted board bundle."""
from pathlib import Path
from .delivery import read_json, sha256, write_json
from .paths import ROOT
from . import ps_stream_session as base
from .showcase_data import data_binding, validate_binding
from .ps_stream_platform import DATA_BINDING

SOURCES = ('src/finedge_host/showcase_replay.py', 'src/finedge_host/showcase_gui.py',
           'src/finedge_host/showcase_session.py', 'scripts/launch_showcase.py',
           'scripts/launch_fpga_stream.py', 'src/finedge_host/ui_widgets.py',
           'src/finedge_host/databento_mbo.py', 'src/finedge_host/showcase_data.py',
           'requirements-lock.txt',
           '启动作品演示.cmd')
MODE = 'fpga_windows_board_npu_only'


def prepare_showcase(directory, *, dataset_id=None):
    data = data_binding(dataset_id)
    scope = dict(dataset_binding=dict(DATA_BINDING)) if data['dataset_id']=='databento20251215' else {}
    bundle = base.prepare(directory, **scope)
    snapshots = bundle.parent/'showcase_sources'
    snapshots.mkdir()
    hashes = {}
    for name in SOURCES:
        target = snapshots/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
        hashes[name] = sha256(target)
    write_json(bundle.parent/'showcase_contract.json', dict(schema=2, mode=MODE,
        bundle_sha256=sha256(bundle), source_hashes=hashes, cpu_reference_enabled=False,
        host_simulation_enabled=False, data_source=data))
    validate_showcase_bundle(bundle)
    return bundle


def validate_showcase_bundle(bundle):
    bundle = Path(bundle).resolve()
    base.validate_bundle(bundle)
    contract = read_json(bundle.parent/'showcase_contract.json')
    if (contract.get('schema') != 2 or contract.get('mode') != MODE
            or contract.get('bundle_sha256') != sha256(bundle)
            or contract.get('cpu_reference_enabled') is not False
            or contract.get('host_simulation_enabled') is not False
            or set(contract.get('source_hashes', {})) != set(SOURCES)):
        raise ValueError('Showcase contract differs')
    data = validate_binding(contract.get('data_source'))
    cfg = read_json(bundle)
    if data['dataset_id']=='databento20251215' and (
            cfg.get('dataset_binding')!=DATA_BINDING or cfg.get('max_windows')!=data['expected_counts']['windows']):
        raise ValueError('Dataset source and service window budget differ')
    for name, digest in contract['source_hashes'].items():
        if sha256(ROOT/name) != digest or sha256(bundle.parent/'showcase_sources'/name) != digest:
            raise ValueError('Showcase frozen source changed: '+name)
    return contract


def deploy_showcase(bundle, confirmation):
    bundle = Path(bundle).resolve()
    validate_showcase_bundle(bundle)
    session = base.deploy(bundle, confirmation)
    cfg = read_json(session)
    write_json(bundle.parent/'showcase_session_binding.json', dict(mode=MODE,
        session_sha256=sha256(session), bundle_sha256=sha256(bundle), boot_id=cfg['boot_id'],
        contract_sha256=sha256(bundle.parent/'showcase_contract.json')))
    return session


def validate_showcase_session(session):
    session = Path(session).resolve()
    validate_showcase_bundle(session.parent/'bundle.json')
    cfg = base.validate_session(session)
    binding = read_json(session.parent/'showcase_session_binding.json')
    expected = dict(mode=MODE, session_sha256=sha256(session),
        bundle_sha256=sha256(session.parent/'bundle.json'), boot_id=cfg['boot_id'],
        contract_sha256=sha256(session.parent/'showcase_contract.json'))
    if binding != expected: raise ValueError('Showcase session binding changed')
    return binding


def showcase_client(session, log_path=None):
    validate_showcase_session(session)
    return base.ManagedStreamClient(session, log_path)
