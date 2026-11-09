"""Explicit, frozen product data selection. Unknown/unavailable data never fall back."""
from importlib.metadata import version
from software.runtime.delivery import read_json, sha256
from shared.model_paths import ROOT

# Switched only after the step-2 data qualification decision is recorded.
DEFAULT_DATASET = 'databento20251215'
LABELS = {'itch20190730': '2019-07-30 · AAPL · 跨时期工程行情',
          'databento20251215': '2025-12-15 · AAPL · 开发验证日原始行情（当前重建）'}


def data_binding(dataset_id=None):
    dataset_id = DEFAULT_DATASET if dataset_id is None else dataset_id
    if dataset_id == 'databento20251215':
        from experiments.reference.runtime.databento_mbo import DatabentoMboSource, DATASET, PROFILE
        source = DatabentoMboSource()
        admission = source.admission
        files = {str(path.relative_to(ROOT)).replace('\\','/'): sha256(path)
                 for path in (source.path, source.status_path, DATASET/'admission.json')}
        return dict(dataset_id=dataset_id, label=LABELS[dataset_id], profile=PROFILE,
            date='2025-12-15', symbol='AAPL', role='development_validation_date_reconstructed',
            original_validation_alignment=False, files=files,
            expected_counts=dict(admission['expected_counts']),
            decoder_versions={name: version(name) for name in ('databento-dbn','zstandard','sortedcontainers')})
    if dataset_id == 'itch20190730':
        from experiments.reference.runtime.raw_itch import RawItchSource
        from experiments.reference.runtime.ps_stream_replay import EXPECTED_DAY
        source = RawItchSource()
        return dict(dataset_id=dataset_id, label=LABELS[dataset_id], profile=source.provenance['profile'],
            date='2019-07-30', symbol='AAPL', role='cross_period_engineering',
            original_validation_alignment=False,
            files={str(path.relative_to(ROOT)).replace('\\','/'):sha256(path) for path in
                (source.manifest_path,source.path/'aapl.itch',source.path/'positions.u64le.bin')},
            expected_counts=dict(EXPECTED_DAY), decoder_versions={})
    raise ValueError('Unknown explicit showcase dataset')


def validate_binding(binding):
    if not isinstance(binding,dict) or binding != data_binding(binding.get('dataset_id')):
        raise ValueError('Frozen showcase dataset/code/decoder admission changed')
    return binding


def open_source(binding):
    validate_binding(binding)
    if binding['dataset_id'] == 'databento20251215':
        from experiments.reference.runtime.databento_mbo import DatabentoMboSource
        return DatabentoMboSource()
    from experiments.reference.runtime.raw_itch import RawItchSource
    return RawItchSource()
