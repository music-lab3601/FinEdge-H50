"""Review-only TCN01 model; no training or dataset access entry point.

Independent implementation of the causal dilated residual-convolution principle
in Bai, Kolter and Koltun (2018), https://arxiv.org/abs/1803.01271.
The author's reference repository, https://github.com/locuslab/TCN, is MIT licensed.
No reference source is copied here. This draft omits its weight normalization and
dropout, retains the project's 40 x 100 contract, and emits three class logits.
Structural checks do not establish predictive quality or NPU compatibility.

Causal implementation 1.1 expands each dilated k3 operation into the sum of
three lagged pointwise convolutions. This uses exactly the same parameters and
causal lag positions; it avoids the slow dilated CUDA kernel observed on this
machine with the existing deterministic FP32 settings. Native NPU compilation
of these padding/slice/pointwise/add operations remains unverified.
"""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'config/research/tcn_candidate_draft.json'
IMPLEMENTATION_VERSION = 'causal-lagwise-pointwise-1.1'
MODEL_SPEC = {
    'version': 'temporal-tcn-candidate-1.0',
    'feature_count': 40,
    'window_length': 100,
    'hidden_channels': 32,
    'blocks': 5,
    'convolutions_per_block': 2,
    'kernel_size': 3,
    'dilations': [1, 2, 4, 8, 16],
    'stride': 1,
    'padding': 'causal_left_zero',
    'activation': 'ReLU',
    'residual_projection': '1x1_when_channels_differ',
    'readout': 'last_time',
    'classes': 3,
    'dropout': 0,
    'weight_norm': False,
    'bias': True,
    'initialization': 'pytorch_default',
}
DATA_CONTRACT = {
    'protocol_version': 'W3-data-protocol-1.0',
    'protocol_path': 'configs/w3_protocol_frozen_v1.json',
    'symbol': 'AAPL',
    'train_dates': ['2025-12-08', '2025-12-09'],
    'tuning_date': '2025-12-10',
    'feature_count': 40,
    'window_length': 100,
    'feature_order': 'ask_price,ask_size,bid_price,bid_size for each level 1..10',
    'sampling': 'every target visible-book change, including unchanged top ten',
    'label': 'future endpoint mid-price relative to current mid-price; no smoothing',
    'horizon': 50,
    'threshold': 0.0,
    'class_mapping': {'down': 0, 'stationary': 1, 'up': 2},
    'normalization': 'existing per-feature mean/std fitted on training records only; zero variance std=1',
    'cross_date_or_segment_windows': False,
    'additional_factors': False,
    'adaptive_input_normalization': False,
}
PRESERVED_SETTINGS = {
    'seed': 42,
    'optimizer': 'AdamW',
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'dropout': 0,
    'loss': 'unweighted_cross_entropy',
    'train_batch_size': 256,
    'evaluation_batch_size': 256,
    'deployment_batch_size': 1,
    'max_epochs': 10,
    'early_stop_patience': 3,
    'device': 'cuda',
    'precision': 'float32',
    'shuffle_train_windows': True,
    'drop_last': False,
    'formal_training_uses_all_training_windows': True,
    'validation_split': 'tuning_validation',
    'forbidden_splits': ['comparison_validation', 'final_test'],
    'selection_rules_path': 'configs/w4_selection_rules.json',
    'selection_rules_version': 'W4-selection-1.0',
}


def validate_config(config):
    """Accept only this review draft; an old approval cannot authorize TCN01.

    This checks declarations, not the current manifest or source-file hashes.
    A separate preflight is required before any subsequently authorized training.
    Device is a future training declaration; build_model leaves the model on CPU.
    """
    if not isinstance(config, dict):
        raise ValueError('TCN candidate configuration must be a dictionary')
    if config.get('version') != 'TCN01-candidate-draft-1.0' or config.get('experiment') != 'TCN01':
        raise ValueError('Unsupported candidate identity; old W4/W5 approval is not reusable')
    if config.get('status') != 'review_only_not_training_approved':
        raise ValueError('This module accepts the review-only draft status')
    if config.get('implementation_version') != IMPLEMENTATION_VERSION:
        raise ValueError('Candidate numerical implementation version differs')
    for key in ('approved_for_formal_training', 'inherit_training_approval', 'resume_old_cnn_checkpoint'):
        if config.get(key) is not False:
            raise ValueError(f'{key} must explicitly remain false for this draft')
    if config.get('approval_date') is not None or config.get('approval_evidence') is not None:
        raise ValueError('No formal-training approval is recorded for this review draft')
    if config.get('formal_training_budget') != {'approved': False, 'maximum_runs': None,
                                               'proposed_maximum_runs': 1,
                                               'proposed_maximum_epochs_per_run': 10,
                                               'proposed_seeds': [42], 'decision': None}:
        raise ValueError('A new formal-training budget is still pending approval')
    if config.get('model') != MODEL_SPEC:
        raise ValueError('Unsupported TCN architecture; review a new candidate version before changing it')
    if config.get('data_contract') != DATA_CONTRACT:
        raise ValueError('TCN01 must preserve the existing data, labels, sampling and normalization')
    for key, expected in PRESERVED_SETTINGS.items():
        if config.get(key) != expected:
            raise ValueError(f'TCN01 must preserve the baseline setting: {key}')
        if isinstance(expected, bool) and config.get(key) is not expected:
            raise ValueError(f'{key} must be a JSON boolean')
    return config


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open('r', encoding='utf-8-sig') as stream:
        config = json.load(stream)
    return validate_config(config)


class CausalConv1d(nn.Module):
    """Same past-only k3 cross-correlation, with a pointwise sum for dilation > 1.

    Tap order follows Conv1d: tap0 reads t-2d, tap1 reads t-d, tap2 reads t.
    Bias is added once. The underlying conv keeps its original state_dict keys,
    weight shape and parameter count; there are no new weights or checkpoints.
    Equivalent FP32 accumulation orders can have small roundoff differences.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=1, padding=0, dilation=dilation, bias=True)

    def forward(self, x):
        padded = F.pad(x, (self.left_padding, 0), mode='constant', value=0.0)
        if self.conv.dilation[0] == 1:
            return self.conv(padded)
        length = x.size(-1)
        dilation = self.conv.dilation[0]
        result = F.conv1d(padded[:, :, :length], self.conv.weight[:, :, 0:1])
        result = result + F.conv1d(padded[:, :, dilation:dilation + length], self.conv.weight[:, :, 1:2])
        result = result + F.conv1d(padded[:, :, 2 * dilation:2 * dilation + length], self.conv.weight[:, :, 2:3])
        bias = self.conv.bias
        if bias is not None:
            result = result + bias.view(1, -1, 1)
        return result


class CausalResidualBlock(nn.Module):
    """Two causal Conv-ReLU layers, channel projection if needed, residual ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.temporal = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
        )
        self.shortcut = (nn.Conv1d(in_channels, out_channels, 1, bias=True)
                         if in_channels != out_channels else nn.Identity())
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.temporal(x) + self.shortcut(x))


class TemporalConvCandidate(nn.Module):
    """Pure temporal CNN candidate for [B, 40, 100], producing [B, 3] logits."""

    def __init__(self):
        super().__init__()
        in_channels = 40
        blocks = []
        for dilation in (1, 2, 4, 8, 16):
            blocks.append(CausalResidualBlock(in_channels, 32, 3, dilation))
            in_channels = 32
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Linear(32, 3)

    @torch.jit.export
    def forward_features(self, x):
        if x.dim() != 3:
            raise ValueError('Expected input [B, 40, 100]')
        if x.size(0) < 1 or x.size(1) != 40 or x.size(2) != 100:
            raise ValueError('Expected a nonempty batch with exactly 40 channels and 100 records')
        return self.features(x)

    def forward(self, x):
        return self.classifier(self.forward_features(x)[:, :, -1])


def build_model(config):
    validate_config(config)
    return TemporalConvCandidate()


def model_summary(config):
    """Arithmetic structure estimate, not measured latency, memory or accuracy."""
    validate_config(config)
    spec = config['model']
    channels = spec['feature_count']
    hidden = spec['hidden_channels']
    kernel = spec['kernel_size']
    length = spec['window_length']
    parameters = 0
    dense_macs = 0
    for _ in spec['dilations']:
        parameters += channels * hidden * kernel + hidden
        parameters += hidden * hidden * kernel + hidden
        dense_macs += length * (channels * hidden * kernel + hidden * hidden * kernel)
        if channels != hidden:
            parameters += channels * hidden + hidden
            dense_macs += length * channels * hidden
        channels = hidden
    parameters += hidden * spec['classes'] + spec['classes']
    dense_macs += hidden * spec['classes']
    receptive_field = 1 + spec['convolutions_per_block'] * (kernel - 1) * sum(spec['dilations'])
    return {
        'model_version': spec['version'],
        'implementation_version': IMPLEMENTATION_VERSION,
        'parameter_count': parameters,
        'theoretical_receptive_field_records': receptive_field,
        'input_window_records': length,
        'oldest_accessible_record_index': max(0, length - receptive_field),
        'dense_macs_per_window': dense_macs,
        'dense_macs_scope': 'Convolution/linear multiply-accumulates for the 100 semantic outputs; lagwise implementation computes the same count and no extra cropped outputs; excludes padding, bias adds, tap-sum adds, residual adds and ReLU',
        'float32_parameter_bytes': parameters * 4,
        'formal_training_approved': False,
        'predictive_quality_verified': False,
        'hardware_verified': False,
    }


if __name__ == '__main__':
    raise SystemExit('TCN01 is a review-only model module. It has no train or resume entry point.')
