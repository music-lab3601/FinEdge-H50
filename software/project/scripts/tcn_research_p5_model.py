"""Independent P5 model factory; import itself is standard-library only.

The baseline returns the unchanged frozen TemporalConvCandidate.  The optional
weight-normalized model parametrizes just the ten causal temporal convolution
weights.  The residual projection and classifier are unchanged.  Modern
parametrizations are necessary: the lagwise implementation reads .weight
directly and bypasses Conv1d.forward for dilation greater than one.
"""
from __future__ import annotations


MODEL_FACTORY_VERSION = 'TCN-research-P5-model-factory-1.0'
BASELINE_PARAMETERS = 33219
WEIGHT_NORM_PARAMETERS = 33539
TEMPORAL_CONVOLUTIONS = 10


def build_model(weight_norm=False):
    """Build fresh model after the caller's metadata/authorization admission.

    Caller owns RNG seeding. No checkpoint, optimizer or data is accessed here.
    Weight normalization uses dim=0 and introduces one 32-value magnitude per
    temporal convolution, initialized to preserve its effective random weight.
    """
    if type(weight_norm) is not bool:
        raise ValueError('weight_norm must be an explicit boolean')

    from tcn_candidate import CausalConv1d, TemporalConvCandidate
    if not weight_norm:
        return TemporalConvCandidate()

    from torch.nn.utils import parametrize
    from torch.nn.utils.parametrizations import weight_norm as apply_weight_norm

    class WeightNormalizedTemporalConvCandidate(TemporalConvCandidate):
        """Cache parametrized weights only within one complete forward call."""

        def forward(self, x):
            # Each lagwise convolution accesses the same effective weight for
            # three taps. Clear the cache at forward exit so updated v/g are
            # observed by the next call, including after a manual/test update.
            with parametrize.cached():
                return super().forward(x)

    model = WeightNormalizedTemporalConvCandidate()
    temporal = [layer for layer in model.modules() if isinstance(layer, CausalConv1d)]
    if len(temporal) != TEMPORAL_CONVOLUTIONS:
        raise ValueError('Frozen temporal architecture changed')
    for layer in temporal:
        apply_weight_norm(layer.conv, name='weight', dim=0)
    if sum(parameter.numel() for parameter in model.parameters()) != WEIGHT_NORM_PARAMETERS:
        raise ValueError('Unexpected weight-normalized parameter count')
    return model
