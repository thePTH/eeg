from __future__ import annotations

from features.categories import FeatureCategory
from features.context import FeatureExtractionContext
from features.definitions.base import EEGFeature, register_feature
from maths.measures.complexity import ComplexityMeasures


@register_feature
class SampleEntropyFeature(EEGFeature):
    """EEG feature computing sample entropy."""

    name = "sample_entropy"
    category = FeatureCategory.ENTROPY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract sample entropy from the signal."""
        r = ctx.cfg.entropy_r_factor * ctx.stats.std

        return ComplexityMeasures.sample_entropy(ctx.x, m=ctx.cfg.entropy_m, r=r)


@register_feature
class ApproximateEntropyFeature(EEGFeature):
    """EEG feature computing approximate entropy."""

    name = "approximate_entropy"
    category = FeatureCategory.ENTROPY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract approximate entropy from the signal."""
        r = ctx.cfg.entropy_r_factor * ctx.stats.std

        return ComplexityMeasures.approximate_entropy(ctx.x, m=ctx.cfg.entropy_m, r=r)


@register_feature
class PermutationEntropyFeature(EEGFeature):
    """EEG feature computing permutation entropy."""

    name = "permutation_entropy"
    category = FeatureCategory.ENTROPY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract permutation entropy from the signal."""
        return ComplexityMeasures.permutation_entropy(
            ctx.x,
            order=ctx.cfg.permutation_order,
            delay=ctx.cfg.permutation_delay,
            normalize=True,
        )


@register_feature
class StateSpaceCorrelationEntropyFeature(EEGFeature):
    """EEG feature computing state-space correlation entropy."""

    name = "state_space_correlation_entropy"
    category = FeatureCategory.ENTROPY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract state-space correlation entropy from the signal."""
        r = ctx.cfg.entropy_r_factor * max(ctx.stats.std, 0)

        return ComplexityMeasures.state_space_correlation_entropy(
            ctx.x,
            emb_dim=3,
            tau=1,
            r=r,
        )