from __future__ import annotations


from features.definitions.base import EEGFeature, register_feature
from features.context import FeatureExtractionContext

from maths.measures.complexity import ComplexityMeasures
from features.categories import FeatureCategory

@register_feature
class SampleEntropyFeature(EEGFeature):
    name = "sample_entropy"
    category = FeatureCategory.ENTROPY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        r = ctx.cfg.entropy_r_factor * ctx.stats.std
        return ComplexityMeasures.sample_entropy(ctx.x, m=ctx.cfg.entropy_m, r=r)


@register_feature
class ApproximateEntropyFeature(EEGFeature):
    name = "approximate_entropy"
    category = FeatureCategory.ENTROPY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        r = ctx.cfg.entropy_r_factor * ctx.stats.std
        return ComplexityMeasures.approximate_entropy(ctx.x, m=ctx.cfg.entropy_m, r=r)


@register_feature
class PermutationEntropyFeature(EEGFeature):
    name = "permutation_entropy"
    category = FeatureCategory.ENTROPY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ComplexityMeasures.permutation_entropy(
            ctx.x,
            order=ctx.cfg.permutation_order,
            delay=ctx.cfg.permutation_delay,
            normalize=True,
        )

@register_feature
class StateSpaceCorrelationEntropyFeature(EEGFeature):
    name = "state_space_correlation_entropy"
    category = FeatureCategory.ENTROPY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        r = ctx.cfg.entropy_r_factor * max(ctx.stats.std, 0)
        return ComplexityMeasures.state_space_correlation_entropy(ctx.x, emb_dim=3, tau=1, r=r)


