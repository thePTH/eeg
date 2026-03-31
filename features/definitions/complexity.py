from __future__ import annotations


from features.definitions.base import EEGFeature, register_feature
from features.context import FeatureExtractionContext

from maths.measures.complexity import ComplexityMeasures
from maths.measures.fractal import FractalMeasures
from maths.measures.chaos import ChaosMeasures
from maths.measures.hjorth import HjorthMeasures

from features.categories import FeatureCategory


@register_feature
class CorrelationDimensionFeature(EEGFeature):
    name = "correlation_dimension"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return FractalMeasures.correlation_dimension(
            ctx.x,
            emb_dim=ctx.cfg.corr_dim_emb_dim,
            tau=ctx.cfg.corr_dim_tau,
            n_radii=10,
        )


@register_feature
class HiguchiFractalDimensionFeature(EEGFeature):
    name = "higuchi_fractal_dimension"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return FractalMeasures.higuchi_fd(ctx.x, kmax=ctx.cfg.higuchi_kmax)


@register_feature
class KatzFractalDimensionFeature(EEGFeature):
    name = "katz_fractal_dimension"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return FractalMeasures.katz_fd(ctx.x)


@register_feature
class LyapunovExponentFeature(EEGFeature):
    name = "lyapunov_exponent"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ChaosMeasures.lyapunov_rosenstein(
            ctx.x,
            emb_dim=ctx.cfg.lyapunov_emb_dim,
            tau=ctx.cfg.lyapunov_tau,
            max_t=ctx.cfg.lyapunov_max_t,
        )

@register_feature
class HurstExponentFeature(EEGFeature):
    name = "hurst_exponent"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return FractalMeasures.hurst_rs(ctx.x)

@register_feature
class LempelZivComplexityFeature(EEGFeature):
    name = "lempel_ziv_complexity"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ComplexityMeasures.lz_complexity(ctx.x)

@register_feature
class HjorthActivityFeature(EEGFeature):
    name = "hjorth_activity"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        a, _, _ = HjorthMeasures.hjorth_parameters(ctx.x)
        return a

@register_feature
class HjorthMobilityFeature(EEGFeature):
    name = "hjorth_mobility"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        _, m, _ = HjorthMeasures.hjorth_parameters(ctx.x)
        return m

@register_feature
class HjorthComplexityFeature(EEGFeature):
    name = "hjorth_complexity"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        _, _, c = HjorthMeasures.hjorth_parameters(ctx.x)
        return c



