from __future__ import annotations

from features.categories import FeatureCategory
from features.context import FeatureExtractionContext
from features.definitions.base import EEGFeature, register_feature
from maths.measures.chaos import ChaosMeasures
from maths.measures.complexity import ComplexityMeasures
from maths.measures.fractal import FractalMeasures
from maths.measures.hjorth import HjorthMeasures


@register_feature
class CorrelationDimensionFeature(EEGFeature):
    """EEG feature computing the correlation dimension."""

    name = "correlation_dimension"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the correlation dimension from the signal."""
        return FractalMeasures.correlation_dimension(
            ctx.x,
            emb_dim=ctx.cfg.corr_dim_emb_dim,
            tau=ctx.cfg.corr_dim_tau,
            n_radii=10,
        )


@register_feature
class HiguchiFractalDimensionFeature(EEGFeature):
    """EEG feature computing the Higuchi fractal dimension."""

    name = "higuchi_fractal_dimension"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the Higuchi fractal dimension from the signal."""
        return FractalMeasures.higuchi_fd(ctx.x, kmax=ctx.cfg.higuchi_kmax)


@register_feature
class KatzFractalDimensionFeature(EEGFeature):
    """EEG feature computing the Katz fractal dimension."""

    name = "katz_fractal_dimension"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the Katz fractal dimension from the signal."""
        return FractalMeasures.katz_fd(ctx.x)


@register_feature
class LyapunovExponentFeature(EEGFeature):
    """EEG feature computing a Rosenstein-like Lyapunov exponent."""

    name = "lyapunov_exponent"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the Lyapunov exponent from the signal."""
        return ChaosMeasures.lyapunov_rosenstein(
            ctx.x,
            emb_dim=ctx.cfg.lyapunov_emb_dim,
            tau=ctx.cfg.lyapunov_tau,
            max_t=ctx.cfg.lyapunov_max_t,
        )


@register_feature
class HurstExponentFeature(EEGFeature):
    """EEG feature computing the Hurst exponent."""

    name = "hurst_exponent"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the Hurst exponent from the signal."""
        return FractalMeasures.hurst_rs(ctx.x)


@register_feature
class LempelZivComplexityFeature(EEGFeature):
    """EEG feature computing Lempel-Ziv complexity."""

    name = "lempel_ziv_complexity"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract Lempel-Ziv complexity from the signal."""
        return ComplexityMeasures.lz_complexity(ctx.x)


@register_feature
class HjorthActivityFeature(EEGFeature):
    """EEG feature computing Hjorth activity."""

    name = "hjorth_activity"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract Hjorth activity from the signal."""
        a, _, _ = HjorthMeasures.hjorth_parameters(ctx.x)

        return a


@register_feature
class HjorthMobilityFeature(EEGFeature):
    """EEG feature computing Hjorth mobility."""

    name = "hjorth_mobility"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract Hjorth mobility from the signal."""
        _, m, _ = HjorthMeasures.hjorth_parameters(ctx.x)

        return m


@register_feature
class HjorthComplexityFeature(EEGFeature):
    """EEG feature computing Hjorth complexity."""

    name = "hjorth_complexity"
    category = FeatureCategory.COMPLEXITY

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract Hjorth complexity from the signal."""
        _, _, c = HjorthMeasures.hjorth_parameters(ctx.x)

        return c