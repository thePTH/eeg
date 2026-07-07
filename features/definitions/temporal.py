from __future__ import annotations

import numpy as np
import scipy

from features.categories import FeatureCategory
from features.context import FeatureExtractionContext
from features.definitions.base import EEGFeature, register_feature


@register_feature
class VarianceFeature(EEGFeature):
    """EEG feature computing signal variance."""

    name = "variance"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract signal variance."""
        return float(np.var(ctx.x, ddof=1))


@register_feature
class SkewnessFeature(EEGFeature):
    """EEG feature computing signal skewness."""

    name = "skewness"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract signal skewness."""
        return float(scipy.stats.skew(ctx.x, bias=False))


@register_feature
class KurtosisFeature(EEGFeature):
    """EEG feature computing signal kurtosis."""

    name = "kurtosis"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract signal kurtosis."""
        return float(scipy.stats.kurtosis(ctx.x, fisher=False, bias=False))


@register_feature
class PeakAmplitudeFeature(EEGFeature):
    """EEG feature computing peak amplitude."""

    name = "peak_amplitude"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract peak amplitude."""
        return ctx.stats.peak_amplitude


@register_feature
class ShapeFactorFeature(EEGFeature):
    """EEG feature computing the shape factor."""

    name = "shape_factor"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the shape factor."""
        return ctx.stats.rms / ctx.stats.abs_mean


@register_feature
class ImpulseFactorFeature(EEGFeature):
    """EEG feature computing the impulse factor."""

    name = "impulse_factor"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the impulse factor."""
        return ctx.stats.peak_amplitude / ctx.stats.abs_mean


@register_feature
class CrestFactorFeature(EEGFeature):
    """EEG feature computing the crest factor."""

    name = "crest_factor"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the crest factor."""
        return ctx.stats.peak_amplitude / ctx.stats.rms


@register_feature
class ClearanceFactorFeature(EEGFeature):
    """EEG feature computing the clearance factor."""

    name = "clearance_factor"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the clearance factor."""
        denom = np.mean(np.sqrt(np.abs(ctx.x))) ** 2

        return ctx.stats.peak_amplitude / denom


@register_feature
class WillisonAmplitudeFeature(EEGFeature):
    """EEG feature computing Willison amplitude."""

    name = "willison_amplitude"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract Willison amplitude."""
        dx = np.abs(np.diff(ctx.x))

        return float(np.sum(dx > ctx.cfg.wamp_threshold))


@register_feature
class ZeroCrossingRateFeature(EEGFeature):
    """EEG feature computing the zero-crossing rate."""

    name = "zero_crossing_rate"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the zero-crossing rate."""
        s = np.sign(ctx.x)
        s[s == 0] = 1

        return float(np.sum(np.abs(np.diff(s))) / (2 * len(ctx.x)))