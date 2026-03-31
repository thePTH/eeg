from __future__ import annotations


from features.definitions.base import EEGFeature, register_feature
from features.context import FeatureExtractionContext

from features.categories import FeatureCategory

import numpy as np
import scipy

@register_feature
class VarianceFeature(EEGFeature):
    name = "variance"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return float(np.var(ctx.x, ddof=1))


@register_feature
class SkewnessFeature(EEGFeature):
    name = "skewness"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return float(scipy.stats.skew(ctx.x, bias=False))

@register_feature
class KurtosisFeature(EEGFeature):
    name = "kurtosis"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return float(scipy.stats.kurtosis(ctx.x, fisher=False, bias=False))

@register_feature
class PeakAmplitudeFeature(EEGFeature):
    name = "peak_amplitude"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.stats.peak_amplitude

@register_feature
class ShapeFactorFeature(EEGFeature):
    name = "shape_factor"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.stats.rms /  ctx.stats.abs_mean

@register_feature
class ImpulseFactorFeature(EEGFeature):
    name = "impulse_factor"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.stats.peak_amplitude / ctx.stats.abs_mean

@register_feature
class CrestFactorFeature(EEGFeature):
    name = "crest_factor"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.stats.peak_amplitude / ctx.stats.rms

@register_feature
class ClearanceFactorFeature(EEGFeature):
    name = "clearance_factor"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        denom = np.mean(np.sqrt(np.abs(ctx.x))) ** 2
        return ctx.stats.peak_amplitude / denom

@register_feature
class WillisonAmplitudeFeature(EEGFeature):
    name = "willison_amplitude"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        dx = np.abs(np.diff(ctx.x))
        return float(np.sum(dx > ctx.cfg.wamp_threshold))

@register_feature
class ZeroCrossingRateFeature(EEGFeature):
    name = "zero_crossing_rate"
    category = FeatureCategory.TEMPORAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        s = np.sign(ctx.x)
        s[s == 0] = 1
        return float(np.sum(np.abs(np.diff(s))) / (2 * len(ctx.x)))
    


