from __future__ import annotations

from features.definitions.base import EEGFeature, register_feature
from features.context import FeatureExtractionContext

from features.categories import FeatureCategory

@register_feature
class ThetaBetaRatioFeature(EEGFeature):
    name = "theta_beta_ratio"
    category = FeatureCategory.POWER_RATIO

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.spectral.theta_beta_ratio


@register_feature
class ThetaAlphaRatioFeature(EEGFeature):
    name = "theta_alpha_ratio"
    category = FeatureCategory.POWER_RATIO

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.spectral.theta_alpha_ratio

@register_feature
class GammaAlphaRatioFeature(EEGFeature):
    name = "gamma_alpha_ratio"
    category = FeatureCategory.POWER_RATIO

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.spectral.gamma_alpha_ratio

@register_feature
class SpectralPowerRatioFeature(EEGFeature):
    name = "spectral_power_ratio"
    category = FeatureCategory.POWER_RATIO

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.spectral.spectral_power_ratio
    


