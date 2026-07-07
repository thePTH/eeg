from __future__ import annotations

from features.categories import FeatureCategory
from features.context import FeatureExtractionContext
from features.definitions.base import EEGFeature, register_feature


@register_feature
class ThetaBetaRatioFeature(EEGFeature):
    """EEG feature computing the theta-to-beta power ratio."""

    name = "theta_beta_ratio"
    category = FeatureCategory.POWER_RATIO

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the theta-to-beta power ratio."""
        return ctx.spectral.theta_beta_ratio


@register_feature
class ThetaAlphaRatioFeature(EEGFeature):
    """EEG feature computing the theta-to-alpha power ratio."""

    name = "theta_alpha_ratio"
    category = FeatureCategory.POWER_RATIO

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the theta-to-alpha power ratio."""
        return ctx.spectral.theta_alpha_ratio


@register_feature
class GammaAlphaRatioFeature(EEGFeature):
    """EEG feature computing the gamma-to-alpha power ratio."""

    name = "gamma_alpha_ratio"
    category = FeatureCategory.POWER_RATIO

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the gamma-to-alpha power ratio."""
        return ctx.spectral.gamma_alpha_ratio


@register_feature
class SpectralPowerRatioFeature(EEGFeature):
    """EEG feature computing the spectral power ratio."""

    name = "spectral_power_ratio"
    category = FeatureCategory.POWER_RATIO

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the spectral power ratio."""
        return ctx.spectral.spectral_power_ratio