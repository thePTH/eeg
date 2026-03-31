from __future__ import annotations

from features.definitions.base import EEGFeature, register_feature
from features.context import FeatureExtractionContext
from features.categories import FeatureCategory


@register_feature
class AlphaDominantFrequencyFeature(EEGFeature):
    name = "alpha_dominant_frequency"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.spectral.dominant_frequency_alpha

@register_feature
class GammaDominantFrequencyFeature(EEGFeature):
    name = "gamma_dominant_frequency"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        
        return ctx.spectral.dominant_frequency_gamma

@register_feature
class SpectralRolloffFeature(EEGFeature):
    name = "spectral_rolloff"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.spectral.rolloff_95

@register_feature
class SpectralCentroidFeature(EEGFeature):
    name = "spectral_centroid"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.spectral.centroid

@register_feature
class SpectralSpreadFeature(EEGFeature):
    name = "spectral_spread"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.spectral.spread

@register_feature
class SpectralFluxFeature(EEGFeature):
    name = "spectral_flux"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.spectral.flux

@register_feature
class SpectralSkewnessFeature(EEGFeature):
    name = "spectral_skewness"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.spectral.skewness

@register_feature
class SpectralKurtosisFeature(EEGFeature):
    name = "spectral_kurtosis"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.spectral.kurtosis


