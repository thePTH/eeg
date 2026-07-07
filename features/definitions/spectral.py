from __future__ import annotations

from features.categories import FeatureCategory
from features.context import FeatureExtractionContext
from features.definitions.base import EEGFeature, register_feature


@register_feature
class AlphaDominantFrequencyFeature(EEGFeature):
    """EEG feature computing the dominant frequency in the alpha band."""

    name = "alpha_dominant_frequency"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the dominant alpha frequency."""
        return ctx.spectral.dominant_frequency_alpha


@register_feature
class GammaDominantFrequencyFeature(EEGFeature):
    """EEG feature computing the dominant frequency in the gamma band."""

    name = "gamma_dominant_frequency"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the dominant gamma frequency."""
        return ctx.spectral.dominant_frequency_gamma


@register_feature
class SpectralRolloffFeature(EEGFeature):
    """EEG feature computing the 95% spectral rolloff frequency."""

    name = "spectral_rolloff"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the spectral rolloff frequency."""
        return ctx.spectral.rolloff_95


@register_feature
class SpectralCentroidFeature(EEGFeature):
    """EEG feature computing the spectral centroid."""

    name = "spectral_centroid"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the spectral centroid."""
        return ctx.spectral.centroid


@register_feature
class SpectralSpreadFeature(EEGFeature):
    """EEG feature computing the spectral spread."""

    name = "spectral_spread"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the spectral spread."""
        return ctx.spectral.spread


@register_feature
class SpectralFluxFeature(EEGFeature):
    """EEG feature computing the spectral flux."""

    name = "spectral_flux"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the spectral flux."""
        return ctx.spectral.flux


@register_feature
class SpectralSkewnessFeature(EEGFeature):
    """EEG feature computing the spectral skewness."""

    name = "spectral_skewness"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the spectral skewness."""
        return ctx.spectral.skewness


@register_feature
class SpectralKurtosisFeature(EEGFeature):
    """EEG feature computing the spectral kurtosis."""

    name = "spectral_kurtosis"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the spectral kurtosis."""
        return ctx.spectral.kurtosis