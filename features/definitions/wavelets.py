from __future__ import annotations

from features.categories import FeatureCategory
from features.context import FeatureExtractionContext
from features.definitions.base import EEGFeature, register_feature


@register_feature
class WaveletEnergyApproximateFeature(EEGFeature):
    """EEG feature computing the approximation wavelet energy."""

    name = "wavelet_energy_approximate"
    category = FeatureCategory.WAVELET

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the approximation wavelet energy."""
        return ctx.wavelet.approximate_energy


@register_feature
class WaveletEnergyDetailFeature(EEGFeature):
    """EEG feature computing the detail wavelet energy."""

    name = "wavelet_energy_detail"
    category = FeatureCategory.WAVELET

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the detail wavelet energy."""
        return ctx.wavelet.detail_energy


@register_feature
class RelativeWaveletEnergyFeature(EEGFeature):
    """EEG feature computing the relative wavelet energy."""

    name = "relative_wavelet_energy"
    category = FeatureCategory.WAVELET

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the relative wavelet energy."""
        return ctx.wavelet.relative_wavelet_energy


@register_feature
class WaveletPacketEnergyApproximateFeature(EEGFeature):
    """EEG feature computing the approximation wavelet packet energy."""

    name = "wavelet_packet_energy_approximate"
    category = FeatureCategory.WAVELET

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the approximation wavelet packet energy."""
        return ctx.wavelet.packet_approximate_energy


@register_feature
class WaveletPacketEnergyDetailFeature(EEGFeature):
    """EEG feature computing the detail wavelet packet energy."""

    name = "wavelet_packet_energy_detail"
    category = FeatureCategory.WAVELET

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the detail wavelet packet energy."""
        return ctx.wavelet.packet_detail_energy


@register_feature
class RelativeWaveletPacketEnergyFeature(EEGFeature):
    """EEG feature computing the relative wavelet packet energy."""

    name = "relative_wavelet_packet_energy"
    category = FeatureCategory.WAVELET

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        """Extract the relative wavelet packet energy."""
        return ctx.wavelet.relative_wavelet_packet_energy