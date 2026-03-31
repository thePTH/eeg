from __future__ import annotations

from features.definitions.base import EEGFeature, register_feature
from features.context import FeatureExtractionContext

from features.categories import FeatureCategory

@register_feature
class WaveletEnergyApproximateFeature(EEGFeature):
    name = "wavelet_energy_approximate"
    category = FeatureCategory.WAVELET

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.wavelet.approximate_energy

@register_feature
class WaveletEnergyDetailFeature(EEGFeature):
    name = "wavelet_energy_detail"
    category = FeatureCategory.WAVELET


    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.wavelet.detail_energy

@register_feature
class RelativeWaveletEnergyFeature(EEGFeature):
    name = "relative_wavelet_energy"
    category = FeatureCategory.WAVELET


    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.wavelet.relative_wavelet_energy

@register_feature
class WaveletPacketEnergyApproximateFeature(EEGFeature):
    name = "wavelet_packet_energy_approximate"
    category = FeatureCategory.WAVELET


    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.wavelet.packet_approximate_energy

@register_feature
class WaveletPacketEnergyDetailFeature(EEGFeature):
    name = "wavelet_packet_energy_detail"
    category = FeatureCategory.WAVELET

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.wavelet.packet_detail_energy

@register_feature
class RelativeWaveletPacketEnergyFeature(EEGFeature):
    name = "relative_wavelet_packet_energy"
    category = FeatureCategory.WAVELET

    @staticmethod
    def _extract(ctx: FeatureExtractionContext) -> float:
        return ctx.wavelet.relative_wavelet_packet_energy

