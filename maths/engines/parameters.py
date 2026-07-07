from features.config import FeatureExtractionConfig
from maths.engines.spectral import SignalSpectralAnalysisParameters
from maths.engines.wavelets import SignalWaveletAnalysisParameters


class SignalAnalysisEngineParametersFactory:
    """Factory used to build analysis engine parameters from a feature extraction configuration."""

    @staticmethod
    def build_spectral_engine_parameters(
        config: FeatureExtractionConfig,
    ) -> SignalSpectralAnalysisParameters:
        """
        Build the parameters required by the spectral analysis engine.

        Parameters
        ----------
        config : FeatureExtractionConfig
            Global feature extraction configuration.

        Returns
        -------
        SignalSpectralAnalysisParameters
            Spectral analysis parameters.
        """
        return SignalSpectralAnalysisParameters(
            bands=config.bands,
            spectral_flux_segment_sec=config.spectral_flux_segment_sec,
            psd_time_halfbandwidth_product=config.psd_time_halfbandwidth_product,
        )

    @staticmethod
    def build_wavelet_engine_parameters(
        config: FeatureExtractionConfig,
    ) -> SignalWaveletAnalysisParameters:
        """
        Build the parameters required by the wavelet analysis engine.

        Parameters
        ----------
        config : FeatureExtractionConfig
            Global feature extraction configuration.

        Returns
        -------
        SignalWaveletAnalysisParameters
            Wavelet analysis parameters.
        """
        return SignalWaveletAnalysisParameters(
            wavelet=config.wavelet,
            wavelet_level=config.wavelet_level,
        )