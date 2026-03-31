from features.config import FeatureExtractionConfig
from maths.engines.spectral import SignalSpectralAnalysisParameters
from maths.engines.wavelets import SignalWaveletAnalysisParameters

class SignalAnalysisEngineParametersFactory:
    @staticmethod
    def build_spectral_engine_parameters(config:FeatureExtractionConfig):
        return SignalSpectralAnalysisParameters(bands=config.bands, spectral_flux_segment_sec=config.spectral_flux_segment_sec, psd_time_halfbandwidth_product=config.psd_time_halfbandwidth_product)
    @staticmethod
    def build_wavelet_engine_parameters(config:FeatureExtractionConfig):
        return SignalWaveletAnalysisParameters(wavelet=config.wavelet, wavelet_level=config.wavelet_level)
    

