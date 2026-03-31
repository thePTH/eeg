from maths.engines.wavelets import SignalWaveletAnalysisResult, SignalWaveletAnalysisEngine
from maths.engines.statistics import SignalStatisticsAnalysisResult, SignalStatisticsAnalysisEngine
from maths.engines.spectral import SignalSpectralAnalysisResult, SignalSpectralAnalysisEngine
from maths.engines.parameters import SignalAnalysisEngineParametersFactory
from dataclasses import dataclass
from features.config import FeatureExtractionConfig
import numpy as np
from functools import cached_property


class SampledSignal:
    def __init__(self, sampling_frequency:float, points:list[float], name:str):
        self._sampling_frequency = sampling_frequency
        self._points = points
        self._name = name

    @property
    def points(self):
        return self._points
    
    @property
    def sampling_frequency(self):
        return self._sampling_frequency
        
    @property
    def time_axis(self):
        return [k / self.sampling_frequency for k in range(len(self.points))]
    
    @property
    def name(self):
        return self._name
       


@dataclass(frozen=True)
class SignalAnalysisResults:
    signal:SampledSignal
    config:FeatureExtractionConfig
    stats: SignalStatisticsAnalysisResult
    spectral: SignalSpectralAnalysisResult
    wavelet: SignalWaveletAnalysisResult

class SignalAnalysisEngine:
    def __init__(self, signal:SampledSignal, config:FeatureExtractionConfig):
        self.signal = signal
        self.x = np.array(signal.points)
        self.fs = signal.sampling_frequency
        self.config = config


    @cached_property
    def stats(self) -> SignalStatisticsAnalysisResult:
        return SignalStatisticsAnalysisEngine(self.x).compute()

    @cached_property
    def spectral(self) -> SignalSpectralAnalysisResult:
        params = SignalAnalysisEngineParametersFactory.build_spectral_engine_parameters(self.config)
        return SignalSpectralAnalysisEngine(self.x, self.fs, params).compute()

    @cached_property
    def wavelet(self) -> SignalWaveletAnalysisResult:
        params = SignalAnalysisEngineParametersFactory.build_wavelet_engine_parameters(self.config)
        return SignalWaveletAnalysisEngine(self.x, params).compute()

    def compute(self) -> SignalAnalysisResults:
        return SignalAnalysisResults(
            signal=self.signal,
            config=self.config,
            stats=self.stats,
            spectral=self.spectral,
            wavelet=self.wavelet
        )




@dataclass(frozen=True)
class SpectralBand:
    """
    Représente une bande de fréquences utilisée pour le calcul de connectivité.
    """
    name: str
    fmin: float
    fmax: float

    @property
    def label(self) -> str:
        return f"{self.name} [{self.fmin:.1f}-{self.fmax:.1f} Hz]"