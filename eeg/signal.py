from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from features.config import FeatureExtractionConfig
from maths.engines.parameters import SignalAnalysisEngineParametersFactory
from maths.engines.spectral import (
    SignalSpectralAnalysisEngine,
    SignalSpectralAnalysisResult,
)
from maths.engines.statistics import (
    SignalStatisticsAnalysisEngine,
    SignalStatisticsAnalysisResult,
)
from maths.engines.wavelets import (
    SignalWaveletAnalysisEngine,
    SignalWaveletAnalysisResult,
)


class SampledSignal:
    """
    Représente un signal échantillonné 1D.

    Amélioration mémoire
    --------------------
    Les points sont stockés en `np.ndarray` plutôt qu'en `list[float]`.
    Cela évite des conversions coûteuses et garde une meilleure compatibilité
    avec les moteurs de calcul scientifiques.
    """

    def __init__(self, sampling_frequency: float, points, name: str):
        self._sampling_frequency = float(sampling_frequency)
        self._points = np.asarray(points, dtype=float)
        self._name = name

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def sampling_frequency(self) -> float:
        return self._sampling_frequency

    @property
    def time_axis(self) -> np.ndarray:
        return np.arange(len(self.points), dtype=float) / self.sampling_frequency

    @property
    def name(self) -> str:
        return self._name


@dataclass(frozen=True)
class SignalAnalysisResults:
    signal: SampledSignal
    config: FeatureExtractionConfig
    stats: SignalStatisticsAnalysisResult
    spectral: SignalSpectralAnalysisResult
    wavelet: SignalWaveletAnalysisResult


class SignalAnalysisEngine:
    def __init__(self, signal: SampledSignal, config: FeatureExtractionConfig):
        self.signal = signal
        self.x = np.asarray(signal.points, dtype=float)
        self.fs = signal.sampling_frequency
        self.config = config

    @cached_property
    def stats(self) -> SignalStatisticsAnalysisResult:
        return SignalStatisticsAnalysisEngine(self.x).compute()

    @cached_property
    def spectral(self) -> SignalSpectralAnalysisResult:
        params = SignalAnalysisEngineParametersFactory.build_spectral_engine_parameters(
            self.config
        )
        return SignalSpectralAnalysisEngine(self.x, self.fs, params).compute()

    @cached_property
    def wavelet(self) -> SignalWaveletAnalysisResult:
        params = SignalAnalysisEngineParametersFactory.build_wavelet_engine_parameters(
            self.config
        )
        return SignalWaveletAnalysisEngine(self.x, params).compute()

    def compute(self) -> SignalAnalysisResults:
        return SignalAnalysisResults(
            signal=self.signal,
            config=self.config,
            stats=self.stats,
            spectral=self.spectral,
            wavelet=self.wavelet,
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