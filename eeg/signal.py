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
    Represent a one-dimensional sampled signal.

    Memory optimization:
    points are stored as a NumPy array instead of a list of floats. This avoids
    costly conversions and improves compatibility with scientific computation
    engines.
    """

    def __init__(self, sampling_frequency: float, points, name: str):
        self._sampling_frequency = float(sampling_frequency)
        self._points = np.asarray(points, dtype=float)
        self._name = name

    @property
    def points(self) -> np.ndarray:
        """Return the sampled signal values."""
        return self._points

    @property
    def sampling_frequency(self) -> float:
        """Return the sampling frequency of the signal."""
        return self._sampling_frequency

    @property
    def time_axis(self) -> np.ndarray:
        """Return the time axis associated with the sampled points."""
        return np.arange(len(self.points), dtype=float) / self.sampling_frequency

    @property
    def name(self) -> str:
        """Return the signal name."""
        return self._name


@dataclass(frozen=True)
class SignalAnalysisResults:
    """Container gathering all analysis results computed for one signal."""

    signal: SampledSignal
    config: FeatureExtractionConfig
    stats: SignalStatisticsAnalysisResult
    spectral: SignalSpectralAnalysisResult
    wavelet: SignalWaveletAnalysisResult


class SignalAnalysisEngine:
    """Compute statistical, spectral, and wavelet features for one sampled signal."""

    def __init__(self, signal: SampledSignal, config: FeatureExtractionConfig):
        self.signal = signal
        self.x = np.asarray(signal.points, dtype=float)
        self.fs = signal.sampling_frequency
        self.config = config

    @cached_property
    def stats(self) -> SignalStatisticsAnalysisResult:
        """Compute and cache statistical analysis results."""
        return SignalStatisticsAnalysisEngine(self.x).compute()

    @cached_property
    def spectral(self) -> SignalSpectralAnalysisResult:
        """Compute and cache spectral analysis results."""
        params = SignalAnalysisEngineParametersFactory.build_spectral_engine_parameters(
            self.config
        )

        return SignalSpectralAnalysisEngine(self.x, self.fs, params).compute()

    @cached_property
    def wavelet(self) -> SignalWaveletAnalysisResult:
        """Compute and cache wavelet analysis results."""
        params = SignalAnalysisEngineParametersFactory.build_wavelet_engine_parameters(
            self.config
        )

        return SignalWaveletAnalysisEngine(self.x, params).compute()

    def compute(self) -> SignalAnalysisResults:
        """Compute all signal analysis results."""
        return SignalAnalysisResults(
            signal=self.signal,
            config=self.config,
            stats=self.stats,
            spectral=self.spectral,
            wavelet=self.wavelet,
        )


@dataclass(frozen=True)
class SpectralBand:
    """Represent a frequency band used for connectivity computation."""

    name: str
    fmin: float
    fmax: float

    @property
    def label(self) -> str:
        """Return a human-readable frequency band label."""
        return f"{self.name} [{self.fmin:.1f}-{self.fmax:.1f} Hz]"