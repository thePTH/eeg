from __future__ import annotations

import numpy as np

from eeg.signal import SignalAnalysisResults


class FeatureExtractionContext:
    """
    Lightweight context passed to EEG feature extractors.

    Purpose
    -------
    This class exposes the results of a previously computed signal analysis,
    allowing feature implementations to reuse common computations instead of
    recomputing them independently.

    It provides convenient access to:
    - the raw signal;
    - sampling information;
    - extraction configuration;
    - statistical descriptors;
    - spectral analysis results;
    - wavelet analysis results.
    """

    def __init__(self, signal_analysis_result: SignalAnalysisResults):
        """
        Initialize the feature extraction context.

        Parameters
        ----------
        signal_analysis_result
            Complete analysis result associated with a signal.
        """
        self._analysis_result = signal_analysis_result

    @property
    def analysis_result(self) -> SignalAnalysisResults:
        """Return the complete signal analysis result."""
        return self._analysis_result

    @property
    def signal(self):
        """Return the underlying signal object."""
        return self._analysis_result.signal

    @property
    def signal_name(self) -> str:
        """Return the signal name."""
        return self.signal.name

    @property
    def x(self) -> np.ndarray:
        """
        Return the signal samples as a NumPy array.
        """
        return np.asarray(self.signal.points, dtype=float)

    @property
    def fs(self) -> float:
        """
        Return the signal sampling frequency in Hz.
        """
        return float(self.signal.sampling_frequency)

    @property
    def cfg(self):
        """
        Return the feature extraction configuration.
        """
        return self._analysis_result.config

    @property
    def stats(self):
        """
        Return the statistical analysis results.
        """
        return self._analysis_result.stats

    @property
    def spectral(self):
        """
        Return the spectral analysis results.
        """
        return self._analysis_result.spectral

    @property
    def wavelet(self):
        """
        Return the wavelet analysis results.
        """
        return self._analysis_result.wavelet