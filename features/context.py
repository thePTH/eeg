from __future__ import annotations

import numpy as np

from eeg.signal import SignalAnalysisResults


class FeatureExtractionContext:
    """
    Contexte léger transmis aux providers de features.

    Objectif
    --------
    Encapsuler proprement le résultat d'analyse d'un signal
    (statistiques, spectral, wavelet, config, etc.) afin de :
    - garder une API orientée objet claire ;
    - éviter que chaque feature ne reconstruise sa propre analyse ;
    - centraliser les accès aux données utiles.
    """

    def __init__(self, signal_analysis_result: SignalAnalysisResults):
        self._analysis_result = signal_analysis_result

    @property
    def analysis_result(self) -> SignalAnalysisResults:
        return self._analysis_result

    @property
    def signal(self):
        return self._analysis_result.signal

    @property
    def signal_name(self) -> str:
        return self.signal.name

    @property
    def x(self) -> np.ndarray:
        """
        Vue NumPy du signal.
        """
        return np.asarray(self.signal.points, dtype=float)

    @property
    def fs(self) -> float:
        return float(self.signal.sampling_frequency)

    @property
    def cfg(self):
        return self._analysis_result.config

    @property
    def stats(self):
        return self._analysis_result.stats

    @property
    def spectral(self):
        return self._analysis_result.spectral

    @property
    def wavelet(self):
        return self._analysis_result.wavelet