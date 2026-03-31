from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from maths.tools import SignalTools

@dataclass(slots=True, frozen=True)
class SignalStatisticsAnalysisResult:
    """
    Résultat exposant les statistiques élémentaires d'un signal.

    Cette classe ne calcule rien. Elle se contente d'exposer des valeurs
    déjà calculées par `SignalStatisticsEngine`.
    """
    n: int
    mean: float
    std: float
    abs_mean: float
    peak_amplitude: float
    rms: float


class SignalStatisticsAnalysisEngine:
    """
    Moteur de calcul des statistiques élémentaires d'un signal.

    Cette classe est responsable du calcul.
    Les résultats sont encapsulés dans `SignalStatisticsResult`.
    """

    def __init__(self, x: np.ndarray):
        self.x = np.asarray(x, dtype=float)

    def compute(self) -> SignalStatisticsAnalysisResult:
        """
        Calcule les statistiques de base du signal.
        """
        n = len(self.x)
        mean = float(np.mean(self.x))
        std = float(np.std(self.x, ddof=1)) if n > 1 else 0.0
        abs_mean = float(np.mean(np.abs(self.x)))
        peak_amplitude = float(np.max(np.abs(self.x))) if n > 0 else 0.0
        rms = SignalTools.rms(self.x) if n > 0 else 0.0

        return SignalStatisticsAnalysisResult(
            n=n,
            mean=mean,
            std=std,
            abs_mean=abs_mean,
            peak_amplitude=peak_amplitude,
            rms=rms,
        )