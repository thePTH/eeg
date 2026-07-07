from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maths.tools import SignalTools


@dataclass(slots=True, frozen=True)
class SignalStatisticsAnalysisResult:
    """
    Container exposing elementary signal statistics.

    This class does not perform any computation. It only stores values already
    computed by `SignalStatisticsAnalysisEngine`.
    """

    n: int
    mean: float
    std: float
    abs_mean: float
    peak_amplitude: float
    rms: float


class SignalStatisticsAnalysisEngine:
    """Engine used to compute elementary statistics from a signal."""

    def __init__(self, x: np.ndarray):
        self.x = np.asarray(x, dtype=float)

    def compute(self) -> SignalStatisticsAnalysisResult:
        """Compute basic signal statistics."""
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