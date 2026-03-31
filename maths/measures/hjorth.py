from __future__ import annotations
from constants import EPS

import numpy as np
import math

class HjorthMeasures:
    """Mesures de Hjorth."""

    @staticmethod
    def hjorth_parameters(x: np.ndarray) -> tuple[float, float, float]:
        """
        Retourne :
        - activity
        - mobility
        - complexity
        """
        x = np.asarray(x, dtype=float)
        dx = np.diff(x)
        ddx = np.diff(dx)

        var0 = np.var(x, ddof=1)
        var1 = np.var(dx, ddof=1) if len(dx) > 1 else 0.0
        var2 = np.var(ddx, ddof=1) if len(ddx) > 1 else 0.0

        activity = float(var0)
        mobility = math.sqrt(var1 / (var0))
        complexity = math.sqrt(var2 / (var1)) / (mobility)

        return activity, float(mobility), float(complexity)