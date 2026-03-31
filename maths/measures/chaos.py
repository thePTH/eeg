from __future__ import annotations


import numpy as np
from maths.tools import EmbeddingTools
from scipy.spatial.distance import cdist

class ChaosMeasures:
    """Mesures issues de l'analyse dynamique / chaotique."""

    @staticmethod
    def lyapunov_rosenstein(
        x: np.ndarray,
        emb_dim: int = 6,
        tau: int = 1,
        max_t: int = 20,
    ) -> float:
        """
        Estimation simple du plus grand exposant de Lyapunov
        (méthode de type Rosenstein).
        """
        y = EmbeddingTools.sliding_embed(x, emb_dim, tau)
        n = len(y)

        if n < max_t + 5:
            return 0.0

        dist = cdist(y, y, metric="euclidean")
        np.fill_diagonal(dist, np.inf)

        theiler = max(emb_dim * tau, 5)
        for i in range(n):
            lo = max(0, i - theiler)
            hi = min(n, i + theiler + 1)
            dist[i, lo:hi] = np.inf

        nn = np.argmin(dist, axis=1)

        div = []
        valid_t = []

        for t in range(1, max_t + 1):
            vals = []
            for i in range(n - t):
                j = nn[i]
                if j + t < n:
                    d = np.linalg.norm(y[i + t] - y[j + t])
                    if d > 0:
                        vals.append(np.log(d))

            if len(vals) >= 5:
                div.append(np.mean(vals))
                valid_t.append(t)

        if len(valid_t) < 3:
            return 0.0

        slope, _ = np.polyfit(np.asarray(valid_t), np.asarray(div), 1)
        return float(slope)