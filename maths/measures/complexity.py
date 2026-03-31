from __future__ import annotations

import math
import numpy as np
from scipy.spatial.distance import cdist

from maths.tools import EmbeddingTools, SignalTools


class ComplexityMeasures:
    """Mesures d'entropie et de complexité."""

    @staticmethod
    def permutation_entropy(
        x: np.ndarray,
        order: int = 3,
        delay: int = 1,
        normalize: bool = True,
    ) -> float:
        """
        Implémentation standard de la permutation entropy de Bandt-Pompe.
        """
        n = len(x)

        if order < 2:
            raise ValueError("order doit être >= 2.")
        if n < (order - 1) * delay + 1:
            return 0.0

        embedded = np.array(
            [x[i:i + order * delay:delay] for i in range(n - (order - 1) * delay)]
        )
        perms = np.argsort(embedded, axis=1)
        _, counts = np.unique(perms, axis=0, return_counts=True)

        p = counts / counts.sum()
        pe = -np.sum(p * np.log(p))

        if normalize:
            pe /= np.log(math.factorial(order))

        return float(pe)

    @staticmethod
    def sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
        """
        Sample entropy standard.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)

        if n <= m + 1:
            return 0.0

        if r is None:
            r = 0.2 * np.std(x, ddof=1)

        def _phi(mm: int) -> float:
            emb = EmbeddingTools.sliding_embed(x, mm, 1)
            dist = cdist(emb, emb, metric="chebyshev")
            count = np.sum(dist <= r, axis=0) - 1
            return np.sum(count)

        a = _phi(m + 1)
        b = _phi(m)

        if b <= 0 or a <= 0:
            return 0.0

        return float(-np.log(a / b))

    @staticmethod
    def approximate_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
        """
        Approximate entropy standard.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)

        if n <= m + 1:
            return 0.0

        if r is None:
            r = 0.2 * np.std(x, ddof=1)

        def _phi(mm: int) -> float:
            emb = EmbeddingTools.sliding_embed(x, mm, 1)
            dist = cdist(emb, emb, metric="chebyshev")
            c = np.mean(dist <= r, axis=0)
            return float(np.mean(np.log(c)))

        return float(_phi(m) - _phi(m + 1))

    @staticmethod
    def lz_complexity(x: np.ndarray) -> float:
        """
        Lempel-Ziv Complexity normalisée.
        """
        b = SignalTools.normalized_binary_sequence(x)
        s = "".join(map(str, b.tolist()))
        n = len(s)

        if n <= 1:
            return 0.0

        i, k, l = 0, 1, 1
        c = 1
        k_max = 1

        while True:
            if i + k >= n or l + k >= n:
                c += 1
                break

            if s[i + k] == s[l + k]:
                k += 1
                if l + k >= n:
                    c += 1
                    break
            else:
                if k > k_max:
                    k_max = k
                i += 1
                if i == l:
                    c += 1
                    l += k_max
                    if l >= n:
                        break
                    i = 0
                    k = 1
                    k_max = 1
                else:
                    k = 1

        return float(c * np.log2(n) / n)

    @staticmethod
    def state_space_correlation_entropy(
        x: np.ndarray,
        emb_dim: int = 3,
        tau: int = 1,
        r: float | None = None,
    ) -> float:
        """
        Approximation robuste de la state-space correlation entropy.

        Version pratique :
        - reconstruction d'état,
        - probabilité de voisinage sous rayon r,
        - entropie moyenne locale.
        """
        y = EmbeddingTools.sliding_embed(x, emb_dim, tau)
        d = cdist(y, y, metric="euclidean")

        if r is None:
            r = 0.2 * np.std(y)

        probs = np.mean(d <= r, axis=1)
        probs = np.clip(probs, 0, 1.0)

        return float(-np.mean(np.log(probs)))