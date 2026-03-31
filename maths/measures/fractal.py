from __future__ import annotations


import numpy as np
from maths.tools import EmbeddingTools
from scipy.spatial.distance import cdist


class FractalMeasures:
    """Mesures fractales et géométriques."""

    @staticmethod
    def correlation_dimension(
        x: np.ndarray,
        emb_dim: int = 3,
        tau: int = 1,
        n_radii: int = 10,
    ) -> float:
        """
        Estimation simple de la correlation dimension par pente
        log(C(r)) vs log(r).
        """
        y = EmbeddingTools.sliding_embed(x, emb_dim, tau)
        d = cdist(y, y, metric="euclidean")
        d = d[np.triu_indices_from(d, k=1)]
        d = d[d > 0]

        if d.size < 10:
            return 0.0

        r_min = np.percentile(d, 5)
        r_max = np.percentile(d, 50)

        if r_max <= r_min:
            return 0.0

        radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
        c_r = np.array([(d < r).mean() for r in radii], dtype=float)

        mask = c_r > 0
        if mask.sum() < 3:
            return 0.0

        slope, _ = np.polyfit(np.log(radii[mask]), np.log(c_r[mask]), deg=1)
        return float(slope)

    @staticmethod
    def higuchi_fd(x: np.ndarray, kmax: int = 10) -> float:
        """
        Higuchi Fractal Dimension.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)

        lk = []
        k_values = range(1, kmax + 1)

        for k in k_values:
            lm = []
            for m in range(k):
                idx = np.arange(m, n, k)
                if len(idx) < 2:
                    continue

                ll = np.sum(np.abs(np.diff(x[idx])))
                norm = (n - 1) / (((len(idx) - 1) * k) * k)
                lm.append(ll * norm)

            if len(lm) > 0:
                lk.append(np.mean(lm))
            else:
                lk.append(np.nan)

        lk = np.asarray(lk, dtype=float)
        mask = np.isfinite(lk) & (lk > 0)

        if mask.sum() < 3:
            return 0.0

        slope, _ = np.polyfit(
            np.log(1.0 / np.arange(1, kmax + 1)[mask]),
            np.log(lk[mask]),
            1,
        )
        return float(slope)

    @staticmethod
    def katz_fd(x: np.ndarray) -> float:
        """
        Katz Fractal Dimension.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)

        if n < 2:
            return 0.0

        ll = np.sum(np.sqrt(1 + np.diff(x) ** 2))
        d = np.max(np.sqrt((np.arange(n) - 0) ** 2 + (x - x[0]) ** 2))

        if d <= 0 or ll <= 0:
            return 0.0

        return float(np.log10(n) / (np.log10(d / ll + 0) + np.log10(n)))

    @staticmethod
    def hurst_rs(
        x: np.ndarray,
        min_chunk: int = 8,
        max_chunk: int | None = None,
        n_chunks: int = 10,
    ) -> float:
        """
        Hurst exponent par méthode R/S.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)

        if max_chunk is None:
            max_chunk = max(min(n // 4, 512), min_chunk + 1)

        sizes = np.unique(
            np.logspace(np.log10(min_chunk), np.log10(max_chunk), n_chunks).astype(int)
        )
        rs_vals = []

        for size in sizes:
            if size >= n:
                continue

            nseg = n // size
            if nseg < 2:
                continue

            segments = x[:nseg * size].reshape(nseg, size)
            rs_seg = []

            for seg in segments:
                y = seg - seg.mean()
                z = np.cumsum(y)
                r = z.max() - z.min()
                s = seg.std(ddof=1)

                if s > 0:
                    rs_seg.append(r / s)

            if len(rs_seg) > 0:
                rs_vals.append(np.mean(rs_seg))

        sizes = sizes[:len(rs_vals)]
        rs_vals = np.asarray(rs_vals, dtype=float)
        mask = rs_vals > 0

        if mask.sum() < 3:
            return 0.5

        slope, _ = np.polyfit(np.log(sizes[mask]), np.log(rs_vals[mask]), 1)
        return float(slope)