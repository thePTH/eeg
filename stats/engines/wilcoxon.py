from __future__ import annotations

from scipy.stats import ranksums

from stats.bundles import GroupComparisonSampleBundle, SampleBundle
from stats.engines.base import StatisticalTestEngine


class WilcoxonRankSumEngine(StatisticalTestEngine):
    """Engine used to compute the Wilcoxon rank-sum (Mann–Whitney) test."""

    test_name = "wilcoxon-rank-sum"

    def compute(self, bundle: SampleBundle, *, target: str, key: str):
        """Compute the Wilcoxon rank-sum test from a group-comparison bundle."""
        if not isinstance(bundle, GroupComparisonSampleBundle):
            raise TypeError("WilcoxonRankSumEngine expects a GroupComparisonSampleBundle")

        x = self._to_numeric_series(bundle.x)
        y = self._to_numeric_series(bundle.y)

        statistic, p_value = ranksums(x, y)

        return self._build_pairwise_result(
            bundle,
            statistic=float(statistic),
            p_value=float(p_value),
            target=target,
            key=key,
        )