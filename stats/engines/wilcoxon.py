from __future__ import annotations

from scipy.stats import ranksums

from stats.engines.base import StatisticalTestEngine
from stats.bundles import GroupComparisonSampleBundle, SampleBundle
from stats.results import StatisticalTestResult


class WilcoxonRankSumEngine(StatisticalTestEngine):
    test_name = "wilcoxon-rank-sum"

    def compute(self, bundle: SampleBundle, *, target: str, key: str) -> StatisticalTestResult:
        if not isinstance(bundle, GroupComparisonSampleBundle):
            raise TypeError("WilcoxonRankSumEngine expects a GroupComparisonSampleBundle")

        statistic, p_value = ranksums(bundle.x, bundle.y)

        return self._build_result(
            bundle,
            statistic=float(statistic),
            p_value=float(p_value),
            target=target,
            key=key,
        )