from __future__ import annotations

from scipy.stats import spearmanr

from stats.engines.base import StatisticalTestEngine
from stats.bundles import CorrelationSampleBundle, SampleBundle
from stats.results import StatisticalTestResult


class SpearmanEngine(StatisticalTestEngine):
    test_name = "spearman"

    def compute(self, bundle: SampleBundle, *, target: str, key: str) -> StatisticalTestResult:
        if not isinstance(bundle, CorrelationSampleBundle):
            raise TypeError("SpearmanEngine expects a CorrelationSampleBundle")

        statistic, p_value = spearmanr(bundle.x, bundle.y, nan_policy="omit")

        return self._build_result(
            bundle,
            statistic=float(statistic),
            p_value=float(p_value),
            target=target,
            key=key,
        )