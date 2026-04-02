from __future__ import annotations

from scipy.stats import ttest_ind

from stats.engines.base import StatisticalTestEngine
from stats.bundles import GroupComparisonSampleBundle, SampleBundle
from stats.results import StatisticalTestResult


class TTestEngine(StatisticalTestEngine):
    test_name = "t-test"

    def __init__(self, equal_var: bool = False):
        self.equal_var = equal_var

    def compute(self, bundle: SampleBundle, *, target: str, key: str) -> StatisticalTestResult:
        if not isinstance(bundle, GroupComparisonSampleBundle):
            raise TypeError("TTestEngine expects a GroupComparisonSampleBundle")

        statistic, p_value = ttest_ind(
            bundle.x,
            bundle.y,
            equal_var=self.equal_var,
            nan_policy="omit",
        )

        return self._build_result(
            bundle,
            statistic=float(statistic),
            p_value=float(p_value),
            target=target,
            key=key,
        )