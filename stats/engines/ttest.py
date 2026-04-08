from __future__ import annotations

from scipy.stats import ttest_ind

from stats.bundles import GroupComparisonSampleBundle, SampleBundle
from stats.engines.base import StatisticalTestEngine


class TTestEngine(StatisticalTestEngine):
    test_name = "t-test"

    def __init__(self, equal_var: bool = False):
        self.equal_var = equal_var

    def compute(self, bundle: SampleBundle, *, target: str, key: str):
        if not isinstance(bundle, GroupComparisonSampleBundle):
            raise TypeError("TTestEngine expects a GroupComparisonSampleBundle")

        x = self._to_numeric_series(bundle.x)
        y = self._to_numeric_series(bundle.y)

        statistic, p_value = ttest_ind(
            x,
            y,
            equal_var=self.equal_var,
            nan_policy="omit",
        )

        return self._build_pairwise_result(
            bundle,
            statistic=float(statistic),
            p_value=float(p_value),
            target=target,
            key=key,
            metadata={"equal_var": self.equal_var},
        )