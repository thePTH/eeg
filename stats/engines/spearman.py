from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from stats.bundles import CorrelationSampleBundle, SampleBundle
from stats.engines.base import StatisticalTestEngine


class SpearmanEngine(StatisticalTestEngine):
    test_name = "spearman"

    def compute(self, bundle: SampleBundle, *, target: str, key: str):
        if not isinstance(bundle, CorrelationSampleBundle):
            raise TypeError("SpearmanEngine expects a CorrelationSampleBundle")

        merged = pd.DataFrame(
            {
                "x": self._to_numeric_series(bundle.x),
                "y": self._to_numeric_series(bundle.y),
            }
        ).dropna()

        statistic, p_value = spearmanr(merged["x"], merged["y"], nan_policy="omit")

        return self._build_pairwise_result(
            bundle,
            statistic=float(statistic),
            p_value=float(p_value),
            target=target,
            key=key,
        )