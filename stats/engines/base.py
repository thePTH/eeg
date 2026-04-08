from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from stats.bundles import (
    CorrelationSampleBundle,
    GroupComparisonSampleBundle,
    OneWayANOVASampleBundle,
    SampleBundle,
)
from stats.results import (
    OneWayANOVAResult,
    PairwiseStatisticalResult,
    StatisticalResult,
)


class StatisticalTestEngine(ABC):
    """
    Un engine sait appliquer une formule statistique à un SampleBundle.
    """

    test_name: str

    @abstractmethod
    def compute(self, bundle: SampleBundle, *, target: str, key: str) -> StatisticalResult:
        raise NotImplementedError

    @staticmethod
    def _to_numeric_series(values: pd.Series | list[float] | np.ndarray) -> pd.Series:
        s = pd.Series(values)
        s = pd.to_numeric(s, errors="coerce")
        s = s.dropna()
        return s

    def _build_pairwise_result(
        self,
        bundle: GroupComparisonSampleBundle | CorrelationSampleBundle,
        *,
        statistic: float,
        p_value: float,
        target: str,
        key: str,
        metadata: dict[str, Any] | None = None,
    ) -> PairwiseStatisticalResult:
        if isinstance(bundle, GroupComparisonSampleBundle):
            n_x = bundle.n_x
            n_y = bundle.n_y
        else:
            n_x = bundle.n_x
            n_y = bundle.n_y

        return PairwiseStatisticalResult(
            statistic=float(statistic),
            p_value=float(p_value),
            target=target,
            key=key,
            test_name=self.test_name,
            n_observations=bundle.n_observations,
            metadata=metadata or {},
            n_x=n_x,
            n_y=n_y,
            x_name=bundle.x_name,
            y_name=bundle.y_name,
        )

    def _build_one_way_anova_result(
        self,
        bundle: OneWayANOVASampleBundle,
        *,
        statistic: float,
        p_value: float,
        target: str,
        key: str,
        df_between: int,
        df_within: int,
        eta_squared: float | None,
        metadata: dict[str, Any] | None = None,
    ) -> OneWayANOVAResult:
        return OneWayANOVAResult(
            statistic=float(statistic),
            p_value=float(p_value),
            target=target,
            key=key,
            test_name=self.test_name,
            n_observations=bundle.n_observations,
            metadata=metadata or {},
            factor_name=bundle.factor_name,
            group_sizes=bundle.group_sizes,
            df_between=int(df_between),
            df_within=int(df_within),
            eta_squared=None if eta_squared is None else float(eta_squared),
        )