from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ======================================================================================
#                                  BASE RESULTS
# ======================================================================================

@dataclass(frozen=True, kw_only=True)
class StatisticalResult(ABC):
    target: str
    key: str
    test_name: str

    @property
    def result_kind(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class ScalarStatisticalResult(StatisticalResult):
    statistic: float
    p_value: float
    n_observations: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def result_kind(self) -> str:
        return "scalar"


@dataclass(frozen=True, kw_only=True)
class PairwiseStatisticalResult(ScalarStatisticalResult):
    n_x: int = 0
    n_y: int = 0
    x_name: str = ""
    y_name: str = ""

    @property
    def label(self) -> str:
        return f"{self.x_name} vs {self.y_name}"

    @property
    def result_kind(self) -> str:
        return "pairwise"


@dataclass(frozen=True, kw_only=True)
class OneWayANOVAResult(ScalarStatisticalResult):
    factor_name: str = ""
    group_sizes: dict[str, int] = field(default_factory=dict)
    df_between: int = 0
    df_within: int = 0
    eta_squared: float | None = None

    @property
    def label(self) -> str:
        return f"{self.target} ~ {self.factor_name}"

    @property
    def result_kind(self) -> str:
        return "one_way_anova"


@dataclass(frozen=True, kw_only=True)
class FactorialEffectResult:
    effect_name: str
    statistic: float
    p_value: float
    df_num: float
    df_den: float
    sum_sq: float | None = None
    mean_sq: float | None = None


@dataclass(frozen=True, kw_only=True)
class TwoWayANOVAResult(StatisticalResult):
    dependent_name: str
    factor_a_name: str
    factor_b_name: str
    effects: dict[str, FactorialEffectResult]
    n_observations: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def result_kind(self) -> str:
        return "factorial"

    @property
    def label(self) -> str:
        return f"{self.dependent_name} ~ {self.factor_a_name} * {self.factor_b_name}"


# ======================================================================================
#                               CORRECTED RESULTS
# ======================================================================================

@dataclass(frozen=True, kw_only=True)
class CorrectedScalarMixin:
    p_value_corrected: float = 1.0
    correction_method: str = ""
    alpha: float = 0.05
    reject_null: bool = False


@dataclass(frozen=True, kw_only=True)
class CorrectedScalarStatisticalResult(ScalarStatisticalResult, CorrectedScalarMixin):
    @property
    def result_kind(self) -> str:
        return "scalar_corrected"


@dataclass(frozen=True, kw_only=True)
class CorrectedPairwiseStatisticalResult(PairwiseStatisticalResult, CorrectedScalarMixin):
    @property
    def result_kind(self) -> str:
        return "pairwise_corrected"


@dataclass(frozen=True, kw_only=True)
class CorrectedOneWayANOVAResult(OneWayANOVAResult, CorrectedScalarMixin):
    @property
    def result_kind(self) -> str:
        return "one_way_anova_corrected"


# ======================================================================================
#                                POST-HOC RESULTS
# ======================================================================================

@dataclass(frozen=True, kw_only=True)
class PostHocComparisonResult:
    group_a: str
    group_b: str
    mean_diff: float
    p_value_adjusted: float
    conf_low: float
    conf_high: float
    reject_null: bool


@dataclass(frozen=True, kw_only=True)
class PostHocResultSet:
    key: str
    target: str
    method: str
    comparisons: list[PostHocComparisonResult]

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for cmp in self.comparisons:
            rows.append(
                {
                    "key": self.key,
                    "target": self.target,
                    "method": self.method,
                    "group_a": cmp.group_a,
                    "group_b": cmp.group_b,
                    "mean_diff": cmp.mean_diff,
                    "p_value_adjusted": cmp.p_value_adjusted,
                    "conf_low": cmp.conf_low,
                    "conf_high": cmp.conf_high,
                    "reject_null": cmp.reject_null,
                }
            )
        return pd.DataFrame(rows)


# ======================================================================================
#                                  RESULT SETS
# ======================================================================================

@dataclass(frozen=True, kw_only=True)
class StatisticalResultSet:
    results: dict[str, StatisticalResult]
    test_name: str
    target: str

    def keys(self) -> list[str]:
        return list(self.results.keys())

    def is_scalar_only(self) -> bool:
        return all(isinstance(result, ScalarStatisticalResult) for result in self.results.values())

    def scalar_p_values(self) -> dict[str, float]:
        if not self.is_scalar_only():
            raise TypeError("scalar_p_values is only available for scalar result sets")
        return {
            key: result.p_value
            for key, result in self.results.items()
            if isinstance(result, ScalarStatisticalResult)
        }

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for key, result in self.results.items():
            if isinstance(result, PairwiseStatisticalResult):
                rows.append(
                    {
                        "key": key,
                        "target": result.target,
                        "test_name": result.test_name,
                        "result_kind": result.result_kind,
                        "statistic": result.statistic,
                        "p_value": result.p_value,
                        "n_observations": result.n_observations,
                        "n_x": result.n_x,
                        "n_y": result.n_y,
                        "x_name": result.x_name,
                        "y_name": result.y_name,
                        "label": result.label,
                        **result.metadata,
                    }
                )
            elif isinstance(result, OneWayANOVAResult):
                rows.append(
                    {
                        "key": key,
                        "target": result.target,
                        "test_name": result.test_name,
                        "result_kind": result.result_kind,
                        "statistic": result.statistic,
                        "p_value": result.p_value,
                        "n_observations": result.n_observations,
                        "factor_name": result.factor_name,
                        "group_sizes": result.group_sizes,
                        "df_between": result.df_between,
                        "df_within": result.df_within,
                        "eta_squared": result.eta_squared,
                        "label": result.label,
                        **result.metadata,
                    }
                )
            elif isinstance(result, TwoWayANOVAResult):
                for effect_name, effect in result.effects.items():
                    rows.append(
                        {
                            "key": key,
                            "target": result.target,
                            "test_name": result.test_name,
                            "result_kind": result.result_kind,
                            "dependent_name": result.dependent_name,
                            "factor_a_name": result.factor_a_name,
                            "factor_b_name": result.factor_b_name,
                            "effect_name": effect_name,
                            "statistic": effect.statistic,
                            "p_value": effect.p_value,
                            "df_num": effect.df_num,
                            "df_den": effect.df_den,
                            "sum_sq": effect.sum_sq,
                            "mean_sq": effect.mean_sq,
                            "n_observations": result.n_observations,
                            "label": result.label,
                            **result.metadata,
                        }
                    )
            elif isinstance(result, ScalarStatisticalResult):
                rows.append(
                    {
                        "key": key,
                        "target": result.target,
                        "test_name": result.test_name,
                        "result_kind": result.result_kind,
                        "statistic": result.statistic,
                        "p_value": result.p_value,
                        "n_observations": result.n_observations,
                        **result.metadata,
                    }
                )
            else:
                raise TypeError(f"Unsupported result type: {type(result).__name__}")

        return pd.DataFrame(rows)


@dataclass(frozen=True, kw_only=True)
class CorrectedStatisticalResultSet:
    results: dict[
        str,
        CorrectedScalarStatisticalResult
        | CorrectedPairwiseStatisticalResult
        | CorrectedOneWayANOVAResult
    ]
    test_name: str
    target: str
    correction_method: str
    alpha: float
    family_name: str

    def keys(self) -> list[str]:
        return list(self.results.keys())

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for key, result in self.results.items():
            base = {
                "key": key,
                "target": result.target,
                "test_name": result.test_name,
                "result_kind": result.result_kind,
                "statistic": result.statistic,
                "p_value": result.p_value,
                "p_value_corrected": result.p_value_corrected,
                "correction_method": result.correction_method,
                "alpha": result.alpha,
                "reject_null": result.reject_null,
                "n_observations": result.n_observations,
                **result.metadata,
            }

            if isinstance(result, CorrectedPairwiseStatisticalResult):
                base.update(
                    {
                        "n_x": result.n_x,
                        "n_y": result.n_y,
                        "x_name": result.x_name,
                        "y_name": result.y_name,
                        "label": result.label,
                    }
                )
            elif isinstance(result, CorrectedOneWayANOVAResult):
                base.update(
                    {
                        "factor_name": result.factor_name,
                        "group_sizes": result.group_sizes,
                        "df_between": result.df_between,
                        "df_within": result.df_within,
                        "eta_squared": result.eta_squared,
                        "label": result.label,
                    }
                )

            rows.append(base)

        return pd.DataFrame(rows)


@dataclass(frozen=True, kw_only=True)
class StatisticalAnalysisOutcome:
    primary_results: StatisticalResultSet
    corrected_results: CorrectedStatisticalResultSet | None = None
    posthoc_results: dict[str, PostHocResultSet] | None = None

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        payload = {
            "primary_results": self.primary_results.to_dataframe(),
        }

        if self.corrected_results is not None:
            payload["corrected_results"] = self.corrected_results.to_dataframe()

        if self.posthoc_results is not None:
            payload["posthoc_results"] = (
                pd.concat(
                    [result.to_dataframe() for result in self.posthoc_results.values()],
                    ignore_index=True,
                )
                if self.posthoc_results
                else pd.DataFrame()
            )

        return payload