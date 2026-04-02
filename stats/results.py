from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class StatisticalTestResult:
    statistic: float
    p_value: float
    target: str
    key: str
    test_name: str
    n_x: int
    n_y: int
    x_name: str
    y_name: str

    @property
    def label(self) -> str:
        """
        Description standardisée de ce qui est testé.
        Exemple :
        - 'reaction_time (patients) vs reaction_time (controls)'
        - 'theta_power (Fz) vs age'
        """
        return f"{self.x_name} vs {self.y_name}"


@dataclass(frozen=True)
class CorrectedStatisticalTestResult(StatisticalTestResult):
    p_value_corrected: float
    correction_method: str
    alpha: float
    reject_null: bool


@dataclass(frozen=True)
class StatisticalTestResultSet:
    results: dict[str, StatisticalTestResult]
    test_name: str
    target: str

    def p_values(self) -> dict[str, float]:
        return {key: result.p_value for key, result in self.results.items()}

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for key, result in self.results.items():
            rows.append({
                "key": key,
                "target": result.target,
                "statistic": result.statistic,
                "p_value": result.p_value,
                "test_name": result.test_name,
                "n_x": result.n_x,
                "n_y": result.n_y,
                "x_name": result.x_name,
                "y_name": result.y_name,
                "label": result.label,
            })
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class CorrectedStatisticalTestResultSet:
    results: dict[str, CorrectedStatisticalTestResult]
    test_name: str
    target: str
    correction_method: str
    alpha: float

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for key, result in self.results.items():
            rows.append({
                "key": key,
                "target": result.target,
                "statistic": result.statistic,
                "p_value": result.p_value,
                "p_value_corrected": result.p_value_corrected,
                "correction_method": result.correction_method,
                "alpha": result.alpha,
                "reject_null": result.reject_null,
                "test_name": result.test_name,
                "n_x": result.n_x,
                "n_y": result.n_y,
                "x_name": result.x_name,
                "y_name": result.y_name,
                "label": result.label,
            })
        return pd.DataFrame(rows)