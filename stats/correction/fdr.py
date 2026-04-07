from __future__ import annotations

from statsmodels.stats.multitest import multipletests

from stats.results import  StatisticalTestResultSet, CorrectedStatisticalTestResult, CorrectedStatisticalTestResultSet

from collections.abc import Mapping
from typing import Union, Any

class FDRCorrector:
    @staticmethod
    def correct(result_set:Union[StatisticalTestResultSet, dict[Any, StatisticalTestResultSet]], alpha: float = 0.05) -> Union[CorrectedStatisticalTestResultSet, dict[Any, CorrectedStatisticalTestResultSet]]:
        if isinstance(result_set, Mapping):
            return {
                key: FDRCorrector.correct(sub_result_set, alpha=alpha)
                for key, sub_result_set in result_set.items()
            }

        keys = list(result_set.results.keys())
        p_values = [result_set.results[key].p_value for key in keys]

        rejected, corrected_p_values, _, _ = multipletests(
            p_values,
            alpha=alpha,
            method="fdr_bh",
        )

        corrected_results = {}

        for key, reject, p_value_corrected in zip(keys, rejected, corrected_p_values):
            raw_result = result_set.results[key]

            corrected_results[key] = CorrectedStatisticalTestResult(
                statistic=raw_result.statistic,
                p_value=raw_result.p_value,
                target=raw_result.target,
                key=raw_result.key,
                test_name=raw_result.test_name,
                n_x=raw_result.n_x,
                n_y=raw_result.n_y,
                x_name=raw_result.x_name,
                y_name=raw_result.y_name,
                p_value_corrected=float(p_value_corrected),
                correction_method="fdr_bh",
                alpha=alpha,
                reject_null=bool(reject),
            )

        return CorrectedStatisticalTestResultSet(
            results=corrected_results,
            test_name=result_set.test_name,
            target=result_set.target,
            correction_method="fdr_bh",
            alpha=alpha,
        )