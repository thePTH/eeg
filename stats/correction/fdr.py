from __future__ import annotations

from statsmodels.stats.multitest import multipletests

from stats.correction.base import MultipleComparisonCorrector
from stats.results import (
    CorrectedOneWayANOVAResult,
    CorrectedPairwiseStatisticalResult,
    CorrectedScalarStatisticalResult,
    CorrectedStatisticalResultSet,
    OneWayANOVAResult,
    PairwiseStatisticalResult,
    ScalarStatisticalResult,
    StatisticalResultSet,
)


class FDRCorrector(MultipleComparisonCorrector):
    method_name = "fdr_bh"

    def correct(
        self,
        result_set: StatisticalResultSet,
        *,
        alpha: float,
        family_name: str,
    ) -> CorrectedStatisticalResultSet:
        if not result_set.is_scalar_only():
            raise TypeError(
                "FDR correction can only be applied to scalar result sets "
                "(e.g. t-test, rank-sum, spearman, one-way ANOVA)."
            )

        keys = result_set.keys()
        p_values = [result_set.results[key].p_value for key in keys]

        reject, pvals_corrected, _, _ = multipletests(
            pvals=p_values,
            alpha=alpha,
            method=self.method_name,
        )

        corrected_results = {}

        for key, rej, p_corr in zip(keys, reject.tolist(), pvals_corrected.tolist()):
            original = result_set.results[key]

            if isinstance(original, PairwiseStatisticalResult):
                corrected = CorrectedPairwiseStatisticalResult(
                    statistic=original.statistic,
                    p_value=original.p_value,
                    target=original.target,
                    key=original.key,
                    test_name=original.test_name,
                    n_observations=original.n_observations,
                    metadata=dict(original.metadata),
                    n_x=original.n_x,
                    n_y=original.n_y,
                    x_name=original.x_name,
                    y_name=original.y_name,
                    p_value_corrected=float(p_corr),
                    correction_method=self.method_name,
                    alpha=float(alpha),
                    reject_null=bool(rej),
                )
            elif isinstance(original, OneWayANOVAResult):
                corrected = CorrectedOneWayANOVAResult(
                    statistic=original.statistic,
                    p_value=original.p_value,
                    target=original.target,
                    key=original.key,
                    test_name=original.test_name,
                    n_observations=original.n_observations,
                    metadata=dict(original.metadata),
                    factor_name=original.factor_name,
                    group_sizes=dict(original.group_sizes),
                    df_between=original.df_between,
                    df_within=original.df_within,
                    eta_squared=original.eta_squared,
                    p_value_corrected=float(p_corr),
                    correction_method=self.method_name,
                    alpha=float(alpha),
                    reject_null=bool(rej),
                )
            elif isinstance(original, ScalarStatisticalResult):
                corrected = CorrectedScalarStatisticalResult(
                    statistic=original.statistic,
                    p_value=original.p_value,
                    target=original.target,
                    key=original.key,
                    test_name=original.test_name,
                    n_observations=original.n_observations,
                    metadata=dict(original.metadata),
                    p_value_corrected=float(p_corr),
                    correction_method=self.method_name,
                    alpha=float(alpha),
                    reject_null=bool(rej),
                )
            else:
                raise TypeError(
                    f"Unsupported scalar result type for FDR correction: {type(original).__name__}"
                )

            corrected_results[key] = corrected

        return CorrectedStatisticalResultSet(
            results=corrected_results,
            test_name=result_set.test_name,
            target=result_set.target,
            correction_method=self.method_name,
            alpha=float(alpha),
            family_name=family_name,
        )