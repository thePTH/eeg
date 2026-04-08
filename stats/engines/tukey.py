from __future__ import annotations

import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from stats.bundles import OneWayANOVASampleBundle
from stats.results import PostHocComparisonResult, PostHocResultSet


class TukeyHSDPostHocEngine:
    """
    Post-hoc Tukey HSD pour une ANOVA à un facteur.

    On garde cette classe séparée des StatisticalTestEngine
    car un post-hoc ne renvoie pas un résultat scalaire unique.
    """

    method_name = "tukey-hsd"

    def compute(
        self,
        bundle: OneWayANOVASampleBundle,
        *,
        target: str,
        key: str,
        alpha: float = 0.05,
    ) -> PostHocResultSet:
        df = pd.DataFrame(
            {
                "value": pd.to_numeric(bundle.values, errors="coerce"),
                "group": bundle.groups.astype(str),
            }
        ).dropna()

        if df["group"].nunique() < 2:
            raise ValueError("Tukey HSD requires at least two groups")

        tukey = pairwise_tukeyhsd(
            endog=df["value"].to_numpy(),
            groups=df["group"].to_numpy(),
            alpha=alpha,
        )

        comparisons: list[PostHocComparisonResult] = []

        # statsmodels renvoie un SimpleTable ; on exploite directement les données
        raw_rows = tukey.summary().data[1:]

        for row in raw_rows:
            group_a, group_b, mean_diff, p_adj, conf_low, conf_high, reject = row

            comparisons.append(
                PostHocComparisonResult(
                    group_a=str(group_a),
                    group_b=str(group_b),
                    mean_diff=float(mean_diff),
                    p_value_adjusted=float(p_adj),
                    conf_low=float(conf_low),
                    conf_high=float(conf_high),
                    reject_null=bool(reject),
                )
            )

        return PostHocResultSet(
            key=key,
            target=target,
            method=self.method_name,
            comparisons=comparisons,
        )