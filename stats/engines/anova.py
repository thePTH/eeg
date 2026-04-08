from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols

from stats.bundles import OneWayANOVASampleBundle, SampleBundle, TwoWayANOVASampleBundle
from stats.engines.base import StatisticalTestEngine
from stats.results import FactorialEffectResult, TwoWayANOVAResult


class OneWayANOVAEngine(StatisticalTestEngine):
    test_name = "one-way-anova"

    def compute(self, bundle: SampleBundle, *, target: str, key: str):
        if not isinstance(bundle, OneWayANOVASampleBundle):
            raise TypeError("OneWayANOVAEngine expects a OneWayANOVASampleBundle")

        df = pd.DataFrame(
            {
                "value": self._to_numeric_series(bundle.values),
                "group": bundle.groups.astype(str),
            }
        ).dropna()

        grouped = [group_df["value"].to_numpy() for _, group_df in df.groupby("group")]

        if len(grouped) < 2:
            raise ValueError("One-way ANOVA requires at least two non-empty groups")

        statistic, p_value = f_oneway(*grouped)

        # Degrees of freedom
        k = len(grouped)
        n = int(len(df))
        df_between = k - 1
        df_within = n - k

        # eta-squared
        grand_mean = float(df["value"].mean())
        ss_between = 0.0
        ss_total = float(((df["value"] - grand_mean) ** 2).sum())

        for _, group_df in df.groupby("group"):
            n_g = len(group_df)
            mean_g = float(group_df["value"].mean())
            ss_between += n_g * (mean_g - grand_mean) ** 2

        eta_squared = None
        if ss_total > 0:
            eta_squared = ss_between / ss_total

        return self._build_one_way_anova_result(
            bundle,
            statistic=float(statistic),
            p_value=float(p_value),
            target=target,
            key=key,
            df_between=df_between,
            df_within=df_within,
            eta_squared=eta_squared,
        )


class TwoWayANOVAEngine(StatisticalTestEngine):
    test_name = "two-way-anova"

    def compute(self, bundle: SampleBundle, *, target: str, key: str):
        if not isinstance(bundle, TwoWayANOVASampleBundle):
            raise TypeError("TwoWayANOVAEngine expects a TwoWayANOVASampleBundle")

        df = bundle.dataframe.copy()
        df = df.rename(
            columns={
                "value": "y",
                bundle.factor_a_name: "factor_a",
                bundle.factor_b_name: "factor_b",
            }
        )
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna()

        if df.empty:
            raise ValueError("Two-way ANOVA received an empty dataframe after cleaning")

        model = ols("y ~ C(factor_a) * C(factor_b)", data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        effects: dict[str, FactorialEffectResult] = {}

        row_mapping = {
            "C(factor_a)": bundle.factor_a_name,
            "C(factor_b)": bundle.factor_b_name,
            "C(factor_a):C(factor_b)": f"{bundle.factor_a_name}*{bundle.factor_b_name}",
        }

        for row_name, effect_name in row_mapping.items():
            if row_name not in anova_table.index:
                continue

            row = anova_table.loc[row_name]

            effects[effect_name] = FactorialEffectResult(
                effect_name=effect_name,
                statistic=float(row["F"]),
                p_value=float(row["PR(>F)"]),
                df_num=float(row["df"]),
                df_den=float(anova_table.loc["Residual", "df"]),
                sum_sq=float(row["sum_sq"]) if "sum_sq" in row else None,
                mean_sq=float(row["sum_sq"] / row["df"]) if row["df"] != 0 else None,
            )

        return TwoWayANOVAResult(
            target=target,
            key=key,
            test_name=self.test_name,
            dependent_name=bundle.dependent_name,
            factor_a_name=bundle.factor_a_name,
            factor_b_name=bundle.factor_b_name,
            effects=effects,
            n_observations=bundle.n_observations,
            metadata={"cell_sizes": bundle.cell_sizes},
        )