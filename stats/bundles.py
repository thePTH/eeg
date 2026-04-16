from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any

import pandas as pd

from stats.queries.base import StatisticalQuery
from stats.queries.eeg_queries import (
    EEGFeatureCorrelationQuery,
    EEGFeatureFactorialQuery,
    EEGFeatureGroupComparisonQuery,
)
from stats.queries.ppc_queries import (
    PPCBandCorrelationQuery,
    PPCBandFactorialQuery,
    PPCBandGroupComparisonQuery,
)
from stats.queries.psd_queries import (
    PSDBandCorrelationQuery,
    PSDBandFactorialQuery,
    PSDBandGroupComparisonQuery,
)
from stats.queries.subject_queries import (
    SubjectCorrelationQuery,
    SubjectFactorialQuery,
    SubjectGroupComparisonQuery,
)


# ======================================================================================
#                                      BUNDLES
# ======================================================================================

@dataclass(frozen=True, slots=True)
class SampleBundle(ABC):
    """
    Bundle générique d'échantillons prêt à être consommé par un engine statistique.
    """

    @property
    def n_observations(self) -> int:
        raise NotImplementedError

    @property
    def label(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class GroupComparisonSampleBundle(SampleBundle):
    """
    Bundle pour une comparaison entre deux groupes.
    """
    x: pd.Series
    y: pd.Series
    x_name: str
    y_name: str

    @property
    def n_x(self) -> int:
        return int(len(self.x))

    @property
    def n_y(self) -> int:
        return int(len(self.y))

    @property
    def n_observations(self) -> int:
        return self.n_x + self.n_y

    @property
    def label(self) -> str:
        return f"{self.x_name} vs {self.y_name}"


@dataclass(frozen=True, slots=True)
class CorrelationSampleBundle(SampleBundle):
    """
    Bundle pour une corrélation entre deux variables.
    """
    x: pd.Series
    y: pd.Series
    x_name: str
    y_name: str

    @property
    def n_x(self) -> int:
        return int(len(self.x))

    @property
    def n_y(self) -> int:
        return int(len(self.y))

    @property
    def n_observations(self) -> int:
        return min(self.n_x, self.n_y)

    @property
    def label(self) -> str:
        return f"{self.x_name} vs {self.y_name}"


@dataclass(frozen=True, slots=True)
class OneWayANOVASampleBundle(SampleBundle):
    """
    Bundle pour une ANOVA à un facteur.
    """
    values: pd.Series
    groups: pd.Series
    dependent_name: str
    factor_name: str

    @property
    def n_observations(self) -> int:
        return int(len(self.values))

    @property
    def label(self) -> str:
        return f"{self.dependent_name} ~ {self.factor_name}"

    @property
    def group_sizes(self) -> dict[str, int]:
        grouped = self.groups.astype(str).value_counts(sort=False)
        return {str(level): int(count) for level, count in grouped.items()}


@dataclass(frozen=True, slots=True)
class TwoWayANOVASampleBundle(SampleBundle):
    """
    Bundle pour une ANOVA à deux facteurs.
    """
    dataframe: pd.DataFrame
    dependent_name: str
    factor_a_name: str
    factor_b_name: str

    @property
    def n_observations(self) -> int:
        return int(len(self.dataframe))

    @property
    def label(self) -> str:
        return f"{self.dependent_name} ~ {self.factor_a_name} * {self.factor_b_name}"

    @property
    def cell_sizes(self) -> dict[str, int]:
        grouped = (
            self.dataframe
            .groupby([self.factor_a_name, self.factor_b_name], dropna=False, observed=False)
            .size()
        )
        return {f"{str(a)}__{str(b)}": int(n) for (a, b), n in grouped.items()}


# ======================================================================================
#                                  FACTORY HELPERS
# ======================================================================================

class SampleBundleFactory:
    """
    Factory qui transforme :
    - une query métier
    - un FeaturesDataset
    en bundle prêt pour un engine statistique.

    Philosophie
    -----------
    Cette factory ne reconstruit pas les données métier.
    Elle s'appuie sur les vues déjà exposées par `FeaturesDataset` :
    - dataset.subject_dataframe
    - dataset.long_dataframe
    - dataset.long_psd_dataframe
    - dataset.long_ppc_dataframe
    """

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @staticmethod
    def list_keys(query: StatisticalQuery, dataset: Any) -> list[str]:
        """
        Retourne les clés statistiques à itérer pour la query.

        Exemples
        --------
        - ["subject_level"]
        - ["Fp1", "Fp2", ...]
        - ["Fp1__Fp2", "Fp1__F3", ...]
        """
        match query:
            # ==========================================================
            # SUBJECT
            # ==========================================================
            case SubjectGroupComparisonQuery():
                return ["subject_level"]

            case SubjectCorrelationQuery():
                return ["subject_level"]

            case SubjectFactorialQuery():
                return ["subject_level"]

            # ==========================================================
            # EEG FEATURES
            # ==========================================================
            case EEGFeatureGroupComparisonQuery(scope="single_channel", channel=channel):
                return [channel]

            case EEGFeatureGroupComparisonQuery(scope="all_channels"):
                return list(dataset.ch_names)

            case EEGFeatureCorrelationQuery(scope="single_channel", channel=channel):
                return [channel]

            case EEGFeatureCorrelationQuery(scope="all_channels"):
                return list(dataset.ch_names)

            case EEGFeatureFactorialQuery(scope="single_channel", channel=channel):
                return [channel]

            case EEGFeatureFactorialQuery(scope="all_channels"):
                return list(dataset.ch_names)

            # ==========================================================
            # PSD
            # ==========================================================
            case PSDBandGroupComparisonQuery(scope="single_channel", channel=channel):
                return [channel]

            case PSDBandGroupComparisonQuery(scope="all_channels"):
                return list(dataset.ch_names)

            case PSDBandCorrelationQuery(scope="single_channel", channel=channel):
                return [channel]

            case PSDBandCorrelationQuery(scope="all_channels"):
                return list(dataset.ch_names)

            case PSDBandFactorialQuery(scope="single_channel", channel=channel):
                return [channel]

            case PSDBandFactorialQuery(scope="all_channels"):
                return list(dataset.ch_names)

            # ==========================================================
            # PPC
            # ==========================================================
            case PPCBandGroupComparisonQuery(scope="single_edge", edge=edge):
                return [edge]

            case PPCBandGroupComparisonQuery(scope="all_edges"):
                return list(dataset.ppc_edge_keys)

            case PPCBandCorrelationQuery(scope="single_edge", edge=edge):
                return [edge]

            case PPCBandCorrelationQuery(scope="all_edges"):
                return list(dataset.ppc_edge_keys)

            case PPCBandFactorialQuery(scope="single_edge", edge=edge):
                return [edge]

            case PPCBandFactorialQuery(scope="all_edges"):
                return list(dataset.ppc_edge_keys)

            case _:
                raise ValueError(
                    f"Unsupported query type for key listing: {type(query).__name__}"
                )

    @staticmethod
    def build(query: StatisticalQuery, dataset: Any, key: str) -> SampleBundle:
        """
        Construit le bundle associé à une query et une clé.
        """
        match query:
            # ==========================================================
            # SUBJECT
            # ==========================================================
            case SubjectGroupComparisonQuery():
                return SampleBundleFactory._build_subject_group_comparison_bundle(query, dataset)

            case SubjectCorrelationQuery():
                return SampleBundleFactory._build_subject_correlation_bundle(query, dataset)

            case SubjectFactorialQuery():
                return SampleBundleFactory._build_subject_factorial_bundle(query, dataset)

            # ==========================================================
            # EEG FEATURES
            # ==========================================================
            case EEGFeatureGroupComparisonQuery():
                return SampleBundleFactory._build_eeg_group_comparison_bundle(query, dataset, key)

            case EEGFeatureCorrelationQuery():
                return SampleBundleFactory._build_eeg_correlation_bundle(query, dataset, key)

            case EEGFeatureFactorialQuery():
                return SampleBundleFactory._build_eeg_factorial_bundle(query, dataset, key)

            # ==========================================================
            # PSD
            # ==========================================================
            case PSDBandGroupComparisonQuery():
                return SampleBundleFactory._build_psd_group_comparison_bundle(query, dataset, key)

            case PSDBandCorrelationQuery():
                return SampleBundleFactory._build_psd_correlation_bundle(query, dataset, key)

            case PSDBandFactorialQuery():
                return SampleBundleFactory._build_psd_factorial_bundle(query, dataset, key)

            # ==========================================================
            # PPC
            # ==========================================================
            case PPCBandGroupComparisonQuery():
                return SampleBundleFactory._build_ppc_group_comparison_bundle(query, dataset, key)

            case PPCBandCorrelationQuery():
                return SampleBundleFactory._build_ppc_correlation_bundle(query, dataset, key)

            case PPCBandFactorialQuery():
                return SampleBundleFactory._build_ppc_factorial_bundle(query, dataset, key)

            case _:
                raise ValueError(
                    f"Unsupported query type for bundle creation: {type(query).__name__}"
                )

    # ------------------------------------------------------------------
    # generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: list[str], *, context: str) -> None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise KeyError(
                f"Missing required columns {missing} in {context}. "
                f"Available columns are: {list(df.columns)}"
            )

    @staticmethod
    def _subset_and_dropna(
        df: pd.DataFrame,
        *,
        mask: pd.Series | None = None,
        columns: list[str],
    ) -> pd.DataFrame:
        """
        Sélectionne éventuellement un sous-ensemble de lignes puis garde
        uniquement les colonnes utiles avant `dropna()`.

        Cela réduit la mémoire temporaire et accélère les traitements.
        """
        if mask is not None:
            df = df.loc[mask, columns]
        else:
            df = df.loc[:, columns]
        return df.dropna()

    @staticmethod
    def _build_factorial_bundle_from_dataframe(
        *,
        df: pd.DataFrame,
        value_col: str,
        factors: tuple[str, ...],
        dependent_name: str,
    ) -> SampleBundle:
        """
        Construit automatiquement un bundle one-way ou two-way ANOVA
        à partir d'un DataFrame long déjà préparé.
        """
        SampleBundleFactory._require_columns(
            df,
            [value_col, *factors],
            context=f"factorial analysis for '{dependent_name}'",
        )

        if len(factors) == 1:
            factor = factors[0]
            sub = SampleBundleFactory._subset_and_dropna(
                df,
                columns=[value_col, factor],
            )

            if sub.empty:
                raise ValueError(
                    f"No valid observations found for one-way ANOVA on '{dependent_name}'."
                )

            return OneWayANOVASampleBundle(
                values=sub[value_col],
                groups=sub[factor].astype(str),
                dependent_name=dependent_name,
                factor_name=factor,
            )

        if len(factors) == 2:
            factor_a, factor_b = factors
            sub = SampleBundleFactory._subset_and_dropna(
                df,
                columns=[value_col, factor_a, factor_b],
            )

            if sub.empty:
                raise ValueError(
                    f"No valid observations found for two-way ANOVA on '{dependent_name}'."
                )

            sub = sub.rename(columns={value_col: "value"})

            return TwoWayANOVASampleBundle(
                dataframe=sub,
                dependent_name=dependent_name,
                factor_a_name=factor_a,
                factor_b_name=factor_b,
            )

        raise ValueError("Factorial bundle only supports one-way or two-way designs.")

    @staticmethod
    def _build_group_comparison_bundle_from_dataframe(
        *,
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        group_a: str,
        group_b: str,
        x_name: str,
        y_name: str,
        error_context: str,
    ) -> GroupComparisonSampleBundle:
        """
        Helper générique pour construire un bundle de comparaison de groupes.
        """
        SampleBundleFactory._require_columns(
            df,
            [value_col, group_col],
            context=error_context,
        )

        sub = SampleBundleFactory._subset_and_dropna(
            df,
            columns=[value_col, group_col],
        )

        if sub.empty:
            raise ValueError(f"No samples found for {error_context}.")

        x = sub.loc[sub[group_col] == group_a, value_col]
        y = sub.loc[sub[group_col] == group_b, value_col]

        if x.empty or y.empty:
            raise ValueError(
                f"No samples found for {error_context}: "
                f"group_a='{group_a}', group_b='{group_b}'."
            )

        return GroupComparisonSampleBundle(
            x=x,
            y=y,
            x_name=x_name,
            y_name=y_name,
        )

    @staticmethod
    def _build_correlation_bundle_from_dataframe(
        *,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        x_name: str,
        y_name: str,
        error_context: str,
    ) -> CorrelationSampleBundle:
        """
        Helper générique pour construire un bundle de corrélation.
        """
        SampleBundleFactory._require_columns(
            df,
            [x_col, y_col],
            context=error_context,
        )

        sub = SampleBundleFactory._subset_and_dropna(
            df,
            columns=[x_col, y_col],
        )

        if sub.empty:
            raise ValueError(f"No samples found for {error_context}.")

        return CorrelationSampleBundle(
            x=sub[x_col],
            y=sub[y_col],
            x_name=x_name,
            y_name=y_name,
        )

    # ------------------------------------------------------------------
    # subject bundles
    # ------------------------------------------------------------------

    @staticmethod
    def _build_subject_group_comparison_bundle(
        query: SubjectGroupComparisonQuery,
        dataset: Any,
    ) -> GroupComparisonSampleBundle:
        df = dataset.subject_dataframe

        return SampleBundleFactory._build_group_comparison_bundle_from_dataframe(
            df=df,
            value_col=query.variable,
            group_col=query.group_col,
            group_a=query.group_a,
            group_b=query.group_b,
            x_name=f"{query.variable} ({query.group_a})",
            y_name=f"{query.variable} ({query.group_b})",
            error_context=(
                f"subject-level comparison for variable='{query.variable}'"
            ),
        )

    @staticmethod
    def _build_subject_correlation_bundle(
        query: SubjectCorrelationQuery,
        dataset: Any,
    ) -> CorrelationSampleBundle:
        df = dataset.subject_dataframe

        return SampleBundleFactory._build_correlation_bundle_from_dataframe(
            df=df,
            x_col=query.x_variable,
            y_col=query.y_variable,
            x_name=query.x_variable,
            y_name=query.y_variable,
            error_context=(
                f"subject-level correlation: {query.x_variable} vs {query.y_variable}"
            ),
        )

    @staticmethod
    def _build_subject_factorial_bundle(
        query: SubjectFactorialQuery,
        dataset: Any,
    ) -> SampleBundle:
        df = dataset.subject_dataframe

        return SampleBundleFactory._build_factorial_bundle_from_dataframe(
            df=df,
            value_col=query.variable,
            factors=query.factor_names,
            dependent_name=query.variable,
        )

    # ------------------------------------------------------------------
    # EEG bundles
    # ------------------------------------------------------------------

    @staticmethod
    def _build_eeg_group_comparison_bundle(
        query: EEGFeatureGroupComparisonQuery,
        dataset: Any,
        key: str,
    ) -> GroupComparisonSampleBundle:
        df = dataset.long_dataframe

        SampleBundleFactory._require_columns(
            df,
            ["feature", "channel", "value", query.group_col],
            context=f"EEG group comparison for feature='{query.feature}'",
        )

        mask = (df["feature"] == query.feature) & (df["channel"] == key)
        sub = SampleBundleFactory._subset_and_dropna(
            df,
            mask=mask,
            columns=["value", query.group_col],
        )

        if sub.empty:
            raise ValueError(
                f"No samples found for EEG group comparison: "
                f"feature='{query.feature}', channel='{key}'."
            )

        x = sub.loc[sub[query.group_col] == query.group_a, "value"]
        y = sub.loc[sub[query.group_col] == query.group_b, "value"]

        if x.empty or y.empty:
            raise ValueError(
                f"No samples found for EEG group comparison: "
                f"feature='{query.feature}', channel='{key}', "
                f"group_a='{query.group_a}', group_b='{query.group_b}'."
            )

        return GroupComparisonSampleBundle(
            x=x,
            y=y,
            x_name=f"{query.feature} ({query.group_a}, {key})",
            y_name=f"{query.feature} ({query.group_b}, {key})",
        )

    @staticmethod
    def _build_eeg_correlation_bundle(
        query: EEGFeatureCorrelationQuery,
        dataset: Any,
        key: str,
    ) -> CorrelationSampleBundle:
        df = dataset.long_dataframe

        SampleBundleFactory._require_columns(
            df,
            ["feature", "channel", "value", query.covariate],
            context=f"EEG correlation for feature='{query.feature}'",
        )

        mask = (df["feature"] == query.feature) & (df["channel"] == key)
        sub = SampleBundleFactory._subset_and_dropna(
            df,
            mask=mask,
            columns=["value", query.covariate],
        )

        if sub.empty:
            raise ValueError(
                f"No samples found for EEG correlation: "
                f"feature='{query.feature}', channel='{key}', covariate='{query.covariate}'."
            )

        return CorrelationSampleBundle(
            x=sub["value"],
            y=sub[query.covariate],
            x_name=f"{query.feature} ({key})",
            y_name=query.covariate,
        )

    @staticmethod
    def _build_eeg_factorial_bundle(
        query: EEGFeatureFactorialQuery,
        dataset: Any,
        key: str,
    ) -> SampleBundle:
        df = dataset.long_dataframe

        SampleBundleFactory._require_columns(
            df,
            ["feature", "channel", "value", *query.factor_names],
            context=f"EEG factorial analysis for feature='{query.feature}'",
        )

        mask = (df["feature"] == query.feature) & (df["channel"] == key)
        sub = df.loc[mask, ["value", *query.factor_names]]

        return SampleBundleFactory._build_factorial_bundle_from_dataframe(
            df=sub,
            value_col="value",
            factors=query.factor_names,
            dependent_name=f"{query.feature} ({key})",
        )

    # ------------------------------------------------------------------
    # PSD bundles
    # ------------------------------------------------------------------

    @staticmethod
    def _build_psd_group_comparison_bundle(
        query: PSDBandGroupComparisonQuery,
        dataset: Any,
        key: str,
    ) -> GroupComparisonSampleBundle:
        df = dataset.long_psd_dataframe

        SampleBundleFactory._require_columns(
            df,
            ["band", "channel", "value", query.group_col],
            context=f"PSD group comparison for band='{query.band}'",
        )

        mask = (df["band"] == query.band) & (df["channel"] == key)
        sub = SampleBundleFactory._subset_and_dropna(
            df,
            mask=mask,
            columns=["value", query.group_col],
        )

        if sub.empty:
            raise ValueError(
                f"No samples found for PSD group comparison: "
                f"band='{query.band}', channel='{key}'."
            )

        x = sub.loc[sub[query.group_col] == query.group_a, "value"]
        y = sub.loc[sub[query.group_col] == query.group_b, "value"]

        if x.empty or y.empty:
            raise ValueError(
                f"No samples found for PSD group comparison: "
                f"band='{query.band}', channel='{key}', "
                f"group_a='{query.group_a}', group_b='{query.group_b}'."
            )

        return GroupComparisonSampleBundle(
            x=x,
            y=y,
            x_name=f"{query.band} ({query.group_a}, {key})",
            y_name=f"{query.band} ({query.group_b}, {key})",
        )

    @staticmethod
    def _build_psd_correlation_bundle(
        query: PSDBandCorrelationQuery,
        dataset: Any,
        key: str,
    ) -> CorrelationSampleBundle:
        df = dataset.long_psd_dataframe

        SampleBundleFactory._require_columns(
            df,
            ["band", "channel", "value", query.covariate],
            context=f"PSD correlation for band='{query.band}'",
        )

        mask = (df["band"] == query.band) & (df["channel"] == key)
        sub = SampleBundleFactory._subset_and_dropna(
            df,
            mask=mask,
            columns=["value", query.covariate],
        )

        if sub.empty:
            raise ValueError(
                f"No samples found for PSD correlation: "
                f"band='{query.band}', channel='{key}', covariate='{query.covariate}'."
            )

        return CorrelationSampleBundle(
            x=sub["value"],
            y=sub[query.covariate],
            x_name=f"{query.band} ({key})",
            y_name=query.covariate,
        )

    @staticmethod
    def _build_psd_factorial_bundle(
        query: PSDBandFactorialQuery,
        dataset: Any,
        key: str,
    ) -> SampleBundle:
        df = dataset.long_psd_dataframe

        SampleBundleFactory._require_columns(
            df,
            ["band", "channel", "value", *query.factor_names],
            context=f"PSD factorial analysis for band='{query.band}'",
        )

        mask = (df["band"] == query.band) & (df["channel"] == key)
        sub = df.loc[mask, ["value", *query.factor_names]]

        return SampleBundleFactory._build_factorial_bundle_from_dataframe(
            df=sub,
            value_col="value",
            factors=query.factor_names,
            dependent_name=f"{query.band} ({key})",
        )

    # ------------------------------------------------------------------
    # PPC bundles
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ppc_group_comparison_bundle(
        query: PPCBandGroupComparisonQuery,
        dataset: Any,
        key: str,
    ) -> GroupComparisonSampleBundle:
        df = dataset.long_ppc_dataframe

        SampleBundleFactory._require_columns(
            df,
            ["band", "edge", "value", query.group_col],
            context=f"PPC group comparison for band='{query.band}'",
        )

        mask = (df["band"] == query.band) & (df["edge"] == key)
        sub = SampleBundleFactory._subset_and_dropna(
            df,
            mask=mask,
            columns=["value", query.group_col],
        )

        if sub.empty:
            raise ValueError(
                f"No samples found for PPC group comparison: "
                f"band='{query.band}', edge='{key}'."
            )

        x = sub.loc[sub[query.group_col] == query.group_a, "value"]
        y = sub.loc[sub[query.group_col] == query.group_b, "value"]

        if x.empty or y.empty:
            raise ValueError(
                f"No samples found for PPC group comparison: "
                f"band='{query.band}', edge='{key}', "
                f"group_a='{query.group_a}', group_b='{query.group_b}'."
            )

        return GroupComparisonSampleBundle(
            x=x,
            y=y,
            x_name=f"{query.band} ({query.group_a}, {key})",
            y_name=f"{query.band} ({query.group_b}, {key})",
        )

    @staticmethod
    def _build_ppc_correlation_bundle(
        query: PPCBandCorrelationQuery,
        dataset: Any,
        key: str,
    ) -> CorrelationSampleBundle:
        df = dataset.long_ppc_dataframe

        SampleBundleFactory._require_columns(
            df,
            ["band", "edge", "value", query.covariate],
            context=f"PPC correlation for band='{query.band}'",
        )

        mask = (df["band"] == query.band) & (df["edge"] == key)
        sub = SampleBundleFactory._subset_and_dropna(
            df,
            mask=mask,
            columns=["value", query.covariate],
        )

        if sub.empty:
            raise ValueError(
                f"No samples found for PPC correlation: "
                f"band='{query.band}', edge='{key}', covariate='{query.covariate}'."
            )

        return CorrelationSampleBundle(
            x=sub["value"],
            y=sub[query.covariate],
            x_name=f"{query.band} ({key})",
            y_name=query.covariate,
        )

    @staticmethod
    def _build_ppc_factorial_bundle(
        query: PPCBandFactorialQuery,
        dataset: Any,
        key: str,
    ) -> SampleBundle:
        df = dataset.long_ppc_dataframe

        SampleBundleFactory._require_columns(
            df,
            ["band", "edge", "value", *query.factor_names],
            context=f"PPC factorial analysis for band='{query.band}'",
        )

        mask = (df["band"] == query.band) & (df["edge"] == key)
        sub = df.loc[mask, ["value", *query.factor_names]]

        return SampleBundleFactory._build_factorial_bundle_from_dataframe(
            df=sub,
            value_col="value",
            factors=query.factor_names,
            dependent_name=f"{query.band} ({key})",
        )