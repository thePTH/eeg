from __future__ import annotations

from dataclasses import dataclass
from abc import ABC
from typing import Any

import pandas as pd

from stats.queries import (
    StatisticalQuery,
    SubjectGroupComparisonQuery,
    EEGFeatureGroupComparisonQuery,
    SubjectCorrelationQuery,
    EEGFeatureCorrelationQuery,
)


@dataclass(frozen=True)
class SampleBundle(ABC):
    """
    Classe abstraite racine pour tous les bundles d'échantillons.

    Tous les bundles concrets doivent exposer :
    - x : premier échantillon / première variable
    - y : second échantillon / seconde variable
    - x_name : nom explicite de x
    - y_name : nom explicite de y

    La propriété `label` permet d'obtenir automatiquement une description
    standardisée du test ou de la relation étudiée.
    """

    @property
    def label(self) -> str:
        """
        Libellé standardisé décrivant ce qui est comparé / corrélé.
        Exemple :
        - "reaction_time (patients) vs reaction_time (controls)"
        - "theta_power (Fz) vs age"
        """
        return f"{self.x_name} vs {self.y_name}"


@dataclass(frozen=True)
class GroupComparisonSampleBundle(SampleBundle):
    """
    Bundle utilisé par les tests de comparaison de groupes :
    t-test, Wilcoxon rank-sum, etc.
    """
    x: pd.Series
    y: pd.Series
    x_name: str
    y_name: str

    @property
    def n_x(self) -> int:
        return len(self.x)

    @property
    def n_y(self) -> int:
        return len(self.y)


@dataclass(frozen=True)
class CorrelationSampleBundle(SampleBundle):
    """
    Bundle utilisé par les tests de corrélation :
    Spearman, Pearson, etc.
    """
    x: pd.Series
    y: pd.Series
    x_name: str
    y_name: str

    @property
    def n_x(self) -> int:
        return len(self.x)

    @property
    def n_y(self) -> int:
        return len(self.y)


class SampleBundleFactory:
    """
    Factory interne.

    Rôle conceptuel :
    - à partir d'une query et d'un dataset,
    - lister les unités d'analyse ("keys")
    - construire le bon SampleBundle pour chaque key

    Cette classe n'est pas censée être utilisée directement par l'utilisateur.
    """

    @staticmethod
    def list_keys(query: StatisticalQuery, dataset: Any) -> list[str]:
        match query:
            case SubjectGroupComparisonQuery():
                return ["subject_level"]

            case SubjectCorrelationQuery():
                return ["subject_level"]

            case EEGFeatureGroupComparisonQuery(scope="single_channel", channel=channel):
                return [channel]

            case EEGFeatureGroupComparisonQuery(scope="all_channels"):
                return list(dataset.ch_names)

            case EEGFeatureCorrelationQuery(scope="single_channel", channel=channel):
                return [channel]

            case EEGFeatureCorrelationQuery(scope="all_channels"):
                return list(dataset.ch_names)

            case _:
                raise ValueError(
                    f"Unsupported query type for key listing: {type(query).__name__}"
                )

    @staticmethod
    def build(query: StatisticalQuery, dataset: Any, key: str) -> SampleBundle:
        match query:
            case SubjectGroupComparisonQuery():
                return SampleBundleFactory._build_subject_group_comparison_bundle(
                    query, dataset
                )

            case SubjectCorrelationQuery():
                return SampleBundleFactory._build_subject_correlation_bundle(
                    query, dataset
                )

            case EEGFeatureGroupComparisonQuery():
                return SampleBundleFactory._build_eeg_group_comparison_bundle(
                    query, dataset, key
                )

            case EEGFeatureCorrelationQuery():
                return SampleBundleFactory._build_eeg_correlation_bundle(
                    query, dataset, key
                )

            case _:
                raise ValueError(
                    f"Unsupported query type for bundle creation: {type(query).__name__}"
                )

    @staticmethod
    def _build_subject_group_comparison_bundle(
        query: SubjectGroupComparisonQuery,
        dataset: Any,
    ) -> GroupComparisonSampleBundle:
        df = dataset.to_subject_dataframe()

        x = df[df[query.group_col] == query.group_a][query.variable]
        y = df[df[query.group_col] == query.group_b][query.variable]

        if x.empty or y.empty:
            raise ValueError(
                f"No samples found for subject-level comparison: variable={query.variable}"
            )

        return GroupComparisonSampleBundle(
            x=x,
            y=y,
            x_name=f"{query.variable} ({query.group_a})",
            y_name=f"{query.variable} ({query.group_b})",
        )

    @staticmethod
    def _build_subject_correlation_bundle(
        query: SubjectCorrelationQuery,
        dataset: Any,
    ) -> CorrelationSampleBundle:
        df = dataset.to_subject_dataframe()

        x = df[query.x_variable]
        y = df[query.y_variable]

        if x.empty or y.empty:
            raise ValueError(
                f"No samples found for subject-level correlation: "
                f"{query.x_variable} vs {query.y_variable}"
            )

        return CorrelationSampleBundle(
            x=x,
            y=y,
            x_name=query.x_variable,
            y_name=query.y_variable,
        )

    @staticmethod
    def _build_eeg_group_comparison_bundle(
        query: EEGFeatureGroupComparisonQuery,
        dataset: Any,
        key: str,
    ) -> GroupComparisonSampleBundle:
        df = dataset.to_long_dataframe()
        df = df[df["feature"] == query.feature]
        df = df[df["channel"] == key]

        x = df[df[query.group_col] == query.group_a]["value"]
        y = df[df[query.group_col] == query.group_b]["value"]

        if x.empty or y.empty:
            raise ValueError(
                f"No samples found for EEG group comparison: "
                f"feature={query.feature}, channel={key}"
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
        df_eeg = dataset.to_long_dataframe()
        df_eeg = df_eeg[df_eeg["feature"] == query.feature]
        df_eeg = df_eeg[df_eeg["channel"] == key]

        df_subjects = dataset.to_subject_dataframe()[["subject_id", query.covariate]]

        merged = df_eeg[["subject_id", "value"]].merge(
            df_subjects,
            on="subject_id",
            how="inner",
        )

        x = merged["value"]
        y = merged[query.covariate]

        if x.empty or y.empty:
            raise ValueError(
                f"No samples found for EEG correlation: "
                f"feature={query.feature}, covariate={query.covariate}, channel={key}"
            )

        return CorrelationSampleBundle(
            x=x,
            y=y,
            x_name=f"{query.feature} ({key})",
            y_name=query.covariate,
        )