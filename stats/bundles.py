from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any

import pandas as pd

from stats.queries import (
    EEGFeatureCorrelationQuery,
    EEGFeatureGroupComparisonQuery,
    PPCBandCorrelationQuery,
    PPCBandGroupComparisonQuery,
    PSDBandCorrelationQuery,
    PSDBandGroupComparisonQuery,
    StatisticalQuery,
    SubjectCorrelationQuery,
    SubjectGroupComparisonQuery,
)


@dataclass(frozen=True)
class SampleBundle(ABC):
    """
    Bundle générique d'échantillons prêt à être injecté dans un engine.
    """

    @property
    def label(self) -> str:
        return f"{self.x_name} vs {self.y_name}"


@dataclass(frozen=True)
class GroupComparisonSampleBundle(SampleBundle):
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


@dataclass(frozen=True)
class CorrelationSampleBundle(SampleBundle):
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


class SampleBundleFactory:
    """
    Factory centrale qui transforme :
    - une query métier
    - un dataset
    en bundles exploitables par les engines statistiques.
    """

    @staticmethod
    def list_keys(query: StatisticalQuery, dataset: Any) -> list[str]:
        match query:
            # ==========================================================
            # SUBJECT
            # ==========================================================
            case SubjectGroupComparisonQuery():
                return ["subject_level"]

            case SubjectCorrelationQuery():
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

            case _:
                raise ValueError(
                    f"Unsupported query type for key listing: {type(query).__name__}"
                )

    @staticmethod
    def build(query: StatisticalQuery, dataset: Any, key: str) -> SampleBundle:
        match query:
            # ==========================================================
            # SUBJECT
            # ==========================================================
            case SubjectGroupComparisonQuery():
                return SampleBundleFactory._build_subject_group_comparison_bundle(query, dataset)

            case SubjectCorrelationQuery():
                return SampleBundleFactory._build_subject_correlation_bundle(query, dataset)

            # ==========================================================
            # EEG FEATURES
            # ==========================================================
            case EEGFeatureGroupComparisonQuery():
                return SampleBundleFactory._build_eeg_group_comparison_bundle(query, dataset, key)

            case EEGFeatureCorrelationQuery():
                return SampleBundleFactory._build_eeg_correlation_bundle(query, dataset, key)

            # ==========================================================
            # PSD
            # ==========================================================
            case PSDBandGroupComparisonQuery():
                return SampleBundleFactory._build_psd_group_comparison_bundle(query, dataset, key)

            case PSDBandCorrelationQuery():
                return SampleBundleFactory._build_psd_correlation_bundle(query, dataset, key)

            # ==========================================================
            # PPC
            # ==========================================================
            case PPCBandGroupComparisonQuery():
                return SampleBundleFactory._build_ppc_group_comparison_bundle(query, dataset, key)

            case PPCBandCorrelationQuery():
                return SampleBundleFactory._build_ppc_correlation_bundle(query, dataset, key)

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
                f"No samples found for subject-level correlation: {query.x_variable} vs {query.y_variable}"
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
                f"No samples found for EEG group comparison: feature={query.feature}, channel={key}"
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
                f"No samples found for EEG correlation: feature={query.feature}, channel={key}, covariate={query.covariate}"
            )

        return CorrelationSampleBundle(
            x=x,
            y=y,
            x_name=f"{query.feature} ({key})",
            y_name=query.covariate,
        )

    @staticmethod
    def _build_psd_group_comparison_bundle(
        query: PSDBandGroupComparisonQuery,
        dataset: Any,
        key: str,
    ) -> GroupComparisonSampleBundle:
        df = dataset.to_long_psd_dataframe()
        df = df[df["band"] == query.band]
        df = df[df["channel"] == key]

        x = df[df[query.group_col] == query.group_a]["value"]
        y = df[df[query.group_col] == query.group_b]["value"]

        if x.empty or y.empty:
            raise ValueError(
                f"No samples found for PSD group comparison: band={query.band}, channel={key}"
            )

        return GroupComparisonSampleBundle(
            x=x,
            y=y,
            x_name=f"PSD-{query.band} ({query.group_a}, {key})",
            y_name=f"PSD-{query.band} ({query.group_b}, {key})",
        )

    @staticmethod
    def _build_psd_correlation_bundle(
        query: PSDBandCorrelationQuery,
        dataset: Any,
        key: str,
    ) -> CorrelationSampleBundle:
        df_psd = dataset.to_long_psd_dataframe()
        df_psd = df_psd[df_psd["band"] == query.band]
        df_psd = df_psd[df_psd["channel"] == key]

        df_subjects = dataset.to_subject_dataframe()[["subject_id", query.covariate]]

        merged = df_psd[["subject_id", "value"]].merge(
            df_subjects,
            on="subject_id",
            how="inner",
        )

        x = merged["value"]
        y = merged[query.covariate]

        if x.empty or y.empty:
            raise ValueError(
                f"No samples found for PSD correlation: band={query.band}, channel={key}, covariate={query.covariate}"
            )

        return CorrelationSampleBundle(
            x=x,
            y=y,
            x_name=f"PSD-{query.band} ({key})",
            y_name=query.covariate,
        )

    @staticmethod
    def _build_ppc_group_comparison_bundle(
        query: PPCBandGroupComparisonQuery,
        dataset: Any,
        key: str,
    ) -> GroupComparisonSampleBundle:
        df = dataset.to_long_ppc_dataframe()
        df = df[df["band"] == query.band]
        df = df[df["edge"] == key]

        x = df[df[query.group_col] == query.group_a]["value"]
        y = df[df[query.group_col] == query.group_b]["value"]

        if x.empty or y.empty:
            raise ValueError(
                f"No samples found for PPC group comparison: band={query.band}, edge={key}"
            )

        return GroupComparisonSampleBundle(
            x=x,
            y=y,
            x_name=f"PPC-{query.band} ({query.group_a}, {key})",
            y_name=f"PPC-{query.band} ({query.group_b}, {key})",
        )

    @staticmethod
    def _build_ppc_correlation_bundle(
        query: PPCBandCorrelationQuery,
        dataset: Any,
        key: str,
    ) -> CorrelationSampleBundle:
        df_ppc = dataset.to_long_ppc_dataframe()
        df_ppc = df_ppc[df_ppc["band"] == query.band]
        df_ppc = df_ppc[df_ppc["edge"] == key]

        df_subjects = dataset.to_subject_dataframe()[["subject_id", query.covariate]]

        merged = df_ppc[["subject_id", "value"]].merge(
            df_subjects,
            on="subject_id",
            how="inner",
        )

        x = merged["value"]
        y = merged[query.covariate]

        if x.empty or y.empty:
            raise ValueError(
                f"No samples found for PPC correlation: band={query.band}, edge={key}, covariate={query.covariate}"
            )

        return CorrelationSampleBundle(
            x=x,
            y=y,
            x_name=f"PPC-{query.band} ({key})",
            y_name=query.covariate,
        )