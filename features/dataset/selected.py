from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Iterable

import pandas as pd

from features.name import FeatureNameHelper

from .base import FeaturesDataset

if TYPE_CHECKING:
    from .participant import SingleParticipantProcessedFeatureDataset


@dataclass(frozen=True)
class SelectedFeature:
    """
    Represent a logical feature selected in a SelectedFeaturesDataset.

    Examples
    --------
    - "variance"    -> several EEG columns
    - "cn_alpha"    -> several connectivity columns
    - "subject_age" -> one subject-level explanatory column
    """

    name: str
    columns: list[str]

    def __post_init__(self) -> None:
        """Validate and normalize selected feature columns."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("`name` must be a non-empty string.")

        if not isinstance(self.columns, list) or len(self.columns) == 0:
            raise ValueError(
                f"`columns` for selected feature '{self.name}' must be a non-empty list."
            )

        cleaned_columns: list[str] = []

        for column in self.columns:
            if not isinstance(column, str) or not column.strip():
                raise ValueError(
                    f"All columns for selected feature '{self.name}' "
                    "must be non-empty strings."
                )

            cleaned_columns.append(column.strip())

        unique_columns = list(dict.fromkeys(cleaned_columns))
        object.__setattr__(self, "columns", unique_columns)

    def __repr__(self):
        """Return the selected feature name."""
        return self.name


class SelectedFeaturesConcatEngine:
    """Utility class used to concatenate selected feature columns."""

    @staticmethod
    def concat_columns(selected_features: list[SelectedFeature]) -> list[str]:
        """Concatenate all columns from selected features."""
        cols = []

        for selected_feature in selected_features:
            cols += selected_feature.columns

        return cols


class SelectedFeaturesDataset(FeaturesDataset):
    """
    Restricted view of a FeaturesDataset.

    This class represents a dataset where only selected logical features are
    kept for machine learning.

    Important
    ---------
    - Initialization is based on ``selected_features``.
    - ``selected_columns`` is entirely derived from ``selected_features``.
    - ``wide_dataframe`` is not modified; only the ``X`` view is restricted.
    """

    def __init__(
        self,
        participant_datasets: list["SingleParticipantProcessedFeatureDataset"],
        selected_features: list[SelectedFeature],
    ):
        """
        Initialize a selected-features dataset.

        Parameters
        ----------
        participant_datasets
            Non-empty list of subject-level datasets.
        selected_features
            Logical features selected for machine learning.
        """
        super().__init__(participant_datasets)

        self._selected_features = selected_features

    @property
    def selected_features(self) -> list[SelectedFeature]:
        """Return the logical features selected in this restricted view."""
        return self._selected_features

    @property
    def selected_feature_names(self) -> list[str]:
        """Return the names of the selected logical features."""
        return [
            feature.name
            for feature in self._selected_features
        ]

    @property
    def selected_columns(self) -> list[str]:
        """
        Return the wide columns selected in this restricted view.

        This property is entirely derived from ``selected_features``.
        """
        columns: list[str] = []

        for feature in self._selected_features:
            columns.extend(feature.columns)

        return columns

    @cached_property
    def X(self) -> pd.DataFrame:
        """Return the explanatory matrix restricted to selected columns."""
        return self.wide_dataframe[self.selected_columns]

    @cached_property
    def all_feature_names(self) -> list[str]:
        """
        Return the exact wide columns kept in this restricted view.

        The project convention is preserved: ``all_feature_names`` corresponds
        to the available wide column names in the current dataset view.
        """
        return list(self.selected_columns)

    @cached_property
    def scalar_feature_names(self) -> list[str]:
        """Return scalar feature families still present in the selection."""
        return [
            feature.name
            for feature in self._selected_features
            if (
                not feature.name.startswith(self.CONNECTIVITY_PREFIX)
                and feature.name not in self.SUBJECT_FEATURE_COLUMNS
            )
        ]

    @cached_property
    def connectivity_feature_names(self) -> list[str]:
        """Return connectivity feature families still present in the selection."""
        return [
            feature.name
            for feature in self._selected_features
            if feature.name.startswith(self.CONNECTIVITY_PREFIX)
        ]

    @cached_property
    def subject_feature_names(self) -> list[str]:
        """Return subject-level features still present in the selection."""
        return [
            feature.name
            for feature in self._selected_features
            if feature.name in self.SUBJECT_FEATURE_COLUMNS
        ]

    @property
    def feature_names(self) -> list[str]:
        """Return domain-level feature family names available in this view."""
        return list(self.selected_feature_names)

    def select_rows(self, row_indices) -> "SelectedFeaturesDataset":
        """
        Build a row-level sub-view while preserving the same selected features.
        """
        if row_indices is None:
            raise ValueError("`row_indices` cannot be None.")

        selected_participants = [
            self.participant_datasets[int(i)]
            for i in row_indices
        ]

        if not selected_participants:
            raise ValueError("Row selection produced an empty dataset.")

        return SelectedFeaturesDataset(
            participant_datasets=selected_participants,
            selected_features=self.selected_features,
        )

    @cached_property
    def X_and_eeg(self) -> pd.DataFrame:
        """
        Return a DataFrame useful for debugging and visualization.

        It contains:
        - explanatory columns from X;
        - an ``eeg`` column containing lazy EEGProcessedData objects.
        """
        return pd.concat(
            [
                self.X.reset_index(drop=True),
                pd.Series(self.eegs, name="eeg"),
            ],
            axis=1,
        )


class SelectedFeaturesDatasetFactory:
    """
    Factory used to build SelectedFeaturesDataset objects.

    It converts a domain-level selection expressed through feature families,
    channels, and edges into SelectedFeature objects.
    """

    @staticmethod
    def _unique_preserve_order(values: Iterable[str]) -> list[str]:
        """Remove duplicates while preserving insertion order."""
        return list(dict.fromkeys(values))

    @staticmethod
    def _group_columns_by_feature_name(columns: list[str]) -> list[SelectedFeature]:
        """
        Group wide columns by logical feature.

        Rules
        -----
        - subject_age      -> feature "subject_age"
        - cn_alpha_Fp1_Fp2 -> feature "cn_alpha"
        - Fp1_entropy      -> feature "entropy"
        """
        if not isinstance(columns, list):
            raise TypeError("`columns` must be a list of strings.")

        if len(columns) == 0:
            raise ValueError("`columns` cannot be empty.")

        grouped: dict[str, list[str]] = {}
        order: list[str] = []

        for column in columns:
            if not isinstance(column, str) or not column.strip():
                raise ValueError("All `columns` must be non-empty strings.")

            column = column.strip()

            if column.startswith("subject_"):
                feature_name = column

            elif column.startswith("cn_"):
                parts = column.split("_")

                if len(parts) < 4:
                    raise ValueError(f"Invalid connectivity column format: '{column}'")

                feature_name = f"{parts[0]}_{parts[1]}"

            else:
                parts = column.split("_", 1)

                if len(parts) != 2:
                    raise ValueError(f"Invalid scalar column format: '{column}'")

                feature_name = parts[1]

            if feature_name not in grouped:
                grouped[feature_name] = []
                order.append(feature_name)

            grouped[feature_name].append(column)

        return [
            SelectedFeature(
                name=feature_name,
                columns=SelectedFeaturesDatasetFactory._unique_preserve_order(
                    grouped[feature_name]
                ),
            )
            for feature_name in order
        ]

    @classmethod
    def from_selected_columns(
        cls,
        dataset: FeaturesDataset,
        selected_columns: list[str],
    ) -> SelectedFeaturesDataset:
        """
        Build a SelectedFeaturesDataset from already resolved wide columns.
        """
        if selected_columns is None:
            raise ValueError("`selected_columns` cannot be None.")

        selected_columns = cls._unique_preserve_order(selected_columns)

        if not selected_columns:
            raise ValueError("`selected_columns` cannot be empty.")

        available_columns = set(dataset.wide_dataframe.columns)
        missing = [
            column
            for column in selected_columns
            if column not in available_columns
        ]

        if missing:
            raise KeyError(
                "Some selected columns do not exist in dataset.wide_dataframe: "
                f"{missing[:10]}"
            )

        selected_features = cls._group_columns_by_feature_name(selected_columns)

        return SelectedFeaturesDataset(
            participant_datasets=dataset.participant_datasets,
            selected_features=selected_features,
        )

    @classmethod
    def from_feature_family_names(
        cls,
        dataset: FeaturesDataset,
        feature_family_names: list[str],
        channels: list[str] | None = None,
        edges: list[str] | None = None,
    ) -> SelectedFeaturesDataset:
        """Build a SelectedFeaturesDataset from a domain-level feature selection."""
        name_factory = FeatureNameHelper(available_features=dataset.all_feature_names)
        selection = name_factory.build(
            family_names=feature_family_names,
            channels=channels,
            edges=edges,
        )

        if selection is None:
            raise ValueError("`selection` cannot be None.")

        return cls.from_selected_columns(
            dataset=dataset,
            selected_columns=selection,
        )

    @classmethod
    def from_selected_features_list(
        cls,
        dataset: FeaturesDataset,
        selected_features: list[SelectedFeature],
    ) -> SelectedFeaturesDataset:
        """
        Build a SelectedFeaturesDataset directly from a list of SelectedFeature objects.
        """
        if selected_features is None:
            raise ValueError("`selected_features` cannot be None.")

        if not isinstance(selected_features, list) or len(selected_features) == 0:
            raise ValueError("`selected_features` must be a non-empty list.")

        available_columns = set(dataset.wide_dataframe.columns)

        missing_columns = [
            col
            for feature in selected_features
            for col in feature.columns
            if col not in available_columns
        ]

        if missing_columns:
            raise KeyError(
                "Some columns from selected_features do not exist in dataset.wide_dataframe: "
                f"{missing_columns[:10]}"
            )

        return SelectedFeaturesDataset(
            participant_datasets=dataset.participant_datasets,
            selected_features=selected_features,
        )


class FeaturesDatasetSelector:
    """
    Low-level helper used to build a SelectedFeaturesDataset from a domain selection.

    This class works at the real ``wide_dataframe`` column level, unlike other
    helpers that may work at the domain-family level, such as entropy,
    cn_alpha, or subject_age.
    """

    @staticmethod
    def select(
        dataset: FeaturesDataset,
        feature_family_names: list[str],
        channels: list[str] = None,
        edges: list[str] = None,
    ) -> SelectedFeaturesDataset:
        """
        Select a dataset subset from feature families.

        The public API is intentionally kept unchanged.
        """
        return SelectedFeaturesDatasetFactory.from_feature_family_names(
            dataset=dataset,
            feature_family_names=feature_family_names,
            channels=channels,
            edges=edges,
        )