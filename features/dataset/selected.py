from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Iterable

import pandas as pd

from .base import FeaturesDataset
from features.name import FeatureNameHelper

if TYPE_CHECKING:
    from .participant import SingleParticipantProcessedFeatureDataset


@dataclass(frozen=True)
class SelectedFeature:
    """
    Représente une feature logique effectivement sélectionnée dans un
    `SelectedFeaturesDataset`.

    Exemples
    --------
    - "variance"      -> plusieurs colonnes EEG
    - "cn_alpha"      -> plusieurs colonnes de connectivité
    - "subject_age"   -> une colonne explicative sujet
    """

    name: str
    columns: list[str]

    def __post_init__(self) -> None:
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
        return self.name
    

class SelectedFeaturesConcatEngine:

    @staticmethod
    def concat_columns(selected_features:list[SelectedFeature]) ->list[str]:
        cols = []
        for selected_feature in selected_features:
            cols += selected_feature.columns
        return cols



class SelectedFeaturesDataset(FeaturesDataset):
    """
    Vue restreinte d'un `FeaturesDataset`.

    Cette classe représente un dataset dont seules certaines features logiques
    ont été retenues pour le machine learning.

    Important
    ---------
    - L'initialisation se fait à partir de `selected_features`.
    - `selected_columns` est entièrement déduit de `selected_features`.
    - `wide_dataframe` n'est pas modifié : seule la vue `X` est restreinte.
    """

    def __init__(
        self,
        participant_datasets: list["SingleParticipantProcessedFeatureDataset"],
        selected_features: list[SelectedFeature],
    ):
        """
        Parameters
        ----------
        participant_datasets:
            Liste non vide de datasets sujet-level.

        selected_features:
            Features logiques effectivement retenues pour le ML.
        """
        super().__init__(participant_datasets)

        self._selected_features = selected_features





    # ==========================================================================
    # Représentation des features sélectionnées
    # ==========================================================================

    @property
    def selected_features(self) -> list[SelectedFeature]:
        """
        Features logiques effectivement retenues dans cette vue restreinte.
        """
        return self._selected_features

    @property
    def selected_feature_names(self) -> list[str]:
        """
        Noms des features logiques effectivement retenues.
        """
        return [feature.name for feature in self._selected_features]

    @property
    def selected_columns(self) -> list[str]:
        """
        Colonnes wide effectivement retenues dans cette vue restreinte.

        Cette propriété est entièrement déduite de `selected_features`.
        """
        columns: list[str] = []
        for feature in self._selected_features:
            columns.extend(feature.columns)
        return columns
    
    # ==========================================================================
    # Colonnes effectivement retenues pour le ML
    # ==========================================================================

    @cached_property
    def X(self) -> pd.DataFrame:
        """
        Matrice explicative restreinte aux colonnes sélectionnées.
        """
        return self.wide_dataframe[self.selected_columns]

    @cached_property
    def all_feature_names(self) -> list[str]:
        """
        Liste exacte des colonnes wide retenues dans cette vue restreinte.

        Remarque
        --------
        On conserve ici la convention historique du projet où
        `all_feature_names` correspond aux noms de colonnes wide disponibles
        dans la vue dataset courante.
        """
        return list(self.selected_columns)

    # ==========================================================================
    # Familles métier encore présentes
    # ==========================================================================

    @cached_property
    def scalar_feature_names(self) -> list[str]:
        """
        Familles de features scalaires encore présentes dans la sélection.
        """
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
        """
        Familles de connectivité encore présentes dans la sélection.
        """
        return [
            feature.name
            for feature in self._selected_features
            if feature.name.startswith(self.CONNECTIVITY_PREFIX)
        ]

    @cached_property
    def subject_feature_names(self) -> list[str]:
        """
        Features sujet encore présentes dans la sélection.
        """
        return [
            feature.name
            for feature in self._selected_features
            if feature.name in self.SUBJECT_FEATURE_COLUMNS
        ]

    @property
    def feature_names(self) -> list[str]:
        """
        Noms de familles métier encore disponibles dans cette vue restreinte.
        """
        return list(self.selected_feature_names)

    def select_rows(self, row_indices) -> "SelectedFeaturesDataset":
        """
        Construit une sous-vue ligne par ligne en conservant exactement les
        mêmes features sélectionnées.
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


class SelectedFeaturesDatasetFactory:
    """
    Factory de construction de `SelectedFeaturesDataset`.

    Cette factory transforme une sélection exprimée au niveau métier
    (`feature_family_names`, `channels`, `edges`) en objets `SelectedFeature`.
    """

    @staticmethod
    def _unique_preserve_order(values: Iterable[str]) -> list[str]:
        """
        Supprime les doublons en conservant l'ordre d'apparition.
        """
        return list(dict.fromkeys(values))

    @staticmethod
    def _group_columns_by_feature_name(columns: list[str]) -> list[SelectedFeature]:
        """
        Regroupe des colonnes wide par feature logique.

        Règles
        ------
        - subject_age         -> feature "subject_age"
        - cn_alpha_Fp1_Fp2    -> feature "cn_alpha"
        - Fp1_entropy         -> feature "entropy"
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
        Construit un `SelectedFeaturesDataset` à partir d'une liste de colonnes
        wide déjà résolues.
        """
        if selected_columns is None:
            raise ValueError("`selected_columns` cannot be None.")

        selected_columns = cls._unique_preserve_order(selected_columns)
        if not selected_columns:
            raise ValueError("`selected_columns` cannot be empty.")

        available_columns = set(dataset.wide_dataframe.columns)
        missing = [column for column in selected_columns if column not in available_columns]
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
        """
        Construit un `SelectedFeaturesDataset` à partir d'une sélection métier.
        """
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
        Construit un `SelectedFeaturesDataset` directement à partir
        d'une liste de `SelectedFeature`.
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
    Helper bas niveau pour construire un `SelectedFeaturesDataset`
    à partir d'une sélection métier.

    Cette classe travaille au niveau des colonnes réelles de `wide_dataframe`,
    contrairement à d'autres helpers qui peuvent travailler au niveau des
    familles métier (`entropy`, `cn_alpha`, `subject_age`, etc.).
    """

    @staticmethod
    def select(
        dataset: FeaturesDataset,
        feature_family_names: list[str],
        channels: list[str] = None,
        edges: list[str] = None,
    ) -> SelectedFeaturesDataset:
        """
        Sélectionne un sous-dataset à partir de familles de features.

        API publique conservée volontairement inchangée.
        """
        return SelectedFeaturesDatasetFactory.from_feature_family_names(
            dataset=dataset,
            feature_family_names=feature_family_names,
            channels=channels,
            edges=edges,
        )